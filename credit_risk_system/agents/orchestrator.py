"""
Multi-agent orchestrator.

Builds the GroupChat with a fixed speaker order:
  Orchestrator → RiskAnalyst → PolicyAgent → ExplanationAgent → AuditorAgent → Orchestrator (terminate)

Tool execution runs on the UserProxyAgent (Orchestrator). All AssistantAgents
only have LLM-side tool registration (schema awareness, not execution).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import autogen

from agents.auditor_agent import build_auditor_agent
from agents.explanation_agent import build_explanation_agent
from agents.policy_agent import build_policy_agent
from agents.risk_analyst import build_risk_analyst
from config.settings import settings
from tools.audit_tools import (
    audit_decision_fairness,
    finalize_decision,
    set_db_session,
    validate_decision_consistency,
)
from tools.explanation_tools import format_decision_letter, generate_shap_explanation
from tools.policy_tools import check_lending_policy, get_policy_thresholds
from tools.risk_tools import get_model_metadata, predict_default_probability

logger = logging.getLogger(__name__)

# Speaker rotation order (indices into groupchat.agents list, excluding orchestrator)
_AGENT_ORDER = ["RiskAnalyst", "PolicyAgent", "ExplanationAgent", "AuditorAgent"]


def run_evaluation(
    application: dict,
    db_session=None,
    llm_config: dict | None = None,
) -> dict:
    """
    Run the full 4-agent evaluation pipeline for a single loan application.

    Args:
        application: dict containing all loan application fields
        db_session: SQLAlchemy session (passed to audit tools for DB writes)
        llm_config: optional override for AutoGen LLM config

    Returns:
        Parsed decision dict extracted from AuditorAgent's final message.
    """
    cfg = llm_config or settings.llm_config

    # Inject DB session into audit tools
    set_db_session(db_session)

    # ── Build agents ───────────────────────────────────────────────────────────
    risk_analyst = build_risk_analyst(cfg)
    policy_agent = build_policy_agent(cfg)
    explanation_agent = build_explanation_agent(cfg)
    auditor_agent = build_auditor_agent(cfg)

    orchestrator = autogen.UserProxyAgent(
        name="Orchestrator",
        is_termination_msg=_is_termination_msg,
        human_input_mode="TERMINATE" if settings.HUMAN_IN_LOOP else "NEVER",
        max_consecutive_auto_reply=settings.MAX_ROUNDS,
        code_execution_config={"use_docker": False},
    )

    # ── Register all tools for execution on the UserProxy ─────────────────────
    _all_tools = [
        (predict_default_probability, "Predict default probability using XGBoost model."),
        (get_model_metadata, "Return model metadata."),
        (check_lending_policy, "Check lending policy rules."),
        (get_policy_thresholds, "Return current policy thresholds."),
        (generate_shap_explanation, "Compute SHAP values and build narrative."),
        (format_decision_letter, "Format customer-facing decision letter."),
        (audit_decision_fairness, "Check for demographic bias."),
        (validate_decision_consistency, "Validate decision consistency."),
        (finalize_decision, "Persist decision to database."),
    ]

    _agent_map = {
        "RiskAnalyst": risk_analyst,
        "PolicyAgent": policy_agent,
        "ExplanationAgent": explanation_agent,
        "AuditorAgent": auditor_agent,
    }

    for tool_fn, desc in _all_tools:
        autogen.register_function(
            tool_fn,
            caller=_get_caller_for_tool(tool_fn.__name__, _agent_map),
            executor=orchestrator,
            description=desc,
        )

    # ── Build GroupChat ────────────────────────────────────────────────────────
    groupchat = autogen.GroupChat(
        agents=[orchestrator, risk_analyst, policy_agent, explanation_agent, auditor_agent],
        messages=[],
        max_round=settings.MAX_ROUNDS * 2,
        speaker_selection_method=_make_speaker_selector(orchestrator, _agent_map),
        allow_repeat_speaker=False,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=cfg)

    # ── Seed message ──────────────────────────────────────────────────────────
    seed_message = (
        "Please evaluate the following loan application. "
        "Each agent should call their assigned tools and report results in JSON.\n\n"
        f"APPLICATION:\n{json.dumps(application, indent=2)}"
    )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    orchestrator.initiate_chat(manager, message=seed_message)

    # ── Extract final decision from last AuditorAgent message ─────────────────
    return _extract_final_decision(groupchat.messages)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_termination_msg(message: dict) -> bool:
    content = message.get("content") or ""
    return "DECISION_COMPLETE" in content


def _make_speaker_selector(orchestrator: autogen.UserProxyAgent, agent_map: dict):
    """
    Return a speaker_selection_method callback that enforces fixed round-robin order
    while correctly routing tool calls through the Orchestrator for execution.

    Flow per agent:
      Agent → (if tool_calls) → Orchestrator (execute) → Agent (process result) → next Agent
    """

    def selector(last_speaker: Any, groupchat: autogen.GroupChat) -> autogen.ConversableAgent:
        agents_by_name = {a.name: a for a in groupchat.agents}
        messages = groupchat.messages
        last_msg = messages[-1] if messages else {}

        # ── Priority 1: pending tool calls → Orchestrator must execute ──────────
        if last_msg.get("tool_calls"):
            return orchestrator

        # ── Priority 2: Orchestrator just executed → back to caller agent ───────
        if last_speaker.name == "Orchestrator":
            # Search backwards for the most recent agent message with tool_calls
            for msg in reversed(messages[:-1]):
                sender = msg.get("name", "")
                if sender in _AGENT_ORDER and msg.get("tool_calls"):
                    return agents_by_name[sender]
            # No pending tool calls found → start the round-robin from RiskAnalyst
            return agents_by_name["RiskAnalyst"]

        # ── Priority 3: Normal round-robin after agent sends a text response ────
        try:
            idx = _AGENT_ORDER.index(last_speaker.name)
        except ValueError:
            return orchestrator

        if idx < len(_AGENT_ORDER) - 1:
            return agents_by_name[_AGENT_ORDER[idx + 1]]
        else:
            return orchestrator

    return selector


def _get_caller_for_tool(tool_name: str, agent_map: dict) -> autogen.AssistantAgent:
    """Map tool function name → the agent that 'owns' it (for LLM schema registration)."""
    risk_tools = {"predict_default_probability", "get_model_metadata"}
    policy_tools = {"check_lending_policy", "get_policy_thresholds"}
    explanation_tools = {"generate_shap_explanation", "format_decision_letter"}
    audit_tools = {"audit_decision_fairness", "validate_decision_consistency", "finalize_decision"}

    if tool_name in risk_tools:
        return agent_map["RiskAnalyst"]
    elif tool_name in policy_tools:
        return agent_map["PolicyAgent"]
    elif tool_name in explanation_tools:
        return agent_map["ExplanationAgent"]
    elif tool_name in audit_tools:
        return agent_map["AuditorAgent"]
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


def _extract_final_decision(messages: list[dict]) -> dict:
    """
    Parse the final decision from the agent conversation.

    Priority:
    1. AuditorAgent text message containing a JSON block
    2. Orchestrator tool_responses containing a successful finalize_decision result
    3. Reconstruct from individual tool results scattered through the conversation
    """
    # ── Priority 1: AuditorAgent text with JSON block ──────────────────────────
    for msg in reversed(messages):
        if msg.get("name") == "AuditorAgent":
            content = msg.get("content", "") or ""
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

    # ── Priority 2: finalize_decision tool result in Orchestrator messages ─────
    finalize_result = None
    for msg in reversed(messages):
        for tool_resp in msg.get("tool_responses", []):
            content = tool_resp.get("content", "")
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "decision_id" in data and data.get("status") == "persisted":
                    finalize_result = data
                    break
            except (json.JSONDecodeError, AttributeError):
                pass
        if finalize_result:
            break

    if finalize_result:
        # Reconstruct a minimal result dict from what we know
        result = dict(finalize_result)
        # Pull policy_decision, probability_of_default, narrative, etc. from tool call args
        for msg in messages:
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                if fn.get("name") == "finalize_decision":
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                        result.update({k: args[k] for k in (
                            "policy_decision", "probability_of_default", "narrative",
                            "risk_tier", "violations", "recommended_rate", "shap_top_features",
                        ) if k in args})
                    except (json.JSONDecodeError, TypeError):
                        pass
        logger.info("Extracted decision from finalize_decision tool result: %s", result.get("decision_id"))
        return result

    logger.warning("Could not parse final decision from conversation. Returning raw last message.")
    return {"raw_last_message": messages[-1] if messages else {}, "parse_error": True}
