from __future__ import annotations

"""Risk Analyst Agent — predicts probability of default."""

import autogen

from config.agent_prompts import RISK_ANALYST_PROMPT
from config.settings import settings


def build_risk_analyst(llm_config: dict | None = None) -> autogen.AssistantAgent:
    """Build and return the Risk Analyst AssistantAgent.

    Tool registration (both LLM schema + execution) is handled centrally
    in agents/orchestrator.py to avoid the executor=None crash.
    """
    cfg = llm_config or settings.llm_config

    agent = autogen.AssistantAgent(
        name="RiskAnalyst",
        system_message=RISK_ANALYST_PROMPT,
        llm_config=cfg,
    )

    return agent
