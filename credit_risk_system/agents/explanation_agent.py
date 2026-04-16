from __future__ import annotations

"""Explanation Agent — generates SHAP-based narratives."""

import autogen

from config.agent_prompts import EXPLANATION_AGENT_PROMPT
from config.settings import settings


def build_explanation_agent(llm_config: dict | None = None) -> autogen.AssistantAgent:
    cfg = llm_config or settings.llm_config

    agent = autogen.AssistantAgent(
        name="ExplanationAgent",
        system_message=EXPLANATION_AGENT_PROMPT,
        llm_config=cfg,
    )

    return agent
