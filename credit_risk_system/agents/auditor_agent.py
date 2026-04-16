from __future__ import annotations

"""Auditor Agent — validates fairness, consistency, and persists decisions."""

import autogen

from config.agent_prompts import AUDITOR_AGENT_PROMPT
from config.settings import settings


def build_auditor_agent(llm_config: dict | None = None) -> autogen.AssistantAgent:
    cfg = llm_config or settings.llm_config

    agent = autogen.AssistantAgent(
        name="AuditorAgent",
        system_message=AUDITOR_AGENT_PROMPT,
        llm_config=cfg,
    )

    return agent
