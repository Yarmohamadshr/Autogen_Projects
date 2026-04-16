from __future__ import annotations

"""Policy Agent — enforces lending rules."""

import autogen

from config.agent_prompts import POLICY_AGENT_PROMPT
from config.settings import settings


def build_policy_agent(llm_config: dict | None = None) -> autogen.AssistantAgent:
    cfg = llm_config or settings.llm_config

    agent = autogen.AssistantAgent(
        name="PolicyAgent",
        system_message=POLICY_AGENT_PROMPT,
        llm_config=cfg,
    )

    return agent
