# prompts — per-node prompt modules (canonical).
#
# Import individual prompts from their node module:
#   from src.workflow.prompts.planner import PLANNER_PROMPT_V2
# This package root re-exports all of them for convenience.

from src.workflow.prompts.delegation import DELEGATION_PROMPT
from src.workflow.prompts.extraction import (
    DEPTH_DECIDER_PROMPT,
    OVERVIEW_PROMPT,
    REPORT_COMPOSE_PROMPT,
)
from src.workflow.prompts.navigation import NAVIGATION_AGENT_PROMPT
from src.workflow.prompts.planner import PLANNER_PROMPT_V2
from src.workflow.prompts.verify import VERIFY_PROMPT

__all__ = [
    "PLANNER_PROMPT_V2",
    "DELEGATION_PROMPT",
    "VERIFY_PROMPT",
    "NAVIGATION_AGENT_PROMPT",
    "DEPTH_DECIDER_PROMPT",
    "OVERVIEW_PROMPT",
    "REPORT_COMPOSE_PROMPT",
]
