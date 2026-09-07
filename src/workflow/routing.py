# routing.py — pure routing functions for the delegated architecture graph.
# No LLM, no I/O — deterministic decisions based on AgentState only.

from langgraph.graph import END

from logs import logger
from src.workflow.agent_state import AgentState


def route_from_delegation(state: AgentState) -> str:
    """Route after the delegation node.

    - "finish" -> END
    - max steps reached -> END (global guard)
    - "navigation" / "extract_information" / "wait_for_user" -> respective node
    - anything else -> END with a logged warning (defensive fallback)
    """
    route = state.get("agent_decision", "")
    logger.info(f"[ROUTING] delegation decision: {route}")

    if route == "finish":
        return END

    if state.get("steps", 0) >= state.get("max_steps", 30):
        logger.warning("[ROUTING] Max steps reached, ending run")
        return END

    if route == "navigation":
        return "navigation"
    if route == "extract_information":
        return "extract_information"
    if route == "wait_for_user":
        return "wait_for_user"

    logger.warning(f"[ROUTING] Unknown agent_decision {route!r}, ending run")
    return END


def route_from_verification(state: AgentState) -> str:
    """Route after the verify node.

    - completed=True -> delegation (counters were reset in verify; delegation picks next task)
    - failed and cap reached (next_action == "report_failure") -> delegation
      (delegation deterministically converts this into a finish)
    - otherwise -> navigation (retry the same task with feedback)
    """
    result = state.get("verification_result") or {}
    completed = bool(result.get("completed"))
    next_action = result.get("next_action", "continue_task")

    if completed:
        logger.info("[ROUTING] Verification passed, returning to delegation")
        return "delegation"

    if next_action == "report_failure":
        logger.warning("[ROUTING] Verification reported failure, returning to delegation to finish")
        return "delegation"

    logger.info("[ROUTING] Task incomplete, retrying navigation")
    return "navigation"
