# routing.py — pure routing functions for the delegated architecture graph.
# No LLM, no I/O — deterministic decisions based on AgentState only.
#
# Persistent-loop contract: the graph NEVER terminates on its own.
# - delegation "finish" routes to wait_for_user (user reviews result, issues
#   the next task). The run stays alive with the browser open.
# - wait_for_user is the ONLY place that can reach END, and only when the
#   user explicitly issued an exit command (exit_requested=True).

from langgraph.graph import END

from logs import logger
from src.workflow.agent_state import AgentState


def route_from_delegation(state: AgentState) -> str:
    """Route after the delegation node.

    - "finish" -> wait_for_user (NOT END: ask the user for the next task;
      only an explicit user exit ends the run)
    - max steps reached -> wait_for_user (let the user decide: retry / new task / exit)
    - "navigation" / "extract_information" / "wait_for_user" -> respective node
    - anything else -> wait_for_user (defensive: ask the user instead of ending)
    """
    route = state.get("agent_decision", "")
    logger.info(f"[ROUTING] delegation decision: {route}")

    if route == "finish":
        logger.info("[ROUTING] Goal finished, pausing for user (persistent loop)")
        return "wait_for_user"

    if state.get("steps", 0) >= state.get("max_steps", 30):
        logger.warning("[ROUTING] Max steps reached, pausing for user instead of ending")
        return "wait_for_user"

    if route == "navigation":
        return "navigation"
    if route == "extract_information":
        return "extract_information"
    if route == "wait_for_user":
        return "wait_for_user"

    logger.warning(f"[ROUTING] Unknown agent_decision {route!r}, pausing for user")
    return "wait_for_user"


def route_from_wait_for_user(state: AgentState) -> str:
    """Route after the wait_for_user node — the ONLY gateway to END.

    - exit_requested=True (user typed exit/quit/stop/...) -> END
    - fresh user task (plan cleared by the wait node) -> planner
    - otherwise -> delegation (resume current goal with human guidance)
    """
    if state.get("exit_requested"):
        logger.info("[ROUTING] User requested exit, ending run")
        return END

    if not state.get("entire_plan"):
        logger.info("[ROUTING] New user task pending, replanning")
        return "planner"

    logger.info("[ROUTING] Resuming with user input, returning to delegation")
    return "delegation"


def route_from_verification(state: AgentState) -> str:
    """Route after the verify node.

    - completed=True -> delegation (counters were reset in verify; delegation picks next task)
    - failed with next_action == "report_failure" -> delegation (delegation finishes
      only if the consecutive-failure streak hit its cap, otherwise it re-delegates
      the same task with verification feedback)
    - otherwise -> navigation (retry the same task with feedback)
    """
    result = state.get("verification_result") or {}
    completed = bool(result.get("completed"))
    next_action = result.get("next_action", "continue_task")

    if completed:
        logger.info("[ROUTING] Verification passed, returning to delegation")
        return "delegation"

    if next_action == "report_failure":
        logger.warning("[ROUTING] Verification reported failure, returning to delegation")
        return "delegation"

    logger.info("[ROUTING] Task incomplete, retrying navigation")
    return "navigation"
