# navigation.py — navigation node (canonical; flat navigation.py is a back-compat shim).

import json

from langchain_core.messages import HumanMessage

from src.config import DEEP_AGENT_RECURSION_LIMIT, MAX_NAVIGATION_ITERATIONS
from logs import logger
from src.workflow.state import AgentState
from src.workflow.constants import (
    ACTION_ARGS_PREVIEW_CHARS,
    ACTION_RESULT_PREVIEW_CHARS,
)
from src.workflow.shared.message_utils import message_content_to_str as _content_to_str


def _args_preview(args: dict, limit: int = ACTION_ARGS_PREVIEW_CHARS) -> str:
    import json as _json

    try:
        rendered = _json.dumps(args, ensure_ascii=False)
    except Exception:
        rendered = str(args)
    return rendered if len(rendered) <= limit else rendered[:limit] + "..."


def _result_preview(content, limit: int = ACTION_RESULT_PREVIEW_CHARS) -> str:
    text = _content_to_str(content).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def _collect_actions(result_messages: list) -> list[str]:
    """Pair tool calls with their tool results into `all_actions` history entries."""
    pending_calls = {}
    entries = []
    for msg in result_messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            pending_calls[tc.get("id")] = tc
        if getattr(msg, "type", "") == "tool":
            tc = pending_calls.get(getattr(msg, "tool_call_id", None))
            if tc:
                name = tc.get("name", "?")
                logger.info(f"[NAVIGATION] Browser action: {name}")
                entries.append(
                    f"{name}({_args_preview(tc.get('args', {}))}) -> {_result_preview(msg.content)}"
                )
    return entries


def _current_url_safe() -> str:
    """Read the live browser URL without starting a browser instance."""
    from src.workflow.browser.manager import BrowserSessionManager

    return BrowserSessionManager.current_url_safe()


async def navigation_node(state: AgentState) -> dict:
    """Run the deep navigation agent on the current delegated task."""
    goal = state.get("goal", "")
    task = state.get("current_delegated_task", "")
    criteria = state.get("success_criteria", "")
    iterations = state.get("navigation_iterations", 0)
    verification_result = state.get("verification_result") or {}
    previous_url = state.get("current_url", "")

    # ── Guard: per-task iteration budget exhausted (no agent call) ──
    if iterations >= MAX_NAVIGATION_ITERATIONS:
        logger.warning(
            f"[NAVIGATION] Iteration limit {MAX_NAVIGATION_ITERATIONS} reached for task "
            f"{task!r}, skipping navigation"
        )
        return {
            "navigation_iterations": iterations + 1,
            "navigation_result": f"Skipped: navigation iteration limit ({MAX_NAVIGATION_ITERATIONS}) reached.",
            "verification_result": {
                "completed": False,
                "reason": f"iteration limit of {MAX_NAVIGATION_ITERATIONS} reached",
                "next_action": "report_failure",
            },
        }

    # Verification feedback is present when this is a retry of the same task.
    feedback = ""
    if verification_result and not verification_result.get("completed", False):
        feedback = verification_result.get("reason", "")

    prompt_text = f"""
GOAL:
{goal}

DELEGATED TASK:
{task}

SUCCESS CRITERIA:
{criteria or "(not specified - infer from the task)"}

VERIFICATION FEEDBACK (previous attempt, retry with a different approach if present):
{feedback or "(first attempt)"}

CURRENT URL:
{previous_url or "(no page loaded yet)"}

ATTEMPT:
{iterations + 1} / {MAX_NAVIGATION_ITERATIONS}
"""

    logger.info("[NAVIGATION] Starting navigation task")

    from src.workflow.navigation_agent import get_navigation_agent

    all_actions = []
    navigation_result = ""
    try:
        agent = get_navigation_agent()
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=prompt_text)]},
            config={"recursion_limit": DEEP_AGENT_RECURSION_LIMIT},
        )

        result_messages = list(result.get("messages", []))
        all_actions = _collect_actions(result_messages)

        final_ai = None
        for msg in reversed(result_messages):
            if getattr(msg, "type", "") == "ai":
                final_ai = msg
                break
        navigation_result = _content_to_str(final_ai.content) if final_ai else ""
    except Exception as e:
        # Crash containment: surface the error to verify instead of crashing the graph.
        logger.exception("[NAVIGATION] Deep agent invocation failed")
        navigation_result = f"Navigation agent error: {e}"

    current_url = _current_url_safe() or previous_url
    logger.info(f"[NAVIGATION] Finished navigation attempt {iterations + 1}, url={current_url}")

    updates = {
        "navigation_result": navigation_result,
        "navigation_iterations": iterations + 1,
        "current_url": current_url,
    }
    if all_actions:
        updates["all_actions"] = all_actions
    return updates
