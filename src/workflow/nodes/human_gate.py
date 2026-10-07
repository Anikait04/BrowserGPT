# human_gate.py — wait_for_user node (canonical; flat wait_for_user.py is a back-compat shim).
#
# Persistent-loop semantics: the agent NEVER terminates on its own.
# - delegation "finish" routes here (not to END) so the user sees the result
#   and can issue the next task.
# - Only an explicit exit command from the user ends the run (routes to END).
# - A substantive new instruction after a finished task becomes the new goal
#   and triggers replanning (routes to planner).
# - Anything else resumes the current goal (routes to delegation).

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from logs import logger
from src.workflow.state import AgentState
from src.workflow.shared.commands import (
    CONTINUE_KEYWORDS,
    EXIT_KEYWORDS,
    EXIT_PHRASES,
    is_continue_command,
    is_exit_command,
)


async def wait_for_user_node(state: AgentState) -> dict:
    """Pause execution and wait for a human instruction.

    Routes (via route_from_wait_for_user):
    - exit command -> END (the ONLY way the run terminates)
    - new task after a finished goal -> delegation (empty plan routes to planner there)
    - otherwise -> delegation (resume current goal with human guidance)
    """
    came_from_finish = state.get("agent_decision") == "finish"
    final_response = state.get("final_response", "")
    entire_plan = state.get("entire_plan", []) or []

    # Delegation owns ambiguous/follow-up questions: when it routed here with a
    # user_prompt, display it verbatim instead of the generic summaries below.
    delegation_prompt = ""
    if state.get("agent_decision") == "wait_for_user":
        delegation_decision = state.get("delegation_decision") or {}
        delegation_prompt = str(delegation_decision.get("user_prompt", "") or "").strip()

    if delegation_prompt:
        summary = delegation_prompt
    elif came_from_finish:
        summary = f"""
Agent finished the current goal and is waiting for your next instruction.
The browser stays open — the run does NOT end on its own.

Goal: {state.get('goal', '')}
Plan Progress: {state.get('step_count', 0)} / {len(entire_plan)}
Result: {final_response or '(no summary)'}
Current URL: {state.get('current_url', 'N/A')}

Type your next task (e.g. 'search for ...'), or type 'exit' to end the run.
"""
    else:
        summary = f"""
Agent is pausing for human input.

Goal: {state.get('goal', '')}
Plan Progress: {state.get('step_count', 0)} / {len(entire_plan)}
Delegated Task: {state.get('current_delegated_task', 'N/A')}
Current URL: {state.get('current_url', 'N/A')}

Please provide an instruction, type 'continue' to proceed, or type 'exit' to end the run.
"""
    logger.info("[WAIT] Pausing for user input")

    human_input = interrupt(summary)
    human_input = str(human_input or "").strip()
    logger.info(f"[WAIT] User provided input: {human_input}")

    existing_messages = list(state.get("messages", []) or [])
    messages = existing_messages + [HumanMessage(content=human_input)]

    # ── 1. Explicit exit: the ONLY terminal path ──
    if is_exit_command(human_input):
        logger.info("[WAIT] User requested exit, ending run")
        return {
            "waiting_for_user": False,
            "exit_requested": True,
            "agent_decision": "finish",
            "messages": messages,
            "progress_verification": f"User requested exit: {human_input}",
            "final_response": final_response or "Session ended by user.",
        }

    # ── 2. New task after a finished goal -> replan as a fresh goal ──
    if came_from_finish and human_input and not is_continue_command(human_input):
        logger.info(f"[WAIT] New user task received, replanning: {human_input!r}")
        from src.workflow.state_factory import reset_for_new_task

        return reset_for_new_task(human_input, messages)

    # ── 3. Default: resume current goal with human guidance ──
    return {
        "waiting_for_user": False,
        "exit_requested": False,
        "messages": messages,
        "progress_verification": f"Human instruction: {human_input}",
    }
