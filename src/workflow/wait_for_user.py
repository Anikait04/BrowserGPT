# wait_for_user.py — wait_for_user node: pauses the graph until human input arrives.

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from logs import logger
from src.workflow.agent_state import AgentState


async def wait_for_user_node(state: AgentState) -> dict:
    """Pause execution and wait for a human instruction, then return to delegation."""
    logger.info("[WAIT] Pausing for user input")

    entire_plan = state.get("entire_plan", []) or []
    summary = f"""
Agent is pausing for human input.

Goal: {state.get('goal', '')}
Plan Progress: {state.get('step_count', 0)} / {len(entire_plan)}
Delegated Task: {state.get('current_delegated_task', 'N/A')}
Current URL: {state.get('current_url', 'N/A')}

Please provide an instruction or type 'continue' to proceed.
"""
    state["waiting_for_user"] = True

    human_input = interrupt(summary)
    logger.info(f"[WAIT] User provided input: {human_input}")

    existing_messages = list(state.get("messages", []) or [])

    return {
        "waiting_for_user": False,
        "messages": existing_messages + [HumanMessage(content=str(human_input))],
        "progress_verification": f"Human instruction: {human_input}",
    }
