# state_factory.py — single canonical AgentState construction.
#
# Previously agent.py had _build_initial_state() plus an inline duplicate in
# run_agent() that drifted (missing conversation_history). All runners must use
# initial_state() here.

from __future__ import annotations

from src.workflow.state import AgentState


def initial_state(goal: str, max_steps: int, task_id: str | None) -> AgentState:
    """Build a fresh run state (all 23 keys; single source of truth)."""
    return {
        "goal": goal,
        "entire_plan": [],
        "step_count": 0,
        "agent_decision": "",
        "task_id": task_id,
        "steps": 0,
        "progress_verification": "",
        "max_steps": max_steps,
        "current_url": "",
        "messages": [],
        "current_delegated_task": "",
        "delegation_decision": None,
        "success_criteria": "",
        "navigation_result": "",
        "verification_result": None,
        "extracted_information": None,
        "extraction_format": "",
        "artifact_id": None,
        "artifact_path": None,
        "waiting_for_user": False,
        "exit_requested": False,
        "final_response": "",
        "navigation_iterations": 0,
        "consecutive_failures": 0,
        "all_actions": [],
        "conversation_history": "",
    }


def reset_for_new_task(human_input: str, messages: list) -> dict:
    """State delta for a fresh user task after a finished goal (wait_for_user)."""
    return {
        "waiting_for_user": False,
        "exit_requested": False,
        "goal": human_input,
        "entire_plan": [],
        "step_count": 0,
        "steps": 0,
        "current_delegated_task": "",
        "delegation_decision": None,
        "success_criteria": "",
        "navigation_result": "",
        "verification_result": None,
        "extracted_information": None,
        "extraction_format": "",
        "artifact_id": None,
        "artifact_path": None,
        "navigation_iterations": 0,
        "consecutive_failures": 0,
        "agent_decision": "",
        "final_response": "",
        "messages": messages,
        "progress_verification": f"New user task: {human_input}",
    }
