# planner.py — planner node (canonical; flat planner.py is a back-compat shim).

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from logs import logger
from src.workflow.state import AgentState
from src.workflow.llm import get_llm
from src.workflow.prompts.planner import PLANNER_PROMPT_V2
from src.workflow.schemas import Plan
from src.workflow.shared.json_utils import extract_json as _extract_json_shared
from src.workflow.llm.helpers import ainvoke_structured_with_fallback
from src.workflow.shared.message_utils import (
    format_history as _format_history_shared,
    message_content_to_str as _message_text_shared,
)


def _coerce_to_plan(raw: object, fallback_goal: str) -> Plan:
    """Best-effort conversion of loose LLM output into a valid Plan.

    Handles the legacy prompt shape {"plan": [...], "messages": "..."}
    as well as {"goal": ..., "steps": ...}. Never raises: returns an
    empty-steps Plan on total failure so delegation can finish gracefully.
    """
    if isinstance(raw, Plan):
        return raw
    if isinstance(raw, dict):
        # Direct validation first (covers aliases via Plan.model_config).
        try:
            return Plan.model_validate(raw)
        except ValidationError:
            pass
        steps = raw.get("steps", raw.get("plan", [])) or []
        goal = raw.get("goal", raw.get("messages", "")) or fallback_goal
        if isinstance(steps, str):
            steps = [steps]
        steps = [str(s) for s in steps if s]
        return Plan(goal=str(goal), steps=steps)
    if isinstance(raw, str):
        parsed = _extract_json_shared(raw)
        if parsed is not None:
            return _coerce_to_plan(parsed, fallback_goal)
        return Plan(goal=fallback_goal, steps=[])
    return Plan(goal=fallback_goal, steps=[])


# Back-compat aliases — other nodes historically imported these private helpers
# from planner. New code must import from src.workflow.shared.* directly.
def _extract_json(text: str) -> object | None:
    """Deprecated alias for shared.json_utils.extract_json."""
    return _extract_json_shared(text)


def _message_text(message) -> str:
    """Deprecated alias for shared.message_utils.message_content_to_str."""
    return _message_text_shared(message)


def _format_history(state: AgentState) -> str:
    """Deprecated alias for shared.message_utils.format_history."""
    return _format_history_shared(state)


async def planner_node(state: AgentState) -> dict:
    """Break the user's goal into an ordered list of high-level steps."""
    logger.info("[PLANNER] Creating execution plan")

    goal_text = (state.get("goal") or "").strip()
    history_text = _format_history(state)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLANNER_PROMPT_V2),
            ("human", "Goal: {goal}\n\nCONVERSATION HISTORY:\n{history}"),
        ]
    )

    invoke_args = {"goal": goal_text, "history": history_text}
    result = await ainvoke_structured_with_fallback(
        prompt=prompt,
        invoke_args=invoke_args,
        structured_chain_factory=lambda: prompt | get_llm().with_structured_output(Plan),
        raw_chain_factory=lambda: prompt | get_llm(),
        coerce=lambda raw: _coerce_to_plan(raw, goal_text),
        fallback=lambda: Plan(goal=goal_text, steps=[]),
        log_scope="PLANNER",
    )

    logger.info(f"[PLANNER] Plan created with {len(result.steps)} step(s): {result.steps}")
    logger.info(f"[PLANNER] Goal restated: {result.goal}")

    if not result.steps:
        logger.warning(f"[PLANNER] Empty plan for goal {goal_text!r} — delegation will finish gracefully")

    existing_messages = list(state.get("messages", []) or [])

    return {
        "entire_plan": result.steps,
        "step_count": 0,
        "messages": existing_messages + [AIMessage(content=result.goal)],
    }
