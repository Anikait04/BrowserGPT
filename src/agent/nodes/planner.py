# planner.py — planner node for the delegated architecture.

import json
import re

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from logs import logger
from src.agent.state import AgentState
from src.agent.llm.llm import get_llm
from src.agent.prompts.prompt import PLANNER_PROMPT_V2
from src.agent.schemas import Plan


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
        parsed = _extract_json(raw)
        if parsed is not None:
            return _coerce_to_plan(parsed, fallback_goal)
        return Plan(goal=fallback_goal, steps=[])
    return Plan(goal=fallback_goal, steps=[])


def _extract_json(text: str) -> object | None:
    """Extract the first JSON object from free-form LLM text."""
    if not text:
        return None
    # Strip code fences if present.
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text.strip())
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
    return None


async def planner_node(state: AgentState) -> dict:
    """Break the user's goal into an ordered list of high-level steps."""
    logger.info("[PLANNER] Creating execution plan")

    goal_text = (state.get("goal") or "").strip()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLANNER_PROMPT_V2),
            ("human", "Goal: {goal}"),
        ]
    )

    result: Plan | None = None

    # 1) Preferred path: structured output (aliases handle legacy keys).
    try:
        chain = prompt | get_llm().with_structured_output(Plan)
        result = await chain.ainvoke({"goal": goal_text})
        if isinstance(result, dict):
            result = _coerce_to_plan(result, goal_text)
    except (OutputParserException, ValidationError) as e:
        logger.warning(f"[PLANNER] Structured output failed ({e}); trying raw-JSON fallback")
        result = None
    except Exception as e:
        logger.warning(f"[PLANNER] Planner LLM call failed ({e}); trying raw-JSON fallback")
        result = None

    # 2) Fallback: plain LLM call + manual JSON extraction + alias mapping.
    if result is None:
        try:
            raw_chain = prompt | get_llm()
            raw_msg = await raw_chain.ainvoke({"goal": goal_text})
            raw_text = getattr(raw_msg, "content", str(raw_msg))
            if isinstance(raw_text, list):
                raw_text = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in raw_text
                )
            parsed = _extract_json(str(raw_text))
            result = _coerce_to_plan(parsed if parsed is not None else raw_text, goal_text)
            logger.warning(f"[PLANNER] Raw fallback produced: goal={result.goal!r} steps={result.steps!r}")
        except Exception as e:
            logger.error(f"[PLANNER] Raw fallback also failed ({e}); using empty plan")
            result = Plan(goal=goal_text, steps=[])

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
