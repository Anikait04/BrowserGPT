# planner.py — planner node for the delegated architecture.

import json
import re

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from src.logs import logger
from src.workflow.agent_state import AgentState
from src.workflow.llm import get_llm
from src.workflow.prompt import PLANNER_PROMPT_V2
from src.workflow.schemas import Plan


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


_HISTORY_ENTRIES = 10
_HISTORY_CHARS = 2000


def _message_text(message) -> str:
    """Flatten a message payload to plain text."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return " ".join(p for p in parts if p)
    return str(content or "")


def _format_history(state: AgentState) -> str:
    """Compact conversation context for grounding follow-ups ("this", "it").

    Covers the current page, prior extraction outcomes, and recent messages.
    Bounded in entries and characters to cap token usage.
    """
    lines = []
    current_url = state.get("current_url", "") or ""
    if current_url:
        lines.append(f"[browser] Current page: {current_url}")
    extracted = state.get("extracted_information", "") or ""
    if extracted:
        lines.append(
            f"[extraction] Previous result "
            f"({state.get('extraction_format', '') or 'unknown format'}): "
            f"{extracted[:300]}"
        )
    if state.get("artifact_id"):
        lines.append(
            f"[extraction] Detailed PDF available: /extract/artifact/{state.get('artifact_id')}"
        )
    messages = list(state.get("messages", []) or [])
    for msg in messages[-_HISTORY_ENTRIES:]:
        role = getattr(msg, "type", "") or "message"
        text = _message_text(msg).strip()[:300]
        if text:
            lines.append(f"[{role}] {text}")
    history = "\n".join(lines).strip()
    return history[:_HISTORY_CHARS] or "(no prior conversation)"


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
    history_text = _format_history(state)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLANNER_PROMPT_V2),
            ("human", "Goal: {goal}\n\nCONVERSATION HISTORY:\n{history}"),
        ]
    )

    result: Plan | None = None

    # 1) Preferred path: structured output (aliases handle legacy keys).
    try:
        chain = prompt | get_llm().with_structured_output(Plan)
        result = await chain.ainvoke({"goal": goal_text, "history": history_text})
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
            raw_msg = await raw_chain.ainvoke({"goal": goal_text, "history": history_text})
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
