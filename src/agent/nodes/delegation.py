# delegation.py — delegation node: decides which component handles the next unit of work.

from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate

from config import MAX_CONSECUTIVE_FAILURES
from logs import logger
from src.agent.state import AgentState
from src.agent.llm.llm import get_llm
from src.agent.nodes.planner import _extract_json
from src.agent.prompts.prompt import DELEGATION_PROMPT
from src.agent.schemas import DelegationDecision


_VALID_ACTIONS = ("navigation", "extract_information", "wait_for_user", "finish")

# Trim prompt tokens when plans get long: render a window around the cursor.
_PLAN_BACK = 3
_PLAN_FORWARD = 9


# ── Module-level singletons: built once per process, reused forever ───────
_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", DELEGATION_PROMPT),
        (
            "human",
            """
GOAL:
{goal}

PLAN (step_count={step_count}/{total_steps}):
{plan}

CURRENT DELEGATED TASK:
{current_task}

LAST VERIFICATION RESULT:
{verification_result}

CONSECUTIVE FAILED ATTEMPTS on current task:
{consecutive_failures}
""",
        ),
    ]
)


@lru_cache(maxsize=1)
def _structured_chain():
    """Structured output that also returns the raw message for fallback parsing."""
    return _PROMPT | get_llm().with_structured_output(
        DelegationDecision, include_raw=True
    )


# ── Small helpers ─────────────────────────────────────────────────────────
def _coerce_to_delegation(raw: object, fallback_task: str = "") -> DelegationDecision:
    """Best-effort conversion of loose LLM output into a valid DelegationDecision."""
    if isinstance(raw, DelegationDecision):
        return raw
    if isinstance(raw, dict):
        try:
            return DelegationDecision.model_validate(raw)
        except Exception:
            pass
        action_raw = raw.get("action", raw.get("component", raw.get("route_decision", "")))
        action = str(action_raw or "").strip().lower()
        task = raw.get("task", "") or fallback_task
        reasoning = raw.get("reasoning", "") or ""
        success_criteria = raw.get("success_criteria", "") or ""
        if action not in _VALID_ACTIONS:
            return DelegationDecision(
                action="wait_for_user",
                task=str(task) or "Delegation output was unclear; please guide the next step.",
                reasoning=f"unrecognized action {action_raw!r}, pausing for user guidance",
                success_criteria="",
            )
        return DelegationDecision(
            action=action,  # type: ignore[arg-type]
            task=str(task),
            reasoning=str(reasoning),
            success_criteria=str(success_criteria),
        )
    if isinstance(raw, str):
        parsed = _extract_json(raw)
        if parsed is not None:
            return _coerce_to_delegation(parsed, fallback_task)
    return DelegationDecision(
        action="wait_for_user",
        task=str(fallback_task) or "Delegation output was unreadable; please guide the next step.",
        reasoning="delegation output unparseable, pausing for user guidance",
        success_criteria="",
    )


def _format_plan(entire_plan: list, step_count: int) -> str:
    """Render a cursor-centered window of the plan to keep the prompt bounded."""
    if not entire_plan:
        return "(no plan steps)"
    n = len(entire_plan)
    start = max(0, step_count - _PLAN_BACK)
    end = min(n, step_count + _PLAN_FORWARD)

    lines = []
    if start > 0:
        lines.append(f"... ({start} earlier step(s) omitted)")
    for i in range(start, end):
        marker = "x" if i < step_count else " "
        lines.append(f"[{marker}] {i + 1}. {entire_plan[i]}")
    if end < n:
        lines.append(f"... ({n - end} later step(s) omitted)")
    return "\n".join(lines)


def _finish_update(
    *, task: str, reasoning: str, final_response: str, reset_nav: bool = False
) -> dict:
    """Uniform 'finish' payload — used by both deterministic overrides."""
    updates = {
        "current_delegated_task": task,
        "success_criteria": "",
        "delegation_decision": DelegationDecision(
            action="finish", task=task, reasoning=reasoning
        ).model_dump(),
        "agent_decision": "finish",
        "final_response": final_response,
    }
    if reset_nav:
        updates["navigation_iterations"] = 0
    return updates


# ── The node ──────────────────────────────────────────────────────────────
async def delegation_node(state: AgentState) -> dict:
    """Route the next unit of work to navigation / extract_information / wait_for_user / finish.

    Persistent-loop contract: 'finish' does NOT terminate the process — routing
    sends it to wait_for_user so the user reviews the result and issues the next
    task. Only an explicit user exit command (in wait_for_user) ends the run.
    """ 
    goal = state.get("goal", "")
    entire_plan = state.get("entire_plan", []) or []
    step_count = state.get("step_count", 0)
    current_delegated_task = state.get("current_delegated_task", "")
    consecutive_failures = state.get("consecutive_failures", 0)
    verification_result = state.get("verification_result") or {}
    max_steps = state.get("max_steps", 30)
    steps = state.get("steps", 0)

    # ── Deterministic override 1: global step budget exhausted ───────────
    if steps >= max_steps:
        logger.warning("[DELEGATION] Max steps reached, finishing")
        return _finish_update(
            task="Stop: maximum step count reached.",
            reasoning=f"Global step budget of {max_steps} is exhausted.",
            final_response=f"Stopped after reaching the maximum of {max_steps} steps.",
        )

    # ── Deterministic override 2: failure streak hit the cap ─────────────
    last_failed = bool(verification_result) and not verification_result.get(
        "completed", False
    )
    if last_failed and consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        reason = verification_result.get("reason", "no reason given")
        logger.warning(
            f"[DELEGATION] Failure cap reached for task {current_delegated_task!r} "
            f"(consecutive_failures={consecutive_failures}), finishing with failure"
        )
        return _finish_update(
            task=f"Report failure for task: {current_delegated_task}",
            reasoning=f"Task failed after {consecutive_failures} attempt(s): {reason}",
            final_response=(
                f"Could not complete the delegated task '{current_delegated_task}' "
                f"after {consecutive_failures} attempt(s). Last verification said: {reason}"
            ),
            reset_nav=True,
        )

    # ── LLM delegation decision (single call) ────────────────────────────
    invoke_args = {
        "goal": goal,
        "step_count": step_count,
        "total_steps": len(entire_plan),
        "plan": _format_plan(entire_plan, step_count),
        "current_task": current_delegated_task or "(none yet)",
        "verification_result": verification_result or "(none yet)",
        "consecutive_failures": consecutive_failures,
    }

    decision: DelegationDecision
    try:
        result = await _structured_chain().ainvoke(invoke_args)
        # include_raw=True → {"raw": AIMessage, "parsed": DelegationDecision | None,
        #                     "parsing_error": Exception | None}
        parsed = result.get("parsed")
        if parsed is not None:
            decision = _coerce_to_delegation(parsed, current_delegated_task)
        else:
            # Structured parse failed — mine the raw message instead of
            # paying for a second LLM round-trip.
            raw = result.get("raw")
            raw_text = getattr(raw, "content", "") if raw is not None else ""
            if isinstance(raw_text, list):
                raw_text = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in raw_text
                )
            extracted = _extract_json(str(raw_text))
            decision = _coerce_to_delegation(
                extracted if extracted is not None else raw_text,
                current_delegated_task,
            )
            logger.warning(
                f"[DELEGATION] Structured parse failed; recovered from raw text: "
                f"action={decision.action!r} task={decision.task!r}"
            )
    except Exception as e:
        # Network / provider error — degrade gracefully, don't retry inside
        # the node (the graph already handles pausing).
        logger.error(f"[DELEGATION] Delegation LLM call failed ({e}); pausing for user")
        decision = DelegationDecision(
            action="wait_for_user",
            task=(
                f"Could not determine the next step for goal {goal!r}; "
                "please guide the next action."
            ),
            reasoning="delegation LLM call failed",
            success_criteria="",
        )

    logger.info(
        f"[DELEGATION] Delegating task: {decision.action} | task={decision.task!r} "
        f"| criteria={decision.success_criteria!r} | reasoning={decision.reasoning!r}"
    )

    updates = {
        "current_delegated_task": decision.task,
        "success_criteria": decision.success_criteria,
        "delegation_decision": decision.model_dump(),
        "agent_decision": decision.action,
    }

    if decision.action == "finish":
        updates["final_response"] = decision.task or decision.reasoning
    elif decision.task != current_delegated_task:
        # New task string issued → fresh nav↔verify budget.
        updates["navigation_iterations"] = 0

    return updates