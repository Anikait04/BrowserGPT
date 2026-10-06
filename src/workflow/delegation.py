# delegation.py — delegation node: decides which component handles the next unit of work.

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from config import MAX_CONSECUTIVE_FAILURES
from logs import logger
from src.workflow.agent_state import AgentState
from src.workflow.llm import get_llm
from src.workflow.planner import _extract_json
from src.workflow.prompt import DELEGATION_PROMPT
from src.workflow.schemas import DelegationDecision


_VALID_ACTIONS = ("navigation", "extract_information", "wait_for_user", "finish")


def _coerce_to_delegation(raw: object, fallback_task: str = "") -> DelegationDecision:
    """Best-effort conversion of loose LLM output into a valid DelegationDecision.

    Handles the observed shape {"component": ..., ...} as well as the canonical
    {"action": ..., ...}. Never raises: an unrecognized/missing action degrades
    to "wait_for_user" so the run pauses for guidance instead of crashing.
    """
    if isinstance(raw, DelegationDecision):
        return raw
    if isinstance(raw, dict):
        # Direct validation first (covers aliases via model_config).
        try:
            return DelegationDecision.model_validate(raw)
        except ValidationError:
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
    if not entire_plan:
        return "(no plan steps)"
    lines = []
    for i, step in enumerate(entire_plan):
        marker = "x" if i < step_count else " "
        lines.append(f"[{marker}] {i + 1}. {step}")
    return "\n".join(lines)


async def delegation_node(state: AgentState) -> dict:
    """Route the next unit of work to navigation / extract_information / wait_for_user / finish.

    Persistent-loop contract: "finish" does NOT terminate the process — routing
    sends it to wait_for_user so the user reviews the result and issues the next
    task. Only an explicit user exit command (in wait_for_user) ends the run.

    Deterministic overrides run before any LLM call:
    1. max steps reached -> finish (then wait_for_user asks the user what to do next)
    2. last verification failed AND the consecutive-failure streak hit the cap
       (consecutive_failures >= MAX_CONSECUTIVE_FAILURES) -> finish with failure
       (then wait_for_user asks the user what to do next).
       A failure with retry budget remaining falls through to the LLM, which
       re-delegates the same task with verification feedback.
    """
    goal = state.get("goal", "")
    entire_plan = state.get("entire_plan", []) or []
    step_count = state.get("step_count", 0)
    current_delegated_task = state.get("current_delegated_task", "")
    consecutive_failures = state.get("consecutive_failures", 0)
    verification_result = state.get("verification_result") or {}
    max_steps = state.get("max_steps", 30)
    steps = state.get("steps", 0)

    # ── Deterministic override 1: global step budget exhausted ──
    if steps >= max_steps:
        logger.warning("[DELEGATION] Max steps reached, finishing")
        decision = DelegationDecision(
            action="finish",
            task="Stop: maximum step count reached.",
            reasoning=f"Global step budget of {max_steps} is exhausted.",
        )
        return {
            "current_delegated_task": decision.task,
            "success_criteria": "",
            "delegation_decision": decision.model_dump(),
            "agent_decision": "finish",
            "final_response": f"Stopped after reaching the maximum of {max_steps} steps.",
        }

    # ── Deterministic override 2: last verification failed AND the failure streak
    # hit the cap -> finish with failure. A failure with retry budget remaining is
    # NOT terminal: fall through so the LLM re-delegates the same task with
    # verification feedback (DELEGATION_PROMPT rule 3). The verifier's
    # report_failure alone never finishes the run; only the streak cap does.
    last_failed = bool(verification_result) and not verification_result.get(
        "completed", False
    )
    streak_capped = consecutive_failures >= MAX_CONSECUTIVE_FAILURES
    if last_failed and streak_capped:
        reason = verification_result.get("reason", "no reason given")
        logger.warning(
            f"[DELEGATION] Failure cap reached for task {current_delegated_task!r} "
            f"(consecutive_failures={consecutive_failures}), finishing with failure"
        )
        decision = DelegationDecision(
            action="finish",
            task=f"Report failure for task: {current_delegated_task}",
            reasoning=f"Task failed after {consecutive_failures} attempt(s): {reason}",
        )
        return {
            "current_delegated_task": decision.task,
            "success_criteria": "",
            "delegation_decision": decision.model_dump(),
            "agent_decision": "finish",
            "navigation_iterations": 0,
            "final_response": (
                f"Could not complete the delegated task '{current_delegated_task}' "
                f"after {consecutive_failures} attempt(s). Last verification said: {reason}"
            ),
        }

    # ── LLM delegation decision ──
    prompt = ChatPromptTemplate.from_messages(
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

    chain = prompt | get_llm().with_structured_output(DelegationDecision)

    decision: DelegationDecision | None = None
    invoke_args = {
        "goal": goal,
        "step_count": step_count,
        "total_steps": len(entire_plan),
        "plan": _format_plan(entire_plan, step_count),
        "current_task": current_delegated_task or "(none yet)",
        "verification_result": verification_result or "(none yet)",
        "consecutive_failures": consecutive_failures,
    }

    # 1) Preferred path: structured output (aliases handle legacy keys).
    try:
        decision = await chain.ainvoke(invoke_args)
        if isinstance(decision, dict):
            decision = _coerce_to_delegation(decision, current_delegated_task)
    except (OutputParserException, ValidationError) as e:
        logger.warning(f"[DELEGATION] Structured output failed ({e}); trying raw-JSON fallback")
        decision = None
    except Exception as e:
        logger.warning(f"[DELEGATION] Delegation LLM call failed ({e}); trying raw-JSON fallback")
        decision = None

    # 2) Fallback: plain LLM call + manual JSON extraction + alias mapping.
    if decision is None:
        try:
            raw_chain = prompt | get_llm()
            raw_msg = await raw_chain.ainvoke(invoke_args)
            raw_text = getattr(raw_msg, "content", str(raw_msg))
            if isinstance(raw_text, list):
                raw_text = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in raw_text
                )
            parsed = _extract_json(str(raw_text))
            decision = _coerce_to_delegation(
                parsed if parsed is not None else raw_text, current_delegated_task
            )
            logger.warning(
                f"[DELEGATION] Raw fallback produced: action={decision.action!r} task={decision.task!r}"
            )
        except Exception as e:
            logger.error(f"[DELEGATION] Raw fallback also failed ({e}); pausing for user")
            decision = DelegationDecision(
                action="wait_for_user",
                task=(
                    f"Could not determine the next step for goal {goal!r}; "
                    "please guide the next action."
                ),
                reasoning="delegation parse fallback",
                success_criteria="",
            )

    logger.info(f"[DELEGATION] Delegating task: {decision.action} | task={decision.task!r} | criteria={decision.success_criteria!r} | reasoning={decision.reasoning!r}")

    updates = {
        "current_delegated_task": decision.task,
        "success_criteria": decision.success_criteria,
        "delegation_decision": decision.model_dump(),
        "agent_decision": decision.action,
    }

    if decision.action == "finish":
        updates["final_response"] = decision.task or decision.reasoning
    elif decision.task != current_delegated_task:
        # New task string issued -> fresh nav<->verify budget.
        updates["navigation_iterations"] = 0

    return updates
