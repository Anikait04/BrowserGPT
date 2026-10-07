# delegation.py — delegation node (canonical; flat delegation.py is a back-compat shim).

from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from src.config import MAX_CONSECUTIVE_FAILURES
from logs import logger
from src.workflow.state import AgentState
from src.workflow.constants import VALID_DELEGATION_ACTIONS
from src.workflow.llm import get_llm
from src.workflow.prompts.delegation import DELEGATION_PROMPT
from src.workflow.schemas import DelegationDecision
from src.workflow.shared.json_utils import extract_json as _extract_json
from src.workflow.llm.helpers import ainvoke_structured_with_fallback
from src.workflow.shared.message_utils import format_history as _format_history


_VALID_ACTIONS = VALID_DELEGATION_ACTIONS

# Plan steps containing one of these require the extract_information node.
_EXTRACT_HINTS = (
    "extract",
    "present",
    "save",
    "report",
    "download",
    "pdf",
    "summar",
)

# Goal/task signals that the job involves browser navigation.
_NAV_HINTS = (
    "navigat",
    "browse",
    "open",
    "go to",
    "goto",
    "website",
    "webpage",
    "web page",
    "web site",
    "site",
    "page",
    "click",
    "search",
    "find",
    "look up",
    "lookup",
    "google",
    "wikipedia",
    "url",
    "link",
    "form",
    "fill",
    "type into",
    "button",
)

# Extra goal/task signals that the job involves extraction (beyond _EXTRACT_HINTS).
_EXTRACT_WANT_HINTS = (
    "overview",
    "information",
    "about",
    "tell me",
    "what is",
    "what are",
    "detail",
    "learn",
    "research",
)


def _needs_planning(goal: str, task: str) -> bool:
    """True only when the job involves navigation or extraction.

    The planner plans browser work: how/what happens during navigation plus
    the end goal, or what information to extract for the user. Anything else
    (chit-chat, unsupported requests) skips planning and goes to the human.
    """
    text = f"{goal or ''} {task or ''}".lower()
    if any(hint in text for hint in _EXTRACT_HINTS + _EXTRACT_WANT_HINTS):
        return True
    return any(hint in text for hint in _NAV_HINTS)


def _extract_steps_pending(entire_plan: list) -> list:
    """Plan steps that require extraction (overview text or PDF report)."""
    return [
        step
        for step in (entire_plan or [])
        if any(hint in str(step).lower() for hint in _EXTRACT_HINTS)
    ]


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
        user_prompt = str(raw.get("user_prompt", "") or "")
        if action not in _VALID_ACTIONS:
            return DelegationDecision(
                action="wait_for_user",
                task=str(task) or "Delegation output was unclear; please guide the next step.",
                reasoning=f"unrecognized action {action_raw!r}, pausing for user guidance",
                success_criteria="",
                user_prompt=user_prompt
                or "I wasn't sure what to do next — what would you like me to do?",
            )
        return DelegationDecision(
            action=action,  # type: ignore[arg-type]
            task=str(task),
            reasoning=str(reasoning),
            success_criteria=str(success_criteria),
            user_prompt=user_prompt,
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
        user_prompt="I couldn't work out the next step — what would you like me to do?",
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
    """Route the next unit of work to planner / navigation / extract_information / wait_for_user / finish.

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
    3. no execution plan yet -> planner, but ONLY when the job involves
       navigation or extraction (fresh browsable goal, or a new user task
       after wait_for_user cleared the plan). Anything else -> wait_for_user
       so the human clarifies. No LLM call needed for either decision.
    4. plan contains extract/present/save/report steps AND no extraction has run
       yet (extraction_format empty) AND the current task is not already an
       extraction task -> extract_information (page presence alone never
       satisfies an extraction step; the content must actually be extracted).
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

    # ── Deterministic override 3: no plan yet -> planner (or human) ──
    if not entire_plan:
        if _needs_planning(goal, current_delegated_task):
            logger.info("[DELEGATION] No execution plan yet, routing to planner")
            decision = DelegationDecision(
                action="planner",
                task=f"Create an execution plan for goal: {goal}",
                reasoning="No execution plan exists yet.",
                success_criteria="",
            )
            return {
                "success_criteria": "",
                "delegation_decision": decision.model_dump(),
                "agent_decision": "planner",
            }
        logger.info("[DELEGATION] Goal involves no navigation or extraction, asking user")
        decision = DelegationDecision(
            action="wait_for_user",
            task=(
                f"The goal {goal!r} involves no browser navigation or information "
                "extraction. Ask the user for a task this browser agent can perform "
                "(e.g. navigate a site, or extract a summary/report from a page)."
            ),
            reasoning="Nothing to plan without navigation or extraction.",
            success_criteria="",
            user_prompt=(
                "I can browse websites and extract summaries or detailed PDF reports. "
                f"What would you like me to do? (Your request {goal!r} doesn't "
                "involve navigation or extraction.)"
            ),
        )
        return {
            "success_criteria": "",
            "delegation_decision": decision.model_dump(),
            "agent_decision": "wait_for_user",
        }

    # ── Deterministic override 4: extraction steps pending but nothing extracted ──
    # Being on the right page never satisfies an extract/present/save step.
    # Fires at most once per extraction: after the extract node runs,
    # extraction_format is set (or the current task is itself an extraction
    # task), so this cannot loop extract -> delegation -> extract.
    pending_extracts = _extract_steps_pending(entire_plan)
    if pending_extracts and not state.get("extraction_format"):
        current_lower = (current_delegated_task or "").lower()
        if not any(hint in current_lower for hint in _EXTRACT_HINTS):
            task_text = (
                f"Extract information from the current page per plan step: "
                f"{pending_extracts[0]}"
            )
            logger.info(f"[DELEGATION] Extraction pending, routing to extract: {task_text!r}")
            decision = DelegationDecision(
                action="extract_information",
                task=task_text,
                reasoning="Plan requires extraction and nothing has been extracted yet.",
                success_criteria="",
            )
            return {
                "current_delegated_task": decision.task,
                "success_criteria": "",
                "delegation_decision": decision.model_dump(),
                "agent_decision": "extract_information",
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

CONVERSATION HISTORY:
{history}

LAST VERIFICATION RESULT:
{verification_result}

CONSECUTIVE FAILED ATTEMPTS on current task:
{consecutive_failures}
""",
            ),
        ]
    )

    invoke_args = {
        "goal": goal,
        "step_count": step_count,
        "total_steps": len(entire_plan),
        "plan": _format_plan(entire_plan, step_count),
        "current_task": current_delegated_task or "(none yet)",
        "history": _format_history(state),
        "verification_result": verification_result or "(none yet)",
        "consecutive_failures": consecutive_failures,
    }

    def _fallback_decision() -> DelegationDecision:
        return DelegationDecision(
            action="wait_for_user",
            task=(
                f"Could not determine the next step for goal {goal!r}; "
                "please guide the next action."
            ),
            reasoning="delegation parse fallback",
            success_criteria="",
            user_prompt=(
                "I couldn't work out the next step for your request — "
                "what would you like me to do?"
            ),
        )

    decision = await ainvoke_structured_with_fallback(
        prompt=prompt,
        invoke_args=invoke_args,
        structured_chain_factory=lambda: prompt | get_llm().with_structured_output(DelegationDecision),
        raw_chain_factory=lambda: prompt | get_llm(),
        coerce=lambda raw: _coerce_to_delegation(raw, current_delegated_task),
        fallback=_fallback_decision,
        log_scope="DELEGATION",
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
