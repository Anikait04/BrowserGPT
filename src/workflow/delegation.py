# delegation.py — delegation node: decides which component handles the next unit of work.

from langchain_core.prompts import ChatPromptTemplate

from config import MAX_CONSECUTIVE_FAILURES
from logs import logger
from src.workflow.agent_state import AgentState
from src.workflow.llm import get_llm
from src.workflow.prompt import DELEGATION_PROMPT
from src.workflow.schemas import DelegationDecision


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

    Deterministic overrides run before any LLM call:
    1. max steps reached -> finish
    2. previous verification reported failure or failure streak hit the cap -> finish with failure
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

    # ── Deterministic override 2: unrecoverable failure on the delegated task ──
    prev_reported_failure = (
        bool(verification_result)
        and not verification_result.get("completed", False)
        and verification_result.get("next_action") == "report_failure"
    )
    if prev_reported_failure or consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
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

    decision: DelegationDecision = await chain.ainvoke(
        {
            "goal": goal,
            "step_count": step_count,
            "total_steps": len(entire_plan),
            "plan": _format_plan(entire_plan, step_count),
            "current_task": current_delegated_task or "(none yet)",
            "verification_result": verification_result or "(none yet)",
            "consecutive_failures": consecutive_failures,
        }
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
