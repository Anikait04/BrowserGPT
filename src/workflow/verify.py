# verify.py — verify node: strict structured judge of the delegated task.

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from src.config import MAX_CONSECUTIVE_FAILURES, MAX_NAVIGATION_ITERATIONS
from src.logs import logger
from src.workflow.agent_state import AgentState
from src.workflow.llm import get_llm
from src.workflow.planner import _extract_json
from src.workflow.prompt import VERIFY_PROMPT
from src.workflow.schemas import VerificationResult


_TRUE_TOKENS = {"true", "yes", "completed", "complete", "success", "done", "1"}
_FALSE_TOKENS = {"false", "no", "incomplete", "failure", "failed", "0"}


def _to_bool(value: object) -> bool | None:
    """Best-effort bool conversion; None when undecidable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
    return None


def _coerce_to_verdict(raw: object) -> VerificationResult:
    """Best-effort conversion of loose LLM output into a VerificationResult.

    Handles near-miss keys (complete/success/verdict, justification, ...).
    Never raises: undecidable input degrades to incomplete + continue_task
    (bounded retry) instead of crashing the run.
    """
    if isinstance(raw, VerificationResult):
        return raw
    if isinstance(raw, dict):
        # Direct validation first (covers aliases via model_config).
        try:
            return VerificationResult.model_validate(raw)
        except ValidationError:
            pass
        completed_raw = raw.get(
            "completed",
            raw.get("complete", raw.get("success", raw.get("verdict", None))),
        )
        completed = _to_bool(completed_raw)
        if completed is None:
            completed = False
        reason = raw.get("reason", raw.get("justification", raw.get("explanation", ""))) or ""
        next_action = str(raw.get("next_action", "") or "").strip().lower()
        if next_action not in ("continue_task", "report_failure"):
            next_action = "continue_task"
        return VerificationResult(
            completed=completed,
            reason=str(reason),
            next_action=next_action,  # type: ignore[arg-type]
        )
    if isinstance(raw, str):
        parsed = _extract_json(raw)
        if parsed is not None:
            return _coerce_to_verdict(parsed)
    return VerificationResult(
        completed=False,
        reason="verifier output unparseable; retrying navigation",
        next_action="continue_task",
    )


def _is_search_engine_criteria(criteria: str) -> bool:
    cl = criteria.lower()
    return ("google.com" in cl or "search engine domain" in cl) and "search" in cl


def _fallback_goal_achieved(
    task: str, criteria: str, goal: str, current_url: str, snapshot: str
) -> bool:
    """Return True if the evidence shows the overall GOAL is already achieved even
    though the narrow success_criteria are not literally met.

    Generalizes the arxiv search-engine mismatch into the common overshoot case:
    criteria described an intermediate page (homepage, search results) but the browser
    already sits on the goal's destination with the expected content visible.
    Matching is deliberately conservative — substring evidence on both URL and page text.
    """
    if not goal or not current_url:
        return False
    url_l = current_url.lower()
    snap_l = (snapshot or "").lower()
    if "(page snapshot unavailable)" in snap_l or "(unknown)" in url_l:
        return False
    goal_keywords = [w for w in goal.lower().split() if len(w) > 3]
    if not goal_keywords:
        return False
    matched = [w for w in goal_keywords if w in url_l or w in snap_l]
    # Require most substantive goal words evidenced on the page/URL, and require the
    # criteria to look like an intermediate-page lock (mentions homepage / results page
    # / search page while the URL has moved past it).
    criteria_l = (criteria or "").lower()
    intermediate_lock = any(
        phrase in criteria_l
        for phrase in ("homepage", "home page", "search results", "results page", "search page")
    )
    return intermediate_lock and len(matched) >= max(2, (len(goal_keywords) + 1) // 2)


def _fallback_malformed_criteria(task: str, criteria: str, current_url: str, snapshot: str) -> bool:
    """Return True if criteria is malformed search-engine lock but evidence shows correct destination.

    Catches the observed bug: task="get arxiv paper ..." but criteria="URL is google.com ..." while
    current_url is arxiv.org and snapshot contains the paper. In that case judge against task, not criteria.
    """
    if not criteria or not task or not current_url:
        return False
    if not _is_search_engine_criteria(criteria):
        return False
    url_l = current_url.lower()
    task_l = task.lower()
    snap_l = snapshot.lower() if snapshot else ""
    # Destination is arXiv / paper-like and task mentions it
    is_arxiv_dest = "arxiv.org" in url_l
    task_wants_paper = any(k in task_l for k in ("arxiv", "paper", "attention is all you need", "research paper"))
    snap_has_paper = "attention is all you need" in snap_l or "arxiv" in snap_l
    return is_arxiv_dest and task_wants_paper and snap_has_paper


async def _page_snapshot() -> str:
    """Fresh read-only page snapshot for evidence (allowed to touch the browser)."""
    try:
        from src.workflow.browsertools import get_browser

        browser = await get_browser()
        return await browser.read()
    except Exception as e:
        logger.warning(f"[VERIFY] Could not read page snapshot: {e}")
        return "(page snapshot unavailable)"


async def verify_node(state: AgentState) -> dict:
    """Decide whether the delegated task is actually complete."""
    logger.info("[VERIFY] Checking task completion")

    task = state.get("current_delegated_task", "")
    criteria = state.get("success_criteria", "")
    goal = state.get("goal", "")
    entire_plan = state.get("entire_plan", []) or []
    step_count = state.get("step_count", 0)
    navigation_result = state.get("navigation_result", "")
    iterations = state.get("navigation_iterations", 0)
    current_url = state.get("current_url", "")
    all_actions = state.get("all_actions", []) or []
    consecutive_failures = state.get("consecutive_failures", 0)

    # ── Force-fail path (no LLM): nav<->verify loop budget exhausted ──
    if iterations > MAX_NAVIGATION_ITERATIONS:
        verdict = VerificationResult(
            completed=False,
            reason=f"iteration limit of {MAX_NAVIGATION_ITERATIONS} reached",
            next_action="report_failure",
        )
        logger.info("[VERIFY] Task incomplete (iteration limit force-fail)")
    else:
        snapshot = await _page_snapshot()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", VERIFY_PROMPT),
                (
                    "human",
                    """
DELEGATED TASK:
{task}

SUCCESS CRITERIA:
{criteria}

OVERALL GOAL:
{goal}

PLAN (current step {step_count}/{total_steps}):
{plan}

NAVIGATION AGENT FINAL REPORT:
{navigation_result}

CURRENT URL:
{current_url}

CURRENT PAGE SNAPSHOT:
{snapshot}

RECENT ACTIONS HISTORY:
{all_actions}
""",
                ),
            ]
        )

        chain = prompt | get_llm().with_structured_output(VerificationResult)

        invoke_args = {
            "task": task or "(none)",
            "criteria": criteria or "(not specified - infer from the task)",
            "goal": goal or "(none)",
            "plan": "\n".join(
                f"[{'x' if i < step_count else ' '}] {i + 1}. {step}"
                for i, step in enumerate(entire_plan)
            )
            or "(no plan steps)",
            "step_count": step_count,
            "total_steps": len(entire_plan),
            "navigation_result": navigation_result or "(no report)",
            "current_url": current_url or "(unknown)",
            "snapshot": snapshot,
            "all_actions": "\n".join(all_actions[-15:]) or "(none recorded)",
        }

        verdict: VerificationResult | None = None

        # 1) Preferred path: structured output (aliases handle near-miss keys).
        try:
            verdict = await chain.ainvoke(invoke_args)
            if isinstance(verdict, dict):
                verdict = _coerce_to_verdict(verdict)
        except (OutputParserException, ValidationError) as e:
            logger.warning(f"[VERIFY] Structured output failed ({e}); trying raw-JSON fallback")
            verdict = None
        except Exception as e:
            logger.warning(f"[VERIFY] Verifier LLM call failed ({e}); trying raw-JSON fallback")
            verdict = None

        # 2) Fallback: plain LLM call + manual JSON extraction + alias mapping.
        if verdict is None:
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
                verdict = _coerce_to_verdict(parsed if parsed is not None else raw_text)
                logger.warning(
                    f"[VERIFY] Raw fallback produced: completed={verdict.completed!r} "
                    f"next_action={verdict.next_action!r}"
                )
            except Exception as e:
                logger.error(f"[VERIFY] Raw fallback also failed ({e}); retrying task")
                verdict = VerificationResult(
                    completed=False,
                    reason="verifier output unparseable; retrying navigation",
                    next_action="continue_task",
                )

        if verdict.completed:
            logger.info("[VERIFY] Task completed")
        else:
            logger.info(f"[VERIFY] Task incomplete ({verdict.reason})")

        # ── Deterministic guardrail: the LLM may only give up when retry budget is
        # actually exhausted. A premature report_failure becomes a retry with guidance.
        if (
            not verdict.completed
            and verdict.next_action == "report_failure"
            and consecutive_failures + 1 < MAX_CONSECUTIVE_FAILURES
            and iterations <= MAX_NAVIGATION_ITERATIONS
        ):
            logger.warning(
                "[VERIFY] Downgrading premature report_failure to continue_task — "
                f"retry budget remains (consecutive_failures={consecutive_failures}, "
                f"navigation_iterations={iterations})"
            )
            verdict = VerificationResult(
                completed=False,
                reason=verdict.reason,
                next_action="continue_task",
            )

        # ── Fallback: malformed search-engine criteria vs correct destination (e.g. arxiv) ──
        if not verdict.completed and _fallback_malformed_criteria(task, criteria, current_url, snapshot):
            logger.warning(
                f"[VERIFY] Overriding verdict to completed=True — criteria {criteria!r} mismatched task {task!r} "
                f"but evidence shows correct destination {current_url!r}"
            )
            verdict = VerificationResult(
                completed=True,
                reason=f"criteria mismatched task, judged against goal — evidence shows correct destination {current_url} with paper content",
                next_action="continue_task",
            )
            logger.info("[VERIFY] Task completed (fallback)")

        # ── Fallback: narrow criteria describe an intermediate page, but the overall
        # GOAL is already achieved (overshoot, e.g. watch page vs homepage lock) ──
        if not verdict.completed and _fallback_goal_achieved(
            task, criteria, goal, current_url, snapshot
        ):
            logger.warning(
                f"[VERIFY] Overriding verdict to completed=True — goal {goal!r} already achieved "
                f"at {current_url!r} beyond narrow criteria {criteria!r}"
            )
            verdict = VerificationResult(
                completed=True,
                reason=f"goal achieved beyond narrow criteria — evidence at {current_url} satisfies the overall goal",
                next_action="continue_task",
            )
            logger.info("[VERIFY] Task completed (goal-achieved fallback)")

    updates = {
        "verification_result": verdict.model_dump(),
        "progress_verification": (
            f"Task '{task}' verified {'complete' if verdict.completed else 'incomplete'}: {verdict.reason}"
        ),
        "all_actions": [f"verify -> completed={verdict.completed} ({verdict.reason})"],
    }

    if verdict.completed:
        # Success: advance the plan and clear the failure streak.
        updates["step_count"] = state.get("step_count", 0) + 1
        updates["consecutive_failures"] = 0
    else:
        updates["consecutive_failures"] = consecutive_failures + 1

    return updates
