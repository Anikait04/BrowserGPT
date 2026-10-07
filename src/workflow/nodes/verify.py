# verify.py — verify node (canonical; flat verify.py is a back-compat shim).

from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from src.config import MAX_CONSECUTIVE_FAILURES, MAX_NAVIGATION_ITERATIONS
from logs import logger
from src.workflow.state import AgentState
from src.workflow.constants import RECENT_ACTIONS_FOR_VERIFY
from src.workflow.llm import get_llm
from src.workflow.prompts.verify import VERIFY_PROMPT
from src.workflow.schemas import VerificationResult
from src.workflow.shared.json_utils import extract_json as _extract_json
from src.workflow.llm.helpers import ainvoke_structured_with_fallback


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


_SEARCH_ENGINE_DOMAINS: tuple[str, ...] = (
    "google.com",
    "bing.com",
    "duckduckgo.com",
    "search.yahoo.com",
    "yandex.",
)


def _extract_domains(text: str) -> set[str]:
    """Lowercased host-like tokens found in free text (for mismatch detection)."""
    import re as _re

    return set(m.lower() for m in _re.findall(r"[a-z0-9-]+\.[a-z]{2,}", (text or "").lower()))


def _is_search_engine_criteria(criteria: str) -> bool:
    cl = criteria.lower()
    mentions_engine = any(d in cl for d in _SEARCH_ENGINE_DOMAINS) or "search engine domain" in cl
    return mentions_engine and "search" in cl


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
    """Return True if criteria locks to an intermediate/search page but evidence shows the destination.

    Generalizes the observed arxiv bug (criteria="URL is google.com ..." while
    task wants an arxiv paper and the browser is on arxiv.org): whenever the
    criteria demands staying on a search-engine/intermediate host but the live
    URL is a different host mentioned by (or consistent with) the task and the
    snapshot contains task keywords, judge against task+goal instead of criteria.
    """
    if not criteria or not task or not current_url:
        return False
    if not _is_search_engine_criteria(criteria):
        return False
    url_l = current_url.lower()
    task_l = task.lower()
    snap_l = (snapshot or "").lower()
    if "(page snapshot unavailable)" in snap_l or "(unknown)" in url_l:
        return False

    criteria_domains = _extract_domains(criteria)
    url_domains = _extract_domains(url_l)
    if not url_domains or (criteria_domains & url_domains):
        return False  # same host — not a mismatch
    if any(d in url_l for d in _SEARCH_ENGINE_DOMAINS):
        return False  # still on a search engine — no overshoot

    # Destination host mentioned in the task (e.g. "arxiv.org", "docs.python.org")?
    task_domains = _extract_domains(task_l)
    if task_domains & url_domains:
        task_keywords = [w for w in task_l.split() if len(w) > 3][:8]
        hits = sum(1 for w in task_keywords if w.strip(".,:;()\"'") in snap_l or w.strip(".,:;()\"'") in url_l)
        if hits >= max(1, len(task_keywords) // 3):
            return True

    # Back-compat: arxiv/paper tasks without an explicit host in the wording.
    is_arxiv_dest = "arxiv.org" in url_l
    task_wants_paper = any(k in task_l for k in ("arxiv", "paper", "research paper"))
    snap_has_paper = "arxiv" in snap_l
    return bool(is_arxiv_dest and task_wants_paper and snap_has_paper)


async def _page_snapshot() -> str:
    """Deprecated wrapper — use browser.observation.read_page_snapshot directly."""
    from src.workflow.browser.observation import read_page_snapshot

    return await read_page_snapshot()


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
    # Unified with navigation.py guard (>=): navigation skips the call at the
    # cap, verify force-fails any state that reached it.
    if iterations >= MAX_NAVIGATION_ITERATIONS:
        verdict = VerificationResult(
            completed=False,
            reason=f"iteration limit of {MAX_NAVIGATION_ITERATIONS} reached",
            next_action="report_failure",
        )
        logger.info("[VERIFY] Task incomplete (iteration limit force-fail)")
    else:
        from src.workflow.browser.observation import read_page_snapshot

        snapshot = await read_page_snapshot()

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
            "all_actions": "\n".join(all_actions[-RECENT_ACTIONS_FOR_VERIFY:]) or "(none recorded)",
        }

        verdict = await ainvoke_structured_with_fallback(
            prompt=prompt,
            invoke_args=invoke_args,
            structured_chain_factory=lambda: prompt | get_llm().with_structured_output(VerificationResult),
            raw_chain_factory=lambda: prompt | get_llm(),
            coerce=_coerce_to_verdict,
            fallback=lambda: VerificationResult(
                completed=False,
                reason="verifier output unparseable; retrying navigation",
                next_action="continue_task",
            ),
            log_scope="VERIFY",
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
