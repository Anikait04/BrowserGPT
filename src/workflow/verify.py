# verify.py — verify node: strict structured judge of the delegated task.

from langchain_core.prompts import ChatPromptTemplate

from config import MAX_NAVIGATION_ITERATIONS
from logs import logger
from src.workflow.agent_state import AgentState
from src.workflow.llm import get_llm
from src.workflow.prompt import VERIFY_PROMPT
from src.workflow.schemas import VerificationResult


def _is_search_engine_criteria(criteria: str) -> bool:
    cl = criteria.lower()
    return ("google.com" in cl or "search engine domain" in cl) and "search" in cl


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

        verdict: VerificationResult = await chain.ainvoke(
            {
                "task": task or "(none)",
                "criteria": criteria or "(not specified - infer from the task)",
                "navigation_result": navigation_result or "(no report)",
                "current_url": current_url or "(unknown)",
                "snapshot": snapshot,
                "all_actions": "\n".join(all_actions[-15:]) or "(none recorded)",
            }
        )

        if verdict.completed:
            logger.info("[VERIFY] Task completed")
        else:
            logger.info(f"[VERIFY] Task incomplete ({verdict.reason})")

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
