# extraction.py — depth-aware extraction node (canonical; flat extract_information.py is a shim).
#
# Decides whether the user needs an overview (short text) or detailed
# information (well-written PDF report), asking the human only when ambiguous:
# - explicit intent in goal/task ("brief", "pdf", ...) decides without asking
# - otherwise an LLM DepthDecision picks overview / detailed / ask_user
# - ask_user pauses via interrupt(); unclear answers default to overview

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt
from pydantic import ValidationError

from logs import logger
from src.workflow.state import AgentState
from src.workflow.artifacts.store import ArtifactStore
from src.workflow.browser.observation import read_page_snapshot
from src.workflow.constants import PAGE_SNAPSHOT_CHARS, artifact_url
from src.workflow.llm import get_llm
from src.workflow.prompts.extraction import (
    DEPTH_DECIDER_PROMPT,
    OVERVIEW_PROMPT,
    REPORT_COMPOSE_PROMPT,
)
from src.workflow.schemas import DepthDecision, ExtractionContent, ReportSection
from src.workflow.shared.json_utils import extract_json as _extract_json
from src.workflow.llm.helpers import ainvoke_structured_with_fallback
from src.workflow.shared.message_utils import message_content_to_str as _content_to_text_shared


async def _page_snapshot() -> str:
    """Deprecated wrapper — use browser.observation.read_page_snapshot directly."""
    return await read_page_snapshot()

_VALID_DEPTHS = ("overview", "detailed", "ask_user")

# Explicit-intent keywords (checked against goal + delegated task).
_DETAILED_HINTS = (
    "pdf",
    "download",
    "full report",
    "in-depth",
    "indepth",
    "comprehensive",
    "exhaustive",
    "detailed",
    "in detail",
    "complete guide",
    "deep dive",
    "deep-dive",
)
_OVERVIEW_HINTS = (
    "overview",
    "brief",
    "summar",
    "tldr",
    "quick",
    "short",
    "gist",
    "high level",
    "high-level",
)

# Answers to the ask_user interrupt.
_DETAILED_ANSWER_HINTS = (
    "detailed",
    "detail",
    "full",
    "pdf",
    "report",
    "document",
    "download",
    "comprehensive",
    "in-depth",
    "long",
)
_OVERVIEW_ANSWER_HINTS = (
    "overview",
    "brief",
    "summar",
    "short",
    "quick",
    "tldr",
    "text",
    "here",
    "just tell",
    "no pdf",
    "no,",
)

_SNAPSHOT_CHARS = PAGE_SNAPSHOT_CHARS

# Local alias preserved for back-compat imports (use constants directly in new code).


def _explicit_depth(text: str) -> str | None:
    """Deterministic shortcut for explicit user intent. None when ambiguous."""
    t = (text or "").lower()
    if any(hint in t for hint in _DETAILED_HINTS):
        return "detailed"
    if any(hint in t for hint in _OVERVIEW_HINTS):
        return "overview"
    return None


def _coerce_to_depth(raw: object) -> DepthDecision:
    """Best-effort conversion of loose LLM output into a DepthDecision.

    Never raises: undecidable input degrades to "overview" (cheap,
    non-blocking) instead of crashing the run. Only an explicit "ask_user"
    verdict triggers a human interrupt.
    """
    if isinstance(raw, DepthDecision):
        return raw
    if isinstance(raw, dict):
        try:
            return DepthDecision.model_validate(raw)
        except ValidationError:
            pass
        depth = str(
            raw.get("depth", raw.get("detail_level", raw.get("format", ""))) or ""
        ).strip().lower()
        if depth not in _VALID_DEPTHS:
            depth = "overview"
        return DepthDecision(
            depth=depth,  # type: ignore[arg-type]
            question=str(raw.get("question", "") or ""),
            reasoning=str(raw.get("reasoning", "") or "depth fallback"),
        )
    if isinstance(raw, str):
        parsed = _extract_json(raw)
        if parsed is not None:
            return _coerce_to_depth(parsed)
    return DepthDecision(depth="overview", question="", reasoning="depth fallback")


def _coerce_to_content(raw: object, title_fallback: str) -> ExtractionContent:
    """Best-effort conversion of loose LLM output into ExtractionContent."""
    if isinstance(raw, ExtractionContent):
        return raw
    if isinstance(raw, dict):
        try:
            return ExtractionContent.model_validate(raw)
        except ValidationError:
            pass
        sections = []
        for entry in raw.get("sections", []) or []:
            if isinstance(entry, dict):
                sections.append(
                    ReportSection(
                        heading=str(entry.get("heading", "") or ""),
                        body=str(entry.get("body", "") or ""),
                    )
                )
            elif entry:
                sections.append(ReportSection(heading="", body=str(entry)))
        sources = [str(s) for s in (raw.get("sources", []) or []) if s]
        return ExtractionContent(
            title=str(raw.get("title", "") or title_fallback),
            summary=str(raw.get("summary", "") or ""),
            sections=sections,
            sources=sources,
        )
    if isinstance(raw, str):
        parsed = _extract_json(raw)
        if parsed is not None:
            return _coerce_to_content(parsed, title_fallback)
        # Plain-text report: keep it as the summary.
        return ExtractionContent(title=title_fallback, summary=raw)
    return ExtractionContent(title=title_fallback, summary="")


def _interpret_answer(answer: str) -> str:
    """Map the human's interrupt answer to a depth. Defaults to overview."""
    t = (answer or "").strip().lower()
    if not t:
        return "overview"
    if any(hint in t for hint in _DETAILED_ANSWER_HINTS):
        return "detailed"
    return "overview"


# PDF rendering now lives in artifacts.pdf; thin wrappers remain for back-compat.
from src.workflow.artifacts.pdf import build_pdf as _build_pdf_impl
from src.workflow.artifacts.pdf import sanitize as _sanitize


def _build_pdf(content: ExtractionContent, dest_path: str) -> None:
    """Deprecated wrapper — use artifacts.pdf.build_pdf directly."""
    _build_pdf_impl(content, dest_path)


def _content_to_text(message) -> str:
    """Deprecated alias for shared.message_utils.message_content_to_str."""
    return _content_to_text_shared(message)


async def _decide_depth(goal: str, task: str, snapshot: str) -> DepthDecision:
    """Keyword shortcut first, LLM DepthDecision otherwise (never raises)."""
    explicit = _explicit_depth(f"{goal} {task}")
    if explicit in ("overview", "detailed"):
        logger.info(f"[EXTRACT] Explicit depth hint -> {explicit}")
        return DepthDecision(depth=explicit, reasoning="explicit user intent")  # type: ignore[arg-type]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DEPTH_DECIDER_PROMPT),
            (
                "human",
                "GOAL:\n{goal}\n\nDELEGATED TASK:\n{task}\n\nPAGE CONTENT (truncated):\n{snapshot}",
            ),
        ]
    )
    invoke_args = {
        "goal": goal or "(none)",
        "task": task or "(none)",
        "snapshot": (snapshot or "")[:_SNAPSHOT_CHARS] or "(empty)",
    }
    return await ainvoke_structured_with_fallback(
        prompt=prompt,
        invoke_args=invoke_args,
        structured_chain_factory=lambda: prompt | get_llm().with_structured_output(DepthDecision),
        raw_chain_factory=lambda: prompt | get_llm(),
        coerce=_coerce_to_depth,
        fallback=lambda: DepthDecision(depth="overview", reasoning="depth fallback"),
        log_scope="EXTRACT",
    )


async def _write_overview(goal: str, task: str, snapshot: str) -> str:
    """Single plain-LLM call producing the overview text."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", OVERVIEW_PROMPT),
            ("human", "GOAL:\n{goal}\n\nTASK:\n{task}\n\nPAGE CONTENT:\n{snapshot}"),
        ]
    )
    chain = prompt | get_llm()
    msg = await chain.ainvoke(
        {
            "goal": goal or "(none)",
            "task": task or "(none)",
            "snapshot": (snapshot or "")[:_SNAPSHOT_CHARS] or "(empty)",
        }
    )
    return _content_to_text_shared(msg).strip()


async def _compose_report(goal: str, task: str, snapshot: str, url: str) -> ExtractionContent:
    """Structured report content for the detailed PDF (never raises)."""
    title_fallback = (task or goal or "Report").strip()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REPORT_COMPOSE_PROMPT),
            (
                "human",
                "GOAL:\n{goal}\n\nTASK:\n{task}\n\nSOURCE URL:\n{url}\n\n"
                "PAGE CONTENT:\n{snapshot}",
            ),
        ]
    )
    invoke_args = {
        "goal": goal or "(none)",
        "task": task or "(none)",
        "url": url or "(unknown)",
        "snapshot": (snapshot or "")[:_SNAPSHOT_CHARS] or "(empty)",
    }
    return await ainvoke_structured_with_fallback(
        prompt=prompt,
        invoke_args=invoke_args,
        structured_chain_factory=lambda: prompt | get_llm().with_structured_output(ExtractionContent),
        raw_chain_factory=lambda: prompt | get_llm(),
        coerce=lambda raw: _coerce_to_content(raw, title_fallback),
        fallback=lambda: ExtractionContent(title=title_fallback, summary=""),
        log_scope="EXTRACT",
    )


async def extract_information_node(state: AgentState) -> dict:
    """Extract an overview (text) or detailed report (PDF) from the page.

    Flow: explicit-intent shortcut -> LLM DepthDecision -> interrupt only when
    ask_user -> overview text OR composed PDF report saved under artifacts/.
    """
    goal = state.get("goal", "") or ""
    task = state.get("current_delegated_task", "") or ""
    url = state.get("current_url", "") or ""
    task_id = state.get("task_id")
    existing_messages = list(state.get("messages", []) or [])
    logger.info("[EXTRACT] Extracting information (depth-aware)")

    snapshot = await read_page_snapshot()
    if not snapshot or "(page snapshot unavailable)" in snapshot.lower():
        logger.warning("[EXTRACT] No page content available")
        note = (
            f"Could not read page content for task {(task or goal)!r}; "
            "no overview or report could be produced."
        )
        return {
            "extracted_information": note,
            "extraction_format": "overview",
            "artifact_id": None,
            "artifact_path": None,
            "messages": existing_messages + [AIMessage(content=note)],
            "progress_verification": f"Extraction skipped: {note}",
            "all_actions": ["extract_information -> skipped (no page content)"],
        }

    decision = await _decide_depth(goal, task, snapshot)
    logger.info(f"[EXTRACT] Depth decision: {decision.depth} ({decision.reasoning})")

    # Human-in-the-loop: ask only when the decider says the intent is ambiguous.
    depth = decision.depth
    asked_human = False
    if depth == "ask_user":
        question = decision.question.strip() or (
            "I found relevant page content — want a quick overview here, "
            "or a detailed PDF report?"
        )
        logger.info("[EXTRACT] Pausing to ask overview vs detailed")
        answer = interrupt(question)
        answer = str(answer or "").strip()
        logger.info(f"[EXTRACT] Human depth answer: {answer!r}")
        existing_messages = existing_messages + [HumanMessage(content=answer)]
        asked_human = True
        depth = _interpret_answer(answer)
        if depth == "overview" and answer and not any(
            hint in answer.lower()
            for hint in (*_DETAILED_ANSWER_HINTS, *_OVERVIEW_ANSWER_HINTS)
        ):
            # Unrelated substantive reply: keep it visible as guidance and
            # fall back to the cheap non-blocking option.
            logger.info("[EXTRACT] Unrelated answer; defaulting to overview with note")
            return {
                "extracted_information": None,
                "extraction_format": "",
                "artifact_id": None,
                "artifact_path": None,
                "messages": existing_messages,
                "progress_verification": f"Human instruction: {answer}",
                "all_actions": ["extract_information -> deferred (human guidance noted)"],
            }

    if depth == "detailed":
        content = await _compose_report(goal, task, snapshot, url)
        store = ArtifactStore()
        thread_id = task_id or "default"
        try:
            artifact_id, dest_path = store.save_report(content, thread_id)
        except Exception as e:
            logger.exception(f"[EXTRACT] PDF build failed ({e}); degrading to overview")
            overview = content.summary or await _write_overview(goal, task, snapshot)
            return {
                "extracted_information": overview,
                "extraction_format": "overview",
                "artifact_id": None,
                "artifact_path": None,
                "messages": existing_messages + [AIMessage(content=overview)],
                "progress_verification": f"Extraction complete (overview, PDF failed): {content.title}",
                "all_actions": [f"extract_information -> overview (pdf failed) ({content.title})"],
            }
        logger.info(f"[EXTRACT] Detailed PDF saved: {dest_path}")

        summary = content.summary or "Detailed report ready."
        link_line = f"Detailed PDF report: {artifact_url(artifact_id)}"
        body = f"{content.title}\n\n{summary}\n\n{link_line}" if content.title else f"{summary}\n\n{link_line}"

        if task_id:
            try:
                from src.routers.common import push_artifact

                await push_artifact(
                    task_id,
                    artifact_id=artifact_id,
                    url=artifact_url(artifact_id),
                    title=content.title or "Detailed report",
                    message=summary,
                )
            except Exception as e:
                logger.warning(f"[EXTRACT] Artifact SSE push failed (non-fatal): {e}")

        return {
            "extracted_information": body,
            "extraction_format": "detailed",
            "artifact_id": artifact_id,
            "artifact_path": dest_path,
            "messages": existing_messages + [AIMessage(content=body)],
            "progress_verification": f"Extraction complete (detailed PDF): {content.title}",
            "all_actions": [f"extract_information -> detailed PDF ({content.title})"],
        }

    # Default: overview text.
    overview = await _write_overview(goal, task, snapshot)
    logger.info("[EXTRACT] Overview complete")
    return {
        "extracted_information": overview,
        "extraction_format": "overview",
        "artifact_id": None,
        "artifact_path": None,
        "messages": existing_messages + [AIMessage(content=overview)],
        "progress_verification": "Extraction complete (overview)",
        "all_actions": [
            "extract_information -> overview"
            + (" (after human choice)" if asked_human else "")
        ],
    }
