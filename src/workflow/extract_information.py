# extract_information.py — depth-aware extraction node.
#
# Decides whether the user needs an overview (short text) or detailed
# information (well-written PDF report), asking the human only when ambiguous:
# - explicit intent in goal/task ("brief", "pdf", ...) decides without asking
# - otherwise an LLM DepthDecision picks overview / detailed / ask_user
# - ask_user pauses via interrupt(); unclear answers default to overview

import os
import uuid

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt
from pydantic import ValidationError

import config
from logs import logger
from src.workflow.agent_state import AgentState
from src.workflow.llm import get_llm
from src.workflow.planner import _extract_json
from src.workflow.prompt import (
    DEPTH_DECIDER_PROMPT,
    OVERVIEW_PROMPT,
    REPORT_COMPOSE_PROMPT,
)
from src.workflow.schemas import DepthDecision, ExtractionContent, ReportSection
from src.workflow.verify import _page_snapshot

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

_SNAPSHOT_CHARS = 6000


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


def _sanitize(text: str) -> str:
    """Strip characters the PDF core fonts cannot render."""
    return (text or "").encode("latin-1", "replace").decode("latin-1")


def _build_pdf(content: ExtractionContent, dest_path: str) -> None:
    """Render the report content to a PDF file."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    class ReportPDF(FPDF):
        def footer(self):
            self.set_y(-15)
            self.set_font("helvetica", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    pdf = ReportPDF()
    pdf.alias_nb_pages("{nb}")
    pdf.set_auto_page_break(True, margin=20)
    pdf.add_page()

    def _line(style: str, size: int, text: str, height: int) -> None:
        pdf.set_font("helvetica", style, size)
        pdf.multi_cell(0, height, _sanitize(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    _line("B", 20, content.title or "Report", 10)
    pdf.ln(4)

    if content.summary:
        _line("I", 11, content.summary, 7)
        pdf.ln(4)

    for section in content.sections:
        if section.heading:
            _line("B", 14, section.heading, 8)
        if section.body:
            _line("", 11, section.body, 7)
        pdf.ln(3)

    if content.sources:
        _line("B", 12, "Sources", 8)
        for source in content.sources:
            _line("", 10, f"- {source}", 6)

    pdf.output(dest_path)


def _content_to_text(message) -> str:
    """Flatten an LLM message payload to plain text."""
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

    try:
        chain = prompt | get_llm().with_structured_output(DepthDecision)
        decision = await chain.ainvoke(invoke_args)
        if isinstance(decision, dict):
            return _coerce_to_depth(decision)
        if isinstance(decision, DepthDecision):
            return decision
        return _coerce_to_depth(decision)
    except (OutputParserException, ValidationError) as e:
        logger.warning(f"[EXTRACT] Depth structured output failed ({e}); trying raw fallback")
    except Exception as e:
        logger.warning(f"[EXTRACT] Depth LLM call failed ({e}); trying raw fallback")

    try:
        raw_chain = prompt | get_llm()
        raw_msg = await raw_chain.ainvoke(invoke_args)
        parsed = _extract_json(_content_to_text(raw_msg))
        return _coerce_to_depth(parsed if parsed is not None else _content_to_text(raw_msg))
    except Exception as e:
        logger.error(f"[EXTRACT] Depth fallback failed ({e}); defaulting to overview")
        return DepthDecision(depth="overview", reasoning="depth fallback")


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
    return _content_to_text(msg).strip()


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

    try:
        chain = prompt | get_llm().with_structured_output(ExtractionContent)
        content = await chain.ainvoke(invoke_args)
        if isinstance(content, (dict, str)):
            return _coerce_to_content(content, title_fallback)
        if isinstance(content, ExtractionContent):
            return content
        return _coerce_to_content(content, title_fallback)
    except (OutputParserException, ValidationError) as e:
        logger.warning(f"[EXTRACT] Report structured output failed ({e}); trying raw fallback")
    except Exception as e:
        logger.warning(f"[EXTRACT] Report LLM call failed ({e}); trying raw fallback")

    try:
        raw_chain = prompt | get_llm()
        raw_msg = await raw_chain.ainvoke(invoke_args)
        text = _content_to_text(raw_msg)
        parsed = _extract_json(text)
        return _coerce_to_content(parsed if parsed is not None else text, title_fallback)
    except Exception as e:
        logger.error(f"[EXTRACT] Report fallback failed ({e}); using empty report")
        return ExtractionContent(title=title_fallback, summary="")


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

    snapshot = await _page_snapshot()
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
        artifact_id = uuid.uuid4().hex
        thread_id = task_id or "default"
        dest_dir = os.path.join(config.ARTIFACTS_DIR, thread_id)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, f"{artifact_id}.pdf")
        try:
            _build_pdf(content, dest_path)
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
        link_line = f"Detailed PDF report: /nav/artifact/{artifact_id}"
        body = f"{content.title}\n\n{summary}\n\n{link_line}" if content.title else f"{summary}\n\n{link_line}"

        if task_id:
            try:
                from src.routers.agent_router import push_artifact

                await push_artifact(
                    task_id,
                    artifact_id=artifact_id,
                    url=f"/nav/artifact/{artifact_id}",
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
