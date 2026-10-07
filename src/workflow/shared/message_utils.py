# message_utils.py — LLM message flattening + conversation history formatting.
#
# Canonical home of the three near-identical helpers previously scattered as:
#   planner._message_text, navigation._content_to_str,
#   extract_information._content_to_text.
# All nodes must import from here.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.workflow.constants import HISTORY_MAX_CHARS, HISTORY_MAX_ENTRIES


def message_content_to_str(content: Any) -> str:
    """Flatten an LLM message payload (str | list[blocks] | message) to text.

    Accepts either a raw `content` value or a message object exposing
    `.content`. Dict blocks contribute their "text" key; anything else is
    stringified. Never raises — falls back to str(content).
    """
    raw = getattr(content, "content", content)
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for part in raw:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return " ".join(p for p in parts if p)
    return str(raw or "")


def format_history(state: Mapping[str, Any]) -> str:
    """Compact conversation context for grounding follow-ups ("this", "it").

    Covers the current page, prior extraction outcomes, and recent messages.
    Bounded in entries and characters to cap token usage.
    """
    lines: list[str] = []
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
    for msg in messages[-HISTORY_MAX_ENTRIES:]:
        role = getattr(msg, "type", "") or "message"
        text = message_content_to_str(msg).strip()[:300]
        if text:
            lines.append(f"[{role}] {text}")
    history = "\n".join(lines).strip()
    return history[:HISTORY_MAX_CHARS] or "(no prior conversation)"


def format_plan(entire_plan: list[Any], step_count: int) -> str:
    """Render plan steps with completion markers for LLM prompts."""
    if not entire_plan:
        return "(no plan steps)"
    lines = []
    for i, step in enumerate(entire_plan):
        marker = "x" if i < step_count else " "
        lines.append(f"[{marker}] {i + 1}. {step}")
    return "\n".join(lines)
