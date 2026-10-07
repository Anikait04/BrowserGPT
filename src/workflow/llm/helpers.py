# llm_helpers.py — shared structured-output → raw-JSON fallback helper.
#
# All four LLM nodes (planner / delegation / verify / extraction) previously
# duplicated this try/except dance. Centralizing it gives one retry policy,
# one log shape, and one place to unit-test fallback behavior.

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from logs import logger
from src.workflow.shared.json_utils import extract_json
from src.workflow.shared.message_utils import message_content_to_str

T = TypeVar("T")


async def ainvoke_structured_with_fallback(
    *,
    prompt: ChatPromptTemplate,
    invoke_args: dict[str, Any],
    structured_chain_factory: Callable[[], Any],
    raw_chain_factory: Callable[[], Any],
    coerce: Callable[[object], T],
    fallback: Callable[[], T],
    log_scope: str,
) -> T:
    """Invoke an LLM with structured output, falling back to raw-JSON parsing.

    1. Preferred path: structured output (aliases handle legacy keys).
    2. Fallback: plain LLM call + manual JSON extraction + coerce mapping.
    3. Last resort: `fallback()` (must never raise).

    `structured_chain_factory` / `raw_chain_factory` are thunks so the caller
    controls which LLM client is used (standard vs navigation) and tests can
    inject fakes without touching the global cache.
    """
    # 1) Preferred path: structured output.
    try:
        result = await structured_chain_factory().ainvoke(invoke_args)
        if isinstance(result, dict):
            return coerce(result)
        return coerce(result)
    except (OutputParserException, ValidationError) as e:
        logger.warning(f"[{log_scope}] Structured output failed ({e}); trying raw-JSON fallback")
    except Exception as e:
        logger.warning(f"[{log_scope}] LLM call failed ({e}); trying raw-JSON fallback")

    # 2) Fallback: plain LLM call + manual JSON extraction.
    try:
        raw_msg = await raw_chain_factory().ainvoke(invoke_args)
        raw_text = message_content_to_str(raw_msg)
        parsed = extract_json(str(raw_text))
        coerced = coerce(parsed if parsed is not None else str(raw_text))
        logger.warning(f"[{log_scope}] Raw fallback produced: {coerced!r}")
        return coerced
    except Exception as e:
        logger.error(f"[{log_scope}] Raw fallback also failed ({e}); using safe default")
        return fallback()


def flatten_raw_content(raw_msg: Any) -> str:
    """Back-compat flatten for raw LLM messages (prefers shared helper)."""
    return message_content_to_str(raw_msg)


__all__ = ["ainvoke_structured_with_fallback", "flatten_raw_content"]
