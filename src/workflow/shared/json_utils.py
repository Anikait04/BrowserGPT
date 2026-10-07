# json_utils.py — free-form LLM text → JSON extraction.
#
# Canonical home of `_extract_json` (previously private in planner.py).
# planner / delegation / verify / extraction must import from here.

from __future__ import annotations

import json
import re


def extract_json(text: str) -> object | None:
    """Extract the first JSON object from free-form LLM text.

    Handles code fences (```json ... ```) and surrounding prose.
    Returns None when no parseable object is found. Never raises.
    """
    if not text:
        return None
    # Strip code fences if present.
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text.strip())
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
    return None
