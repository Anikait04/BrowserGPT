# selectors.py — centralized fallback CSS selectors.
#
# Previously three near-identical inline lists in browserplugin.py
# (type vs type_and_enter vs _get_alternative_selectors). One place to tune.

from __future__ import annotations

# Generic search-input fallbacks (Google-oriented; used when the agent's
# selector from read_page is stale or the page changed).
SEARCH_INPUT_ALTERNATIVES: tuple[str, ...] = (
    'textarea[name="q"]',
    'input[name="q"]',
    'input[title="Search"]',
    'textarea[title="Search"]',
    "#APjFqb",
)

BUTTON_ALTERNATIVES: tuple[str, ...] = (
    'button[type="submit"]',
    'input[type="submit"]',
    "button.submit",
    '[role="button"]',
)

SEARCH_ALTERNATIVES: tuple[str, ...] = (
    'input[name="q"]',
    'textarea[name="q"]',
    'input[type="search"]',
    '[aria-label*="Search"]',
)


def alternative_selectors(selector: str) -> list[str]:
    """Heuristic fallbacks for a failed selector (button/search aware)."""
    alternatives: list[str] = []
    lowered = (selector or "").lower()
    if "button" in lowered:
        alternatives.extend(BUTTON_ALTERNATIVES)
    if "search" in lowered:
        alternatives.extend(SEARCH_ALTERNATIVES)
    return alternatives
