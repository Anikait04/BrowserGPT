# page_reader.py — LLM-free observation tool.
#
# Extracts the element-scraping logic that used to live inside
# `observe_and_choose_node` (nodes.py) into plain functions and exposes it as a
# `read_page` tool for the deep navigation agent. There is deliberately NO LLM
# anywhere in this module — the agent decides what to do with the observation.

import re

from langchain.tools import tool

from logs import logger
from src.workflow.browsertools import get_browser


async def _get_stable_selector(el) -> str:
    """Pick the most stable CSS selector available for an element."""
    # 1. aria-label (most stable)
    aria = await el.get_attribute("aria-label")
    if aria:
        return f"[aria-label='{aria}']"

    # 2. name attribute
    name = await el.get_attribute("name")
    if name:
        return f"[name='{name}']"

    # 3. placeholder
    placeholder = await el.get_attribute("placeholder")
    if placeholder:
        return f"[placeholder='{placeholder}']"

    # 4. data-testid
    testid = await el.get_attribute("data-testid")
    if testid:
        return f"[data-testid='{testid}']"

    # 5. type + role combo
    el_type = await el.get_attribute("type")
    role = await el.get_attribute("role")
    if el_type:
        return f"input[type='{el_type}']"
    if role:
        return f"[role='{role}']"

    # 6. Last resort: escape the ID properly
    el_id = await el.get_attribute("id")
    if el_id:
        return f'[id="{el_id}"]'  # attribute selector, not #id — avoids CSS parsing issues

    tag = await el.evaluate("e => e.tagName.toLowerCase()")
    return tag


async def _build_candidate(el, el_type: str, idx: int):
    """Build an element candidate dict, or return None if it has no usable label."""
    async def safe_text(e):
        try:
            return (await e.inner_text()).strip()
        except Exception:
            return ""

    label = (
        await el.get_attribute("aria-label")
        or await el.get_attribute("placeholder")
        or await el.get_attribute("name")
        or await safe_text(el)
    )

    if not label:
        return None, idx

    label = re.sub(r"\s+", " ", label).strip()

    selector = await _get_stable_selector(el)

    candidate = {
        "id": idx,
        "type": el_type,
        "label": label,
        "selector": selector,
    }

    return candidate, idx + 1


async def get_interactable_elements(page) -> list[dict]:
    """Scrape visible interactive elements from a Playwright page.

    Returns candidates shaped like {id, type, label, selector}.
    """
    async def visible(el):
        try:
            return await el.is_visible()
        except Exception:
            return False

    elements = []
    idx = 1

    groups = [
        ("input, textarea", "input"),
        ("button, [role='button'], [onclick]", "button"),
        ("a[href]", "link"),
    ]

    for css, el_type in groups:
        for el in await page.locator(css).all():
            if not await visible(el):
                continue
            candidate, idx = await _build_candidate(el, el_type, idx)
            if candidate:
                elements.append(candidate)

    return elements


@tool
async def read_page(dummy: str = "") -> str:
    """Observe the current page: URL, title, readable text and numbered interactive elements."""
    logger.info("[NAVIGATION] read_page called")
    browser = await get_browser()
    page_content = await browser.read()
    elements = await get_interactable_elements(browser.page)

    url = ""
    try:
        url = browser.page.url
    except Exception:
        pass
    logger.info(f"[NAVIGATION] Read page: url={url} elements={len(elements)}")

    lines = [page_content, "", "=== INTERACTIVE ELEMENTS ==="]
    if elements:
        for c in elements:
            lines.append(f"[{c['id']}] {c['type']} - {c['label']} - {c['selector']}")
    else:
        lines.append("(no interactive elements found)")

    return "\n".join(lines)
