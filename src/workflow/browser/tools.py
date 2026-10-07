# tools.py — LangChain browser tools for the deep navigation agent.
#
# Canonical home of the @tool wrappers (moved from browsertools.py, kept as a
# shim). Tools acquire the browser via browser.manager.get_browser().

from __future__ import annotations

from langchain.tools import tool
from pydantic import BaseModel

from logs import logger
from src.workflow.browser.manager import get_browser


@tool
async def navigate(url: str) -> str:
    """Navigate to a specific URL in the browser."""
    logger.info(f"Navigate called with URL: {url}")
    browser = await get_browser()
    result = await browser.go_to(url)
    logger.info(f"Navigate result: {result[:100]}")  # Log preview
    return result


class TypeTextInput(BaseModel):
    selector: str
    value: str
    press_enter: bool = False


@tool(args_schema=TypeTextInput)
async def type_text(selector: str, value: str, press_enter: bool = False) -> str:
    """Type text into an input field or textarea."""
    logger.info(f"type_text called with selector={selector}, value={value}, press_enter={press_enter}")

    browser = await get_browser()
    result = await browser.type(selector, value, press_enter=press_enter)

    logger.info(f"type_text result: {result}")
    return result


class TypeAndEnterInput(BaseModel):
    selector: str
    value: str


@tool(args_schema=TypeAndEnterInput)
async def type_and_enter(selector: str, value: str) -> str:
    """Type text into an input field and immediately press Enter."""
    logger.info(f"type_and_enter called with selector={selector}, value={value}")

    browser = await get_browser()
    result = await browser.type_and_enter(selector, value)

    logger.info(f"type_and_enter result: {result}")
    return result


@tool
async def click_element(selector: str) -> str:
    """Click an element on the page using CSS selector."""
    logger.info(f"click_element called with selector: {selector}")
    browser = await get_browser()
    result = await browser.click(selector)
    logger.info(f"click_element result: {result}")
    return result


@tool
async def wait_seconds(seconds: str) -> str:
    """Wait for specified number of seconds."""
    logger.info(f"wait_seconds called for {seconds} seconds")
    browser = await get_browser()
    try:
        wait_time = float(seconds)
        result = await browser.wait(wait_time)
        logger.info(f"Waited for {wait_time} seconds")
        return result
    except ValueError:
        logger.error(f"wait_seconds error: invalid number '{seconds}'")
        return f"Error: '{seconds}' is not a valid number"


__all__ = [
    "navigate",
    "type_text",
    "TypeTextInput",
    "type_and_enter",
    "TypeAndEnterInput",
    "click_element",
    "wait_seconds",
]
