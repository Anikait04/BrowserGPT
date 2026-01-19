from langchain.tools import tool
from browserplugin import Browser
from logs import logger

_browser_instance = None

async def get_browser():
    """Get or create browser instance"""
    global _browser_instance
    if _browser_instance is None:
        logger.info("Starting new browser instance")
        _browser_instance = Browser()
        await _browser_instance.start()
        logger.info("Browser instance started")
    return _browser_instance


@tool
async def navigate(url: str) -> str:
    """Navigate to a specific URL in the browser."""
    logger.info(f"Navigate called with URL: {url}")
    browser = await get_browser()
    result = await browser.go_to(url)
    logger.info(f"Navigate result: {result[:100]}")  # Log preview
    return result


@tool
async def read_page(dummy: str = "") -> str:
    """Read the current page content."""
    logger.info("read_page called")
    browser = await get_browser()
    content = await browser.read()
    logger.debug(f"Page content preview: {content[:200]}")
    return content


@tool
async def type_text(selector_and_text: str) -> str:
    """Type text into an input field or textarea."""
    logger.info(f"type_text called with input: {selector_and_text}")
    browser = await get_browser()
    
    parts = selector_and_text.split("|")
    if len(parts) < 2:
        logger.error("type_text error: invalid format")
        return "Error: Format must be 'selector|text' or 'selector|text|enter'"
    
    selector = parts[0].strip()
    text = parts[1].strip()
    press_enter = len(parts) > 2 and parts[2].strip().lower() == 'enter'

    result = await browser.type(selector, text, press_enter=press_enter)
    logger.info(f"type_text result: {result}")
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
async def take_screenshot(filename: str = "screenshot.png") -> str:
    """Take a screenshot of the current browser view."""
    logger.info(f"take_screenshot called, saving to: {filename}")
    browser = await get_browser()
    result = await browser.screenshot(filename)
    logger.info(f"Screenshot saved: {filename}")
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


@tool
async def finish_task(summary: str = "Task completed") -> str:
    """Mark the task as finished."""
    logger.info(f"finish_task called with summary: {summary}")
    return f"TASK COMPLETED: {summary}"


# Export all tools
tools = [
    navigate,
    read_page,
    type_text,
    click_element,
    take_screenshot,
    wait_seconds,
    finish_task
]
