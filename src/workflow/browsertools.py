from langchain.tools import tool
from src.workflow.browserplugin import Browser
from logs import logger

_browser_instance = None

async def get_browser():
    """Get or create browser instance"""
    global _browser_instance

    if _browser_instance is not None:
        try:
            _ = _browser_instance.page.url  # throws if browser/page is closed
        except Exception:
            logger.warning("Browser instance is stale, reinitializing...")
            _browser_instance = None

    if _browser_instance is None:
        logger.info("Starting new browser instance")
        _browser_instance = Browser()
        await _browser_instance.start()
        logger.info("Browser instance started")

    return _browser_instance

async def close_browser():
    """Close browser and reset singleton so next call gets a fresh instance."""
    global _browser_instance
    if _browser_instance is not None:
        try:
            await _browser_instance.close()
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")
        finally:
            _browser_instance = None


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


from pydantic import BaseModel

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
    click_element,
    type_text,
    read_page,
    type_and_enter,
    finish_task
]
