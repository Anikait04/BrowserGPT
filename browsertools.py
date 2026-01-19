from langchain.tools import tool
from browserplugin import Browser
from logs import logger
from typing import Optional

_browser_instance = None

async def get_browser():
    """Get or create browser instance"""
    global _browser_instance
    if _browser_instance is None:
        logger.info("Starting new browser instance")
        _browser_instance = Browser()
        await _browser_instance.start(headless=False)
        logger.info("Browser instance started")
    return _browser_instance


async def cleanup_browser():
    """Cleanup browser instance"""
    global _browser_instance
    if _browser_instance:
        await _browser_instance.close()
        _browser_instance = None
        logger.info("Browser instance cleaned up")


@tool
async def navigate(url: str) -> str:
    """
    Navigate to a specific URL in the browser.
    
    Args:
        url: The URL to navigate to (e.g., 'https://google.com' or 'google.com')
    
    Returns:
        Success message with page title and current URL
    
    Example: navigate('https://github.com')
    """
    logger.info(f"Navigate called with URL: {url}")
    browser = await get_browser()
    result = await browser.go_to(url)
    logger.info(f"Navigate result: {result[:100]}")
    return result


@tool
async def read_page(selector: str = "") -> str:
    """
    Read the current page content or content of a specific element.
    
    Args:
        selector: Optional CSS selector to read specific element. Leave empty to read entire page.
    
    Returns:
        Page content including URL, title, and text content
    
    Example: read_page() or read_page('.main-content')
    """
    logger.info(f"read_page called with selector: '{selector}'")
    browser = await get_browser()
    content = await browser.read(selector if selector else None)
    logger.debug(f"Page content preview: {content[:200]}")
    return content


@tool
async def type_text(input_string: str) -> str:
    """
    Type text into an input field or textarea.
    
    Args:
        input_string: Format 'selector|text' or 'selector|text|enter'
                     - selector: CSS selector for the input field
                     - text: Text to type
                     - enter: Optional, include 'enter' to press Enter after typing
    
    Returns:
        Success or error message
    
    Examples:
        type_text('input[name="q"]|hello world')
        type_text('textarea#message|Hello there|enter')
    """
    logger.info(f"type_text called with input: {input_string}")
    browser = await get_browser()
    
    parts = input_string.split("|")
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
    """
    Click an element on the page using CSS selector.
    
    Args:
        selector: CSS selector for the element to click
    
    Returns:
        Success or error message
    
    Examples:
        click_element('button.submit')
        click_element('#login-button')
        click_element('a[href="/about"]')
    """
    logger.info(f"click_element called with selector: {selector}")
    browser = await get_browser()
    result = await browser.click(selector)
    logger.info(f"click_element result: {result}")
    return result


@tool
async def click_text(text: str) -> str:
    """
    Click an element that contains specific text.
    
    Args:
        text: The text content to search for and click
    
    Returns:
        Success or error message
    
    Examples:
        click_text('Sign In')
        click_text('Submit')
        click_text('Learn More')
    """
    logger.info(f"click_text called with text: {text}")
    browser = await get_browser()
    result = await browser.click_text(text)
    logger.info(f"click_text result: {result}")
    return result


@tool
async def scroll_page(direction: str) -> str:
    """
    Scroll the page in a specified direction.
    
    Args:
        direction: One of 'down', 'up', 'bottom', 'top'
    
    Returns:
        Success message
    
    Examples:
        scroll_page('down')
        scroll_page('bottom')
        scroll_page('top')
    """
    logger.info(f"scroll_page called with direction: {direction}")
    browser = await get_browser()
    result = await browser.scroll(direction=direction.lower())
    logger.info(f"scroll_page result: {result}")
    return result


@tool
async def hover_element(selector: str) -> str:
    """
    Hover over an element to reveal hidden menus or tooltips.
    
    Args:
        selector: CSS selector for the element to hover over
    
    Returns:
        Success or error message
    
    Example: hover_element('.dropdown-menu')
    """
    logger.info(f"hover_element called with selector: {selector}")
    browser = await get_browser()
    result = await browser.hover(selector)
    logger.info(f"hover_element result: {result}")
    return result


@tool
async def select_dropdown(input_string: str) -> str:
    """
    Select an option from a dropdown menu.
    
    Args:
        input_string: Format 'selector|value=X' or 'selector|label=X' or 'selector|index=X'
    
    Returns:
        Success or error message
    
    Examples:
        select_dropdown('select#country|value=US')
        select_dropdown('select.language|label=English')
        select_dropdown('select#options|index=2')
    """
    logger.info(f"select_dropdown called with: {input_string}")
    browser = await get_browser()
    
    parts = input_string.split("|")
    if len(parts) != 2:
        return "Error: Format must be 'selector|value=X' or 'selector|label=X' or 'selector|index=X'"
    
    selector = parts[0].strip()
    option = parts[1].strip()
    
    if option.startswith("value="):
        result = await browser.select_dropdown(selector, value=option[6:])
    elif option.startswith("label="):
        result = await browser.select_dropdown(selector, label=option[6:])
    elif option.startswith("index="):
        try:
            index = int(option[6:])
            result = await browser.select_dropdown(selector, index=index)
        except ValueError:
            return f"Error: Invalid index '{option[6:]}'"
    else:
        return "Error: Option must start with 'value=', 'label=', or 'index='"
    
    logger.info(f"select_dropdown result: {result}")
    return result


@tool
async def wait_for_element(input_string: str) -> str:
    """
    Wait for a specific element to appear on the page.
    
    Args:
        input_string: Format 'selector' or 'selector|timeout' (timeout in milliseconds)
    
    Returns:
        Success or timeout message
    
    Examples:
        wait_for_element('.loading-spinner')
        wait_for_element('#content|5000')
    """
    logger.info(f"wait_for_element called with: {input_string}")
    browser = await get_browser()
    
    parts = input_string.split("|")
    selector = parts[0].strip()
    timeout = 15000
    
    if len(parts) > 1:
        try:
            timeout = int(parts[1].strip())
        except ValueError:
            return f"Error: Invalid timeout '{parts[1]}'"
    
    result = await browser.wait_for_selector(selector, timeout=timeout)
    logger.info(f"wait_for_element result: {result}")
    return result


@tool
async def wait_for_navigation(timeout: str = "30000") -> str:
    """
    Wait for page navigation to complete.
    
    Args:
        timeout: Timeout in milliseconds (default: 30000)
    
    Returns:
        Success or timeout message
    
    Example: wait_for_navigation('10000')
    """
    logger.info(f"wait_for_navigation called with timeout: {timeout}")
    browser = await get_browser()
    
    try:
        timeout_ms = int(timeout)
        result = await browser.wait_for_navigation(timeout=timeout_ms)
        logger.info(f"wait_for_navigation result: {result}")
        return result
    except ValueError:
        return f"Error: Invalid timeout '{timeout}'"


@tool
async def go_back() -> str:
    """
    Go back to the previous page in browser history.
    
    Returns:
        Success message with new URL
    
    Example: go_back()
    """
    logger.info("go_back called")
    browser = await get_browser()
    result = await browser.back()
    logger.info(f"go_back result: {result}")
    return result


@tool
async def go_forward() -> str:
    """
    Go forward in browser history.
    
    Returns:
        Success message with new URL
    
    Example: go_forward()
    """
    logger.info("go_forward called")
    browser = await get_browser()
    result = await browser.forward()
    logger.info(f"go_forward result: {result}")
    return result


@tool
async def reload_page() -> str:
    """
    Reload the current page.
    
    Returns:
        Success message
    
    Example: reload_page()
    """
    logger.info("reload_page called")
    browser = await get_browser()
    result = await browser.reload()
    logger.info(f"reload_page result: {result}")
    return result


@tool
async def get_element_attribute(input_string: str) -> str:
    """
    Get an attribute value from an element.
    
    Args:
        input_string: Format 'selector|attribute'
    
    Returns:
        Attribute value or error message
    
    Examples:
        get_element_attribute('a.download|href')
        get_element_attribute('img#logo|src')
        get_element_attribute('input#email|value')
    """
    logger.info(f"get_element_attribute called with: {input_string}")
    browser = await get_browser()
    
    parts = input_string.split("|")
    if len(parts) != 2:
        return "Error: Format must be 'selector|attribute'"
    
    selector = parts[0].strip()
    attribute = parts[1].strip()
    
    value = await browser.get_attribute(selector, attribute)
    result = f"Attribute '{attribute}' value: {value}" if value else f"Could not get attribute '{attribute}'"
    logger.info(f"get_element_attribute result: {result}")
    return result


@tool
async def execute_javascript(script: str) -> str:
    """
    Execute custom JavaScript code on the page.
    
    Args:
        script: JavaScript code to execute
    
    Returns:
        Result of JavaScript execution or error message
    
    Examples:
        execute_javascript('document.title')
        execute_javascript('window.scrollTo(0, 500)')
        execute_javascript('document.querySelector(".price").innerText')
    """
    logger.info(f"execute_javascript called with script: {script[:100]}")
    browser = await get_browser()
    result = await browser.evaluate(script)
    logger.info(f"execute_javascript result: {result}")
    return str(result)


@tool
async def take_screenshot(filename: str = "screenshot.png") -> str:
    """
    Take a screenshot of the current browser view.
    
    Args:
        filename: Path/filename to save screenshot (default: 'screenshot.png')
    
    Returns:
        Success message with filename
    
    Examples:
        take_screenshot()
        take_screenshot('page_capture.png')
    """
    logger.info(f"take_screenshot called, saving to: {filename}")
    browser = await get_browser()
    result = await browser.screenshot(filename, full_page=False)
    logger.info(f"Screenshot saved: {filename}")
    return result


@tool
async def wait_seconds(seconds: str) -> str:
    """
    Wait for specified number of seconds.
    
    Args:
        seconds: Number of seconds to wait (can be decimal like '1.5')
    
    Returns:
        Success message
    
    Examples:
        wait_seconds('2')
        wait_seconds('0.5')
    """
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
    """
    Mark the task as finished and cleanup browser.
    
    Args:
        summary: Summary of what was accomplished
    
    Returns:
        Completion message
    
    Example: finish_task('Successfully logged in and extracted data')
    """
    logger.info(f"finish_task called with summary: {summary}")
    await cleanup_browser()
    return f"TASK COMPLETED: {summary}"


# Export all tools
tools = [
    navigate,
    read_page,
    type_text,
    click_element,
    click_text,
    scroll_page,
    hover_element,
    select_dropdown,
    wait_for_element,
    wait_for_navigation,
    go_back,
    go_forward,
    reload_page,
    get_element_attribute,
    execute_javascript,
    take_screenshot,
    wait_seconds,
    finish_task
]