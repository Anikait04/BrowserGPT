from langchain.tools import tool
from browserplugin import Browser

_browser_instance = None

async def get_browser():
    """Get or create browser instance"""
    global _browser_instance
    if _browser_instance is None:
        _browser_instance = Browser()
        await _browser_instance.start()
    return _browser_instance

@tool
async def navigate(url: str) -> str:
    """Navigate to a specific URL in the browser.
    
    Args:
        url: The URL to navigate to (e.g., 'https://google.com' or 'github.com')
    
    Returns:
        Success message with page title and URL
    
    Examples:
        navigate('https://google.com')
        navigate('github.com')
    """
    browser = await get_browser()
    return await browser.go_to(url)

@tool
async def read_page(dummy: str = "") -> str:
    """Read the current page content to see what's displayed on the page.
    ALWAYS call this after navigating or clicking to understand what's on the page.
    
    Args:
        dummy: Not used, just required for tool signature
    
    Returns:
        The visible text content of the current page
    
    Examples:
        read_page()
    """
    browser = await get_browser()
    return await browser.read()

@tool
async def type_text(selector_and_text: str) -> str:
    """Type text into an input field or textarea.
    
    Args:
        selector_and_text: Format is 'CSS_SELECTOR|TEXT_TO_TYPE'
                          Optionally add '|enter' to press Enter after typing
    
    Returns:
        Success or error message
    
    Examples:
        type_text('input[name="q"]|LangGraph tutorial')
        type_text('textarea[name="q"]|Python asyncio|enter')
        type_text('#search-box|web scraping')
    
    Common selectors:
        - Google search: 'textarea[name="q"]' or 'input[name="q"]'
        - Generic search: 'input[type="search"]'
        - By ID: '#element-id'
        - By class: '.class-name'
    """
    browser = await get_browser()
    
    parts = selector_and_text.split("|")
    if len(parts) < 2:
        return "❌ Error: Format must be 'selector|text' or 'selector|text|enter'"
    
    selector = parts[0].strip()
    text = parts[1].strip()
    press_enter = len(parts) > 2 and parts[2].strip().lower() == 'enter'
    
    return await browser.type(selector, text, press_enter=press_enter)

@tool
async def click_element(selector: str) -> str:
    """Click an element on the page using a CSS selector.
    
    Args:
        selector: CSS selector for the element to click
    
    Returns:
        Success or error message
    
    Examples:
        click_element('button[type="submit"]')
        click_element('input[value="Google Search"]')
        click_element('.search-button')
        click_element('#submit-btn')
    
    Common patterns:
        - Submit button: 'button[type="submit"]' or 'input[type="submit"]'
        - By text: 'button:has-text("Search")'
        - Google search button: 'input[value="Google Search"]'
    """
    browser = await get_browser()
    return await browser.click(selector)

@tool
async def take_screenshot(filename: str = "screenshot.png") -> str:
    """Take a screenshot of the current browser view.
    Useful for debugging or capturing results.
    
    Args:
        filename: Name of the screenshot file (default: screenshot.png)
    
    Returns:
        Success message with filename
    
    Examples:
        take_screenshot()
        take_screenshot('search_results.png')
    """
    browser = await get_browser()
    return await browser.screenshot(filename)

@tool
async def wait_seconds(seconds: str) -> str:
    """Wait for a specified number of seconds.
    Useful when page is loading or after triggering actions.
    
    Args:
        seconds: Number of seconds to wait (can be decimal like '1.5')
    
    Returns:
        Confirmation message
    
    Examples:
        wait_seconds('2')
        wait_seconds('0.5')
    """
    browser = await get_browser()
    try:
        wait_time = float(seconds)
        return await browser.wait(wait_time)
    except ValueError:
        return f"❌ Error: '{seconds}' is not a valid number"

@tool
async def finish_task(summary: str = "Task completed") -> str:
    """Call this tool when you have successfully completed the user's goal.
    This will stop the agent and mark the task as finished.
    
    Args:
        summary: Brief summary of what was accomplished
    
    Returns:
        Completion message
    
    Examples:
        finish_task('Successfully searched Google for LangGraph')
        finish_task('Navigated to GitHub homepage')
    """
    return f"🎉 TASK COMPLETED: {summary}"

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