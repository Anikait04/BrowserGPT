from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import asyncio
from logs import logger  # import your centralized logger here

class Browser:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None
        self.current_url = None
    
    async def start(self):
        """Initialize browser instance"""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=False,
                args=[
                    '--start-maximized',
                    '--disable-blink-features=AutomationControlled'  # Avoid detection
                ]
            )
            self.page = await self.browser.new_page(no_viewport=True)
            self.page.set_default_timeout(20000)  # 20 second default timeout
            logger.info("Browser started successfully")
        except Exception:
            logger.exception("Failed to start browser")
            raise

    async def go_to(self, url: str) -> str:
        """Navigate to a URL"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = f'https://{url}'
            
            logger.info(f"Navigating to: {url}")
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(1.5)
            
            self.current_url = self.page.url
            title = await self.page.title()
            
            logger.info(f"Loaded: {title}")
            return f"Successfully navigated to {url}\nPage title: {title}\nCurrent URL: {self.current_url}"
        
        except PlaywrightTimeoutError:
            logger.error(f"Timeout while loading URL: {url}")
            return f"Timeout: Could not load {url} within 30 seconds"
        except Exception:
            logger.exception(f"Navigation error for URL: {url}")
            return f"Navigation error: {url}"

    async def click(self, selector: str) -> str:
        """Click an element by CSS selector"""
        try:
            logger.info(f"Attempting to click: {selector}")
            
            await self.page.wait_for_selector(selector, state="visible", timeout=10000)
            await self.page.locator(selector).scroll_into_view_if_needed()
            await asyncio.sleep(0.3)
            
            try:
                await self.page.click(selector, timeout=5000)
            except Exception:
                await self.page.locator(selector).click(force=True)
            
            await asyncio.sleep(0.8)
            logger.info(f"Clicked: {selector}")
            return f"Successfully clicked element: {selector}"
        
        except PlaywrightTimeoutError:
            alternatives = self._get_alternative_selectors(selector)
            for alt in alternatives:
                try:
                    await self.page.wait_for_selector(alt, state="visible", timeout=3000)
                    await self.page.click(alt)
                    logger.info(f"Clicked alternative: {alt}")
                    return f"Clicked element using alternative selector: {alt}"
                except Exception:
                    continue
            
            logger.error(f"Could not find clickable element: {selector}")
            return f"Could not find clickable element: {selector}"
        except Exception:
            logger.exception(f"Click error on selector: {selector}")
            return f"Click error: {selector}"

    async def type(self, selector: str, text: str, press_enter: bool = False) -> str:
        """Type text into an input field"""
        try:
            logger.info(f"Typing '{text}' into: {selector}")
            
            await self.page.wait_for_selector(selector, state="visible", timeout=10000)
            await self.page.fill(selector, "")
            await asyncio.sleep(0.2)
            await self.page.type(selector, text, delay=50)
            
            if press_enter:
                await self.page.keyboard.press("Enter")
                await asyncio.sleep(1)
                logger.info("Pressed Enter")
            
            await asyncio.sleep(0.5)
            logger.info(f"Typed into: {selector}")
            return f"Successfully typed '{text}' into {selector}" + (" and pressed Enter" if press_enter else "")
        
        except PlaywrightTimeoutError:
            alternatives = [
                'textarea[name="q"]',
                'input[name="q"]',
                'input[title="Search"]',
                'textarea[title="Search"]',
                '#APjFqb'
            ]
            
            for alt in alternatives:
                try:
                    await self.page.wait_for_selector(alt, state="visible", timeout=3000)
                    await self.page.fill(alt, text)
                    if press_enter:
                        await self.page.keyboard.press("Enter")
                        await asyncio.sleep(1)
                    logger.info(f"Typed into alternative: {alt}")
                    return f"Typed using alternative selector: {alt}"
                except Exception:
                    continue
            
            logger.error(f"Could not find input field: {selector}. Tried alternatives: {alternatives}")
            return f"Could not find input field: {selector}. Tried alternatives: {alternatives}"
        except Exception:
            logger.exception(f"Type error on selector: {selector}")
            return f"Type error: {selector}"

    async def read(self) -> str:
        """Read visible page content"""
        try:
            logger.info("Reading page content...")
            content = await self.page.evaluate("""
                () => {
                    const clone = document.body.cloneNode(true);
                    clone.querySelectorAll('script, style, noscript').forEach(el => el.remove());
                    return clone.innerText;
                }
            """)
            content = content[:4000]
            title = await self.page.title()
            url = self.page.url
            
            logger.info(f"Read {len(content)} characters")
            return f"=== PAGE CONTENT ===\nURL: {url}\nTitle: {title}\n\nContent:\n{content}"
        
        except Exception:
            logger.exception("Read error")
            return "Read error occurred"

    async def screenshot(self, path: str = "screenshot.png") -> str:
        """Take a screenshot"""
        try:
            logger.info(f"Taking screenshot: {path}")
            await self.page.screenshot(path=path, full_page=False)
            logger.info(f"Screenshot saved: {path}")
            return f"Screenshot saved to {path}"
        except Exception:
            logger.exception("Screenshot error")
            return "Screenshot error occurred"

    async def wait(self, seconds: float = 1.0) -> str:
        logger.debug(f"Waiting for {seconds} seconds")
        await asyncio.sleep(seconds)
        return f"Waited {seconds} seconds"

    async def close(self):
        try:
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("Browser closed successfully")
        except Exception:
            logger.exception("Error closing browser")

    def _get_alternative_selectors(self, selector: str) -> list:
        alternatives = []
        if 'button' in selector.lower():
            alternatives.extend([
                'button[type="submit"]',
                'input[type="submit"]',
                'button.submit',
                '[role="button"]'
            ])
        if 'search' in selector.lower():
            alternatives.extend([
                'input[name="q"]',
                'textarea[name="q"]',
                'input[type="search"]',
                '[aria-label*="Search"]'
            ])
        return alternatives
