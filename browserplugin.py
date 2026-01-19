from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import asyncio
from logs import logger
from typing import Optional, List, Dict, Any
import re

class Browser:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.current_url = None
    
    async def start(self, headless: bool = False, user_agent: Optional[str] = None):
        """Initialize browser instance with enhanced settings"""
        try:
            self.playwright = await async_playwright().start()
            
            # Launch browser with stealth settings
            self.browser = await self.playwright.chromium.launch(
                headless=headless,
                args=[
                    '--start-maximized',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process'
                ]
            )
            
            # Create context with realistic settings
            self.context = await self.browser.new_context(
                no_viewport=True,
                user_agent=user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                locale='en-US',
                timezone_id='America/New_York',
                permissions=['geolocation', 'notifications'],
                java_script_enabled=True,
                ignore_https_errors=True
            )
            
            # Add stealth scripts
            await self.context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
                window.chrome = {runtime: {}};
            """)
            
            self.page = await self.context.new_page()
            self.page.set_default_timeout(30000)
            
            # Set up request interception to block unnecessary resources
            await self.page.route("**/*", lambda route: (
                route.abort() if route.request.resource_type in ["image", "font"] 
                else route.continue_()
            ))
            
            logger.info("Browser started successfully")
        except Exception:
            logger.exception("Failed to start browser")
            raise

    async def go_to(self, url: str, wait_until: str = "domcontentloaded") -> str:
        """Navigate to a URL with enhanced error handling"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = f'https://{url}'
            
            logger.info(f"Navigating to: {url}")
            
            # Try multiple wait strategies
            wait_strategies = [wait_until, "load", "networkidle"]
            for strategy in wait_strategies:
                try:
                    await self.page.goto(url, wait_until=strategy, timeout=45000)
                    break
                except PlaywrightTimeoutError:
                    if strategy == wait_strategies[-1]:
                        raise
                    continue
            
            await asyncio.sleep(2)
            
            self.current_url = self.page.url
            title = await self.page.title()
            
            # Handle cookie consent popups
            await self._handle_popups()
            
            logger.info(f"Loaded: {title}")
            return f"Successfully navigated to {url}\nPage title: {title}\nCurrent URL: {self.current_url}"
        
        except PlaywrightTimeoutError:
            logger.error(f"Timeout while loading URL: {url}")
            return f"Timeout: Could not load {url} within timeout period"
        except Exception:
            logger.exception(f"Navigation error for URL: {url}")
            return f"Navigation error: {url}"

    async def click(self, selector: str, method: str = "auto") -> str:
        """Click an element with multiple fallback strategies"""
        try:
            logger.info(f"Attempting to click: {selector}")
            
            # Wait for element
            await self.page.wait_for_selector(selector, state="visible", timeout=15000)
            
            # Scroll into view
            await self.page.locator(selector).scroll_into_view_if_needed()
            await asyncio.sleep(0.5)
            
            # Try different click methods
            click_methods = {
                "auto": lambda: self.page.click(selector, timeout=5000),
                "force": lambda: self.page.locator(selector).click(force=True),
                "js": lambda: self.page.evaluate(f'document.querySelector("{selector}").click()'),
                "dispatch": lambda: self.page.dispatch_event(selector, "click")
            }
            
            if method == "auto":
                for method_name, click_func in click_methods.items():
                    try:
                        await click_func()
                        break
                    except Exception as e:
                        if method_name == "dispatch":
                            raise
                        continue
            else:
                await click_methods.get(method, click_methods["auto"])()
            
            await asyncio.sleep(1)
            logger.info(f"Clicked: {selector}")
            return f"Successfully clicked element: {selector}"
        
        except PlaywrightTimeoutError:
            # Try alternative selectors
            alternatives = self._get_smart_alternatives(selector)
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

    async def click_text(self, text: str, exact: bool = False) -> str:
        """Click element containing specific text"""
        try:
            logger.info(f"Clicking element with text: {text}")
            
            if exact:
                selector = f"text='{text}'"
            else:
                selector = f"text={text}"
            
            await self.page.locator(selector).first.click(timeout=10000)
            await asyncio.sleep(1)
            
            logger.info(f"Clicked text: {text}")
            return f"Successfully clicked element with text: {text}"
        except Exception:
            logger.exception(f"Click text error: {text}")
            return f"Could not click text: {text}"

    async def type(self, selector: str, text: str, press_enter: bool = False, clear_first: bool = True) -> str:
        """Type text into an input field with enhanced handling"""
        try:
            logger.info(f"Typing '{text}' into: {selector}")
            
            await self.page.wait_for_selector(selector, state="visible", timeout=15000)
            
            # Focus the element
            await self.page.focus(selector)
            await asyncio.sleep(0.3)
            
            if clear_first:
                # Clear existing content
                await self.page.fill(selector, "")
                await asyncio.sleep(0.2)
            
            # Type with human-like delay
            await self.page.type(selector, text, delay=50)
            
            if press_enter:
                await self.page.keyboard.press("Enter")
                await asyncio.sleep(1.5)
                logger.info("⏎ Pressed Enter")
            
            await asyncio.sleep(0.5)
            logger.info(f"Typed into: {selector}")
            return f"Successfully typed '{text}' into {selector}" + (" and pressed Enter" if press_enter else "")
        
        except PlaywrightTimeoutError:
            # Try smart alternatives
            alternatives = self._get_smart_alternatives(selector)
            
            for alt in alternatives:
                try:
                    await self.page.wait_for_selector(alt, state="visible", timeout=3000)
                    await self.page.fill(alt, text)
                    if press_enter:
                        await self.page.keyboard.press("Enter")
                        await asyncio.sleep(1.5)
                    logger.info(f"Typed into alternative: {alt}")
                    return f"Typed using alternative selector: {alt}"
                except Exception:
                    continue
            
            logger.error(f"Could not find input field: {selector}")
            return f"Could not find input field: {selector}"
        except Exception:
            logger.exception(f"Type error on selector: {selector}")
            return f"Type error: {selector}"

    async def select_dropdown(self, selector: str, value: str = None, label: str = None, index: int = None) -> str:
        """Select option from dropdown"""
        try:
            logger.info(f"Selecting from dropdown: {selector}")
            
            await self.page.wait_for_selector(selector, state="visible", timeout=10000)
            
            if value:
                await self.page.select_option(selector, value=value)
            elif label:
                await self.page.select_option(selector, label=label)
            elif index is not None:
                await self.page.select_option(selector, index=index)
            else:
                return "Must provide value, label, or index"
            
            await asyncio.sleep(0.5)
            logger.info(f"Selected from dropdown: {selector}")
            return f"Successfully selected option from {selector}"
        except Exception:
            logger.exception(f"Dropdown selection error: {selector}")
            return f"Dropdown selection error: {selector}"

    async def hover(self, selector: str) -> str:
        """Hover over an element"""
        try:
            logger.info(f"Hovering over: {selector}")
            await self.page.wait_for_selector(selector, state="visible", timeout=10000)
            await self.page.hover(selector)
            await asyncio.sleep(0.5)
            logger.info(f"Hovered: {selector}")
            return f"Successfully hovered over: {selector}"
        except Exception:
            logger.exception(f"Hover error: {selector}")
            return f"Hover error: {selector}"

    async def scroll(self, direction: str = "down", amount: int = 500) -> str:
        """Scroll the page"""
        try:
            if direction.lower() == "down":
                await self.page.evaluate(f"window.scrollBy(0, {amount})")
            elif direction.lower() == "up":
                await self.page.evaluate(f"window.scrollBy(0, -{amount})")
            elif direction.lower() == "bottom":
                await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            elif direction.lower() == "top":
                await self.page.evaluate("window.scrollTo(0, 0)")
            
            await asyncio.sleep(0.5)
            logger.info(f"Scrolled {direction}")
            return f"Scrolled {direction}"
        except Exception:
            logger.exception("Scroll error")
            return "Scroll error"

    async def wait_for_navigation(self, timeout: int = 30000) -> str:
        """Wait for navigation to complete"""
        try:
            await self.page.wait_for_load_state("networkidle", timeout=timeout)
            await asyncio.sleep(1)
            return "Navigation completed"
        except Exception:
            logger.exception("Wait for navigation error")
            return "Navigation wait timeout"

    async def wait_for_selector(self, selector: str, timeout: int = 15000, state: str = "visible") -> str:
        """Wait for a specific selector to appear"""
        try:
            await self.page.wait_for_selector(selector, state=state, timeout=timeout)
            return f"Element {selector} is now {state}"
        except Exception:
            logger.exception(f"Wait for selector error: {selector}")
            return f"Timeout waiting for {selector}"

    async def read(self, selector: Optional[str] = None) -> str:
        """Read visible page content or specific element"""
        try:
            logger.info("Reading page content...")
            
            if selector:
                content = await self.page.locator(selector).inner_text()
            else:
                content = await self.page.evaluate("""
                    () => {
                        const clone = document.body.cloneNode(true);
                        clone.querySelectorAll('script, style, noscript, iframe, svg').forEach(el => el.remove());
                        return clone.innerText;
                    }
                """)
            
            content = content[:6000]
            title = await self.page.title()
            url = self.page.url
            
            logger.info(f"Read {len(content)} characters")
            return f"=== PAGE CONTENT ===\nURL: {url}\nTitle: {title}\n\nContent:\n{content}"
        
        except Exception:
            logger.exception("Read error")
            return "Read error occurred"

    async def get_attribute(self, selector: str, attribute: str) -> Optional[str]:
        """Get attribute value from element"""
        try:
            value = await self.page.get_attribute(selector, attribute)
            logger.info(f"Got attribute {attribute} from {selector}: {value}")
            return value
        except Exception:
            logger.exception(f"Get attribute error: {selector}")
            return None

    async def evaluate(self, script: str) -> Any:
        """Execute JavaScript on the page"""
        try:
            result = await self.page.evaluate(script)
            logger.info("Executed JavaScript")
            return result
        except Exception:
            logger.exception("JavaScript evaluation error")
            return None

    async def screenshot(self, path: str = "screenshot.png", full_page: bool = False) -> str:
        """Take a screenshot"""
        try:
            logger.info(f"Taking screenshot: {path}")
            await self.page.screenshot(path=path, full_page=full_page)
            logger.info(f"Screenshot saved: {path}")
            return f"Screenshot saved to {path}"
        except Exception:
            logger.exception("Screenshot error")
            return "Screenshot error occurred"

    async def get_cookies(self) -> List[Dict]:
        """Get all cookies"""
        try:
            cookies = await self.context.cookies()
            return cookies
        except Exception:
            logger.exception("Get cookies error")
            return []

    async def set_cookies(self, cookies: List[Dict]):
        """Set cookies"""
        try:
            await self.context.add_cookies(cookies)
            logger.info("Cookies set successfully")
        except Exception:
            logger.exception("Set cookies error")

    async def wait(self, seconds: float = 1.0) -> str:
        """Wait for specified seconds"""
        logger.debug(f"Waiting for {seconds} seconds")
        await asyncio.sleep(seconds)
        return f"Waited {seconds} seconds"

    async def back(self) -> str:
        """Go back in browser history"""
        try:
            await self.page.go_back(wait_until="domcontentloaded")
            await asyncio.sleep(1)
            self.current_url = self.page.url
            return f"Navigated back to {self.current_url}"
        except Exception:
            logger.exception("Back navigation error")
            return "Cannot go back"

    async def forward(self) -> str:
        """Go forward in browser history"""
        try:
            await self.page.go_forward(wait_until="domcontentloaded")
            await asyncio.sleep(1)
            self.current_url = self.page.url
            return f"Navigated forward to {self.current_url}"
        except Exception:
            logger.exception("Forward navigation error")
            return "Cannot go forward"

    async def reload(self) -> str:
        """Reload current page"""
        try:
            await self.page.reload(wait_until="domcontentloaded")
            await asyncio.sleep(1)
            return "Page reloaded"
        except Exception:
            logger.exception("Reload error")
            return "Reload error"

    async def close(self):
        """Close browser and cleanup"""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("Browser closed successfully")
        except Exception:
            logger.exception("Error closing browser")

    async def _handle_popups(self):
        """Handle common cookie consent and popup dialogs"""
        popup_selectors = [
            'button:has-text("Accept")',
            'button:has-text("Accept all")',
            'button:has-text("I agree")',
            'button:has-text("Allow all")',
            'button:has-text("OK")',
            '[id*="accept"]',
            '[class*="accept"]',
            '[aria-label*="Accept"]',
            '[aria-label*="Close"]',
            'button:has-text("✕")',
            'button:has-text("×")'
        ]
        
        for selector in popup_selectors:
            try:
                element = self.page.locator(selector).first
                if await element.is_visible(timeout=2000):
                    await element.click(timeout=2000)
                    logger.info(f"Dismissed popup: {selector}")
                    await asyncio.sleep(0.5)
                    break
            except Exception:
                continue

    def _get_smart_alternatives(self, selector: str) -> List[str]:
        """Generate smart alternative selectors based on the original"""
        alternatives = []
        
        # Extract parts of the selector
        if '#' in selector:
            id_part = re.search(r'#([\w-]+)', selector)
            if id_part:
                alternatives.append(f'[id="{id_part.group(1)}"]')
        
        if '.' in selector:
            class_part = re.search(r'\.([\w-]+)', selector)
            if class_part:
                alternatives.append(f'[class*="{class_part.group(1)}"]')
        
        # Common input alternatives
        if any(word in selector.lower() for word in ['input', 'search', 'query']):
            alternatives.extend([
                'input[name="q"]',
                'textarea[name="q"]',
                'input[type="search"]',
                'input[type="text"]',
                '[aria-label*="Search"]',
                '[placeholder*="Search"]',
                '#search',
                '[name="search"]'
            ])
        
        # Button alternatives
        if 'button' in selector.lower():
            alternatives.extend([
                'button[type="submit"]',
                'input[type="submit"]',
                'button.submit',
                '[role="button"]',
                'a.button',
                'a.btn'
            ])
        
        return alternatives