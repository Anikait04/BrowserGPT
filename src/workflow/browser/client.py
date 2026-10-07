# client.py — Playwright browser client + typed results.
#
# Canonical home of the Browser class (moved from browserplugin.py, kept as a
# shim). Browser keeps its string tool contract for the deep agent; the
# BrowserResult / interpret_result layer below gives new code a typed way to
# branch programmatically instead of substring-matching human text.

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

from logs import logger


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
                    '--disable-blink-features=AutomationControlled',
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
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

            # Use first match to avoid strict mode violation
            locator = self.page.locator(selector).first

            await locator.wait_for(state="visible", timeout=10000)
            await locator.scroll_into_view_if_needed()
            await asyncio.sleep(0.3)

            try:
                await locator.click(timeout=5000)
            except Exception:
                await locator.click(force=True)

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
        """Type text into an input field."""
        from src.workflow.browser.selectors import SEARCH_INPUT_ALTERNATIVES

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
            alternatives = list(SEARCH_INPUT_ALTERNATIVES)

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

    async def type_and_enter(self, selector: str, text: str) -> str:
        """Type text into an input field and always press Enter.

        Thin wrapper over type(..., press_enter=True) — kept for the
        deep-agent tool contract. Single implementation lives in type().
        """
        return await self.type(selector, text, press_enter=True)

    async def read(self) -> str:
        """Read visible page content"""
        try:
            from src.workflow.constants import BROWSER_READ_CHAR_LIMIT

            logger.info("Reading page content...")
            content = await self.page.evaluate("""
                () => {
                    const clone = document.body.cloneNode(true);
                    clone.querySelectorAll('script, style, noscript').forEach(el => el.remove());
                    return clone.innerText;
                }
            """)
            content = content[:BROWSER_READ_CHAR_LIMIT]
            title = await self.page.title()
            url = self.page.url

            logger.info(f"Read {len(content)} characters")
            return f"=== PAGE CONTENT ===\nURL: {url}\nTitle: {title}\n\nContent:\n{content}"

        except Exception:
            logger.exception("Read error")
            return "Read error occurred"

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
        from src.workflow.browser.selectors import alternative_selectors

        return alternative_selectors(selector)


@dataclass(frozen=True)
class BrowserConfig:
    """Tunable browser behavior (replaces hardcoded timeouts/sleeps)."""

    headless: bool = False
    default_timeout_ms: int = 20000
    navigation_timeout_ms: int = 30000
    action_timeout_ms: int = 10000
    click_timeout_ms: int = 5000
    fallback_timeout_ms: int = 3000
    settle_sleep_s: float = 1.5
    after_click_sleep_s: float = 0.8
    type_delay_ms: int = 50
    read_char_limit: int = 4000


@dataclass(frozen=True)
class BrowserResult:
    """Typed outcome for a browser operation."""

    ok: bool
    message: str
    data: dict = field(default_factory=dict)

    def __str__(self) -> str:  # keep LLM-tool compat (tools return str)
        return self.message


_FAILURE_MARKERS: tuple[str, ...] = (
    "timeout",
    "could not find",
    "could not load",
    "navigation error",
    "click error",
    "type error",
    "type_and_enter error",
    "read error",
)


def interpret_result(message: str) -> BrowserResult:
    """Best-effort classification of legacy string results.

    New code: `result = interpret_result(await browser.go_to(url))`
    then branch on `result.ok` instead of substring checks.
    """
    text = message or ""
    lowered = text.lower()
    ok = not any(marker in lowered for marker in _FAILURE_MARKERS)
    return BrowserResult(ok=ok, message=text)


__all__ = [
    "Browser",
    "BrowserConfig",
    "BrowserResult",
    "interpret_result",
]
