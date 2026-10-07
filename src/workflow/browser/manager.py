# manager.py — browser session lifecycle + injectable manager.
#
# Canonical owner of the process browser singleton (moved from browsertools.py,
# kept as a shim). tools.py and observation.py acquire the browser through
# get_browser() here. BrowserSessionManager adds a lock + injectable factory
# so tests can inject a fake and concurrent server tasks do not race.

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from logs import logger
from src.workflow.browser.client import Browser

_browser_instance = None


async def get_browser():
    """Get or create browser instance (health-checked, stale-safe)."""
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


def reset_browser_singleton() -> None:
    """Drop the cached instance without closing (used by tests)."""
    global _browser_instance
    _browser_instance = None


class BrowserSessionManager:
    """Locked, injectable accessor over the module-level browser session."""

    def __init__(
        self,
        factory: Callable[[], Awaitable[object]] | None = None,
    ):
        self._factory = factory
        self._lock = asyncio.Lock()

    async def get(self):
        """Get or create the browser instance (locked)."""
        async with self._lock:
            if self._factory is not None:
                return await self._factory()
            return await get_browser()

    async def close(self) -> None:
        """Close and reset the session (locked)."""
        async with self._lock:
            try:
                if self._factory is not None:
                    return
                await close_browser()
            except Exception as e:
                logger.warning(f"[BROWSER] Session close failed (non-fatal): {e}")

    @staticmethod
    def current_url_safe() -> str:
        """Read live browser URL without starting an instance (never raises)."""
        try:
            browser = _browser_instance
            if browser is not None:
                return browser.page.url
        except Exception:
            pass
        return ""


# Default process-wide manager (keeps old call sites working).
default_manager = BrowserSessionManager()

__all__ = [
    "get_browser",
    "close_browser",
    "reset_browser_singleton",
    "BrowserSessionManager",
    "default_manager",
]
