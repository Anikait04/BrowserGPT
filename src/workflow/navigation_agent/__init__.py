# navigation_agent — deep browser-driving agent factory.
#
# Canonical package replacing navigation_agent.py. Existing imports keep
# working via these re-exports.

from src.workflow.navigation_agent.factory import (
    build_navigation_agent,
    get_navigation_agent,
    reset_navigation_agent,
)

__all__ = [
    "build_navigation_agent",
    "get_navigation_agent",
    "reset_navigation_agent",
]
