# constants.py — single source of truth for workflow-wide literals.
#
# Node names, safety limits, truncation budgets, and artifact routing live here
# so graph wiring (agent.py / graph.py), routing decisions, and nodes cannot
# drift apart. Import from here instead of hardcoding strings/numbers.

from __future__ import annotations

import os

# ── LangGraph node names (must match graph.add_node keys) ──
NODE_PLANNER = "planner"
NODE_DELEGATION = "delegation"
NODE_NAVIGATION = "navigation"
NODE_VERIFY = "verify"
NODE_EXTRACT = "extract_information"
NODE_WAIT_FOR_USER = "wait_for_user"

# Valid delegation actions (mirrors DelegationDecision.action literal).
VALID_DELEGATION_ACTIONS: tuple[str, ...] = (
    NODE_PLANNER,
    NODE_NAVIGATION,
    NODE_EXTRACT,
    NODE_WAIT_FOR_USER,
    "finish",
)

# ── Safety limits (defaults mirror src.config; nodes should read these via
# config injection where possible, but fall back here for pure-function use) ──
DEFAULT_MAX_STEPS = 30
DEFAULT_MAX_NAVIGATION_ITERATIONS = 8
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3
DEFAULT_DEEP_AGENT_RECURSION_LIMIT = 40

# ── Truncation budgets ──
HISTORY_MAX_ENTRIES = 10
HISTORY_MAX_CHARS = 2000
PAGE_SNAPSHOT_CHARS = 6000
BROWSER_READ_CHAR_LIMIT = 4000
ACTION_ARGS_PREVIEW_CHARS = 120
ACTION_RESULT_PREVIEW_CHARS = 160
RECENT_ACTIONS_FOR_VERIFY = 15

# ── Extraction artifacts ──
ARTIFACT_ROUTE_PREFIX = "/extract/artifact"

# ── Graph diagram export (opt-in; agent.get_app exported unconditionally before) ──
EXPORT_GRAPH_PNG = os.getenv("EXPORT_GRAPH_PNG", "false").lower() in ("1", "true", "yes")
GRAPH_PNG_PATH = os.getenv("GRAPH_PNG_PATH", "agent_flow.png")


def artifact_url(artifact_id: str) -> str:
    """Public download URL for a detailed-PDF artifact id."""
    return f"{ARTIFACT_ROUTE_PREFIX}/{artifact_id}"
