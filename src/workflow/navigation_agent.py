# navigation_agent.py — builds the slim deep agent that owns the browser tools.
#
# Uses `deepagents.create_deep_agent` with a harness profile that excludes the
# built-in todo / filesystem / execute tools and disables the general-purpose
# subagent, so the agent is exactly: browser tools + read_page + the LLM.

from logs import logger
from src.workflow.browsertools import (
    click_element,
    navigate,
    type_and_enter,
    type_text,
    wait_seconds,
)
from src.workflow.llm import get_llm
from src.workflow.page_reader import read_page
from src.workflow.prompt import NAVIGATION_AGENT_PROMPT

_profile_registered = False


def _register_slim_harness_profile() -> None:
    """Register a slim harness profile so `create_deep_agent` ships without
    todo/filesystem/execute tools or the general-purpose subagent.

    The profile is keyed by LLM provider (openai / ollama) which matches how
    `get_llm()` instances resolve. Re-registration merges, so this is idempotent.
    """
    global _profile_registered
    if _profile_registered:
        return

    from deepagents import (
        GeneralPurposeSubagentProfile,
        HarnessProfileConfig,
        register_harness_profile,
    )

    slim_profile = HarnessProfileConfig(
        excluded_tools={
            "write_todos",
            "ls",
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "grep",
            "execute",
            "task",
        },
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    )

    for provider_key in ("openai", "ollama"):
        register_harness_profile(provider_key, slim_profile)

    _profile_registered = True


def build_navigation_agent():
    """Build (and cache) the deep navigation agent."""
    from deepagents import create_deep_agent

    _register_slim_harness_profile()

    logger.info("[NAVIGATION] Building deep navigation agent")

    agent = create_deep_agent(
        model=get_llm(),
        tools=[
            navigate,
            click_element,
            type_text,
            type_and_enter,
            wait_seconds,
            read_page,
        ],
        system_prompt=NAVIGATION_AGENT_PROMPT,
    )

    # TODO(roadmap 2.7): stream screenshots from deep-agent inner turns via SSE.
    return agent


_cached_agent = None


def get_navigation_agent():
    """Module-level cache wrapper — rebuilds only if never built."""
    global _cached_agent
    if _cached_agent is None:
        _cached_agent = build_navigation_agent()
    return _cached_agent


def reset_navigation_agent():
    """Drop the cached agent (used by tests)."""
    global _cached_agent
    _cached_agent = None
