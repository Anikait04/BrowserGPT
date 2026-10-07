# factory.py — builds the slim deep agent that owns the browser tools (canonical).
#
# Uses `deepagents.create_deep_agent` with a harness profile that excludes the
# built-in todo / filesystem / execute tools and disables the general-purpose
# subagent, so the agent is exactly: browser tools + read_page + the LLM.

from logs import logger
from src.workflow.browser.tools import (
    click_element,
    navigate,
    type_and_enter,
    type_text,
    wait_seconds,
)
from src.workflow.llm import get_navigation_llm, get_session_id
from src.workflow.browser.observation import read_page
from src.workflow.prompts.navigation import NAVIGATION_AGENT_PROMPT

# Provider keys needing the slim harness profile. Core keys mirror
# llm.SUPPORTED_PROVIDERS; legacy aliases kept for back-compat models.
_HARNESS_PROVIDER_KEYS: tuple[str, ...] = (
    "openai",
    "ollama",
    "groq",
    "gemini",
    "jev",
    "codiv",
    "typesafe",
)

_profile_registered = False


def _register_slim_harness_profile() -> None:
    """Register a slim harness profile so `create_deep_agent` ships without
    todo/filesystem/execute tools or the general-purpose subagent.

    The profile is keyed by LLM provider (openai / ollama / groq / gemini / jev)
    which matches how `get_llm()` / `get_navigation_llm()` instances resolve.
    Re-registration merges, so this is idempotent.
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

    for provider_key in _HARNESS_PROVIDER_KEYS:
        register_harness_profile(provider_key, slim_profile)

    _profile_registered = True


def build_navigation_agent(model=None, tools=None, system_prompt: str | None = None):
    """Build the deep navigation agent (JEV / codiv.ai model).

    Args are injectable for tests; defaults preserve production behavior.
    """
    from deepagents import create_deep_agent

    _register_slim_harness_profile()

    logger.info("[NAVIGATION] Building deep navigation agent (jev)")

    agent = create_deep_agent(
        model=model if model is not None else get_navigation_llm(),
        tools=(
            tools
            if tools is not None
            else [
                navigate,
                click_element,
                type_text,
                type_and_enter,
                wait_seconds,
                read_page,
            ]
        ),
        system_prompt=system_prompt or NAVIGATION_AGENT_PROMPT,
    )

    # TODO(roadmap 2.7): stream screenshots from deep-agent inner turns via SSE.
    return agent


_cached_agent = None
_cached_session_id = None


def get_navigation_agent():
    """Module-level cache wrapper — rebuilds on first use or session change.

    The agent holds a session-bound LLM client, so a new run (new thread_id)
    needs a fresh agent; otherwise calls would carry the previous run's session.
    """
    global _cached_agent, _cached_session_id
    session_id = get_session_id()
    if _cached_agent is None or _cached_session_id != session_id:
        _cached_agent = build_navigation_agent()
        _cached_session_id = session_id
    return _cached_agent


def reset_navigation_agent():
    """Drop the cached agent (used by tests)."""
    global _cached_agent, _cached_session_id
    _cached_agent = None
    _cached_session_id = None
