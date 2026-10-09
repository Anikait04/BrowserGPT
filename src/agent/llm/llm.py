from __future__ import annotations

import contextvars
from typing import Any
import uuid

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

import config


# ── Run-scoped LLM session ────────────────────────────────────────────────────
# run_agent() sets this to the LangGraph thread_id, so every LLM call in the run
# carries that thread_id as its session id (x-opencode-session header).
_current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_session_id", default=None
)


def set_session_id(session_id: str | None) -> contextvars.Token:
    """Bind subsequent get_llm() clients to session_id. Returns a token for reset."""
    return _current_session_id.set(session_id)


def reset_session_id(token: contextvars.Token) -> None:
    """Undo a set_session_id() call."""
    _current_session_id.reset(token)


def get_session_id() -> str | None:
    """Current run's session id (thread_id), or None outside a run."""
    return _current_session_id.get()


class LLMService:

    def __init__(self, session_id: str | None = None):
        # Prefer an explicit id, then the ambient run session; mint one only
        # outside a run so the header is always present.
        self.session_id = session_id or get_session_id() or str(uuid.uuid4())

    # ==========================================================================
    # OpenCode / OpenAI-compatible
    # ==========================================================================
    def _create_openai(
        self,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatOpenAI:

        return ChatOpenAI(
            model=config.MODEL_NAME_OPENCODE,
            api_key=config.OPENCODE_API_KEY,
            base_url=config.OPENCODE_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort="high",

            # Required by OpenCode Go
            default_headers={
                "x-opencode-session": self.session_id,
                "x-opencode-client": "langchain",
            },
        )

    # ==========================================================================
    # Ollama
    # ==========================================================================
    def _create_ollama(
        self,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatOllama:

        return ChatOllama(
            model=config.MODEL_NAME_OLLAMA,
            temperature=temperature,
            api_key=config.OLLAMA_API_KEY,
            base_url=config.OLLAMA_BASE_URL,
        )

    # ==========================================================================
    # Groq (OpenAI-compatible endpoint — no extra dependency needed)
    # ==========================================================================
    def _create_groq(
        self,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatOpenAI:

        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set (required for provider 'groq')")

        return ChatOpenAI(
            model=config.GROQ_MODEL_NAME,
            api_key=config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            # NOTE: no reasoning_effort (OpenAI-only) and no x-opencode-*
            # headers (OpenCode-Go-only) — Groq rejects unknown fields.
        )

    # ==========================================================================
    # JEV / codiv.ai (OpenAI-compatible endpoint — browser navigation only)
    # ==========================================================================
    def _create_jev(
        self,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatOpenAI:

        if not config.JEV_API_KEY:
            raise ValueError("JEV_API_KEY is not set (required for provider 'jev')")

        # Normalize: codiv.ai OpenAI-compatible API lives under /v1. A bare
        # host (e.g. legacy TYPESAFE_BASE_URL without /v1) gets it appended.
        base_url = (config.JEV_BASE_URL or "").rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"

        # NOTE: no response_format={"type": "json_object"} here. Probes against
        # diffusiongemma-26b show a global json_object response_format breaks
        # bind_tools ("only `strict` function tools can be auto-parsed"), while
        # a plain client supports BOTH bind_tools and with_structured_output.
        # LangChain structures JSON via function-calling under the hood, so the
        # snippet's response_format pattern only applies to raw OpenAI usage.
        # Also no reasoning_effort / x-opencode-* headers — codiv rejects them.
        return ChatOpenAI(
            model=config.JEV_MODEL_NAME,
            api_key=config.JEV_API_KEY,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens if max_tokens is not None else config.JEV_MAX_TOKENS,
        )

    # ==========================================================================
    # Google Gemini (native SDK)
    # ==========================================================================
    def _create_gemini(
        self,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatGoogleGenerativeAI:

        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set (required for provider 'gemini')")

        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL_NAME,
            google_api_key=config.GEMINI_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
            # NOTE: no reasoning_effort / x-opencode-* headers — provider-specific.
        )

    # ==========================================================================
    # LLM Factory
    # ==========================================================================
    def create_llm(
        self,
        provider: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Any:

        providers = {
            "openai": self._create_openai,
            "ollama": self._create_ollama,
            "groq": self._create_groq,
            "gemini": self._create_gemini,
            "jev": self._create_jev,
        }

        provider = (provider or config.LLM_PROVIDER).lower()

        if provider not in providers:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        return providers[provider](
            temperature=temperature,
            max_tokens=max_tokens,
        )


# Clients are cached per (provider, session_id): cheap (no I/O at build time),
# bounded (oldest run's client is evicted), and each run's calls always carry
# that run's thread_id. Outside a run a single "default" client is reused.
_LLM_CACHE_MAX = 8
_llm_cache: dict[tuple[str, str], Any] = {}


def get_llm() -> Any:
    """Return an LLM client bound to the current run's session (thread_id)."""
    provider = (config.LLM_PROVIDER or "openai").lower()
    session_id = get_session_id() or "default"
    key = (provider, session_id)
    llm = _llm_cache.get(key)
    if llm is None:
        llm = LLMService(session_id=get_session_id()).create_llm(provider)
        if len(_llm_cache) >= _LLM_CACHE_MAX:
            _llm_cache.pop(next(iter(_llm_cache)))
        _llm_cache[key] = llm
    return llm


def get_navigation_llm() -> Any:
    """Return the JEV (codiv.ai) client for the browser navigation deep-agent.

    Pinned to provider 'jev' regardless of the global LLM_PROVIDER, so
    planner / delegation / verify keep their current model while only
    navigation drives diffusiongemma-26b. Cached per session like get_llm().
    """
    session_id = get_session_id() or "default"
    key = ("jev", session_id)
    llm = _llm_cache.get(key)
    if llm is None:
        llm = LLMService(session_id=get_session_id()).create_llm("jev")
        if len(_llm_cache) >= _LLM_CACHE_MAX:
            _llm_cache.pop(next(iter(_llm_cache)))
        _llm_cache[key] = llm
    return llm


def reset_llm_cache() -> None:
    """Drop cached clients (used by tests)."""
    _llm_cache.clear()