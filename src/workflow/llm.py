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


def reset_llm_cache() -> None:
    """Drop cached clients (used by tests)."""
    _llm_cache.clear()