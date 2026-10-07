# session.py — run-scoped LLM session binding.
#
# run_agent() sets this to the LangGraph thread_id, so every LLM call in the
# run carries that thread_id as its session id (x-opencode-session header).

from __future__ import annotations

import contextvars

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


__all__ = ["set_session_id", "reset_session_id", "get_session_id"]
