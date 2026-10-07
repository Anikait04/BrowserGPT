# llm — provider clients (session-bound cache) + invocation helpers.
#
# Canonical package replacing llm.py. `from src.workflow.llm import get_llm`
# keeps working via these re-exports.

from src.workflow.llm.helpers import (
    ainvoke_structured_with_fallback,
    flatten_raw_content,
)
from src.workflow.llm.registry import (
    SUPPORTED_PROVIDERS,
    LLMService,
    get_llm,
    get_navigation_llm,
    reset_llm_cache,
)
from src.workflow.llm.session import (
    get_session_id,
    reset_session_id,
    set_session_id,
)

__all__ = [
    "LLMService",
    "SUPPORTED_PROVIDERS",
    "get_llm",
    "get_navigation_llm",
    "reset_llm_cache",
    "get_session_id",
    "set_session_id",
    "reset_session_id",
    "ainvoke_structured_with_fallback",
    "flatten_raw_content",
]
