"""
src.workflow.llm
~~~~~~~~~~~~~~~~
LLM factory and custom chat-model adapter.

This module serves two purposes:

1. **Provider factory** – ``LLMService`` / ``get_llm()`` expose a
   unified way to obtain a LangChain chat model for the configured
   provider (Gemini, OpenRouter, OpenAI).  The OpenAI path is the
   primary one for the BrowserGPT deep agent: it resolves
   ``model``, ``api_key`` and ``base_url`` via
   ``settings.resolve_openai_config()`` so that ``CUSTOM_PROVIDER``
   indirection (e.g. ``opencode``) works transparently.

2. **Legacy adapter** – ``CustomLLMClient`` / ``CustomChatModel``
   wrap the previous self-hosted ``MODEL_URL`` endpoint (``gpt-oss:
   120b-cloud``) as a ``BaseChatModel`` so that ``create_deep_agent``
   can ``bind_tools`` and drive the ReAct loop via ``.ainvoke``.
   ``get_chat_model()`` prefers the OpenAI provider when its
   credentials are present and falls back to the custom adapter
   otherwise, so the deep agent in ``nodes.agent_node`` always has a
   tool-calling model.

All public helpers are intentionally small and heavily commented for
readability.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Standard library
# --------------------------------------------------------------------------- #
import asyncio
import json
import os
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Third-party — optional Google / OpenRouter deps are imported lazily so
# that the module still loads when only OpenAI is installed.
# --------------------------------------------------------------------------- #
from dotenv import load_dotenv
from pydantic import ConfigDict, Field

# LangChain core abstractions — always required for the deep agent.
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

# Provider-specific chat models.
from langchain_openai import ChatOpenAI

try:  # optional — only needed when provider == "gemini"
    from google.oauth2 import service_account  # type: ignore
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
except Exception:  # pragma: no cover - import guard
    service_account = None  # type: ignore
    ChatGoogleGenerativeAI = None  # type: ignore

try:  # optional — only needed when provider == "openrouter"
    from langchain_openrouter import ChatOpenRouter  # type: ignore
except Exception:  # pragma: no cover
    ChatOpenRouter = None  # type: ignore

# --------------------------------------------------------------------------- #
# Internal imports — settings & legacy config
# --------------------------------------------------------------------------- #
# New settings object (backend.core.settings) is the canonical source;
# config.py re-exports the same values for legacy callers.
try:
    from backend.core.settings import settings  # preferred
except ImportError:  # fallback when backend package is not on PYTHONPATH
    from config import settings as settings  # type: ignore

# Legacy self-hosted LLM endpoint (kept for backward compat).
import httpx
import cloudpickle  # noqa: F401  (kept for legacy payload handling)
import dill  # noqa: F401
import base64  # noqa: F401
import io  # noqa: F401
from config import MODEL_NAME_OLLAMA, USERNAME, PASSWORD, LOGIN_URL, MODEL_URL

# Ensure .env is loaded before any settings are accessed.
load_dotenv()


# =========================================================================== #
# 1. Provider factory — LLMService
# =========================================================================== #


class LLMService:
    """
    Factory for LangChain chat models.

    Each ``_create_*`` method returns a *configured* chat model
    instance.  The public ``create_llm`` dispatcher selects the right
    one based on ``provider`` (or ``settings.LLM_PROVIDER`` when the
    caller passes ``None``).
    """

    def __init__(self) -> None:
        # Hold a reference to the global settings singleton so that
        # ``resolve_openai_config`` always sees the latest env snapshot.
        self.settings = settings

    # ------------------------------------------------------------------ #
    # Gemini (Vertex AI)
    # ------------------------------------------------------------------ #
    def _create_gemini(
        self,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Any:
        """
        Create a Gemini / Vertex AI chat model.

        Requires ``GOOGLE_APPLICATION_CREDENTIALS`` (service-account
        JSON) plus ``GOOGLE_CLOUD_PROJECT`` / ``GOOGLE_CLOUD_LOCATION``.
        """
        if ChatGoogleGenerativeAI is None or service_account is None:
            raise ImportError(
                "langchain-google-genai and google-auth are required for provider='gemini'. "
                "Install with: pip install langchain-google-genai google-auth"
            )

        if not self.settings.GOOGLE_APPLICATION_CREDENTIALS:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set")

        # Build credentials from the service-account file.
        credentials = service_account.Credentials.from_service_account_file(
            self.settings.GOOGLE_APPLICATION_CREDENTIALS,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        # Return a Vertex AI-backed Gemini model.
        return ChatGoogleGenerativeAI(
            model=self.settings.GOOGLE_MODEL,
            credentials=credentials,
            project=self.settings.GOOGLE_CLOUD_PROJECT,
            location=self.settings.GOOGLE_CLOUD_LOCATION,
            vertexai=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------ #
    # OpenRouter
    # ------------------------------------------------------------------ #
    def _create_openrouter(
        self,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Any:
        """
        Create an OpenRouter chat model.

        OpenRouter is OpenAI-compatible; the model name is expected to
        be of the form ``openai/gpt-oss-20b:free``.
        """
        if ChatOpenRouter is None:
            raise ImportError(
                "langchain-openrouter is required for provider='openrouter'. "
                "Install with: pip install langchain-openrouter"
            )

        return ChatOpenRouter(
            model=self.settings.OPENROUTER_MODEL,
            api_key=self.settings.OPENROUTER_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------ #
    # OpenAI — primary provider for the deep agent
    # ------------------------------------------------------------------ #
    def _create_openai(
        self,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatOpenAI:
        """
        Create an OpenAI-compatible chat model.

        This is the **main entry point** used by the deep agent
        (``src.workflow.nodes.get_deep_agent``).  It explicitly
        resolves ``model``, ``api_key`` and ``base_url`` via
        ``settings.resolve_openai_config()`` so that the three
        parameters travel together and ``CUSTOM_PROVIDER`` mapping is
        honoured.

        Example ``.env`` (standard):
            OPENAI_MODEL=gpt-4o-mini
            OPENAI_API_KEY=sk-...
            OPENAI_BASE_URL=https://api.openai.com/v1

        Example ``.env`` (custom provider ``opencode``):
            CUSTOM_PROVIDER=opencode
            OPENCODE_MODEL=opencode/gpt-oss
            OPENCODE_API_KEY=sk-opencode
            OPENCODE_BASE_URL=https://opencode.ai/api/v1
            # model_configuration.json maps OPENAI_* → OPENCODE_*
        """
        # Resolve the effective triple (model, api_key, base_url).
        # This handles both the standard case and the CUSTOM_PROVIDER
        # indirection (see Settings.resolve_openai_config docstring).
        config = self.settings.resolve_openai_config()

        # ``config`` always contains the three keys; values may be None
        # when the user has not configured the provider — ChatOpenAI
        # will raise a clear error in that case.
        return ChatOpenAI(
            model=config["model"],  # e.g. "gpt-4o-mini" or "opencode/gpt-oss"
            api_key=config["api_key"],  # bearer token
            base_url=config["base_url"],  # e.g. "https://api.openai.com/v1" or custom
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------ #
    # Public dispatcher
    # ------------------------------------------------------------------ #
    def create_llm(
        self,
        provider: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Any:
        """
        Factory dispatcher.

        Args:
            provider: One of ``"gemini"``, ``"openrouter"``, ``"openai"``.
                When ``None``, falls back to ``settings.LLM_PROVIDER``.
            temperature: Sampling temperature.
            max_tokens: Optional cap on generated tokens.

        Returns:
            A LangChain ``BaseChatModel`` instance.

        Raises:
            ValueError: if the provider string is unknown.
        """
        # Normalise the provider name; default to the globally configured one.
        provider = (provider or self.settings.LLM_PROVIDER).lower()

        # Map provider names to their constructors.
        providers: Dict[str, Callable[..., Any]] = {
            "gemini": self._create_gemini,
            "openrouter": self._create_openrouter,
            "openai": self._create_openai,
        }

        if provider not in providers:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        # Call the selected constructor with the supplied sampling args.
        return providers[provider](
            temperature=temperature,
            max_tokens=max_tokens,
        )


# --------------------------------------------------------------------------- #
# Singleton accessor — cached so that the same ChatOpenAI instance is
# reused across the deep-agent and verifier nodes.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_llm() -> Any:
    """
    Return the default LLM as configured by ``LLM_PROVIDER``.

    The result is cached (``lru_cache``) so repeated calls are cheap.
    To force a re-creation after changing ``.env``, clear the cache
    with ``get_llm.cache_clear()``.
    """
    return LLMService().create_llm()


# =========================================================================== #
# 2. Legacy self-hosted client (kept for backward compat & fallback)
# =========================================================================== #


class CustomLLMClient:
    """
    Thin async wrapper around the legacy self-hosted ``MODEL_URL``.

    The service expects a POST to ``MODEL_URL`` with
    ``{model, system_prompt, user_prompt, structured, output_schema}`` and
    returns ``{response: ...}``.  Authentication is via ``LOGIN_URL``.
    This client is retained so that older ``planner`` / ``verifier``
    prompts continue to work when ``CUSTOM_PROVIDER`` is not set.
    """

    def __init__(self) -> None:
        self.token: str | None = None

    async def login(self) -> None:
        """Obtain a bearer token from ``LOGIN_URL``."""
        async with httpx.AsyncClient() as client:
            res = await client.post(
                LOGIN_URL,
                json={"username": USERNAME, "password": PASSWORD},
                headers={"accept": "application/json"},
            )
            res.raise_for_status()
            data = res.json()
            self.token = data.get("access_token") or data.get("token")

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        structured: bool = False,
        schema: dict | None = None,
    ) -> dict:
        """
        Call the self-hosted model.

        Args:
            system_prompt: System instruction.
            user_prompt: User turn.
            structured: When True, ``schema`` is sent as ``output_schema``.
            schema: JSON schema for structured output.

        Returns:
            Parsed JSON response (``{response: ...}``).
        """
        if not self.token:
            await self.login()

        payload: Dict[str, Any] = {
            "model": MODEL_NAME_OLLAMA,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "structured": structured,
        }
        if structured and schema:
            payload["output_schema"] = schema

        async with httpx.AsyncClient(timeout=120) as client:
            res = await client.post(
                MODEL_URL,
                headers={
                    "accept": "application/json",
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

            # Token may have expired → re-login once.
            if res.status_code == 401:
                await self.login()
                return await self.generate(system_prompt, user_prompt, structured, schema)

            res.raise_for_status()
            return res.json()


# =========================================================================== #
# 3. Adapter — CustomLLMClient as a LangChain BaseChatModel
# =========================================================================== #


class CustomChatModel(BaseChatModel):
    """
    Adapts ``CustomLLMClient`` to the LangChain ``BaseChatModel`` API.

    This enables ``create_deep_agent(model=CustomChatModel(...))`` to
    ``bind_tools`` and drive the ReAct tool-calling loop via
    ``.ainvoke``.  When no tools are bound the adapter falls back to
    plain text generation; when tools are bound it augments the system
    prompt with tool descriptions and requests a structured
    ``{tool_calls, content}`` JSON object from the self-hosted model.

    The implementation is intentionally defensive: it handles both
    structured and unstructured responses and normalises tool calls to
    the ``{id, name, args}`` shape expected by LangGraph.
    """

    # Pydantic config — allow arbitrary ``CustomLLMClient`` instance.
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    client: CustomLLMClient = Field(default_factory=CustomLLMClient)
    bound_tools: Optional[List[Any]] = Field(default=None)
    _llm_type: str = "custom"

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return "custom"

    # ------------------------------------------------------------------ #
    # Tool binding — called by ``create_deep_agent`` / LangGraph.
    # ------------------------------------------------------------------ #
    def bind_tools(
        self,
        tools: Sequence[Dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Runnable:
        """
        Bind tools to this model.

        LangGraph calls ``model.bind_tools(tools)`` during agent
        compilation.  We return a *new* ``CustomChatModel`` instance
        that remembers the tools so that ``_agenerate`` can render
        them into the prompt.  Extra ``kwargs`` (e.g. ``tool_choice``)
        are forwarded via ``Runnable.bind``.
        """
        new_instance = CustomChatModel(client=self.client, bound_tools=list(tools))
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if kwargs:
            return new_instance.bind(**kwargs)  # type: ignore[return-value]
        return new_instance  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Sync wrapper — delegates to the async implementation.
    # ------------------------------------------------------------------ #
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous ``_generate`` — runs ``_agenerate`` in an event loop."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    coro = self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                    future = asyncio.run_coroutine_threadsafe(coro, loop)
                    return future.result(timeout=130)
            else:
                return loop.run_until_complete(
                    self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                )
        except RuntimeError:
            return asyncio.run(self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs))

    # ------------------------------------------------------------------ #
    # Core async generation
    # ------------------------------------------------------------------ #
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Convert LangChain messages → ``system_prompt`` / ``user_prompt``,
        optionally augment the system prompt with tool descriptions, and
        call ``CustomLLMClient.generate``.
        """
        # Merge tools that may have been passed via ``Runnable.bind``.
        bound_tools = self.bound_tools
        if not bound_tools and "tools" in kwargs:
            bound_tools = kwargs.get("tools")

        # ---- Flatten the LangChain message list ----------------------- #
        system_parts: List[str] = []
        user_parts: List[str] = []
        for m in messages:
            content = m.content if isinstance(m.content, str) else str(m.content) if m.content is not None else ""
            if isinstance(m, SystemMessage):
                system_parts.append(content)
            elif isinstance(m, HumanMessage):
                user_parts.append(content)
            elif isinstance(m, AIMessage):
                if getattr(m, "tool_calls", None):
                    try:
                        user_parts.append(f"Assistant tool_calls: {json.dumps(m.tool_calls)}")
                    except Exception:
                        user_parts.append(f"Assistant tool_calls: {str(m.tool_calls)}")
                if content:
                    user_parts.append(f"Assistant: {content}")
            elif isinstance(m, ToolMessage):
                tname = getattr(m, "name", "tool")
                user_parts.append(f"Tool [{tname}] result: {content}")
                if getattr(m, "tool_call_id", None):
                    user_parts[-1] += f" (call_id={m.tool_call_id})"
            else:
                user_parts.append(content)

        system_prompt = "\n\n".join([p for p in system_parts if p]) if system_parts else "You are a helpful assistant."
        user_prompt = "\n\n".join([p for p in user_parts if p]) if user_parts else "Respond."

        # ---- No tools → plain generation ------------------------------ #
        if not bound_tools:
            result = await self.client.generate(system_prompt=system_prompt, user_prompt=user_prompt, structured=False)
            text = result.get("response", result) if isinstance(result, dict) else result
            if isinstance(text, dict):
                text = text.get("content") or text.get("message") or text.get("text") or json.dumps(text)
            if not isinstance(text, str):
                text = str(text)
            message = AIMessage(content=text)
            return ChatResult(generations=[ChatGeneration(message=message)])

        # ---- Tool-calling path --------------------------------------- #
        # Render tool descriptions into the system prompt so the
        # self-hosted model knows which tools are available.
        tools_desc: List[str] = []
        tool_names: List[str] = []
        for t in bound_tools:
            try:
                oai = convert_to_openai_tool(t)
                func = oai.get("function", oai)
                name = func.get("name") or oai.get("name") or getattr(t, "name", "tool")
                desc = func.get("description") or getattr(t, "description", "")
                params = func.get("parameters", {})
                tools_desc.append(f"- {name}: {desc} Parameters: {json.dumps(params, ensure_ascii=False)}")
                tool_names.append(name)
            except Exception:
                name = getattr(t, "name", str(t))
                desc = getattr(t, "description", "")
                tools_desc.append(f"- {name}: {desc}")
                tool_names.append(name)

        system_prompt += "\n\nYou have access to the following tools:\n" + "\n".join(tools_desc)
        system_prompt += (
            "\n\nWhen you need to use a tool, you MUST output ONLY valid JSON with keys 'tool_calls' (array) and 'content' (string). "
            "Each tool_call must have 'name' (one of [" + ", ".join(f'\"{n}\"' for n in tool_names) + "]) and 'args' (object matching the tool's parameters). "
            'Example: {"tool_calls": [{"name": "navigate", "args": {"url": "https://example.com"}}], "content": ""} '
            'If no tool is needed, output {"tool_calls": [], "content": "your final answer"}. '
            "Output ONLY JSON, no markdown, no extra text."
        )

        # Structured schema to guide the self-hosted model.
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "enum": tool_names} if tool_names else {"type": "string"},
                            "args": {"type": "object"},
                        },
                        "required": ["name", "args"],
                    },
                },
                "content": {"type": "string"},
            },
            "required": ["tool_calls", "content"],
        }

        # Try structured first, fall back to unstructured on error.
        try:
            result = await self.client.generate(system_prompt=system_prompt, user_prompt=user_prompt, structured=True, schema=schema)
        except Exception:
            result = await self.client.generate(system_prompt=system_prompt, user_prompt=user_prompt, structured=False)
            text = result.get("response", result) if isinstance(result, dict) else result
            if isinstance(text, dict):
                text = text.get("content") or text.get("message") or json.dumps(text)
            if not isinstance(text, str):
                text = str(text)
            try:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    parsed = json.loads(text[start : end + 1])
                    if isinstance(parsed, dict) and "tool_calls" in parsed:
                        result = {"response": parsed}
                    else:
                        raise ValueError
                else:
                    raise ValueError
            except Exception:
                message = AIMessage(content=text)
                return ChatResult(generations=[ChatGeneration(message=message)])

        payload = result.get("response", result) if isinstance(result, dict) else result

        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                message = AIMessage(content=payload)
                return ChatResult(generations=[ChatGeneration(message=message)])

        if not isinstance(payload, dict):
            payload = {"tool_calls": [], "content": str(payload)}

        tool_calls_raw = payload.get("tool_calls") or []
        content = payload.get("content") or payload.get("message") or ""

        # Normalise to LangChain {id, name, args} shape.
        tool_calls: List[Dict[str, Any]] = []
        for idx, tc in enumerate(tool_calls_raw):
            if not isinstance(tc, dict):
                continue
            name = tc.get("name") or tc.get("tool") or tc.get("function")
            args = tc.get("args") or tc.get("arguments") or tc.get("parameters") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"value": args}
            if not name:
                continue
            if not isinstance(args, dict):
                args = {"value": str(args)}
            tool_calls.append({"id": f"call_{idx}_{name}", "name": name, "args": args})

        if not tool_calls and not content:
            if "name" in payload and "args" in payload:
                tool_calls.append({"id": "call_0_" + payload["name"], "name": payload["name"], "args": payload["args"]})
            else:
                content = json.dumps(payload) if payload else ""

        message = AIMessage(content=content or "", tool_calls=tool_calls if tool_calls else [])
        return ChatResult(generations=[ChatGeneration(message=message)])


# --------------------------------------------------------------------------- #
# Singleton for the legacy adapter — kept for nodes that still import
# ``get_chat_model`` directly.  New code should prefer ``get_llm()``
# with ``provider="openai"``.
# --------------------------------------------------------------------------- #
_chat_model_singleton: Optional[CustomChatModel] = None


def get_chat_model() -> BaseChatModel:
    """
    Return a chat model suitable for ``create_deep_agent``.

    Preference order:
      1. OpenAI via ``LLMService`` (uses ``base_url`` / ``api_key`` /
         ``model`` from ``resolve_openai_config``) when those
         credentials are present.
      2. Legacy ``CustomChatModel`` (self-hosted ``MODEL_URL``) as a
         fallback so that existing deployments without OpenAI still work.

    The ``agent_node`` in ``nodes.py`` calls this helper to obtain the
    model that is then passed to ``create_deep_agent(..., tools=deep_tools)``
    and driven via ``.ainvoke``.
    """
    # Try the modern OpenAI path first — it natively supports
    # ``bind_tools`` and yields the most reliable tool-calling.
    try:
        # Use LLMService so that CUSTOM_PROVIDER mapping is honoured.
        llm = LLMService().create_llm(provider="openai")
        # Sanity-check that we actually got credentials; if the base
        # model is empty, fall through to the legacy adapter.
        cfg = settings.resolve_openai_config()
        if cfg.get("model") and cfg.get("api_key"):
            return llm  # type: ignore[return-value]
    except Exception:
        # Swallow — will fall back to CustomChatModel below.
        pass

    # Fallback: self-hosted custom model.
    global _chat_model_singleton
    if _chat_model_singleton is None:
        _chat_model_singleton = CustomChatModel(client=CustomLLMClient())
    return _chat_model_singleton
