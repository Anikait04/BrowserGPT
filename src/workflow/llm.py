from __future__ import annotations

import os
from typing import Any
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

import config


class LLMService:

    # ==========================================================================
    # OpenAI
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
            reasoning_effort="high"
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
        }

        provider = (provider or config.LLM_PROVIDER).lower()

        if provider not in providers:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        return providers[provider](
            temperature=temperature,
            max_tokens=max_tokens,
        )


@lru_cache(maxsize=1)
def get_llm() -> Any:
    return LLMService().create_llm()
