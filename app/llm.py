"""Configurable LLM provider layer.

Providers are selected by configuration (BOOKSMART_LLM_PROVIDER /
BOOKSMART_LLM_MODEL), never hardcoded, so ingestion stages that need a model
(profile generation, knowledge extraction, embeddings) all share one seam.
API keys come from settings or fall back to the SDKs' standard environment
variables (ANTHROPIC_API_KEY, OPENAI_API_KEY).
"""

import os
from dataclasses import dataclass
from typing import Protocol

import anthropic
import openai
from openai.types.chat import ChatCompletionMessageParam

from app.config import Settings

MAX_COMPLETION_TOKENS = 16000

# Gemini is served through Google's OpenAI-compatible endpoint, so the OpenAI
# SDK covers both and Gemini needs no dependency of its own.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-5.5",
    "gemini": "gemini-2.5-pro",
    # Deterministic canned responses, no keys or network (CI, local dev).
    "fake": "fake-llm-1",
}


class LLMError(RuntimeError):
    """The provider returned no usable completion (refusal, empty response)."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str


class LLMProvider(Protocol):
    model: str

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse: ...


class AnthropicProvider:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self._client = client or anthropic.Anthropic(api_key=api_key)

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_COMPLETION_TOKENS,
            system=system if system is not None else anthropic.omit,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise LLMError(f"{self.model} refused the request")
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(text=text, model=response.model)


class OpenAIProvider:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: openai.OpenAI | None = None,
    ) -> None:
        self.model = model
        self._client = client or openai.OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        messages: list[ChatCompletionMessageParam] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            messages=messages,
        )
        if not response.choices or response.choices[0].message.content is None:
            raise LLMError(f"{self.model} returned an empty completion")
        return LLMResponse(text=response.choices[0].message.content, model=response.model)


class GeminiProvider(OpenAIProvider):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        client: openai.OpenAI | None = None,
    ) -> None:
        # The OpenAI SDK only knows OPENAI_API_KEY, so resolve Gemini's own
        # conventional variable here instead of leaving it to the SDK.
        super().__init__(
            model,
            api_key=api_key or os.environ.get("GEMINI_API_KEY"),
            base_url=GEMINI_BASE_URL,
            client=client,
        )


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider not in DEFAULT_MODELS:
        raise ValueError(
            f"Unknown LLM provider {settings.llm_provider!r}; "
            f"expected one of {sorted(DEFAULT_MODELS)}"
        )
    model = settings.llm_model or DEFAULT_MODELS[settings.llm_provider]
    if settings.llm_provider == "fake":
        # Imported lazily: fakes imports stage prompts, whose modules import
        # this one.
        from app.fakes import FakeLLMProvider

        return FakeLLMProvider(model=model)
    if settings.llm_provider == "anthropic":
        return AnthropicProvider(model=model, api_key=settings.anthropic_api_key)
    if settings.llm_provider == "gemini":
        return GeminiProvider(model=model, api_key=settings.gemini_api_key)
    return OpenAIProvider(model=model, api_key=settings.openai_api_key)
