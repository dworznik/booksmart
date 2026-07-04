"""Configurable LLM provider layer.

Providers are selected by configuration (BOOKSMART_LLM_PROVIDER /
BOOKSMART_LLM_MODEL), never hardcoded, so ingestion stages that need a model
(profile generation, knowledge extraction, embeddings) all share one seam.
API keys come from settings or fall back to the SDKs' standard environment
variables (ANTHROPIC_API_KEY, OPENAI_API_KEY).
"""

from dataclasses import dataclass
from typing import Protocol

import anthropic
import openai
from openai.types.chat import ChatCompletionMessageParam

from app.config import Settings

MAX_COMPLETION_TOKENS = 16000

DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-5.5",
}


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str


class LLMProvider(Protocol):
    model: str

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse: ...


class AnthropicProvider:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_COMPLETION_TOKENS,
            system=system if system is not None else anthropic.omit,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"{self.model} refused the request")
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(text=text, model=response.model)


class OpenAIProvider:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self._client = openai.OpenAI(api_key=api_key)

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        messages: list[ChatCompletionMessageParam] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(model=self.model, messages=messages)
        if not response.choices or response.choices[0].message.content is None:
            raise RuntimeError(f"{self.model} returned an empty completion")
        return LLMResponse(text=response.choices[0].message.content, model=response.model)


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider not in DEFAULT_MODELS:
        raise ValueError(
            f"Unknown LLM provider {settings.llm_provider!r}; "
            f"expected one of {sorted(DEFAULT_MODELS)}"
        )
    model = settings.llm_model or DEFAULT_MODELS[settings.llm_provider]
    if settings.llm_provider == "anthropic":
        return AnthropicProvider(model=model, api_key=settings.anthropic_api_key)
    return OpenAIProvider(model=model, api_key=settings.openai_api_key)
