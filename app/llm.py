"""Configurable LLM provider layer.

Providers are selected by configuration (BOOKSMART_LLM_PROVIDER /
BOOKSMART_LLM_MODEL), never hardcoded, so ingestion stages that need a model
(profile generation, knowledge extraction, embeddings) all share one seam.
API keys come from settings or fall back to the SDKs' standard environment
variables (ANTHROPIC_API_KEY, OPENAI_API_KEY).

Vendor API facts live here too (see CONTEXT.md): a Limit is a
provider-declared API fact resolved per (vendor, model) at construction and
exposed as plain instance attributes; a Preference is a user choice validated
against Limits before any call is made. Consumers ask the provider
(`embedder.max_batch`), never a vendor. Limits change only via the tables in
this module — an env-overridable Limit would just be a Preference with a
scarier name.
"""

import logging
import os
from dataclasses import dataclass
from typing import ClassVar, Protocol, TypeVar

import anthropic
import openai
from openai.types.chat import ChatCompletionMessageParam

from app.config import Settings

logger = logging.getLogger(__name__)

# Both SDKs retry rate limits and transient errors with exponential backoff,
# honoring Retry-After; the default of 2 attempts gives up too easily for a
# long unattended ingestion run.
CLIENT_MAX_RETRIES = 4

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

# Anthropic offers no embeddings API, so the embedding provider is configured
# separately (BOOKSMART_EMBEDDING_PROVIDER / BOOKSMART_EMBEDDING_MODEL).
DEFAULT_EMBEDDING_MODELS = {
    "openai": "text-embedding-3-small",
    "gemini": "gemini-embedding-001",
    # Deterministic fixed-size vectors, no keys or network (CI, local dev).
    "fake": "fake-embed-1",
}


class ProviderConfigError(ValueError):
    """A Preference conflicts with a Limit (or the configuration is otherwise
    deterministically wrong) — at provider construction or at the first write
    the model-locked vector collection rejects. Retrying cannot fix it, so it
    maps to a non-retriable error in durable execution."""


@dataclass(frozen=True)
class LLMLimits:
    """Provider-declared API facts for one (vendor, model)."""

    max_output_tokens: int
    # None means unknown: reasoning-effort Preferences pass through to the API
    # unvalidated (a logged gamble) instead of being rejected here.
    valid_reasoning_efforts: tuple[str, ...] | None = None


@dataclass(frozen=True)
class EmbeddingLimits:
    """Provider-declared API facts for one (vendor, embedding model)."""

    max_batch: int
    embedding_dimensions: int | None = None  # None means unknown


# Per-model Limits tables plus a conservative per-vendor default, so unknown
# (new) models stay usable day one with one log line. If we know the Limit we
# enforce it; if we don't, we say so and defer to the API.

_ANTHROPIC_LLM_LIMITS = {
    # valid_reasoning_efforts stays None: the Anthropic provider does not take
    # the reasoning-effort Preference, so there is nothing to validate.
    "claude-opus-4-8": LLMLimits(max_output_tokens=32000),
    "claude-sonnet-5": LLMLimits(max_output_tokens=64000),
}
_ANTHROPIC_LLM_DEFAULT = LLMLimits(max_output_tokens=32000)

_OPENAI_LLM_LIMITS = {
    "gpt-5.5": LLMLimits(
        max_output_tokens=128000,
        valid_reasoning_efforts=("none", "minimal", "low", "medium", "high"),
    ),
}
_OPENAI_LLM_DEFAULT = LLMLimits(max_output_tokens=32000)

_GEMINI_LLM_LIMITS = {
    # 2.5 Pro rejects "none" (thinking cannot be disabled); Flash accepts it.
    "gemini-2.5-pro": LLMLimits(
        max_output_tokens=65536,
        valid_reasoning_efforts=("low", "medium", "high"),
    ),
    "gemini-2.5-flash": LLMLimits(
        max_output_tokens=65536,
        valid_reasoning_efforts=("none", "low", "medium", "high"),
    ),
}
_GEMINI_LLM_DEFAULT = LLMLimits(max_output_tokens=32000)

_OPENAI_EMBEDDING_LIMITS = {
    "text-embedding-3-small": EmbeddingLimits(max_batch=2048, embedding_dimensions=1536),
    "text-embedding-3-large": EmbeddingLimits(max_batch=2048, embedding_dimensions=3072),
}
# 2048 inputs per request is an endpoint-wide fact, not per-model.
_OPENAI_EMBEDDING_DEFAULT = EmbeddingLimits(max_batch=2048)

_GEMINI_EMBEDDING_LIMITS = {
    "gemini-embedding-001": EmbeddingLimits(max_batch=100, embedding_dimensions=3072),
}
_GEMINI_EMBEDDING_DEFAULT = EmbeddingLimits(max_batch=100)

LimitsT = TypeVar("LimitsT", LLMLimits, EmbeddingLimits)


def resolve_limits(
    vendor: str, model: str, table: dict[str, LimitsT], default: LimitsT
) -> LimitsT:
    """Look up a model's Limits in its provider module's table, falling back to
    the conservative vendor default (with one log line) for unknown models."""
    limits = table.get(model)
    if limits is not None:
        return limits
    logger.warning(
        "unknown %s model %r: assuming conservative vendor-default limits %s",
        vendor,
        model,
        default,
    )
    return default


def _validate_reasoning_effort(
    vendor: str, model: str, effort: str | None, valid: tuple[str, ...] | None
) -> None:
    if effort is None:
        return
    if valid is None:
        logger.warning(
            "cannot validate reasoning_effort %r for unknown %s model %r; "
            "passing it through — the API may reject it",
            effort,
            vendor,
            model,
        )
        return
    if effort not in valid:
        raise ProviderConfigError(
            f"reasoning_effort {effort!r} is not valid for {model}; valid: {', '.join(valid)}"
        )


def strip_fences(text: str) -> str:
    """Models occasionally wrap JSON in ``` fences despite instructions."""
    if not text.startswith("```"):
        return text
    lines = text.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


class LLMError(RuntimeError):
    """The provider returned no usable completion (refusal, empty response)."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    # Billable token counts as reported by the provider; None when the
    # provider did not report usage.
    input_tokens: int | None = None
    output_tokens: int | None = None


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
        limits = resolve_limits("anthropic", model, _ANTHROPIC_LLM_LIMITS, _ANTHROPIC_LLM_DEFAULT)
        self.max_output_tokens = limits.max_output_tokens
        self._client = client or anthropic.Anthropic(
            api_key=api_key, max_retries=CLIENT_MAX_RETRIES
        )

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_output_tokens,
            system=system if system is not None else anthropic.omit,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise LLMError(f"{self.model} refused the request")
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class OpenAIProvider:
    # Overridden by OpenAI-compatible subclasses so limit resolution and
    # error messages speak about the actual vendor.
    _vendor: ClassVar[str] = "openai"
    _llm_limits: ClassVar[dict[str, LLMLimits]] = _OPENAI_LLM_LIMITS
    _llm_default_limits: ClassVar[LLMLimits] = _OPENAI_LLM_DEFAULT

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: openai.OpenAI | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        limits = resolve_limits(self._vendor, model, self._llm_limits, self._llm_default_limits)
        self.max_output_tokens = limits.max_output_tokens
        self.valid_reasoning_efforts = limits.valid_reasoning_efforts
        _validate_reasoning_effort(
            self._vendor, model, reasoning_effort, limits.valid_reasoning_efforts
        )
        self.reasoning_effort = reasoning_effort
        self._client = client or openai.OpenAI(
            api_key=api_key, base_url=base_url, max_retries=CLIENT_MAX_RETRIES
        )

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        messages: list[ChatCompletionMessageParam] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        # Via extra_body because Gemini's compat layer accepts "none", which
        # the OpenAI SDK's reasoning_effort type does not.
        extra_body = (
            {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else None
        )
        response = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_output_tokens,
            messages=messages,
            extra_body=extra_body,
        )
        if not response.choices or response.choices[0].message.content is None:
            raise LLMError(f"{self.model} returned an empty completion")
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content,
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )


def _resolve_gemini_key(api_key: str | None) -> str | None:
    # The OpenAI SDK only knows OPENAI_API_KEY, so resolve Gemini's own
    # conventional variable here instead of leaving it to the SDK.
    return api_key or os.environ.get("GEMINI_API_KEY")


class GeminiProvider(OpenAIProvider):
    _vendor: ClassVar[str] = "gemini"
    _llm_limits: ClassVar[dict[str, LLMLimits]] = _GEMINI_LLM_LIMITS
    _llm_default_limits: ClassVar[LLMLimits] = _GEMINI_LLM_DEFAULT

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        client: openai.OpenAI | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(
            model,
            api_key=_resolve_gemini_key(api_key),
            base_url=GEMINI_BASE_URL,
            client=client,
            reasoning_effort=reasoning_effort,
        )


class EmbeddingProvider(Protocol):
    model: str
    max_batch: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbeddingProvider:
    _vendor: ClassVar[str] = "openai"
    _embedding_limits: ClassVar[dict[str, EmbeddingLimits]] = _OPENAI_EMBEDDING_LIMITS
    _embedding_default_limits: ClassVar[EmbeddingLimits] = _OPENAI_EMBEDDING_DEFAULT

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: openai.OpenAI | None = None,
    ) -> None:
        self.model = model
        limits = resolve_limits(
            self._vendor, model, self._embedding_limits, self._embedding_default_limits
        )
        self.max_batch = limits.max_batch
        self.embedding_dimensions = limits.embedding_dimensions
        self._client = client or openai.OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


class GeminiEmbeddingProvider(OpenAIEmbeddingProvider):
    _vendor: ClassVar[str] = "gemini"
    _embedding_limits: ClassVar[dict[str, EmbeddingLimits]] = _GEMINI_EMBEDDING_LIMITS
    _embedding_default_limits: ClassVar[EmbeddingLimits] = _GEMINI_EMBEDDING_DEFAULT

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        client: openai.OpenAI | None = None,
    ) -> None:
        super().__init__(
            model,
            api_key=_resolve_gemini_key(api_key),
            base_url=GEMINI_BASE_URL,
            client=client,
        )


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider not in DEFAULT_EMBEDDING_MODELS:
        raise ProviderConfigError(
            f"Unknown embedding provider {settings.embedding_provider!r}; "
            f"expected one of {sorted(DEFAULT_EMBEDDING_MODELS)}"
        )
    model = settings.embedding_model or DEFAULT_EMBEDDING_MODELS[settings.embedding_provider]
    if settings.embedding_provider == "fake":
        # Lazy for the same import-cycle reason as in build_llm_provider.
        from app.fakes import FakeEmbeddingProvider

        return FakeEmbeddingProvider(model=model)
    if settings.embedding_provider == "gemini":
        return GeminiEmbeddingProvider(model=model, api_key=settings.gemini_api_key)
    return OpenAIEmbeddingProvider(model=model, api_key=settings.openai_api_key)


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider not in DEFAULT_MODELS:
        raise ProviderConfigError(
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
        return GeminiProvider(
            model=model,
            api_key=settings.gemini_api_key,
            reasoning_effort=settings.llm_reasoning_effort,
        )
    return OpenAIProvider(
        model=model,
        api_key=settings.openai_api_key,
        reasoning_effort=settings.llm_reasoning_effort,
    )
