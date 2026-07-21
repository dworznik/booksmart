"""Configurable LLM provider layer.

Providers are selected by ``Settings`` (llm_provider / llm_model), never
hardcoded, so ingestion stages that need a model (profile generation, knowledge
extraction, embeddings) all share one seam. API keys come from ``Settings``
alone — there is no environment fallback; ``build_llm_provider`` /
``build_embedding_provider`` raise ``MissingAPIKeyError`` when the selected
provider's key is absent.

Vendor API facts live here too (see CONTEXT.md): a Limit is a
provider-declared API fact resolved per (vendor, model) at construction and
exposed as plain instance attributes; a Preference is a user choice validated
against Limits before any call is made. Consumers ask the provider
(`embedder.max_batch`), never a vendor. Limits change only via the tables in
this module — an env-overridable Limit would just be a Preference with a
scarier name.
"""

import logging
from dataclasses import dataclass
from typing import ClassVar, Protocol, TypeVar

import anthropic
import openai
from openai.types.chat import ChatCompletionMessageParam

from booksmart_core.config import Settings

# ProviderConfigError is defined in booksmart_core.errors now; re-exported here because
# provider construction (and booksmart_core.vectors) has always raised it from this seam.
from booksmart_core.errors import (
    MissingAPIKeyError,
    ProviderConfigError,
    ProviderResponseError,
)

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
    # A stable id, not a preview one: this is what a deployment gets when it
    # names no model, so it should be the entry least likely to move under it.
    "gemini": "gemini-3.5-flash",
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
    # Each model declares a different output Limit (Opus 4.8 and Sonnet 5 at
    # 128k, Haiku 4.5 at 64k), but none of them is the one that binds: this
    # provider calls the API non-streaming, and the SDK refuses non-streaming
    # requests above ~21.3k tokens ("streaming is required for operations that
    # may take longer than 10 minutes"). That threshold is a hardcoded
    # throughput heuristic which never consults the model, so the Limit we can
    # actually use is the same for every Claude model — which is why Haiku 4.5
    # takes the same 20000 despite declaring half the output of the other two.
    # valid_reasoning_efforts stays None: the Anthropic provider does not take
    # the reasoning-effort Preference, so there is nothing to validate.
    "claude-opus-4-8": LLMLimits(max_output_tokens=20000),
    "claude-sonnet-5": LLMLimits(max_output_tokens=20000),
    "claude-haiku-4-5": LLMLimits(max_output_tokens=20000),
}
_ANTHROPIC_LLM_DEFAULT = LLMLimits(max_output_tokens=20000)

_OPENAI_LLM_LIMITS = {
    # gpt-5.5 does not accept "minimal" (unlike earlier gpt-5 models).
    "gpt-5.5": LLMLimits(
        max_output_tokens=128000,
        valid_reasoning_efforts=("none", "low", "medium", "high", "xhigh"),
    ),
    # The gpt-5 generation is the mirror image of gpt-5.5: it takes "minimal",
    # but predates both "none" (gpt-5.1) and "xhigh" (gpt-5.1-codex-max). Its
    # output Limit is the same 128k as gpt-5.5 — 4x what the vendor default
    # guesses for an unknown model.
    "gpt-5-mini": LLMLimits(
        max_output_tokens=128000,
        valid_reasoning_efforts=("minimal", "low", "medium", "high"),
    ),
    "gpt-5-nano": LLMLimits(
        max_output_tokens=128000,
        valid_reasoning_efforts=("minimal", "low", "medium", "high"),
    ),
}
_OPENAI_LLM_DEFAULT = LLMLimits(max_output_tokens=32000)

# Gemini's roster churns faster than a frontier-only table can track, and an id
# it retires stops answering rather than degrading: gemini-2.5-pro (in this
# table until #53) and gemini-2.5-flash-lite both return 404 "no longer
# available to new users". So every entry here is one a live call answered on
# the probe date; retired ids are removed rather than kept as history, which
# leaves them resolving to the vendor default with the usual warning.
# All current models share the same 65536 output Limit (models.list
# outputTokenLimit, probed 2026-07-21 — see docs/research/gemini-llm-limits.md).
_GEMINI_LLM_LIMITS = {
    # The flash tiers accept every effort the compat endpoint defines. The
    # endpoint's own rejection message enumerates that universe: "Valid values
    # are: high, low, medium, minimal, none" — note it has no "xhigh", which is
    # OpenAI's alone.
    "gemini-3.1-flash-lite": LLMLimits(
        max_output_tokens=65536,
        valid_reasoning_efforts=("none", "minimal", "low", "medium", "high"),
    ),
    "gemini-3.5-flash": LLMLimits(
        max_output_tokens=65536,
        valid_reasoning_efforts=("none", "minimal", "low", "medium", "high"),
    ),
    # Still answering, but the next row to fall: the vendor gives it a shutdown
    # date of 2026-10-16 (earliest possible) and names gemini-3.5-flash as its
    # replacement — which is why that is the default above.
    "gemini-2.5-flash": LLMLimits(
        max_output_tokens=65536,
        valid_reasoning_efforts=("none", "minimal", "low", "medium", "high"),
    ),
    # The pro tier cannot stop thinking, and 3.1 Pro is stricter about it than
    # the 2.5 Pro it replaces: that one refused only "none", this one has no
    # thinking level below "low" either ("Budget 0 is invalid. This model only
    # works in thinking mode." / "Thinking level MINIMAL is not supported for
    # this model."). A preview id, so these numbers are the more refreshable.
    "gemini-3.1-pro-preview": LLMLimits(
        max_output_tokens=65536,
        valid_reasoning_efforts=("low", "medium", "high"),
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


class LLMError(ProviderResponseError):
    """The provider returned no usable completion (refusal, empty response).

    A descriptive subclass of ProviderResponseError, so it is retriable and a
    Runner classifies it the same way as any other bad model response."""


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
        # Via extra_body because a Preference is a plain str here, while the
        # SDK's reasoning_effort param is a Literal of the efforts OpenAI itself
        # accepts — too narrow for a seam that also serves Gemini's compat layer,
        # whose valid efforts are its own. The table, not the SDK type, is what
        # validates this value (see _validate_reasoning_effort).
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
            api_key=api_key,
            base_url=GEMINI_BASE_URL,
            client=client,
            reasoning_effort=reasoning_effort,
        )


@dataclass(frozen=True)
class EmbeddingResponse:
    """One batch of vectors plus what the endpoint billed for producing them.

    The embeddings side of LLMResponse: usage the provider reports is ground
    truth a consumer needs to cost a run, and embedding calls outnumber
    completion calls by far, so dropping it makes any cost estimate a guess.
    ``input_tokens`` is None only when the provider genuinely reported nothing
    — never an estimate of our own."""

    vectors: list[list[float]]
    input_tokens: int | None = None


class EmbeddingProvider(Protocol):
    model: str
    max_batch: int

    def embed(self, texts: list[str]) -> EmbeddingResponse: ...


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

    def embed(self, texts: list[str]) -> EmbeddingResponse:
        response = self._client.embeddings.create(model=self.model, input=texts)
        # OpenAI marks usage required on this response and the SDK types it
        # non-optionally, but the SDK parses responses leniently (an omitted
        # field reads back as None) and this class also serves Gemini's compat
        # endpoint, which is free to omit it. Hence the None branch the type
        # says is unreachable.
        usage = response.usage
        return EmbeddingResponse(
            vectors=[item.embedding for item in sorted(response.data, key=lambda item: item.index)],
            input_tokens=usage.prompt_tokens if usage else None,
        )


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
            api_key=api_key,
            base_url=GEMINI_BASE_URL,
            client=client,
        )


# Which Settings field holds each vendor's key, and the vendor's display name
# for MissingAPIKeyError messages.
_KEY_FIELDS = {
    "anthropic": ("Anthropic", "anthropic_api_key"),
    "openai": ("OpenAI", "openai_api_key"),
    "gemini": ("Gemini", "gemini_api_key"),
}


def _require_key(settings: Settings, vendor: str) -> str:
    display_name, field = _KEY_FIELDS[vendor]
    key: str | None = getattr(settings, field)
    if key is None:
        raise MissingAPIKeyError(display_name, field)
    return key


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider not in DEFAULT_EMBEDDING_MODELS:
        raise ProviderConfigError(
            f"Unknown embedding provider {settings.embedding_provider!r}; "
            f"expected one of {sorted(DEFAULT_EMBEDDING_MODELS)}"
        )
    model = settings.embedding_model or DEFAULT_EMBEDDING_MODELS[settings.embedding_provider]
    if settings.embedding_provider == "fake":
        # Lazy for the same import-cycle reason as in build_llm_provider.
        from booksmart_core.fakes import FakeEmbeddingProvider

        return FakeEmbeddingProvider(model=model)
    if settings.embedding_provider == "gemini":
        return GeminiEmbeddingProvider(model=model, api_key=_require_key(settings, "gemini"))
    return OpenAIEmbeddingProvider(model=model, api_key=_require_key(settings, "openai"))


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
        from booksmart_core.fakes import FakeLLMProvider

        return FakeLLMProvider(model=model)
    if settings.llm_provider == "anthropic":
        return AnthropicProvider(model=model, api_key=_require_key(settings, "anthropic"))
    if settings.llm_provider == "gemini":
        return GeminiProvider(
            model=model,
            api_key=_require_key(settings, "gemini"),
            reasoning_effort=settings.llm_reasoning_effort,
        )
    return OpenAIProvider(
        model=model,
        api_key=_require_key(settings, "openai"),
        reasoning_effort=settings.llm_reasoning_effort,
    )
