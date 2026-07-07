"""Unit tests for provider-owned Limits and Preference validation.

Vocabulary (CONTEXT.md): a Limit is a provider-declared API fact resolved per
(vendor, model) at construction; a Preference is a user choice validated
against Limits before any call is made.
"""

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from app.config import Settings
from app.fakes import FakeEmbeddingProvider, FakeLLMProvider
from app.llm import (
    AnthropicProvider,
    GeminiEmbeddingProvider,
    GeminiProvider,
    OpenAIEmbeddingProvider,
    OpenAIProvider,
    ProviderConfigError,
    build_llm_provider,
)


class TestLLMLimitResolution:
    def test_known_gemini_models_resolve_per_model_limits(self) -> None:
        flash = GeminiProvider(model="gemini-2.5-flash", api_key="test")
        pro = GeminiProvider(model="gemini-2.5-pro", api_key="test")

        assert flash.max_output_tokens == 65536
        assert flash.valid_reasoning_efforts is not None
        assert "none" in flash.valid_reasoning_efforts
        assert pro.valid_reasoning_efforts is not None
        assert "none" not in pro.valid_reasoning_efforts

    def test_known_anthropic_model_resolves_max_output_tokens(self) -> None:
        provider = AnthropicProvider(model="claude-opus-4-8", api_key="test")

        assert provider.max_output_tokens == 32000

    def test_known_openai_model_resolves_max_output_tokens(self) -> None:
        provider = OpenAIProvider(model="gpt-5.5", api_key="test")

        assert provider.max_output_tokens == 128000

    def test_unknown_model_falls_back_to_vendor_defaults_with_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="app.llm"):
            provider = GeminiProvider(model="gemini-9-experimental", api_key="test")

        assert provider.max_output_tokens == 32000
        assert provider.valid_reasoning_efforts is None
        assert any("gemini-9-experimental" in record.message for record in caplog.records)

    def test_known_model_resolves_silently(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="app.llm"):
            GeminiProvider(model="gemini-2.5-flash", api_key="test")

        assert caplog.records == []

    def test_complete_sends_the_models_own_output_token_limit(self) -> None:
        captured: dict[str, Any] = {}

        def fake_create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                model="gemini-2.5-flash",
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=None,
            )

        provider = GeminiProvider(
            model="gemini-2.5-flash",
            client=SimpleNamespace(  # type: ignore[arg-type]
                chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
            ),
        )

        provider.complete("Extract.")

        assert captured["max_completion_tokens"] == 65536

    def test_anthropic_complete_sends_the_models_own_output_token_limit(self) -> None:
        captured: dict[str, Any] = {}

        def fake_create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                model="claude-opus-4-8",
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text="ok")],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            )

        provider = AnthropicProvider(
            model="claude-opus-4-8",
            client=SimpleNamespace(messages=SimpleNamespace(create=fake_create)),  # type: ignore[arg-type]
        )

        provider.complete("Describe.")

        assert captured["max_tokens"] == 32000


class TestReasoningEffortValidation:
    def test_invalid_effort_for_known_model_raises_naming_both_sides(self) -> None:
        with pytest.raises(ProviderConfigError) as excinfo:
            GeminiProvider(model="gemini-2.5-pro", api_key="test", reasoning_effort="none")

        message = str(excinfo.value)
        assert "'none'" in message
        assert "gemini-2.5-pro" in message
        assert "low, medium, high" in message

    def test_valid_effort_for_known_model_is_accepted(self) -> None:
        provider = GeminiProvider(
            model="gemini-2.5-flash", api_key="test", reasoning_effort="none"
        )

        assert provider.reasoning_effort == "none"

    def test_effort_on_unknown_model_passes_through_as_logged_gamble(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="app.llm"):
            provider = GeminiProvider(
                model="gemini-9-experimental", api_key="test", reasoning_effort="none"
            )

        assert provider.reasoning_effort == "none"
        assert any(
            "reasoning_effort" in record.message and "'none'" in record.message
            for record in caplog.records
        )

    def test_build_llm_provider_surfaces_construction_time_validation(self) -> None:
        settings = Settings(
            llm_provider="gemini",
            llm_model="gemini-2.5-pro",
            llm_reasoning_effort="none",
            gemini_api_key="test",
        )

        with pytest.raises(ProviderConfigError):
            build_llm_provider(settings)


class TestEmbeddingLimitResolution:
    def test_known_openai_models_resolve_batch_and_dimensions(self) -> None:
        small = OpenAIEmbeddingProvider(model="text-embedding-3-small", api_key="test")
        large = OpenAIEmbeddingProvider(model="text-embedding-3-large", api_key="test")

        assert small.max_batch == 2048
        assert small.embedding_dimensions == 1536
        assert large.embedding_dimensions == 3072

    def test_known_gemini_model_resolves_batch_and_dimensions(self) -> None:
        provider = GeminiEmbeddingProvider(model="gemini-embedding-001", api_key="test")

        assert provider.max_batch == 100
        assert provider.embedding_dimensions == 3072

    def test_unknown_model_falls_back_to_vendor_defaults_with_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="app.llm"):
            provider = GeminiEmbeddingProvider(model="gemini-embedding-999", api_key="test")

        assert provider.max_batch == 100
        assert provider.embedding_dimensions is None
        assert any("gemini-embedding-999" in record.message for record in caplog.records)


class TestFakeProviderLimits:
    def test_fake_llm_exposes_limits(self) -> None:
        provider = FakeLLMProvider()

        assert provider.max_output_tokens == 32000

    def test_fake_embedder_exposes_limits(self) -> None:
        provider = FakeEmbeddingProvider()

        assert provider.max_batch == 100
        assert provider.embedding_dimensions == 8
