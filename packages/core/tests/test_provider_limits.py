"""Unit tests for provider-owned Limits and Preference validation.

Vocabulary (CONTEXT.md): a Limit is a provider-declared API fact resolved per
(vendor, model) at construction; a Preference is a user choice validated
against Limits before any call is made.
"""

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from booksmart_core.config import Settings
from booksmart_core.fakes import FakeEmbeddingProvider, FakeLLMProvider
from booksmart_core.llm import (
    AnthropicProvider,
    GeminiEmbeddingProvider,
    GeminiProvider,
    OpenAIEmbeddingProvider,
    OpenAIProvider,
    ProviderConfigError,
    build_llm_provider,
)


class TestLLMLimitResolution:
    def test_current_gemini_tiers_resolve_from_the_table_not_the_vendor_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The point of #53: the tiers callers now select were absent, so they
        # fell through to a default that under-reported output by half and
        # warned on every run.
        with caplog.at_level(logging.WARNING, logger="booksmart_core.llm"):
            flash_lite = GeminiProvider(model="gemini-3.1-flash-lite", api_key="test")
            pro = GeminiProvider(model="gemini-3.1-pro-preview", api_key="test")
            flash = GeminiProvider(model="gemini-3.5-flash", api_key="test")

        assert flash_lite.max_output_tokens == 65536
        assert pro.max_output_tokens == 65536
        assert flash.max_output_tokens == 65536
        assert caplog.records == []

    def test_gemini_flash_tiers_accept_every_effort_the_endpoint_allows(self) -> None:
        full_range = ("none", "minimal", "low", "medium", "high")

        for model in ("gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash"):
            provider = GeminiProvider(model=model, api_key="test")

            assert provider.valid_reasoning_efforts == full_range

    def test_gemini_pro_tier_takes_neither_none_nor_minimal(self) -> None:
        # 3.1 Pro is stricter than the 2.5 Pro it replaces: that one refused
        # only "none", this one has no thinking level below "low" either.
        pro = GeminiProvider(model="gemini-3.1-pro-preview", api_key="test")

        assert pro.valid_reasoning_efforts == ("low", "medium", "high")

    def test_retired_gemini_model_falls_back_to_the_vendor_default_with_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The table holds live models only. gemini-2.5-pro was removed once the
        # API started answering 404 for it, so it now resolves like any id the
        # table has never heard of — a guess, said out loud.
        with caplog.at_level(logging.WARNING, logger="booksmart_core.llm"):
            provider = GeminiProvider(model="gemini-2.5-pro", api_key="test")

        assert provider.max_output_tokens == 32000
        assert provider.valid_reasoning_efforts is None
        assert any("gemini-2.5-pro" in record.message for record in caplog.records)

    def test_the_gemini_default_model_is_one_the_table_knows(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Selecting the provider without naming a model must not land on an id
        # the API has retired — which is what the old gemini-2.5-pro default did.
        settings = Settings(llm_provider="gemini", gemini_api_key="test")

        with caplog.at_level(logging.WARNING, logger="booksmart_core.llm"):
            provider = build_llm_provider(settings)

        assert isinstance(provider, GeminiProvider)
        assert provider.max_output_tokens == 65536
        assert provider.valid_reasoning_efforts is not None
        assert caplog.records == []

    def test_known_anthropic_model_resolves_max_output_tokens(self) -> None:
        # The SDK's non-streaming ceiling, not the model's 128k API cap.
        provider = AnthropicProvider(model="claude-opus-4-8", api_key="test")

        assert provider.max_output_tokens == 20000

    def test_known_openai_model_resolves_max_output_tokens(self) -> None:
        provider = OpenAIProvider(model="gpt-5.5", api_key="test")

        assert provider.max_output_tokens == 128000

    def test_cheap_tier_models_resolve_from_the_tables_not_the_vendor_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The cheap tiers are the point of #47: before they were tabulated they
        # fell through to a vendor default that under-reported OpenAI's output
        # Limit 4x, and said so in a warning on every run.
        with caplog.at_level(logging.WARNING, logger="booksmart_core.llm"):
            mini = OpenAIProvider(model="gpt-5-mini", api_key="test")
            nano = OpenAIProvider(model="gpt-5-nano", api_key="test")
            haiku = AnthropicProvider(model="claude-haiku-4-5", api_key="test")

        assert mini.max_output_tokens == 128000
        assert nano.max_output_tokens == 128000
        # Haiku declares half the output of Opus and Sonnet, but resolves to the
        # same 20000: the SDK's non-streaming threshold binds before the model's
        # own Limit does, and that threshold ignores the model.
        assert haiku.max_output_tokens == 20000
        assert caplog.records == []

    def test_gpt_5_generation_accepts_minimal_effort_and_gpt_5_5_does_not(self) -> None:
        # The two tuples are mirror images: the gpt-5 generation takes "minimal"
        # but predates both "none" (gpt-5.1) and "xhigh" (gpt-5.1-codex-max).
        mini = OpenAIProvider(model="gpt-5-mini", api_key="test")
        nano = OpenAIProvider(model="gpt-5-nano", api_key="test")
        gpt55 = OpenAIProvider(model="gpt-5.5", api_key="test")

        assert mini.valid_reasoning_efforts == ("minimal", "low", "medium", "high")
        assert nano.valid_reasoning_efforts == ("minimal", "low", "medium", "high")
        assert gpt55.valid_reasoning_efforts is not None
        assert "minimal" not in gpt55.valid_reasoning_efforts

    def test_unknown_model_falls_back_to_vendor_defaults_with_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="booksmart_core.llm"):
            provider = GeminiProvider(model="gemini-9-experimental", api_key="test")

        assert provider.max_output_tokens == 32000
        assert provider.valid_reasoning_efforts is None
        assert any("gemini-9-experimental" in record.message for record in caplog.records)

    def test_unknown_anthropic_model_falls_back_to_vendor_defaults_with_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="booksmart_core.llm"):
            provider = AnthropicProvider(model="claude-10", api_key="test")

        assert provider.max_output_tokens == 20000
        assert any("claude-10" in record.message for record in caplog.records)

    def test_unknown_openai_model_falls_back_to_vendor_defaults_with_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="booksmart_core.llm"):
            provider = OpenAIProvider(model="gpt-7", api_key="test")

        assert provider.max_output_tokens == 32000
        assert provider.valid_reasoning_efforts is None
        assert any("gpt-7" in record.message for record in caplog.records)

    def test_known_model_resolves_silently(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="booksmart_core.llm"):
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

        assert captured["max_tokens"] == 20000


class TestReasoningEffortValidation:
    def test_invalid_effort_for_known_model_raises_naming_both_sides(self) -> None:
        with pytest.raises(ProviderConfigError) as excinfo:
            GeminiProvider(
                model="gemini-3.1-pro-preview", api_key="test", reasoning_effort="none"
            )

        message = str(excinfo.value)
        assert "'none'" in message
        assert "gemini-3.1-pro-preview" in message
        assert "low, medium, high" in message

    def test_minimal_is_rejected_on_the_gemini_pro_tier(self) -> None:
        # The test above covers "none"; "minimal" is the value 3.1 Pro dropped
        # that 2.5 Pro allowed ("Thinking level MINIMAL is not supported for
        # this model."), so porting the old tuple forward would have missed it.
        with pytest.raises(ProviderConfigError) as excinfo:
            GeminiProvider(
                model="gemini-3.1-pro-preview", api_key="test", reasoning_effort="minimal"
            )

        assert "'minimal'" in str(excinfo.value)

    def test_xhigh_is_rejected_on_every_gemini_tier(self) -> None:
        # "Valid values are: high, low, medium, minimal, none" — the compat
        # endpoint's own enumeration has no xhigh; that one is OpenAI's alone.
        for model in ("gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.1-pro-preview"):
            with pytest.raises(ProviderConfigError) as excinfo:
                GeminiProvider(model=model, api_key="test", reasoning_effort="xhigh")

            assert "'xhigh'" in str(excinfo.value)

    def test_efforts_the_current_gemini_tiers_accept_are_accepted(self) -> None:
        # The other half of validation: a Preference the model does take must
        # survive construction rather than being rejected with its neighbours.
        flash_lite = GeminiProvider(
            model="gemini-3.1-flash-lite", api_key="test", reasoning_effort="none"
        )
        pro = GeminiProvider(
            model="gemini-3.1-pro-preview", api_key="test", reasoning_effort="low"
        )

        assert flash_lite.reasoning_effort == "none"
        assert pro.reasoning_effort == "low"

    def test_valid_effort_for_known_model_is_accepted(self) -> None:
        provider = GeminiProvider(
            model="gemini-2.5-flash", api_key="test", reasoning_effort="none"
        )

        assert provider.reasoning_effort == "none"

    def test_effort_the_gpt_5_generation_predates_is_rejected(self) -> None:
        # "none" arrived with gpt-5.1 and "xhigh" after gpt-5.1-codex-max, so a
        # Preference carrying either is a config error on mini/nano — before #47
        # both passed through unvalidated for the API to reject at call time.
        for effort in ("none", "xhigh"):
            with pytest.raises(ProviderConfigError) as excinfo:
                OpenAIProvider(model="gpt-5-mini", api_key="test", reasoning_effort=effort)

            message = str(excinfo.value)
            assert f"{effort!r}" in message
            assert "gpt-5-mini" in message
            assert "minimal, low, medium, high" in message

    def test_minimal_effort_is_accepted_on_the_gpt_5_generation(self) -> None:
        provider = OpenAIProvider(model="gpt-5-nano", api_key="test", reasoning_effort="minimal")

        assert provider.reasoning_effort == "minimal"

    def test_effort_on_unknown_model_passes_through_as_logged_gamble(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="booksmart_core.llm"):
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
            llm_model="gemini-3.1-pro-preview",
            llm_reasoning_effort="none",
            gemini_api_key="test",
        )

        with pytest.raises(ProviderConfigError):
            build_llm_provider(settings)

    def test_anthropic_ignores_the_effort_preference_rather_than_validating_it(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The None in the Anthropic table is not "unknown, so gamble" — it is
        # "nothing to validate": build_llm_provider never forwards the effort to
        # AnthropicProvider, which takes no such argument. So an effort that no
        # Claude model would accept is neither rejected nor warned about; it
        # simply cannot reach the API. Pins the reason the table entry is None.
        settings = Settings(
            llm_provider="anthropic",
            llm_model="claude-haiku-4-5",
            llm_reasoning_effort="xhigh",
            anthropic_api_key="test",
        )

        with caplog.at_level(logging.WARNING, logger="booksmart_core.llm"):
            provider = build_llm_provider(settings)

        assert isinstance(provider, AnthropicProvider)
        assert caplog.records == []


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
        with caplog.at_level(logging.WARNING, logger="booksmart_core.llm"):
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

    def test_unknown_fake_models_fall_back_to_vendor_defaults_with_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="booksmart_core.llm"):
            llm = FakeLLMProvider(model="fake-llm-2")
            embedder = FakeEmbeddingProvider(model="fake-embed-2")

        assert llm.max_output_tokens == 32000
        assert embedder.max_batch == 100
        assert embedder.embedding_dimensions is None
        assert len(caplog.records) == 2
