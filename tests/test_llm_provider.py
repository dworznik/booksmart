"""Unit tests for the configurable LLM provider layer."""

from types import SimpleNamespace
from typing import Any

import pytest

from app.config import Settings
from app.extraction import EXTRACTION_SYSTEM_PROMPT, parse_extraction_response
from app.fakes import FakeEmbeddingProvider, FakeLLMProvider
from app.profile import PROFILE_SYSTEM_PROMPT
from app.summaries import SUMMARY_SYSTEM_PROMPT, parse_summary_response
from app.llm import (
    GEMINI_BASE_URL,
    AnthropicProvider,
    GeminiEmbeddingProvider,
    GeminiProvider,
    LLMError,
    OpenAIEmbeddingProvider,
    OpenAIProvider,
    build_embedding_provider,
    build_llm_provider,
)


def make_settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "anthropic_api_key": "sk-ant-test",
        "openai_api_key": "sk-openai-test",
        "gemini_api_key": "sk-gemini-test",
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestProviderSelection:
    def test_default_configuration_builds_anthropic_provider(self) -> None:
        provider = build_llm_provider(make_settings())

        assert isinstance(provider, AnthropicProvider)
        assert provider.model == "claude-opus-4-8"

    def test_openai_provider_selected_via_configuration(self) -> None:
        provider = build_llm_provider(make_settings(llm_provider="openai"))

        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-5.5"

    def test_gemini_provider_selected_via_configuration(self) -> None:
        provider = build_llm_provider(make_settings(llm_provider="gemini"))

        assert isinstance(provider, GeminiProvider)
        assert provider.model == "gemini-2.5-pro"
        assert str(provider._client.base_url) == GEMINI_BASE_URL

    def test_configured_model_overrides_provider_default(self) -> None:
        provider = build_llm_provider(
            make_settings(llm_provider="anthropic", llm_model="claude-sonnet-5")
        )

        assert provider.model == "claude-sonnet-5"

    def test_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="hal9000"):
            build_llm_provider(make_settings(llm_provider="hal9000"))

    def test_reasoning_effort_reaches_openai_compatible_providers(self) -> None:
        provider = build_llm_provider(
            make_settings(llm_provider="gemini", llm_reasoning_effort="none")
        )

        assert isinstance(provider, GeminiProvider)
        assert provider.reasoning_effort == "none"


class TestReasoningEffort:
    def _provider(self, reasoning_effort: str | None, captured: dict[str, Any]) -> OpenAIProvider:
        def fake_create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                model="gemini-2.5-flash",
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        return OpenAIProvider(
            model="gemini-2.5-flash",
            reasoning_effort=reasoning_effort,
            client=SimpleNamespace(  # type: ignore[arg-type]
                chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
            ),
        )

    def test_configured_effort_is_sent_with_every_call(self) -> None:
        captured: dict[str, Any] = {}

        self._provider("none", captured).complete("Extract.")

        assert captured["extra_body"] == {"reasoning_effort": "none"}

    def test_unset_effort_sends_nothing(self) -> None:
        captured: dict[str, Any] = {}

        self._provider(None, captured).complete("Extract.")

        assert captured.get("extra_body") is None


class TestEmbeddingProviderSelection:
    def test_default_configuration_builds_openai_embeddings(self) -> None:
        provider = build_embedding_provider(make_settings())

        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.model == "text-embedding-3-small"

    def test_gemini_embeddings_selected_via_configuration(self) -> None:
        provider = build_embedding_provider(make_settings(embedding_provider="gemini"))

        assert isinstance(provider, GeminiEmbeddingProvider)
        assert provider.model == "gemini-embedding-001"
        assert str(provider._client.base_url) == GEMINI_BASE_URL

    def test_configured_model_overrides_default(self) -> None:
        provider = build_embedding_provider(
            make_settings(embedding_model="text-embedding-3-large")
        )

        assert provider.model == "text-embedding-3-large"

    def test_unknown_embedding_provider_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="anthropic"):
            build_embedding_provider(make_settings(embedding_provider="anthropic"))


class TestOpenAIEmbeddingProvider:
    def test_embed_returns_vectors_in_input_order(self) -> None:
        captured: dict[str, Any] = {}

        def fake_create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            # Deliberately out of order: the provider must sort by index.
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=1, embedding=[2.0, 2.0]),
                    SimpleNamespace(index=0, embedding=[1.0, 1.0]),
                ]
            )

        provider = OpenAIEmbeddingProvider(
            model="text-embedding-3-small",
            client=SimpleNamespace(  # type: ignore[arg-type]
                embeddings=SimpleNamespace(create=fake_create)
            ),
        )

        vectors = provider.embed(["first", "second"])

        assert vectors == [[1.0, 1.0], [2.0, 2.0]]
        assert captured["model"] == "text-embedding-3-small"
        assert captured["input"] == ["first", "second"]


class TestFakeProvider:
    def test_fake_provider_selected_via_configuration(self) -> None:
        provider = build_llm_provider(make_settings(llm_provider="fake"))

        assert isinstance(provider, FakeLLMProvider)
        assert provider.model == "fake-llm-1"

    def test_fake_provider_needs_no_api_keys(self) -> None:
        provider = build_llm_provider(Settings(llm_provider="fake"))

        response = provider.complete("anything")

        assert response.text
        assert response.model == "fake-llm-1"

    def test_fake_provider_reports_zero_token_usage(self) -> None:
        provider = build_llm_provider(Settings(llm_provider="fake"))

        response = provider.complete("anything")

        assert response.input_tokens == 0
        assert response.output_tokens == 0

    def test_fake_extraction_response_parses_as_knowledge_objects(self) -> None:
        provider = build_llm_provider(Settings(llm_provider="fake"))

        response = provider.complete("chapter text", system=EXTRACTION_SYSTEM_PROMPT)

        objects, dropped = parse_extraction_response(response.text)
        assert len(objects) == 1
        assert objects[0].type == "Principle"
        assert dropped == []

    def test_fake_provider_answers_profile_stage_deterministically(self) -> None:
        provider = build_llm_provider(Settings(llm_provider="fake"))

        first = provider.complete("prompt", system=PROFILE_SYSTEM_PROMPT)
        second = provider.complete("different prompt", system=PROFILE_SYSTEM_PROMPT)

        assert first.text == second.text
        assert "profile" in first.text.lower()

    def test_fake_summary_response_parses_for_any_section_count(self) -> None:
        provider = build_llm_provider(Settings(llm_provider="fake"))

        response = provider.complete("chapter text", system=SUMMARY_SYSTEM_PROMPT)

        chapter_summary, section_summaries = parse_summary_response(response.text, 3)
        assert chapter_summary
        assert section_summaries == [None, None, None]

    def test_fake_embedding_provider_needs_no_api_keys(self) -> None:
        provider = build_embedding_provider(Settings(embedding_provider="fake"))

        assert isinstance(provider, FakeEmbeddingProvider)
        assert provider.model == "fake-embed-1"

    def test_fake_embeddings_are_deterministic_fixed_size_vectors(self) -> None:
        provider = build_embedding_provider(Settings(embedding_provider="fake"))

        first = provider.embed(["alpha", "a longer text"])
        second = provider.embed(["alpha", "a longer text"])

        assert first == second
        assert {len(vector) for vector in first} == {8}


class TestAnthropicProvider:
    def test_complete_sends_prompt_and_joins_text_blocks(self) -> None:
        captured: dict[str, Any] = {}

        def fake_create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                model="claude-opus-4-8",
                stop_reason="end_turn",
                content=[
                    SimpleNamespace(type="thinking", thinking=""),
                    SimpleNamespace(type="text", text="Part one. "),
                    SimpleNamespace(type="text", text="Part two."),
                ],
                usage=SimpleNamespace(input_tokens=1200, output_tokens=34),
            )

        provider = AnthropicProvider(
            model="claude-opus-4-8",
            client=SimpleNamespace(messages=SimpleNamespace(create=fake_create)),  # type: ignore[arg-type]
        )

        response = provider.complete("Describe the book.", system="You are a librarian.")

        assert response.text == "Part one. Part two."
        assert response.model == "claude-opus-4-8"
        assert response.input_tokens == 1200
        assert response.output_tokens == 34
        assert captured["model"] == "claude-opus-4-8"
        assert captured["system"] == "You are a librarian."
        assert captured["messages"] == [{"role": "user", "content": "Describe the book."}]

    def test_refusal_stop_reason_raises(self) -> None:
        provider = AnthropicProvider(
            model="claude-opus-4-8",
            client=SimpleNamespace(  # type: ignore[arg-type]
                messages=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(
                        model="claude-opus-4-8", stop_reason="refusal", content=[]
                    )
                )
            ),
        )

        with pytest.raises(LLMError, match="refus"):
            provider.complete("Describe the book.")


class TestOpenAIProvider:
    def test_complete_sends_prompt_and_reads_first_choice(self) -> None:
        captured: dict[str, Any] = {}

        def fake_create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                model="gpt-5.5",
                choices=[SimpleNamespace(message=SimpleNamespace(content="The profile."))],
                usage=SimpleNamespace(prompt_tokens=850, completion_tokens=120),
            )

        provider = OpenAIProvider(
            model="gpt-5.5",
            client=SimpleNamespace(  # type: ignore[arg-type]
                chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
            ),
        )

        response = provider.complete("Describe the book.", system="You are a librarian.")

        assert response.text == "The profile."
        assert response.model == "gpt-5.5"
        assert response.input_tokens == 850
        assert response.output_tokens == 120
        assert captured["model"] == "gpt-5.5"
        assert captured["messages"] == [
            {"role": "system", "content": "You are a librarian."},
            {"role": "user", "content": "Describe the book."},
        ]

    def test_missing_usage_yields_unknown_token_counts(self) -> None:
        # The OpenAI SDK types usage as optional; absence must not break calls.
        provider = OpenAIProvider(
            model="gpt-5.5",
            client=SimpleNamespace(  # type: ignore[arg-type]
                chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **kwargs: SimpleNamespace(
                            model="gpt-5.5",
                            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                            usage=None,
                        )
                    )
                )
            ),
        )

        response = provider.complete("Describe the book.")

        assert response.input_tokens is None
        assert response.output_tokens is None

    def test_empty_completion_raises(self) -> None:
        provider = OpenAIProvider(
            model="gpt-5.5",
            client=SimpleNamespace(  # type: ignore[arg-type]
                chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **kwargs: SimpleNamespace(model="gpt-5.5", choices=[])
                    )
                )
            ),
        )

        with pytest.raises(LLMError, match="empty"):
            provider.complete("Describe the book.")
