"""Unit tests for the configurable LLM provider layer."""

from types import SimpleNamespace
from typing import Any

import pytest

from app.config import Settings
from app.llm import AnthropicProvider, OpenAIProvider, build_llm_provider


def make_settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "anthropic_api_key": "sk-ant-test",
        "openai_api_key": "sk-openai-test",
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

    def test_configured_model_overrides_provider_default(self) -> None:
        provider = build_llm_provider(
            make_settings(llm_provider="anthropic", llm_model="claude-sonnet-5")
        )

        assert provider.model == "claude-sonnet-5"

    def test_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="hal9000"):
            build_llm_provider(make_settings(llm_provider="hal9000"))


class TestAnthropicProvider:
    def test_complete_sends_prompt_and_joins_text_blocks(self) -> None:
        provider = AnthropicProvider(model="claude-opus-4-8", api_key="sk-ant-test")
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
            )

        provider._client = SimpleNamespace(  # type: ignore[assignment]
            messages=SimpleNamespace(create=fake_create)
        )

        response = provider.complete("Describe the book.", system="You are a librarian.")

        assert response.text == "Part one. Part two."
        assert response.model == "claude-opus-4-8"
        assert captured["model"] == "claude-opus-4-8"
        assert captured["system"] == "You are a librarian."
        assert captured["messages"] == [{"role": "user", "content": "Describe the book."}]

    def test_refusal_stop_reason_raises(self) -> None:
        provider = AnthropicProvider(model="claude-opus-4-8", api_key="sk-ant-test")
        provider._client = SimpleNamespace(  # type: ignore[assignment]
            messages=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    model="claude-opus-4-8", stop_reason="refusal", content=[]
                )
            )
        )

        with pytest.raises(RuntimeError, match="refus"):
            provider.complete("Describe the book.")


class TestOpenAIProvider:
    def test_complete_sends_prompt_and_reads_first_choice(self) -> None:
        provider = OpenAIProvider(model="gpt-5.5", api_key="sk-openai-test")
        captured: dict[str, Any] = {}

        def fake_create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                model="gpt-5.5",
                choices=[SimpleNamespace(message=SimpleNamespace(content="The profile."))],
            )

        provider._client = SimpleNamespace(  # type: ignore[assignment]
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        )

        response = provider.complete("Describe the book.", system="You are a librarian.")

        assert response.text == "The profile."
        assert response.model == "gpt-5.5"
        assert captured["model"] == "gpt-5.5"
        assert captured["messages"] == [
            {"role": "system", "content": "You are a librarian."},
            {"role": "user", "content": "Describe the book."},
        ]

    def test_empty_completion_raises(self) -> None:
        provider = OpenAIProvider(model="gpt-5.5", api_key="sk-openai-test")
        provider._client = SimpleNamespace(  # type: ignore[assignment]
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(model="gpt-5.5", choices=[])
                )
            )
        )

        with pytest.raises(RuntimeError, match="empty"):
            provider.complete("Describe the book.")
