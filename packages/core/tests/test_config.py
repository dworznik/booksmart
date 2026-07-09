"""Settings is explicit: a plain model that never reads the environment."""

import pytest

from booksmart_core.config import Settings


def test_settings_ignores_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pre-0.2 Settings was a BaseSettings with a BOOKSMART_ prefix; now the
    # caller resolves env (the CLI does, a server would) and core does not.
    monkeypatch.setenv("BOOKSMART_LLM_PROVIDER", "openai")
    monkeypatch.setenv("BOOKSMART_ANTHROPIC_API_KEY", "sk-ambient")

    settings = Settings()

    assert settings.llm_provider == "anthropic"
    assert settings.anthropic_api_key is None
