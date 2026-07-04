"""Deterministic fake providers for CI and local development.

Selected like any real provider (BOOKSMART_LLM_PROVIDER=fake), so the compose
smoke test can drive the whole pipeline with no API keys, no network, and no
cost. Responses are keyed by the stage's system prompt and shaped exactly as
the stage's parser expects.
"""

from app.llm import LLMResponse
from app.profile import PROFILE_SYSTEM_PROMPT

FAKE_LLM_MODEL = "fake-llm-1"

STAGE_RESPONSES: dict[str, str] = {
    PROFILE_SYSTEM_PROMPT: (
        "A deterministic fake book profile: this book covers the smoke-test "
        "topic end to end."
    ),
}

DEFAULT_RESPONSE = "A deterministic fake response."


class FakeLLMProvider:
    def __init__(self, model: str = FAKE_LLM_MODEL) -> None:
        self.model = model

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        text = STAGE_RESPONSES.get(system or "", DEFAULT_RESPONSE)
        return LLMResponse(text=text, model=self.model)
