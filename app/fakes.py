"""Deterministic fake providers for CI and local development.

Selected like any real provider (BOOKSMART_LLM_PROVIDER=fake), so the compose
smoke test can drive the whole pipeline with no API keys, no network, and no
cost. Responses are keyed by the stage's system prompt and shaped exactly as
the stage's parser expects.
"""

import json

from app.extraction import EXTRACTION_SYSTEM_PROMPT
from app.llm import LLMResponse
from app.profile import PROFILE_SYSTEM_PROMPT
from app.summaries import SUMMARY_SYSTEM_PROMPT

FAKE_LLM_MODEL = "fake-llm-1"
FAKE_EMBEDDING_MODEL = "fake-embed-1"
FAKE_EMBEDDING_SIZE = 8

# One well-formed knowledge object per chapter, so the extraction stage's
# parsing and persistence run for real.
FAKE_KNOWLEDGE_OBJECTS = [
    {
        "type": "Principle",
        "title": "Fake determinism",
        "content": "Fake providers return the same output for every call.",
        "summary": "Deterministic canned responses.",
        "confidence": 1.0,
        "section_index": None,
        "page": None,
        "paragraph": None,
    }
]

STAGE_RESPONSES: dict[str, str] = {
    PROFILE_SYSTEM_PROMPT: (
        "A deterministic fake book profile: this book covers the smoke-test "
        "topic end to end."
    ),
    EXTRACTION_SYSTEM_PROMPT: json.dumps(FAKE_KNOWLEDGE_OBJECTS),
    # Missing section summaries are padded with None by the summary parser,
    # so the empty list stays valid for any section count.
    SUMMARY_SYSTEM_PROMPT: json.dumps(
        {"chapter_summary": "A deterministic fake chapter summary.", "section_summaries": []}
    ),
}

DEFAULT_RESPONSE = "A deterministic fake response."


class FakeLLMProvider:
    def __init__(self, model: str = FAKE_LLM_MODEL) -> None:
        self.model = model

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        text = STAGE_RESPONSES.get(system or "", DEFAULT_RESPONSE)
        return LLMResponse(text=text, model=self.model)


class FakeEmbeddingProvider:
    def __init__(self, model: str = FAKE_EMBEDDING_MODEL) -> None:
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Fixed-size vectors derived from text length: deterministic, and
        distinct texts usually get distinct vectors."""
        return [
            [float((len(text) + position) % 7 + 1) for position in range(FAKE_EMBEDDING_SIZE)]
            for text in texts
        ]
