"""Deterministic fake providers for CI and local development.

Selected like any real provider (BOOKSMART_LLM_PROVIDER=fake), so the compose
smoke test can drive the whole pipeline with no API keys, no network, and no
cost. Responses are keyed by the stage's system prompt and shaped exactly as
the stage's parser expects.
"""

import json
import re
import zlib

from booksmart_core.extraction import EXTRACTION_SYSTEM_PROMPT
from booksmart_core.llm import (
    EmbeddingLimits,
    EmbeddingResponse,
    LLMLimits,
    LLMResponse,
    resolve_limits,
)
from booksmart_core.profile import PROFILE_SYSTEM_PROMPT
from booksmart_core.sparse import SparseVector
from booksmart_core.summaries import SUMMARY_SYSTEM_PROMPT

FAKE_LLM_MODEL = "fake-llm-1"
FAKE_EMBEDDING_MODEL = "fake-embed-1"
FAKE_EMBEDDING_SIZE = 8
FAKE_SPARSE_MODEL = "fake-sparse-1"

# Fakes carry Limits like any real provider so pipeline code exercised against
# them (batching, budgets) behaves exactly as it will in production.
_FAKE_LLM_LIMITS = {
    FAKE_LLM_MODEL: LLMLimits(max_output_tokens=32000),
}
_FAKE_LLM_DEFAULT = LLMLimits(max_output_tokens=32000)

_FAKE_EMBEDDING_LIMITS = {
    FAKE_EMBEDDING_MODEL: EmbeddingLimits(
        max_batch=100, embedding_dimensions=FAKE_EMBEDDING_SIZE
    ),
}
_FAKE_EMBEDDING_DEFAULT = EmbeddingLimits(max_batch=100)

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
        limits = resolve_limits("fake", model, _FAKE_LLM_LIMITS, _FAKE_LLM_DEFAULT)
        self.max_output_tokens = limits.max_output_tokens

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        text = STAGE_RESPONSES.get(system or "", DEFAULT_RESPONSE)
        return LLMResponse(text=text, model=self.model, input_tokens=0, output_tokens=0)


class FakeEmbeddingProvider:
    def __init__(self, model: str = FAKE_EMBEDDING_MODEL) -> None:
        self.model = model
        limits = resolve_limits("fake", model, _FAKE_EMBEDDING_LIMITS, _FAKE_EMBEDDING_DEFAULT)
        self.max_batch = limits.max_batch
        self.embedding_dimensions = limits.embedding_dimensions

    def embed(self, texts: list[str]) -> EmbeddingResponse:
        """Fixed-size vectors derived from text length: deterministic, and
        distinct texts usually get distinct vectors. Usage is a truthful zero —
        nothing was billed — matching FakeLLMProvider."""
        return EmbeddingResponse(
            vectors=[
                [float((len(text) + position) % 7 + 1) for position in range(FAKE_EMBEDDING_SIZE)]
                for text in texts
            ],
            input_tokens=0,
        )


_WORD = re.compile(r"\w+")


class FakeSparseEmbeddingProvider:
    """Hashed word counts standing in for BM25: no download, no dependency.

    Real enough for the properties the pipeline depends on — the same word hashes
    to the same term id everywhere, so lexical overlap between a query and a
    document actually retrieves — while staying a pure function of the text, so
    CI never reaches for a model. The weighting is a plain term count rather than
    BM25 saturation; anything asserting on *ranking quality* wants the real
    provider (see the eval in docs/research/), not this.
    """

    def __init__(self, model: str = FAKE_SPARSE_MODEL) -> None:
        self.model = model
        # Shaped like a real recipe (see sparse.py) so lock handling is exercised
        # against the same format in tests as in production.
        self.recipe = f"{model}(k=0,b=0,avg_len=0,language=none)"

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._weighted(text) for text in texts]

    def embed_query(self, text: str) -> SparseVector:
        """Flat weights, like the real provider: the IDF is Qdrant's job."""
        indices = sorted({_term_id(word) for word in _WORD.findall(text.lower())})
        return SparseVector(indices=indices, values=[1.0] * len(indices))

    def _weighted(self, text: str) -> SparseVector:
        counts: dict[int, float] = {}
        for word in _WORD.findall(text.lower()):
            counts[_term_id(word)] = counts.get(_term_id(word), 0.0) + 1.0
        indices = sorted(counts)
        return SparseVector(indices=indices, values=[counts[index] for index in indices])


def _term_id(word: str) -> int:
    """A stable non-negative term id. crc32 (not hash()) because Python salts
    string hashing per process, and these ids are written into stored vectors."""
    return zlib.crc32(word.encode())
