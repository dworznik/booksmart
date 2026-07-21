"""Integration tests: the embedding stage populates Qdrant and links embedding_ids."""

import json
import uuid
from pathlib import Path

from qdrant_client import models as qmodels
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from booksmart_core.config import Settings
from booksmart_core.llm import EmbeddingProvider, EmbeddingResponse
from booksmart_core.models import Chapter, KnowledgeObject, Section
from booksmart_core.runner import execute_run
from booksmart_core.stages import StageReport, run_embeddings
from booksmart_core.storage import BookStorage
from booksmart_core.summaries import SUMMARY_SYSTEM_PROMPT
from booksmart_core.vectors import COLLECTION_NAME, VectorStore

from .conftest import StubEmbeddingProvider, StubLLMProvider, get_run
from .test_ingestion_api import register_book
from .test_knowledge_api import prime_extraction
from .test_profile_api import register_book_with_hints
from .test_structure_api import ingest

CHAPTER_SUMMARIES = [
    {
        "chapter_summary": "Modules should be deep.",
        "section_summaries": ["About deep modules.", "About shallow modules."],
    },
    {
        "chapter_summary": "Symptoms of complexity.",
        "section_summaries": ["How complexity shows up."],
    },
]


def prime_summaries(stub: StubLLMProvider) -> None:
    stub.queue(SUMMARY_SYSTEM_PROMPT, *(json.dumps(payload) for payload in CHAPTER_SUMMARIES))


def count_book_points(store: VectorStore, book_id: str) -> int:
    if not store.client.collection_exists(COLLECTION_NAME):
        return 0  # nothing was ever embedded
    return store.client.count(
        COLLECTION_NAME,
        count_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key="book_id", match=qmodels.MatchValue(value=book_id))]
        ),
    ).count


class ExplodingEmbedder:
    model = "exploding-embed"
    max_batch = 100

    def embed(self, texts: list[str]) -> EmbeddingResponse:
        raise RuntimeError("embedding service down")


class SilentEmbedder:
    """Reports no usage at all. Same model and vector shape as the stub, so it
    can re-embed a collection the stub locked (ADR 0001)."""

    model = "stub-embed-1"
    max_batch = 100

    def embed(self, texts: list[str]) -> EmbeddingResponse:
        return EmbeddingResponse(vectors=[[1.0, 1.0, 0.5] for _ in texts])


def embedded_book(
    session_factory: sessionmaker[Session],
    settings: Settings,
    storage: BookStorage,
    stub_llm: StubLLMProvider,
) -> uuid.UUID:
    """A fully ingested book, ready for the embeddings stage to be re-run."""
    book_id = register_book_with_hints(session_factory, storage)
    prime_extraction(stub_llm)
    prime_summaries(stub_llm)
    ingest(session_factory, settings, book_id)
    return uuid.UUID(book_id)


def embed_again(
    session_factory: sessionmaker[Session],
    book_id: uuid.UUID,
    embedder: EmbeddingProvider,
    vector_store: VectorStore,
) -> StageReport:
    """Re-run the embeddings stage alone, as a Runner would, for its report."""
    with session_factory() as session:
        return run_embeddings(
            session, book_id, embedder=embedder, vector_store=vector_store
        )


class TestEmbeddingStage:
    def test_ingestion_embeds_summaries_and_objects_and_links_ids(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)
        prime_extraction(stub_llm)
        prime_summaries(stub_llm)

        run = ingest(session_factory, settings, book_id)

        assert run["status"] == "succeeded"
        with session_factory() as session:
            chapters = list(
                session.scalars(
                    select(Chapter)
                    .where(Chapter.book_id == uuid.UUID(book_id))
                    .order_by(Chapter.position)
                )
            )
            assert [c.summary for c in chapters] == [
                "Modules should be deep.",
                "Symptoms of complexity.",
            ]
            sections = [s for c in chapters for s in c.sections]
            assert [s.summary for s in sections] == [
                "About deep modules.",
                "About shallow modules.",
                "How complexity shows up.",
            ]
            for summarized in [*chapters, *sections]:
                assert summarized.summary_model == "stub-llm-1"
                assert summarized.summary_prompt_version == "1"
            objects = list(
                session.scalars(
                    select(KnowledgeObject).where(KnowledgeObject.book_id == uuid.UUID(book_id))
                )
            )
            embedded = [*chapters, *sections, *objects]
            assert len(embedded) == 8
            for record in embedded:
                assert record.embedding_id is not None
                assert record.embedding_model == "stub-embed-1"
                assert record.embedded_at is not None

            assert count_book_points(vector_store, book_id) == 8
            chapter = chapters[0]
            point = vector_store.client.retrieve(
                COLLECTION_NAME, ids=[str(chapter.embedding_id)], with_payload=True
            )[0]
            assert point.payload is not None
            assert point.payload["record_type"] == "chapter"
            assert point.payload["record_id"] == str(chapter.id)
            assert point.payload["book_id"] == book_id

    def test_reingest_replaces_book_vectors(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)
        prime_extraction(stub_llm)
        prime_summaries(stub_llm)
        ingest(session_factory, settings, book_id)
        with session_factory() as session:
            old_ids = [
                str(embedding_id)
                for embedding_id in session.scalars(
                    select(Chapter.embedding_id).where(Chapter.book_id == uuid.UUID(book_id))
                )
            ]

        prime_extraction(stub_llm)
        prime_summaries(stub_llm)
        ingest(session_factory, settings, book_id)

        assert count_book_points(vector_store, book_id) == 8
        assert vector_store.client.retrieve(COLLECTION_NAME, ids=old_ids) == []

    def test_summary_prompts_carry_sliced_chapter_text(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)
        prime_summaries(stub_llm)

        ingest(session_factory, settings, book_id)

        summary_calls = [p for p, s in stub_llm.calls if s == SUMMARY_SYSTEM_PROMPT]
        assert len(summary_calls) == 2
        assert "Chapter One: Modules" in summary_calls[0]
        assert "Body text explaining the idea" in summary_calls[0]
        assert "Chapter One: Modules" not in summary_calls[1]

    def test_transient_invalid_summary_response_is_retried(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)
        prime_extraction(stub_llm)
        # Chapter 1 answers garbage once, then its real summary on the retry.
        stub_llm.queue(
            SUMMARY_SYSTEM_PROMPT,
            "not a summary payload",
            *(json.dumps(payload) for payload in CHAPTER_SUMMARIES),
        )

        run = ingest(session_factory, settings, book_id)

        assert run["status"] == "succeeded"
        with session_factory() as session:
            chapters = list(
                session.scalars(
                    select(Chapter)
                    .where(Chapter.book_id == uuid.UUID(book_id))
                    .order_by(Chapter.position)
                )
            )
            assert [c.summary for c in chapters] == [
                "Modules should be deep.",
                "Symptoms of complexity.",
            ]

    def test_records_without_summaries_are_not_embedded(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
        vector_store: VectorStore,
    ) -> None:
        # The default stubbed summary response carries no section summaries.
        book_id = register_book_with_hints(session_factory, storage)
        prime_extraction(stub_llm)

        run = ingest(session_factory, settings, book_id)

        assert run["status"] == "succeeded"
        with session_factory() as session:
            sections = list(
                session.scalars(
                    select(Section)
                    .join(Chapter, Section.chapter_id == Chapter.id)
                    .where(Chapter.book_id == uuid.UUID(book_id))
                )
            )
            assert all(s.summary is None for s in sections)
            assert all(s.embedding_id is None for s in sections)
        # 2 chapter summaries + 3 knowledge objects, no sections
        assert count_book_points(vector_store, book_id) == 5

    def test_book_with_nothing_to_embed_succeeds(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        vector_store: VectorStore,
    ) -> None:
        book_id = register_book(session_factory, storage)  # unstructured: no chapters/objects

        run = ingest(session_factory, settings, book_id)

        assert run["status"] == "succeeded"
        assert count_book_points(vector_store, book_id) == 0
        log = (Path(settings.storage_root) / "logs" / f"{run['id']}.log").read_text(
            encoding="utf-8"
        )
        assert "nothing to embed" in log

    def test_embedding_failure_fails_run(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)
        prime_extraction(stub_llm)
        prime_summaries(stub_llm)

        run_id = execute_run(
            session_factory,
            settings.storage_root,
            uuid.UUID(book_id),
            "full",
            embedder=ExplodingEmbedder(),
        )

        run = get_run(session_factory, str(run_id))
        assert run is not None
        assert run["status"] == "failed"
        assert "embedding" in str(run["error"])

    def test_embedding_batches_respect_the_providers_max_batch_limit(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
        stub_embedder: StubEmbeddingProvider,
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)
        prime_extraction(stub_llm)
        prime_summaries(stub_llm)
        stub_embedder.max_batch = 3

        run = ingest(session_factory, settings, book_id)

        assert run["status"] == "succeeded"
        # 8 embeddable records split into provider-limit-sized batches.
        assert [len(batch) for batch in stub_embedder.batches] == [3, 3, 2]

    def test_reingest_with_a_different_embedding_model_fails_actionably(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        # ADR 0001: the collection is locked to the model that created it.
        book_id = register_book_with_hints(session_factory, storage)
        prime_extraction(stub_llm)
        prime_summaries(stub_llm)
        ingest(session_factory, settings, book_id)

        prime_extraction(stub_llm)
        prime_summaries(stub_llm)
        other_embedder = StubEmbeddingProvider()
        other_embedder.model = "stub-embed-2"

        run_id = execute_run(
            session_factory,
            settings.storage_root,
            uuid.UUID(book_id),
            "full",
            embedder=other_embedder,
        )

        run = get_run(session_factory, str(run_id))
        assert run is not None
        assert run["status"] == "failed"
        assert "stub-embed-1" in str(run["error"])
        assert "stub-embed-2" in str(run["error"])

    def test_embedded_texts_include_knowledge_object_content(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
        stub_embedder: StubEmbeddingProvider,
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)
        prime_extraction(stub_llm)
        prime_summaries(stub_llm)

        ingest(session_factory, settings, book_id)

        assert len(stub_embedder.batches) == 1  # one batched call per run
        batch = stub_embedder.batches[0]
        assert "Modules should be deep." in batch  # chapter summary
        assert "About deep modules." in batch  # section summary
        assert any("Deep modules" in text for text in batch)  # knowledge object


class TestEmbeddingUsageReporting:
    """The provider reports what the embeddings endpoint billed; the stage
    reports it onward, separately from LLM tokens — the two are priced apart."""

    def test_report_sums_the_embedding_tokens_across_batches(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
        stub_embedder: StubEmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id = embedded_book(session_factory, settings, storage, stub_llm)
        stub_embedder.max_batch = 3  # the total must survive batching

        report = embed_again(session_factory, book_id, stub_embedder, vector_store)

        # 8 embeddable records, each reporting a fixed per-text usage.
        assert report.embedding_tokens == 8 * StubEmbeddingProvider.INPUT_TOKENS_PER_TEXT

    def test_embedding_tokens_are_not_counted_as_llm_tokens(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
        stub_embedder: StubEmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id = embedded_book(session_factory, settings, storage, stub_llm)

        report = embed_again(session_factory, book_id, stub_embedder, vector_store)

        assert report.embedding_tokens > 0
        assert report.input_tokens == 0
        assert report.output_tokens == 0

    def test_unreported_usage_counts_as_zero_and_logs_unknown(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id = embedded_book(session_factory, settings, storage, stub_llm)

        report = embed_again(session_factory, book_id, SilentEmbedder(), vector_store)

        assert report.embedding_tokens == 0
        assert any("tokens in=?" in line for line in report.log_lines)
