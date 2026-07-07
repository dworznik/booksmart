"""Integration tests: the embedding stage populates Qdrant and links embedding_ids."""

import json
import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from qdrant_client import models as qmodels
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models import Chapter, KnowledgeObject, Section
from app.summaries import SUMMARY_SYSTEM_PROMPT
from app.vectors import COLLECTION_NAME, VectorStore
from app.worker import process_one_job

from .conftest import StubEmbeddingProvider, StubLLMProvider
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

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding service down")


class TestEmbeddingStage:
    def test_ingestion_embeds_summaries_and_objects_and_links_ids(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id = register_book_with_hints(client)
        prime_extraction(stub_llm)
        prime_summaries(stub_llm)

        job = ingest(client, session_factory, settings, book_id)

        assert job["status"] == "succeeded"
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
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id = register_book_with_hints(client)
        prime_extraction(stub_llm)
        prime_summaries(stub_llm)
        ingest(client, session_factory, settings, book_id)
        with session_factory() as session:
            old_ids = [
                str(embedding_id)
                for embedding_id in session.scalars(
                    select(Chapter.embedding_id).where(Chapter.book_id == uuid.UUID(book_id))
                )
            ]

        prime_extraction(stub_llm)
        prime_summaries(stub_llm)
        ingest(client, session_factory, settings, book_id)

        assert count_book_points(vector_store, book_id) == 8
        assert vector_store.client.retrieve(COLLECTION_NAME, ids=old_ids) == []

    def test_summary_prompts_carry_sliced_chapter_text(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(client)
        prime_summaries(stub_llm)

        ingest(client, session_factory, settings, book_id)

        summary_calls = [p for p, s in stub_llm.calls if s == SUMMARY_SYSTEM_PROMPT]
        assert len(summary_calls) == 2
        assert "Chapter One: Modules" in summary_calls[0]
        assert "Body text explaining the idea" in summary_calls[0]
        assert "Chapter One: Modules" not in summary_calls[1]

    def test_transient_invalid_summary_response_is_retried(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(client)
        prime_extraction(stub_llm)
        # Chapter 1 answers garbage once, then its real summary on the retry.
        stub_llm.queue(
            SUMMARY_SYSTEM_PROMPT,
            "not a summary payload",
            *(json.dumps(payload) for payload in CHAPTER_SUMMARIES),
        )

        job = ingest(client, session_factory, settings, book_id)

        assert job["status"] == "succeeded"
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
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
        vector_store: VectorStore,
    ) -> None:
        # The default stubbed summary response carries no section summaries.
        book_id = register_book_with_hints(client)
        prime_extraction(stub_llm)

        job = ingest(client, session_factory, settings, book_id)

        assert job["status"] == "succeeded"
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
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        vector_store: VectorStore,
    ) -> None:
        book_id = register_book(client)  # unstructured: no chapters, no objects

        job = ingest(client, session_factory, settings, book_id)

        assert job["status"] == "succeeded"
        assert count_book_points(vector_store, book_id) == 0
        log = (Path(settings.storage_root) / "logs" / f"{job['id']}.log").read_text(
            encoding="utf-8"
        )
        assert "nothing to embed" in log

    def test_embedding_failure_fails_job(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(client)
        prime_extraction(stub_llm)
        prime_summaries(stub_llm)
        job_id = client.post(f"/books/{book_id}/ingest").json()["id"]

        assert (
            process_one_job(session_factory, settings.storage_root, embedder=ExplodingEmbedder())
            is True
        )

        job = client.get(f"/jobs/{job_id}").json()
        assert job["status"] == "failed"
        assert "embedding" in str(job["error"])

    def test_embedded_texts_include_knowledge_object_content(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
        stub_embedder: StubEmbeddingProvider,
    ) -> None:
        book_id = register_book_with_hints(client)
        prime_extraction(stub_llm)
        prime_summaries(stub_llm)

        ingest(client, session_factory, settings, book_id)

        assert len(stub_embedder.batches) == 1  # one batched call per run
        batch = stub_embedder.batches[0]
        assert "Modules should be deep." in batch  # chapter summary
        assert "About deep modules." in batch  # section summary
        assert any("Deep modules" in text for text in batch)  # knowledge object
