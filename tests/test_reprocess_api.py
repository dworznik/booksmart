"""Integration tests for incremental reprocessing, version stamps, and history.

Each scope re-runs only its stages as a new tracked job, reusing earlier
artifacts where the scope allows; every run stamps versions and the full job
history stays queryable.
"""

import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models import Chapter, KnowledgeObject
from app.profile import PROFILE_SYSTEM_PROMPT
from app.vectors import VectorStore
from app.worker import process_one_job

from .conftest import StubLLMProvider
from .test_embeddings_api import count_book_points, prime_summaries
from .test_ingestion_api import register_book
from .test_knowledge_api import prime_extraction
from .test_profile_api import register_book_with_hints
from .test_structure_api import ingest


def reprocess(
    client: TestClient,
    session_factory: sessionmaker[Session],
    settings: Settings,
    book_id: str,
    scope: str,
) -> dict[str, object]:
    response = client.post(f"/books/{book_id}/reprocess", json={"scope": scope})
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    assert process_one_job(session_factory, settings.storage_root) is True
    job: dict[str, object] = client.get(f"/jobs/{job_id}").json()
    return job


def full_ingest(
    client: TestClient,
    session_factory: sessionmaker[Session],
    settings: Settings,
    stub_llm: StubLLMProvider,
) -> tuple[str, dict[str, object]]:
    """A structured book taken through the whole pipeline once."""
    book_id = register_book_with_hints(client)
    prime_extraction(stub_llm)
    prime_summaries(stub_llm)
    job = ingest(client, session_factory, settings, book_id)
    assert job["status"] == "succeeded"
    return book_id, job


def knowledge_object_ids(session_factory: sessionmaker[Session], book_id: str) -> set[str]:
    with session_factory() as session:
        return {
            str(ko.id)
            for ko in session.scalars(
                select(KnowledgeObject).where(KnowledgeObject.book_id == uuid.UUID(book_id))
            )
        }


def chapter_rows(
    session_factory: sessionmaker[Session], book_id: str
) -> list[tuple[str, str | None, str | None]]:
    """(id, summary, embedding_id) per chapter, in order."""
    with session_factory() as session:
        return [
            (str(c.id), c.summary, str(c.embedding_id) if c.embedding_id else None)
            for c in session.scalars(
                select(Chapter)
                .where(Chapter.book_id == uuid.UUID(book_id))
                .order_by(Chapter.position)
            )
        ]


class TestReprocessEndpoint:
    def test_reprocess_returns_202_with_queued_scoped_job(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, _ = full_ingest(client, session_factory, settings, stub_llm)

        response = client.post(f"/books/{book_id}/reprocess", json={"scope": "profile"})

        assert response.status_code == 202
        job = response.json()
        assert job["book_id"] == book_id
        assert job["scope"] == "profile"
        assert job["status"] == "queued"

    def test_unknown_book_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/books/00000000-0000-0000-0000-000000000000/reprocess",
            json={"scope": "full"},
        )

        assert response.status_code == 404

    def test_invalid_scope_rejected(self, client: TestClient) -> None:
        book_id = register_book(client)

        response = client.post(f"/books/{book_id}/reprocess", json={"scope": "everything"})

        assert response.status_code == 422

    def test_incremental_scope_without_prior_success_rejected(
        self, client: TestClient
    ) -> None:
        book_id = register_book(client)

        response = client.post(f"/books/{book_id}/reprocess", json={"scope": "extraction"})

        assert response.status_code == 409

    def test_full_scope_allowed_without_prior_success(self, client: TestClient) -> None:
        book_id = register_book(client)

        response = client.post(f"/books/{book_id}/reprocess", json={"scope": "full"})

        assert response.status_code == 202
        assert response.json()["scope"] == "full"


class TestProfileScope:
    def test_profile_rerun_supersedes_profile_and_touches_nothing_else(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, _ = full_ingest(client, session_factory, settings, stub_llm)
        chapters_before = chapter_rows(session_factory, book_id)
        objects_before = knowledge_object_ids(session_factory, book_id)
        stub_llm.queue(PROFILE_SYSTEM_PROMPT, "An updated book profile.")

        job = reprocess(client, session_factory, settings, book_id, "profile")

        assert job["status"] == "succeeded"
        assert client.get(f"/books/{book_id}/profile").json()["content"] == (
            "An updated book profile."
        )
        assert chapter_rows(session_factory, book_id) == chapters_before
        assert knowledge_object_ids(session_factory, book_id) == objects_before

    def test_profile_run_stamps_versions(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, _ = full_ingest(client, session_factory, settings, stub_llm)

        job = reprocess(client, session_factory, settings, book_id, "profile")

        assert job["extraction_version"] == "1"
        assert job["model_version"] == "llm=stub-llm-1"
        assert job["prompt_version"] == "profile=1"
        assert job["started_at"] is not None
        assert job["finished_at"] is not None
        # No parsing happened and no artifact was needed.
        assert job["parser_used"] is None
        assert job["output_path"] is None


class TestExtractionScope:
    def test_extraction_rerun_replaces_objects_reusing_parsed_markdown(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, first_job = full_ingest(client, session_factory, settings, stub_llm)
        chapters_before = chapter_rows(session_factory, book_id)
        objects_before = knowledge_object_ids(session_factory, book_id)
        assert len(objects_before) == 3
        prime_extraction(stub_llm)

        job = reprocess(client, session_factory, settings, book_id, "extraction")

        assert job["status"] == "succeeded"
        objects_after = knowledge_object_ids(session_factory, book_id)
        assert len(objects_after) == 3  # replaced, not duplicated
        assert objects_after.isdisjoint(objects_before)
        # Upstream artifacts reused: no new parse, same parsed markdown.
        assert job["parser_used"] is None
        assert job["output_path"] == first_job["output_path"]
        # Structure untouched (same chapter rows except refreshed embeddings).
        assert [row[0] for row in chapter_rows(session_factory, book_id)] == [
            row[0] for row in chapters_before
        ]

    def test_extraction_prompts_carry_reused_chapter_text(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, _ = full_ingest(client, session_factory, settings, stub_llm)
        calls_before = len(stub_llm.calls)
        prime_extraction(stub_llm)

        reprocess(client, session_factory, settings, book_id, "extraction")

        extraction_calls = [
            prompt
            for prompt, system in stub_llm.calls[calls_before:]
            if system is not None and "knowledge objects" in system
        ]
        assert len(extraction_calls) == 2
        assert "Body text explaining the idea" in extraction_calls[0]

    def test_extraction_rerun_refreshes_embeddings_for_new_objects(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id, _ = full_ingest(client, session_factory, settings, stub_llm)
        assert count_book_points(vector_store, book_id) == 8
        prime_extraction(stub_llm)

        job = reprocess(client, session_factory, settings, book_id, "extraction")

        assert count_book_points(vector_store, book_id) == 8
        with session_factory() as session:
            objects = list(
                session.scalars(
                    select(KnowledgeObject).where(
                        KnowledgeObject.book_id == uuid.UUID(book_id)
                    )
                )
            )
        assert all(ko.embedding_id is not None for ko in objects)
        assert job["model_version"] == "llm=stub-llm-1,embedding=stub-embed-1"
        assert job["prompt_version"] == "extraction=1"

    def test_extraction_rerun_fails_clearly_when_artifact_missing_on_disk(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, first_job = full_ingest(client, session_factory, settings, stub_llm)
        Path(str(first_job["output_path"])).unlink()

        job = reprocess(client, session_factory, settings, book_id, "extraction")

        assert job["status"] == "failed"
        assert "parsed markdown" in str(job["error"])


class TestEmbeddingsScope:
    def test_embeddings_rerun_replaces_vectors_without_duplicates(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id, _ = full_ingest(client, session_factory, settings, stub_llm)
        rows_before = chapter_rows(session_factory, book_id)
        old_embedding_ids = [row[2] for row in rows_before]

        job = reprocess(client, session_factory, settings, book_id, "embeddings")

        assert job["status"] == "succeeded"
        assert count_book_points(vector_store, book_id) == 8
        rows_after = chapter_rows(session_factory, book_id)
        # Same chapters and summaries, fresh embedding linkage.
        assert [row[0] for row in rows_after] == [row[0] for row in rows_before]
        assert [row[1] for row in rows_after] == [row[1] for row in rows_before]
        assert set(row[2] for row in rows_after).isdisjoint(old_embedding_ids)
        assert job["model_version"] == "embedding=stub-embed-1"
        assert job["prompt_version"] is None


class TestFullScope:
    def test_full_rebuild_reruns_whole_pipeline_from_original(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, first_job = full_ingest(client, session_factory, settings, stub_llm)
        outline_before = client.get(f"/books/{book_id}/structure").json()
        prime_extraction(stub_llm)
        prime_summaries(stub_llm)

        job = reprocess(client, session_factory, settings, book_id, "full")

        assert job["status"] == "succeeded"
        assert job["parser_used"] is not None
        assert job["output_path"] != first_job["output_path"]
        outline_after = client.get(f"/books/{book_id}/structure").json()
        assert [c["title"] for c in outline_after] == [c["title"] for c in outline_before]
        # Structure was replaced, not duplicated.
        assert len(outline_after) == len(outline_before)
        assert job["extraction_version"] == "1"
        assert job["model_version"] == "llm=stub-llm-1,embedding=stub-embed-1"
        assert job["prompt_version"] == "profile=1,extraction=1,summary=1"


class TestTokenUsage:
    """The stub LLM reports 100 in / 10 out per call; the structured test book
    has 2 chapters, so a full ingest makes 5 LLM calls (1 profile + 2
    extraction + 2 summaries)."""

    def test_full_ingest_accumulates_token_totals_on_job(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        _, job = full_ingest(client, session_factory, settings, stub_llm)

        assert job["input_tokens"] == 500
        assert job["output_tokens"] == 50

    def test_profile_scope_counts_its_single_call(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, _ = full_ingest(client, session_factory, settings, stub_llm)

        job = reprocess(client, session_factory, settings, book_id, "profile")

        assert job["input_tokens"] == 100
        assert job["output_tokens"] == 10

    def test_llm_free_scope_reports_no_token_usage(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, _ = full_ingest(client, session_factory, settings, stub_llm)

        job = reprocess(client, session_factory, settings, book_id, "embeddings")

        assert job["input_tokens"] is None
        assert job["output_tokens"] is None

    def test_each_call_logs_its_usage_in_the_job_log(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        _, job = full_ingest(client, session_factory, settings, stub_llm)

        log = (Path(settings.storage_root) / "logs" / f"{job['id']}.log").read_text(
            encoding="utf-8"
        )

        assert log.count("tokens in=100 out=10") == 5


class TestIngestionHistory:
    def test_history_accumulates_across_runs_with_scopes_and_versions(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, _ = full_ingest(client, session_factory, settings, stub_llm)
        stub_llm.queue(PROFILE_SYSTEM_PROMPT, "Profile v2")
        reprocess(client, session_factory, settings, book_id, "profile")
        prime_extraction(stub_llm)
        reprocess(client, session_factory, settings, book_id, "extraction")

        history = client.get(f"/books/{book_id}/jobs").json()

        assert [job["scope"] for job in history] == ["full", "profile", "extraction"]
        assert all(job["status"] == "succeeded" for job in history)
        for job in history:
            assert job["extraction_version"] == "1"
            assert job["created_at"] is not None
            assert job["finished_at"] is not None

    def test_failed_runs_stay_in_history(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, first_job = full_ingest(client, session_factory, settings, stub_llm)
        Path(str(first_job["output_path"])).unlink()
        reprocess(client, session_factory, settings, book_id, "extraction")

        history = client.get(f"/books/{book_id}/jobs").json()

        assert [job["status"] for job in history] == ["succeeded", "failed"]

    def test_history_for_unknown_book_returns_404(self, client: TestClient) -> None:
        response = client.get("/books/00000000-0000-0000-0000-000000000000/jobs")

        assert response.status_code == 404

    def test_plain_ingest_appears_as_full_scope(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, _ = full_ingest(client, session_factory, settings, stub_llm)

        history = client.get(f"/books/{book_id}/jobs").json()

        assert [job["scope"] for job in history] == ["full"]
