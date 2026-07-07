"""Integration tests for incremental reprocessing, version stamps, and history.

Each scope re-runs only its stages as a new tracked Run, reusing earlier
artifacts where the scope allows; every run stamps versions and the full run
history stays queryable.
"""

import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from booksmart_core.config import Settings
from booksmart_core.models import Chapter, KnowledgeObject
from booksmart_core.profile import PROFILE_SYSTEM_PROMPT
from booksmart_core.storage import BookStorage
from booksmart_core.vectors import VectorStore

from .conftest import (
    StubLLMProvider,
    book_structure,
    latest_profile,
    run_scope,
    runs_for_book,
)
from .test_embeddings_api import count_book_points, prime_summaries
from .test_ingestion_api import register_book
from .test_knowledge_api import prime_extraction
from .test_profile_api import register_book_with_hints
from .test_structure_api import ingest


def reprocess(
    session_factory: sessionmaker[Session],
    settings: Settings,
    book_id: str,
    scope: str,
) -> dict[str, object]:
    """Run one scope to completion and return its Run. (The prior-success and
    scope-validation guards lived in the HTTP endpoint, now documented for
    consumers; here an incremental scope without upstream output fails at the
    stage that finds nothing to build on.)"""
    return run_scope(session_factory, settings, book_id, scope)


def full_ingest(
    session_factory: sessionmaker[Session],
    settings: Settings,
    storage: BookStorage,
    stub_llm: StubLLMProvider,
) -> tuple[str, dict[str, object]]:
    """A structured book taken through the whole pipeline once."""
    book_id = register_book_with_hints(session_factory, storage)
    prime_extraction(stub_llm)
    prime_summaries(stub_llm)
    run = ingest(session_factory, settings, book_id)
    assert run["status"] == "succeeded"
    return book_id, run


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


class TestScopeGuards:
    def test_reprocess_runs_scoped_run_synchronously(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, _ = full_ingest(session_factory, settings, storage, stub_llm)

        run = reprocess(session_factory, settings, book_id, "profile")

        assert run["book_id"] == book_id
        assert run["scope"] == "profile"
        assert run["status"] == "succeeded"

    def test_unknown_scope_fails_the_run(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage)

        run = reprocess(session_factory, settings, book_id, "everything")

        assert run["status"] == "failed"
        assert "scope" in str(run["error"])

    def test_incremental_scope_without_upstream_output_fails(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        # No prior parse, so extraction finds no parsed markdown to build on.
        book_id = register_book(session_factory, storage)

        run = reprocess(session_factory, settings, book_id, "extraction")

        assert run["status"] == "failed"
        assert "parsed markdown" in str(run["error"])

    def test_full_scope_allowed_without_prior_success(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage)

        run = reprocess(session_factory, settings, book_id, "full")

        assert run["scope"] == "full"
        assert run["status"] == "succeeded"


class TestProfileScope:
    def test_profile_rerun_supersedes_profile_and_touches_nothing_else(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, _ = full_ingest(session_factory, settings, storage, stub_llm)
        chapters_before = chapter_rows(session_factory, book_id)
        objects_before = knowledge_object_ids(session_factory, book_id)
        stub_llm.queue(PROFILE_SYSTEM_PROMPT, "An updated book profile.")

        run = reprocess(session_factory, settings, book_id, "profile")

        assert run["status"] == "succeeded"
        profile = latest_profile(session_factory, book_id)
        assert profile is not None
        assert profile["content"] == "An updated book profile."
        assert chapter_rows(session_factory, book_id) == chapters_before
        assert knowledge_object_ids(session_factory, book_id) == objects_before

    def test_profile_run_stamps_versions(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, _ = full_ingest(session_factory, settings, storage, stub_llm)

        run = reprocess(session_factory, settings, book_id, "profile")

        assert run["extraction_version"] == "1"
        assert run["model_version"] == "llm=stub-llm-1"
        assert run["prompt_version"] == "profile=1"
        assert run["finished_at"] is not None
        # No parsing happened and no artifact was needed.
        assert run["parser_used"] is None
        assert run["output_path"] is None


class TestExtractionScope:
    def test_extraction_rerun_replaces_objects_reusing_parsed_markdown(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, _ = full_ingest(session_factory, settings, storage, stub_llm)
        chapters_before = chapter_rows(session_factory, book_id)
        objects_before = knowledge_object_ids(session_factory, book_id)
        assert len(objects_before) == 3
        prime_extraction(stub_llm)

        run = reprocess(session_factory, settings, book_id, "extraction")

        assert run["status"] == "succeeded"
        objects_after = knowledge_object_ids(session_factory, book_id)
        assert len(objects_after) == 3  # replaced, not duplicated
        assert objects_after.isdisjoint(objects_before)
        # Upstream artifacts reused: this run did not parse, so it records no
        # artifact of its own — extraction read the book's existing parsed_path.
        assert run["parser_used"] is None
        assert run["output_path"] is None
        # Structure untouched (same chapter rows except refreshed embeddings).
        assert [row[0] for row in chapter_rows(session_factory, book_id)] == [
            row[0] for row in chapters_before
        ]

    def test_extraction_prompts_carry_reused_chapter_text(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, _ = full_ingest(session_factory, settings, storage, stub_llm)
        calls_before = len(stub_llm.calls)
        prime_extraction(stub_llm)

        reprocess(session_factory, settings, book_id, "extraction")

        extraction_calls = [
            prompt
            for prompt, system in stub_llm.calls[calls_before:]
            if system is not None and "knowledge objects" in system
        ]
        assert len(extraction_calls) == 2
        assert "Body text explaining the idea" in extraction_calls[0]

    def test_extraction_rerun_refreshes_embeddings_for_new_objects(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id, _ = full_ingest(session_factory, settings, storage, stub_llm)
        assert count_book_points(vector_store, book_id) == 8
        prime_extraction(stub_llm)

        run = reprocess(session_factory, settings, book_id, "extraction")

        assert count_book_points(vector_store, book_id) == 8
        with session_factory() as session:
            objects = list(
                session.scalars(
                    select(KnowledgeObject).where(KnowledgeObject.book_id == uuid.UUID(book_id))
                )
            )
        assert all(ko.embedding_id is not None for ko in objects)
        assert run["model_version"] == "llm=stub-llm-1,embedding=stub-embed-1"
        assert run["prompt_version"] == "extraction=2"

    def test_extraction_rerun_fails_clearly_when_artifact_missing_on_disk(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, first_run = full_ingest(session_factory, settings, storage, stub_llm)
        storage.resolve(str(first_run["output_path"])).unlink()

        run = reprocess(session_factory, settings, book_id, "extraction")

        assert run["status"] == "failed"
        assert "parsed markdown" in str(run["error"])


class TestEmbeddingsScope:
    def test_embeddings_rerun_replaces_vectors_without_duplicates(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id, _ = full_ingest(session_factory, settings, storage, stub_llm)
        rows_before = chapter_rows(session_factory, book_id)
        old_embedding_ids = [row[2] for row in rows_before]

        run = reprocess(session_factory, settings, book_id, "embeddings")

        assert run["status"] == "succeeded"
        assert count_book_points(vector_store, book_id) == 8
        rows_after = chapter_rows(session_factory, book_id)
        # Same chapters and summaries, fresh embedding linkage.
        assert [row[0] for row in rows_after] == [row[0] for row in rows_before]
        assert [row[1] for row in rows_after] == [row[1] for row in rows_before]
        assert set(row[2] for row in rows_after).isdisjoint(old_embedding_ids)
        assert run["model_version"] == "embedding=stub-embed-1"
        assert run["prompt_version"] is None


class TestFullScope:
    def test_full_rebuild_reruns_whole_pipeline_from_original(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, first_run = full_ingest(session_factory, settings, storage, stub_llm)
        outline_before = book_structure(session_factory, book_id)
        prime_extraction(stub_llm)
        prime_summaries(stub_llm)

        run = reprocess(session_factory, settings, book_id, "full")

        assert run["status"] == "succeeded"
        assert run["parser_used"] is not None
        # One parsed artifact per book (replaced wholesale), so a full rebuild
        # reuses the same stable path rather than minting a per-run file.
        assert run["output_path"] == first_run["output_path"]
        outline_after = book_structure(session_factory, book_id)
        assert [c["title"] for c in outline_after] == [c["title"] for c in outline_before]
        # Structure was replaced, not duplicated.
        assert len(outline_after) == len(outline_before)
        assert run["extraction_version"] == "1"
        assert run["model_version"] == "llm=stub-llm-1,embedding=stub-embed-1"
        assert run["prompt_version"] == "profile=1,extraction=2,summary=1"


class TestTokenUsage:
    """The stub LLM reports 100 in / 10 out per call; the structured test book
    has 2 chapters, so a full ingest makes 5 LLM calls (1 profile + 2
    extraction + 2 summaries)."""

    def test_full_ingest_accumulates_token_totals_on_run(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        _, run = full_ingest(session_factory, settings, storage, stub_llm)

        assert run["input_tokens"] == 500
        assert run["output_tokens"] == 50

    def test_profile_scope_counts_its_single_call(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, _ = full_ingest(session_factory, settings, storage, stub_llm)

        run = reprocess(session_factory, settings, book_id, "profile")

        assert run["input_tokens"] == 100
        assert run["output_tokens"] == 10

    def test_llm_free_scope_reports_no_token_usage(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, _ = full_ingest(session_factory, settings, storage, stub_llm)

        run = reprocess(session_factory, settings, book_id, "embeddings")

        assert run["input_tokens"] is None
        assert run["output_tokens"] is None

    def test_each_call_logs_its_usage_in_the_run_log(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        _, run = full_ingest(session_factory, settings, storage, stub_llm)

        log = (Path(settings.storage_root) / "logs" / f"{run['id']}.log").read_text(
            encoding="utf-8"
        )

        assert log.count("tokens in=100 out=10") == 5


class TestRunHistory:
    def test_history_accumulates_across_runs_with_scopes_and_versions(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, _ = full_ingest(session_factory, settings, storage, stub_llm)
        stub_llm.queue(PROFILE_SYSTEM_PROMPT, "Profile v2")
        reprocess(session_factory, settings, book_id, "profile")
        prime_extraction(stub_llm)
        reprocess(session_factory, settings, book_id, "extraction")

        history = runs_for_book(session_factory, book_id)

        assert [run["scope"] for run in history] == ["full", "profile", "extraction"]
        assert all(run["status"] == "succeeded" for run in history)
        for run in history:
            assert run["extraction_version"] == "1"
            assert run["created_at"] is not None
            assert run["finished_at"] is not None

    def test_failed_runs_stay_in_history(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, first_run = full_ingest(session_factory, settings, storage, stub_llm)
        storage.resolve(str(first_run["output_path"])).unlink()
        reprocess(session_factory, settings, book_id, "extraction")

        history = runs_for_book(session_factory, book_id)

        assert [run["status"] for run in history] == ["succeeded", "failed"]

    def test_plain_ingest_appears_as_full_scope(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id, _ = full_ingest(session_factory, settings, storage, stub_llm)

        history = runs_for_book(session_factory, book_id)

        assert [run["scope"] for run in history] == ["full"]
