"""The Stage contract (ADR 0002), proven without Inngest.

Stages are public synchronous functions that take serializable inputs (a
book_id), resolve their own data, commit their own output, and never touch a
Run row. The load-bearing test drives them the way booksmart-api's Inngest
Runner will: a *fresh session per stage*, with only the book_id carried between
stages, and the Run managed on its own separate session — then shows the result
is identical to the in-repo sequential Runner.
"""

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from booksmart_core.config import Settings
from booksmart_core.errors import (
    BooksmartError,
    ProviderConfigError,
    ProviderResponseError,
    StagePreconditionError,
)
from booksmart_core.models import Chapter, KnowledgeObject, Run
from booksmart_core.parsing import build_default_chain
from booksmart_core.runner import SCOPE_STAGES, execute_run, start_run
from booksmart_core.stages import (
    run_embeddings,
    run_extraction,
    run_parse,
    run_profile,
    run_structure,
    run_summaries,
)
from booksmart_core.storage import BookStorage
from booksmart_core.vectors import VectorStore

from .conftest import StubEmbeddingProvider, StubLLMProvider
from .test_embeddings_api import count_book_points, prime_summaries
from .test_knowledge_api import prime_extraction
from .test_profile_api import register_book_with_hints


def _drive_per_stage_session(
    session_factory: sessionmaker[Session],
    settings: Settings,
    book_id: uuid.UUID,
    stub_llm: StubLLMProvider,
    stub_embedder: StubEmbeddingProvider,
    vector_store: VectorStore,
) -> list[object]:
    """Run the full pipeline as an Inngest-shaped Runner would: one fresh
    session per stage, only the serializable book_id crossing the boundary."""
    storage = BookStorage(settings.storage_root)
    chain = build_default_chain()
    reports: list[object] = []
    with session_factory() as session:
        reports.append(run_parse(session, book_id, chain=chain, storage=storage))
    with session_factory() as session:
        reports.append(run_structure(session, book_id, storage=storage))
    with session_factory() as session:
        reports.append(run_profile(session, book_id, llm=stub_llm))
    with session_factory() as session:
        reports.append(run_extraction(session, book_id, llm=stub_llm, storage=storage))
    with session_factory() as session:
        reports.append(run_summaries(session, book_id, llm=stub_llm, storage=storage))
    with session_factory() as session:
        reports.append(
            run_embeddings(session, book_id, embedder=stub_embedder, vector_store=vector_store)
        )
    return reports


class TestPerStageSessionContract:
    def test_fresh_session_per_stage_produces_a_complete_ingest(
        self,
        storage: BookStorage,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
        stub_embedder: StubEmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id = uuid.UUID(register_book_with_hints(session_factory, storage))
        prime_extraction(stub_llm)
        prime_summaries(stub_llm)

        # A Run opened on its own session (as an Inngest step would), never
        # passed into a stage.
        with session_factory() as session:
            run = start_run(session, book_id, "full")
            run_id = run.id

        reports = _drive_per_stage_session(
            session_factory, settings, book_id, stub_llm, stub_embedder, vector_store
        )

        with session_factory() as session:
            run = session.get(Run, run_id)
            assert run is not None
            # Stages are Run-blind: none of them advanced the Run's status.
            assert run.status == "running"
            chapters = list(
                session.scalars(select(Chapter).where(Chapter.book_id == book_id))
            )
            objects = list(
                session.scalars(select(KnowledgeObject).where(KnowledgeObject.book_id == book_id))
            )
        assert len(chapters) == 2
        assert len(objects) == 3
        assert count_book_points(vector_store, str(book_id)) == 8
        # 1 profile + 2 extraction + 2 summary calls, 100 in / 10 out each.
        assert sum(r.input_tokens for r in reports) == 500  # type: ignore[attr-defined]
        assert sum(r.output_tokens for r in reports) == 50  # type: ignore[attr-defined]

    def test_per_stage_result_is_identical_to_the_sequential_runner(
        self,
        storage: BookStorage,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
        stub_embedder: StubEmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id = uuid.UUID(register_book_with_hints(session_factory, storage))
        prime_extraction(stub_llm)
        prime_summaries(stub_llm)

        _drive_per_stage_session(
            session_factory, settings, book_id, stub_llm, stub_embedder, vector_store
        )

        def snapshot() -> tuple[list[str], list[str], int]:
            with session_factory() as session:
                chapter_titles = [
                    c.title
                    for c in session.scalars(
                        select(Chapter)
                        .where(Chapter.book_id == book_id)
                        .order_by(Chapter.position)
                    )
                ]
                object_titles = sorted(
                    ko.title
                    for ko in session.scalars(
                        select(KnowledgeObject).where(KnowledgeObject.book_id == book_id)
                    )
                )
            return chapter_titles, object_titles, count_book_points(vector_store, str(book_id))

        per_stage = snapshot()

        # Now run the very same scope through the in-repo sequential Runner
        # (single session, foreground loop) and confirm it converges on the
        # same state — every stage replaces its output wholesale.
        prime_extraction(stub_llm)
        prime_summaries(stub_llm)
        run_id = execute_run(
            session_factory,
            settings.storage_root,
            book_id,
            "full",
            chain=build_default_chain(),
            llm=stub_llm,
            embedder=stub_embedder,
            vector_store=vector_store,
        )
        with session_factory() as session:
            assert session.get(Run, run_id).status == "succeeded"  # type: ignore[union-attr]

        assert snapshot() == per_stage


class TestStagesAreRunBlind:
    def test_stages_create_no_run_rows(
        self,
        storage: BookStorage,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
        stub_embedder: StubEmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id = uuid.UUID(register_book_with_hints(session_factory, storage))
        prime_extraction(stub_llm)
        prime_summaries(stub_llm)

        _drive_per_stage_session(
            session_factory, settings, book_id, stub_llm, stub_embedder, vector_store
        )

        with session_factory() as session:
            runs = list(session.scalars(select(Run).where(Run.book_id == book_id)))
        assert runs == []


class TestStagePreconditions:
    def test_stage_on_missing_book_raises_non_retriable_precondition(
        self, session_factory: sessionmaker[Session], storage: BookStorage
    ) -> None:
        with session_factory() as session:
            with pytest.raises(StagePreconditionError) as excinfo:
                run_structure(session, uuid.uuid4(), storage=storage)
        assert excinfo.value.retriable is False

    def test_markdown_stage_before_parse_raises_precondition(
        self,
        storage: BookStorage,
        session_factory: sessionmaker[Session],
        stub_llm: StubLLMProvider,
    ) -> None:
        # Registered but never parsed: Book.parsed_path is NULL.
        book_id = uuid.UUID(register_book_with_hints(session_factory, storage))
        with session_factory() as session:
            with pytest.raises(StagePreconditionError, match="parsed markdown"):
                run_extraction(session, book_id, llm=stub_llm, storage=storage)


class TestErrorTaxonomy:
    def test_retriable_flags_are_class_level(self) -> None:
        assert StagePreconditionError.retriable is False
        assert ProviderConfigError.retriable is False
        assert ProviderResponseError.retriable is True
        # The narrower model-response errors inherit the retriable classification.
        from booksmart_core.extraction import ExtractionError
        from booksmart_core.llm import LLMError
        from booksmart_core.summaries import SummaryError

        assert issubclass(LLMError, ProviderResponseError)
        assert issubclass(ExtractionError, ProviderResponseError)
        assert issubclass(SummaryError, ProviderResponseError)
        assert LLMError.retriable is True

    def test_config_error_is_still_a_value_error(self) -> None:
        assert issubclass(ProviderConfigError, ValueError)
        assert issubclass(ProviderConfigError, BooksmartError)

    def test_unparseable_response_after_retry_surfaces_as_retriable(
        self,
        storage: BookStorage,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = uuid.UUID(register_book_with_hints(session_factory, storage))
        chain = build_default_chain()
        with session_factory() as session:
            run_parse(session, book_id, chain=chain, storage=storage)
        with session_factory() as session:
            run_structure(session, book_id, storage=storage)

        # The model returns non-JSON on both the call and its in-stage retry.
        from booksmart_core.extraction import EXTRACTION_SYSTEM_PROMPT

        stub_llm.queue(EXTRACTION_SYSTEM_PROMPT, "not json", "still not json")
        with session_factory() as session:
            with pytest.raises(ProviderResponseError) as excinfo:
                run_extraction(session, book_id, llm=stub_llm, storage=storage)
        assert excinfo.value.retriable is True


def test_scope_stages_cover_every_scope() -> None:
    """Every reprocess scope the API accepts maps to a concrete stage list."""
    assert set(SCOPE_STAGES) == {"full", "profile", "extraction", "embeddings"}
