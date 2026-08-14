"""Where a Run's spend went, and how long each Stage took.

A Run has always recorded what it cost in total. The total is the one number
nobody tuning a pipeline can act on: it cannot say which Stage to move to a
cheaper model, and it cannot say which Stage is the slow one. Embedding tokens
are billed at a different rate again, and were not persisted at all.

Stages stay Run-blind (ADR 0002). What changes is that the Runner keeps the
reports it already receives instead of summing them and throwing them away.
"""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from booksmart_core.config import Settings
from booksmart_core.models import Run, RunStage
from booksmart_core.runner import execute_run
from booksmart_core.storage import BookStorage

from .conftest import StubEmbeddingProvider, StubLLMProvider, store_book
from .test_profile_api import ExplodingLLM
from .test_structure_api import make_structured_pdf_bytes


def stages_of(session_factory: sessionmaker[Session], run_id: uuid.UUID) -> list[RunStage]:
    with session_factory() as session:
        return list(
            session.scalars(
                select(RunStage).where(RunStage.run_id == run_id).order_by(RunStage.position)
            )
        )


def naive(moment: datetime | None) -> datetime | None:
    """A timestamp with its zone dropped, for comparing across both backends.

    `DateTime(timezone=True)` round-trips aware on Postgres and naive on SQLite,
    and this suite runs on both. Everything written here is UTC — `_utcnow` is
    the only clock — so dropping the zone compares like with like rather than
    papering over a conversion.
    """
    return moment.replace(tzinfo=None) if moment is not None else None


def run_row(session_factory: sessionmaker[Session], run_id: uuid.UUID) -> Run:
    with session_factory() as session:
        run = session.get(Run, run_id)
        assert run is not None
        session.expunge(run)
        return run


@pytest.fixture()
def book(session_factory: sessionmaker[Session], storage: BookStorage) -> uuid.UUID:
    return uuid.UUID(
        store_book(
            session_factory,
            storage,
            title="A Book About Craft",
            author="A. N. Author",
            filename="craft.pdf",
            content=make_structured_pdf_bytes(),
        )
    )


class TestEveryStageIsRecorded:
    def test_a_full_run_keeps_one_row_per_stage_in_the_order_they_ran(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        book: uuid.UUID,
    ) -> None:
        run_id = execute_run(session_factory, settings.storage_root, book)

        assert [stage.stage for stage in stages_of(session_factory, run_id)] == [
            "parse", "structure", "profile", "extraction", "summaries", "embeddings"
        ]

    def test_an_incremental_scope_records_only_the_stages_it_ran(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        book: uuid.UUID,
    ) -> None:
        execute_run(session_factory, settings.storage_root, book)

        run_id = execute_run(session_factory, settings.storage_root, book, "embeddings")

        assert [stage.stage for stage in stages_of(session_factory, run_id)] == ["embeddings"]

    def test_a_failed_run_keeps_the_stages_that_did_run(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        book: uuid.UUID,
    ) -> None:
        """A partial run is exactly when "where did it get to, and what did that
        cost" is worth asking. The stages before the failure really did spend
        what they spent, and the failure discards their *output* — the rollback
        — without making their spend untrue.

        The book is a good one and the provider is the thing that breaks, so the
        run reaches its third stage before failing. A fixture that failed at the
        first stage would assert an empty list and prove nothing.
        """
        run_id = execute_run(
            session_factory, settings.storage_root, book, llm=ExplodingLLM()
        )

        run = run_row(session_factory, run_id)
        assert run.status == "failed"
        assert run.error is not None and "profile" in run.error
        assert [stage.stage for stage in stages_of(session_factory, run_id)] == [
            "parse", "structure"
        ]

    def test_a_run_that_fails_at_its_first_stage_keeps_nothing(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
    ) -> None:
        """The other end of the same rule: no stage completed, so there is no
        spend to attribute and no row claiming there was."""
        book_id = uuid.UUID(
            store_book(
                session_factory,
                storage,
                title="A Book About Craft",
                author="A. N. Author",
                filename="craft.pdf",
                content=b"not a book",
            )
        )

        run_id = execute_run(session_factory, settings.storage_root, book_id)

        assert run_row(session_factory, run_id).status == "failed"
        assert stages_of(session_factory, run_id) == []


class TestTheSpendIsAttributed:
    def test_each_stage_carries_the_tokens_it_spent(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        book: uuid.UUID,
    ) -> None:
        """The stub bills a fixed 100 in / 10 out per LLM call, so the split is
        arithmetic rather than a property of some provider."""
        run_id = execute_run(session_factory, settings.storage_root, book)

        spend = {
            stage.stage: (stage.input_tokens, stage.output_tokens)
            for stage in stages_of(session_factory, run_id)
        }

        assert spend["parse"] == (0, 0)
        assert spend["structure"] == (0, 0)
        calls = StubLLMProvider.INPUT_TOKENS_PER_CALL
        assert spend["profile"] == (calls, StubLLMProvider.OUTPUT_TOKENS_PER_CALL)
        assert spend["extraction"][0] > 0
        assert spend["summaries"][0] > 0

    def test_the_stage_totals_add_up_to_the_run_total(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        book: uuid.UUID,
    ) -> None:
        """The run-level number is unchanged and still true; this only says
        where it came from."""
        run_id = execute_run(session_factory, settings.storage_root, book)

        run = run_row(session_factory, run_id)
        stages = stages_of(session_factory, run_id)

        assert run.input_tokens == sum(stage.input_tokens for stage in stages)
        assert run.output_tokens == sum(stage.output_tokens for stage in stages)

    def test_embedding_tokens_are_a_total_of_their_own(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        book: uuid.UUID,
    ) -> None:
        """Billed at a different rate from completion tokens, so adding them to
        `input_tokens` would make the one number nobody can cost from. They were
        not persisted at all before this."""
        run_id = execute_run(session_factory, settings.storage_root, book)

        run = run_row(session_factory, run_id)
        stages = {stage.stage: stage for stage in stages_of(session_factory, run_id)}

        assert run.embedding_tokens is not None
        assert run.embedding_tokens > 0
        assert run.embedding_tokens == stages["embeddings"].embedding_tokens
        assert run.embedding_tokens % StubEmbeddingProvider.INPUT_TOKENS_PER_TEXT == 0
        # And they are kept out of the completion totals.
        assert stages["embeddings"].input_tokens == 0

    def test_a_scope_that_calls_no_provider_records_no_totals(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        book: uuid.UUID,
    ) -> None:
        """"No LLM work" has to keep reading differently from "LLM work that
        reported zero" — the distinction the NULLs carry today, and the reason
        the Runner is told rather than guessing from the reports."""
        run_id = execute_run(session_factory, settings.storage_root, book, "structure")

        run = run_row(session_factory, run_id)

        assert run.input_tokens is None
        assert run.output_tokens is None
        assert run.embedding_tokens is None

    def test_item_counts_survive_onto_the_row(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        book: uuid.UUID,
    ) -> None:
        """A cost per Stage is only readable beside what the Stage got through:
        five hundred tokens over two chapters is a different fact from five
        hundred over fifty."""
        run_id = execute_run(session_factory, settings.storage_root, book)

        stages = {stage.stage: stage for stage in stages_of(session_factory, run_id)}

        assert stages["structure"].counts
        assert all(isinstance(value, int) for value in stages["structure"].counts.values())


class TestTheTimingIsRecorded:
    def test_each_stage_records_when_it_started_and_finished(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        book: uuid.UUID,
    ) -> None:
        before = datetime.now(UTC).replace(tzinfo=None)

        run_id = execute_run(session_factory, settings.storage_root, book)

        stages = stages_of(session_factory, run_id)
        after = datetime.now(UTC).replace(tzinfo=None)
        for stage in stages:
            started, finished = naive(stage.started_at), naive(stage.finished_at)
            assert started is not None and finished is not None
            assert before <= started <= finished <= after

    def test_the_stages_do_not_overlap_and_run_in_order(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        book: uuid.UUID,
    ) -> None:
        """The Runner in this repo is sequential. A row that said otherwise
        would mean the timing is measuring something other than the Stage."""
        run_id = execute_run(session_factory, settings.storage_root, book)

        stages = stages_of(session_factory, run_id)

        for earlier, later in zip(stages, stages[1:], strict=False):
            finished, started = naive(earlier.finished_at), naive(later.started_at)
            assert finished is not None and started is not None
            assert finished <= started

    def test_the_stages_run_inside_the_run(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        book: uuid.UUID,
    ) -> None:
        run_id = execute_run(session_factory, settings.storage_root, book)

        run = run_row(session_factory, run_id)
        stages = stages_of(session_factory, run_id)

        assert run.finished_at is not None
        assert naive(run.created_at) <= naive(stages[0].started_at)
        assert naive(stages[-1].finished_at) <= naive(run.finished_at)
