"""A minimal in-repo sequential Runner over the Stage contract (ADR 0002).

A Runner walks a Scope's Stages in order, owns the Run record (Stages never
touch it), and maps Stage failures to an outcome. This one runs the whole Scope
synchronously in the foreground on a single session — the shape the CLI will
use. booksmart-api brings a different Runner (each Stage wrapped in an Inngest
step); both drive the same public ``run_<stage>`` functions.

The build_default_* seams exist so tests can substitute stubs without every
call site passing providers explicitly.
"""

import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from booksmart_core.config import Settings
from booksmart_core.errors import StagePreconditionError
from booksmart_core.extraction import EXTRACTION_PROMPT_VERSION
from booksmart_core.llm import (
    EmbeddingProvider,
    LLMProvider,
    build_embedding_provider,
    build_llm_provider,
)
from booksmart_core.models import Book, Run
from booksmart_core.parsing import EXTRACTION_VERSION, ParserChain, build_default_chain
from booksmart_core.profile import PROFILE_PROMPT_VERSION
from booksmart_core.stages import (
    LLM_STAGES,
    Stage,
    StageReport,
    run_embeddings,
    run_extraction,
    run_parse,
    run_profile,
    run_structure,
    run_summaries,
)
from booksmart_core.storage import BookStorage
from booksmart_core.summaries import SUMMARY_PROMPT_VERSION
from booksmart_core.vectors import VectorStore, build_vector_store

# Which Stages each Scope runs. Incremental scopes reuse upstream Stage output;
# extraction pulls embeddings along because the replaced knowledge objects
# would otherwise be left with stale or missing vectors.
SCOPE_STAGES: dict[str, tuple[Stage, ...]] = {
    "full": ("parse", "structure", "profile", "extraction", "summaries", "embeddings"),
    "profile": ("profile",),
    "extraction": ("extraction", "embeddings"),
    "embeddings": ("embeddings",),
}

# Module-level so Marker's model load (when installed) happens once per process.
DEFAULT_CHAIN = build_default_chain()


def build_default_llm() -> LLMProvider:
    """Provider used when the caller injects none. A separate function so tests
    can substitute a stub without touching every execute_run call site."""
    return build_llm_provider(Settings())


def build_default_embedder() -> EmbeddingProvider:
    """Test seam, like build_default_llm."""
    return build_embedding_provider(Settings())


def build_default_vector_store() -> VectorStore:
    """Test seam, like build_default_llm. Honors qdrant_path (embedded) or
    qdrant_url (server) from Settings."""
    return build_vector_store(Settings())


def _version_stamps(
    stages: tuple[Stage, ...],
    llm: LLMProvider | None,
    embedder: EmbeddingProvider | None,
) -> tuple[str, str | None, str | None]:
    """(extraction_version, model_version, prompt_version) describing exactly
    what this run used, limited to the stages it actually ran."""
    models = []
    if llm is not None and any(stage in LLM_STAGES for stage in stages):
        models.append(f"llm={llm.model}")
    if embedder is not None and "embeddings" in stages:
        models.append(f"embedding={embedder.model}")
    prompts = []
    if "profile" in stages:
        prompts.append(f"profile={PROFILE_PROMPT_VERSION}")
    if "extraction" in stages:
        prompts.append(f"extraction={EXTRACTION_PROMPT_VERSION}")
    if "summaries" in stages:
        prompts.append(f"summary={SUMMARY_PROMPT_VERSION}")
    return EXTRACTION_VERSION, ",".join(models) or None, ",".join(prompts) or None


def start_run(session: Session, book_id: uuid.UUID, scope: str) -> Run:
    """Create the Run at execution start (there is no queued state) and commit
    it so it is immediately visible as ``running``. A core helper so any Runner,
    across process boundaries, opens a run the same way."""
    run = Run(book_id=book_id, scope=scope, status="running")
    session.add(run)
    session.commit()
    return run


def finalize_run(
    session: Session,
    run: Run,
    reports: Sequence[StageReport],
    *,
    status: str,
    stamps: tuple[str, str | None, str | None],
    error: str | None = None,
    count_tokens: bool = False,
) -> None:
    """Aggregate a run's StageReports onto its Run row and close it: outcome,
    version stamps, summed token spend, and finish time. Token totals stay NULL
    unless ``count_tokens`` (the scope made LLM calls), so "no LLM work" reads
    differently from "LLM work that reported zero". A core helper so any Runner
    — this one, or booksmart-api's Inngest Runner — finalizes a Run the same way
    across process boundaries. Commits."""
    run.status = status
    run.error = error
    run.extraction_version, run.model_version, run.prompt_version = stamps
    if count_tokens:
        run.input_tokens = sum(report.input_tokens for report in reports)
        run.output_tokens = sum(report.output_tokens for report in reports)
    run.finished_at = datetime.now(UTC)
    session.commit()


def has_successful_run(session: Session, book_id: uuid.UUID) -> bool:
    """Whether the book has at least one succeeded Run to build an incremental
    scope on."""
    return (
        session.scalars(
            select(Run.id)
            .where(Run.book_id == book_id, Run.status == "succeeded")
            .limit(1)
        ).first()
        is not None
    )


def _dispatch(
    stage: Stage,
    session: Session,
    book_id: uuid.UUID,
    chain: ParserChain,
    storage: BookStorage,
    llm: LLMProvider | None,
    embedder: EmbeddingProvider | None,
    vector_store: VectorStore | None,
) -> StageReport:
    if stage == "parse":
        return run_parse(session, book_id, chain=chain, storage=storage)
    if stage == "structure":
        return run_structure(session, book_id, storage=storage)
    if stage == "profile":
        assert llm is not None
        return run_profile(session, book_id, llm=llm)
    if stage == "extraction":
        assert llm is not None
        return run_extraction(session, book_id, llm=llm, storage=storage)
    if stage == "summaries":
        assert llm is not None
        return run_summaries(session, book_id, llm=llm, storage=storage)
    assert embedder is not None and vector_store is not None
    return run_embeddings(session, book_id, embedder=embedder, vector_store=vector_store)


def execute_run(
    session_factory: sessionmaker[Session],
    storage_root: Path,
    book_id: uuid.UUID,
    scope: str = "full",
    *,
    chain: ParserChain | None = None,
    llm: LLMProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
    on_stage: Callable[[Stage], None] | None = None,
) -> uuid.UUID:
    """Run a Scope over a book to completion, recording a Run. Returns the Run
    id; the Run's ``status`` reflects the outcome (this never raises for an
    expected Stage failure — the failure is recorded on the Run).

    ``on_stage`` is called with each Stage just before it runs, so a foreground
    Runner (the CLI) can stream progress; it never affects the outcome."""
    storage = BookStorage(storage_root)
    chain = chain or DEFAULT_CHAIN
    stages = SCOPE_STAGES.get(scope)

    with session_factory() as session:
        run = start_run(session, book_id, scope)
        run_id = run.id

        reports: list[StageReport] = []
        stamps: tuple[str, str | None, str | None] = (EXTRACTION_VERSION, None, None)
        status = "succeeded"
        error: str | None = None
        error_line: str | None = None
        current_stage: Stage | None = None
        try:
            if stages is None:
                raise StagePreconditionError(f"unknown scope {scope!r}")
            # Build only the providers this scope needs (and fail fast if a
            # Preference conflicts with a Limit — a non-retriable config error).
            if any(stage in LLM_STAGES for stage in stages):
                llm = llm or build_default_llm()
            if "embeddings" in stages:
                embedder = embedder or build_default_embedder()
                vector_store = vector_store or build_default_vector_store()
            stamps = _version_stamps(stages, llm, embedder)

            for stage in stages:
                current_stage = stage
                if on_stage is not None:
                    on_stage(stage)
                reports.append(
                    _dispatch(
                        stage, session, book_id, chain, storage, llm, embedder, vector_store
                    )
                )
            current_stage = None
            if "parse" in stages:
                # Record what this run parsed (Stages are Run-blind, so the
                # Runner copies the pointer the parse Stage wrote onto the Book).
                book = session.get(Book, book_id)
                if book is not None:
                    run.output_path = book.parsed_path
                    run.parser_used = book.parser_used
        except Exception as exc:
            # Discard the failed stage's partial writes; earlier stages already
            # committed their own output wholesale and stay committed.
            session.rollback()
            status = "failed"
            # Name the failing stage for provenance without wrapping (and so
            # losing the type of) the exception that actually propagated.
            stage_prefix = f"{current_stage}: " if current_stage is not None else ""
            error = f"{stage_prefix}{type(exc).__name__}: {exc}"
            retriable = getattr(exc, "retriable", False)
            error_line = (
                f"{datetime.now(UTC).isoformat()} run failed "
                f"(retriable={retriable}): {error}"
            )

        _write_run_log(storage, run_id, reports, error_line)
        finalize_run(
            session,
            run,
            reports,
            status=status,
            stamps=stamps,
            error=error,
            count_tokens=stages is not None and any(s in LLM_STAGES for s in stages),
        )
        return run_id


def _write_run_log(
    storage: BookStorage,
    run_id: uuid.UUID,
    reports: Sequence[StageReport],
    error_line: str | None,
) -> None:
    """Persist the run's aggregated stage logs (plus a final failure line) —
    diagnostics only, so a filesystem hiccup never sinks the run record."""
    log_lines = [line for report in reports for line in report.log_lines]
    if error_line is not None:
        log_lines.append(error_line)
    try:
        storage.save_log(run_id, "".join(f"{line}\n" for line in log_lines))
    except OSError:
        pass
