"""Building a corpus, and recording exactly what that cost.

The Runner here is bench-owned and deliberately not core's ``execute_run``. Both
drive the same public ``run_<stage>`` functions, but core's Runner aggregates
StageReports onto the Run row and lets them go, and the Run row records a total
spend with no per-Stage breakdown (booksmart#65). The cost dimension needs that
breakdown, so this walks the Stages itself, keeps every StageReport and times
each one in-process. When #65 lands, this becomes the fallback rather than the
source.

Ingest is the only verb that spends money, so the charter is ingest-once,
recall-often. Two guards make that safe rather than merely cheap: a book is
skipped when its bytes and configuration are unchanged, and a source whose
sha256 does not match the one its truth pins is refused outright — truth
authored against one edition must never silently score another.
"""

import json
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from sqlalchemy.orm import Session

from booksmart_core.config import Settings
from booksmart_core.llm import (
    EmbeddingProvider,
    LLMProvider,
    build_embedding_provider,
    build_llm_provider,
)
from booksmart_core.models import Book
from booksmart_core.parsing import ParserChain, build_default_chain
from booksmart_core.runner import SCOPE_STAGES, finalize_run, start_run
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
from booksmart_core.storage import hash_stream

from booksmart_bench.corpus import Corpus
from booksmart_bench.errors import BenchError, SourceMissingError
from booksmart_bench.truth import BookTruth, Truth

PROVENANCE_FILE = "provenance.json"

# The full pipeline, in order. Bench always ingests a whole book — an
# incremental scope is a product affordance, and a benchmark that measured a
# partially-rebuilt corpus would not be measuring anything reproducible.
FULL_SCOPE = "full"

@dataclass(frozen=True)
class StageTiming:
    """One Stage's cost, which the Run row cannot currently carry."""

    stage: str
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class BookIngest:
    slug: str
    status: str  # "succeeded" | "skipped" | "failed"
    sha256: str
    book_id: str | None = None
    run_id: str | None = None
    stages: tuple[StageTiming, ...] = ()
    wall_seconds: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class IngestOutcome:
    snapshot: str
    home: Path
    books: tuple[BookIngest, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Provenance:
    """What this corpus contains, and what it cost to put there."""

    snapshot: str
    books: dict[str, dict[str, Any]] = field(default_factory=dict)


def read_provenance(home: Path) -> Provenance:
    path = home / PROVENANCE_FILE
    if not path.exists():
        return Provenance(snapshot="")
    try:
        document = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise BenchError(f"Could not parse {path}: {exc}") from exc
    return Provenance(
        snapshot=str(document.get("snapshot", "")),
        books=dict(document.get("books", {})),
    )


def write_provenance(home: Path, provenance: Provenance) -> None:
    (home / PROVENANCE_FILE).write_text(
        json.dumps(
            {"snapshot": provenance.snapshot, "books": provenance.books}, indent=2
        )
        + "\n"
    )


# --- scope resolution -------------------------------------------------------


def resolve_scope(truth: Truth, scope: str) -> tuple[BookTruth, ...]:
    """The books a scope names: one slug, or every book in an area."""
    if scope in truth.books:
        return (truth.books[scope],)

    in_area = tuple(
        book for _, book in sorted(truth.books.items()) if book.area == scope
    )
    if in_area:
        return in_area

    areas = sorted({book.area for book in truth.books.values() if book.area})
    raise BenchError(
        f"Unknown scope {scope!r}. Books: {', '.join(sorted(truth.books)) or '(none)'}. "
        f"Areas: {', '.join(areas) or '(none)'}."
    )


# --- sources ----------------------------------------------------------------


def locate_source(assets: Path, book: BookTruth) -> tuple[Path, str]:
    """The book's source file and its actual sha256, verified against truth.

    Refusing a mismatch here is the whole point of pinning: a benchmark that
    scored a different edition than its truth was authored against would report
    a pipeline regression that is really a library change.
    """
    relative = book.source_file or f"sources/{book.slug}.pdf"
    path = assets / relative
    if not path.is_file():
        raise SourceMissingError(
            f"{book.slug}: no source at {path}. Drop the file there (see the "
            f"sources handover) and re-run."
        )

    with path.open("rb") as stream:
        digest = hash_stream(stream)

    if book.is_pinned:
        pinned = book.source_sha256
        if pinned != digest:
            raise SourceMissingError(
                f"{book.slug}: sha256 mismatch. {relative} hashes to {digest}, but "
                f"book.yaml pins {pinned}. Truth was authored against a different "
                f"artifact; ingesting this one would score the wrong edition."
            )
    return path, digest


# --- the bench-owned Runner -------------------------------------------------


def _dispatch(
    stage: Stage,
    session: Session,
    book_id: uuid.UUID,
    corpus: Corpus,
    llm: LLMProvider,
    embedder: EmbeddingProvider,
    chain: ParserChain,
) -> StageReport:
    if stage == "parse":
        return run_parse(session, book_id, chain=chain, storage=corpus.storage)
    if stage == "structure":
        return run_structure(session, book_id, storage=corpus.storage)
    if stage == "profile":
        return run_profile(session, book_id, llm=llm)
    if stage == "extraction":
        return run_extraction(session, book_id, llm=llm, storage=corpus.storage)
    if stage == "summaries":
        return run_summaries(session, book_id, llm=llm, storage=corpus.storage)
    return run_embeddings(
        session, book_id, embedder=embedder, vector_store=corpus.vector_store
    )


def run_pipeline(
    corpus: Corpus,
    book_id: uuid.UUID,
    *,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[str, tuple[StageTiming, ...], float, str | None]:
    """Walk the full Scope, timing each Stage. Returns (run_id, timings, wall,
    error). Never raises for a Stage failure — the failure is recorded on the
    Run, exactly as core's Runner does, and returned for the report."""
    stages: Sequence[Stage] = SCOPE_STAGES[FULL_SCOPE]
    chain = build_default_chain()
    llm = build_llm_provider(corpus.settings)
    embedder = build_embedding_provider(corpus.settings)

    timings: list[StageTiming] = []
    reports: list[StageReport] = []
    error: str | None = None
    status = "succeeded"
    started = time.monotonic()

    with corpus.session_factory() as session:
        run = start_run(session, book_id, FULL_SCOPE)
        run_id = str(run.id)
        try:
            for stage in stages:
                if on_stage is not None:
                    on_stage(stage)
                stage_started = time.monotonic()
                report = _dispatch(stage, session, book_id, corpus, llm, embedder, chain)
                reports.append(report)
                timings.append(
                    StageTiming(
                        stage=stage,
                        seconds=time.monotonic() - stage_started,
                        input_tokens=report.input_tokens,
                        output_tokens=report.output_tokens,
                        embedding_tokens=report.embedding_tokens,
                        counts=dict(report.counts),
                    )
                )
        except Exception as exc:
            # Discard the failed Stage's partial writes before finalizing, or
            # finalize_run's commit would persist them. A Stage replaces its
            # output wholesale (ADR 0002), so a failure between the delete and
            # the write would otherwise leave the corpus permanently emptied for
            # that book — with only a "failed" row to explain it.
            session.rollback()
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"

        wall = time.monotonic() - started
        finalize_run(
            session,
            run,
            reports,
            status=status,
            stamps=(
                corpus.settings.embedding_model or "",
                f"llm={llm.model},embedding={embedder.model}",
                None,
            ),
            error=error,
            count_tokens=any(stage in LLM_STAGES for stage in stages),
        )
    return run_id, tuple(timings), wall, error


# --- the verb ---------------------------------------------------------------


def ingest_scope(
    assets: Path,
    truth: Truth,
    scope: str,
    settings: Settings,
    *,
    force: bool = False,
    on_stage: Callable[[str, str], None] | None = None,
) -> IngestOutcome:
    """Ingest every book a scope names into this configuration's corpus."""
    books = resolve_scope(truth, scope)

    # Sources are checked for all books up front: discovering a missing file
    # after an hour of ingesting the previous book helps nobody.
    located = [(book, *locate_source(assets, book)) for book in books]

    notes: list[str] = [
        f"{book.slug}: source is not pinned; add sha256 {digest} to its book.yaml"
        for book, _, digest in located
        if not book.is_pinned
    ]

    outcomes: list[BookIngest] = []
    with Corpus.open(settings) as corpus:
        provenance = read_provenance(corpus.home)
        provenance = Provenance(snapshot=corpus.home.name, books=dict(provenance.books))

        for book, source, digest in located:
            recorded = provenance.books.get(book.slug)
            if recorded and recorded.get("sha256") == digest and not force:
                outcomes.append(
                    BookIngest(slug=book.slug, status="skipped", sha256=digest,
                               book_id=recorded.get("book_id"))
                )
                continue

            book_id = _register(corpus, book, source, digest)
            run_id, timings, wall, error = run_pipeline(
                corpus, book_id, on_stage=_per_book_callback(on_stage, book.slug)
            )
            status = "failed" if error else "succeeded"
            outcomes.append(
                BookIngest(
                    slug=book.slug,
                    status=status,
                    sha256=digest,
                    book_id=str(book_id),
                    run_id=run_id,
                    stages=timings,
                    wall_seconds=wall,
                    error=error,
                )
            )
            if status == "succeeded":
                provenance.books[book.slug] = _record_of(book_id, run_id, digest, wall, timings)
                # Written per book, not once at the end: an escape anywhere
                # later in the scope would otherwise discard the record of every
                # book already ingested, and re-spend all of it next run.
                write_provenance(corpus.home, provenance)

        home = corpus.home

    return IngestOutcome(
        snapshot=home.name, home=home, books=tuple(outcomes), notes=tuple(notes)
    )


def _record_of(
    book_id: uuid.UUID,
    run_id: str,
    digest: str,
    wall: float,
    timings: tuple[StageTiming, ...],
) -> dict[str, Any]:
    """One book's entry in the corpus provenance: what was ingested, from which
    bytes, and what each Stage cost — the breakdown the Run row cannot hold."""
    return {
        "book_id": str(book_id),
        "run_id": run_id,
        "sha256": digest,
        "ingested_at": datetime.now(UTC).isoformat(),
        "wall_seconds": wall,
        "stages": [
            {
                "stage": timing.stage,
                "seconds": timing.seconds,
                "input_tokens": timing.input_tokens,
                "output_tokens": timing.output_tokens,
                "embedding_tokens": timing.embedding_tokens,
                "counts": timing.counts,
            }
            for timing in timings
        ],
    }


def _per_book_callback(
    on_stage: Callable[[str, str], None] | None, slug: str
) -> Callable[[str], None] | None:
    """Bind the book a stage belongs to, so the caller's progress line can name
    it without every Stage having to carry the slug."""
    if on_stage is None:
        return None

    def report(stage: str) -> None:
        on_stage(slug, stage)

    return report


def _register(corpus: Corpus, book: BookTruth, source: Path, digest: str) -> uuid.UUID:
    """Persist the Book row, reusing the existing one when these bytes are
    already registered — a re-ingest replaces a book's derived rows, it does not
    accumulate copies of the book."""
    with corpus.session_factory() as session:
        existing = session.scalars(select(Book).where(Book.file_hash == digest)).first()
        if existing is not None:
            return existing.id

    book_id = uuid.uuid4()
    with source.open("rb") as stream:
        stored = corpus.storage.save_original(book_id, source.name, stream, digest)

    row = Book(
        id=book_id,
        title=book.title or book.slug,
        author=", ".join(book.authors) or "unknown",
        original_filename=stored.path.name,
        file_format=source.suffix.lstrip(".").lower(),
        storage_path=str(stored.path),
        checksum=stored.checksum,
        file_hash=stored.file_hash,
    )
    try:
        with corpus.session_factory() as session:
            session.add(row)
            session.commit()
    except Exception:
        corpus.storage.discard(book_id)
        raise
    return book_id
