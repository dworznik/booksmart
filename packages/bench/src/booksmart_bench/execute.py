"""The run verb: benchmarks in, one run file out.

This is the only place the three artefacts touch. Truth says what to ask, the
corpus answers through the pipeline's own code paths, and what lands on disk is
a run file the scorer can read without ever seeing the corpus again. Two
consequences shape everything here:

**Locations, never record ids.** Every hit, every judged summary and every
detected node is written as a ToC node id, resolved through
``locations.join``. The scorer is a pure comparison; a run file carrying record
ids would drag the corpus into it.

**The real path, not a re-implementation.** Recall goes through
``booksmart_core.search`` exactly as a product consumer would — same filters,
same limit, same ranking. A benchmark that re-implemented retrieval would
measure the re-implementation.

Two families, because the artefacts they produce are independent:

    recall      the query sets, asked through search(), scored by location
    ingestion   structure, extraction coverage, summary faithfulness, cost

``all`` runs both into one file; scoring reads whatever a file happens to carry.

Nothing measured is silently dropped and nothing unmeasurable is silently
counted. A book that was never ingested, a query the run refused to ask, a hit
whose record could not be joined to a node — each produces a note that travels
into the run file and onto the console. The failure this guards against is a
harness that returns a plausible zero.
"""

import json
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from booksmart_core.config import Settings
from booksmart_core.extraction import chapter_body, iter_chapter_bodies
from booksmart_core.llm import EmbeddingProvider, LLMProvider, build_embedding_provider
from booksmart_core.models import Book, Chapter, KnowledgeObject
from booksmart_core.search import SearchHit, SearchResults, search
from booksmart_core.vectors import RECORD_TYPES

from booksmart_bench.config import snapshot_inputs
from booksmart_bench.corpus import Corpus
from booksmart_bench.errors import BenchError
from booksmart_bench.ingest import Provenance, read_provenance, resolve_scope
from booksmart_bench.judge import (
    JudgeConfig,
    JudgeReport,
    SummaryUnderTest,
    build_judge,
    judge_summaries,
)
from booksmart_bench.locations import Detected, LocationMap, join
from booksmart_bench.truth import BookTruth, Query, Truth

# What each family measures, in one line. Also the set a bad argument is
# checked against, so the two can never disagree.
FAMILIES = {
    "recall": "the query sets, asked through the real search path",
    "ingestion": "structure, extraction coverage, summary faithfulness and cost",
    "all": "both families into one run file",
}

# Core's search path offers exactly one retrieval mode today. It is stamped
# anyway, and named rather than left blank: when a second mode lands
# (booksmart#64) a run from before it must not read as a run of the new default.
SEARCH_MODE = "dense"

# Which families include which half, so membership is decided in one place
# rather than re-spelled at every branch.
RECALL_FAMILIES = frozenset({"recall", "all"})
INGESTION_FAMILIES = frozenset({"ingestion", "all"})

DEFAULT_LIMIT = 10

# How a detected Chapter row's `kind` reads in truth's vocabulary. Front and
# back matter keep their own kind rather than being flattened into "chapter":
# the structure dimension scores chapters and sections, and a detector that
# skips the index is not wrong (SPEC §4).
NODE_KINDS = {
    "chapter": "chapter",
    "front_matter": "front-matter",
    "back_matter": "back-matter",
}


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    path: Path
    family: str
    scope: str
    books: tuple[str, ...]
    asked: int
    judged: int
    notes: tuple[str, ...] = ()


def execute_run(
    assets: Path,
    truth: Truth,
    family: str,
    scope: str,
    settings: Settings,
    *,
    limit: int = DEFAULT_LIMIT,
    out: Path | None = None,
    judge: JudgeConfig | None = None,
    judge_llm: LLMProvider | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> RunOutcome:
    """Execute one family of benchmarks over one scope and write the run file."""
    if family not in FAMILIES:
        raise BenchError(
            f"Unknown family {family!r}; expected one of {', '.join(sorted(FAMILIES))}."
        )
    books, area = _resolve_scope(truth, scope)
    notes: list[str] = []
    if family in RECALL_FAMILIES and area is None and scope not in truth.books:
        notes.append(
            f"{scope} has no truth/areas/{scope}/queries.yaml, so no cross-book "
            "queries were asked; only the books' own sets were"
        )

    def report(line: str) -> None:
        if on_progress is not None:
            on_progress(line)

    with Corpus.open(settings) as corpus:
        provenance = read_provenance(corpus.home)
        known = _in_corpus(truth, provenance)
        scoped = tuple(book for book in books if book.slug in known)
        notes.extend(
            f"{book.slug}: not in this corpus; `booksmart-bench ingest {book.slug}` first"
            for book in books
            if book.slug not in known
        )
        if not scoped:
            raise BenchError(
                f"No book in scope {scope!r} is in this corpus "
                f"({corpus.home}). Run `booksmart-bench ingest {scope}` first."
            )

        with corpus.session_factory() as session:
            # Every known book, not only the scoped ones: an area query is asked
            # over the whole corpus, and a hit in a book outside the scope still
            # has to resolve to a location or the run cannot say what came back.
            maps = {
                slug: _joined(session, slug, book_id, truth.books[slug], notes)
                for slug, book_id in known.items()
            }

            results: list[dict[str, Any]] = []
            search_cost = _SearchCost()
            if family in RECALL_FAMILIES:
                results = _ask_everything(
                    session,
                    corpus,
                    truth,
                    scoped,
                    area,
                    known,
                    maps,
                    limit=limit,
                    cost=search_cost,
                    notes=notes,
                    report=report,
                )

            structure: list[dict[str, Any]] = []
            coverage: list[dict[str, Any]] = []
            judged: JudgeReport | None = None
            if family in INGESTION_FAMILIES:
                structure = [
                    {"book": book.slug, "detected": _detected_nodes(maps[book.slug].nodes)}
                    for book in scoped
                ]
                coverage = [
                    {
                        "book": book.slug,
                        "surfaced": _surfaced(session, known[book.slug]),
                    }
                    for book in scoped
                ]
                judged = _judge(
                    session,
                    corpus,
                    settings,
                    scoped,
                    known,
                    maps,
                    config=judge,
                    provider=judge_llm,
                    notes=notes,
                    report=report,
                )

        run_id = _run_id(family, scope)
        document = {
            "run_id": run_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "family": family,
            "scope": scope,
            "corpus": {
                "snapshot": corpus.home.name,
                "books": [book.slug for book in scoped],
                # The configuration the snapshot is computed from, spelled out:
                # a directory name says two runs differ, this says how.
                "config": snapshot_inputs(settings),
            },
            "search": {
                "mode": SEARCH_MODE,
                "limit": limit,
                "record_types": list(RECORD_TYPES),
            },
            # Only when it ran: a recall run naming a ruler it never picked up
            # would read as a faithfulness measurement that scored nothing.
            "judge": judge.as_dict() if judged is not None and judge is not None else None,
            "results": results,
            "structure": structure,
            "coverage": coverage,
            "faithfulness": [
                {
                    "book": verdict.book,
                    "loc": verdict.loc,
                    "supported": verdict.supported,
                    "claims": verdict.claims,
                    "note": verdict.note,
                }
                for verdict in (judged.verdicts if judged is not None else ())
            ],
            "cost": _cost(family, provenance, scoped, search_cost, judged),
            "notes": notes,
        }

    path = out if out is not None else assets / "runs" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")

    return RunOutcome(
        run_id=run_id,
        path=path,
        family=family,
        scope=scope,
        books=tuple(book.slug for book in scoped),
        asked=len(results),
        judged=len(judged.verdicts) if judged is not None else 0,
        notes=tuple(notes),
    )


# --- scope and corpus -------------------------------------------------------


def _resolve_scope(truth: Truth, scope: str) -> tuple[tuple[BookTruth, ...], str | None]:
    """The books a scope names, and the area whose cross-book queries it also
    asks. A book slug names no area: its own set is the whole of it."""
    books = resolve_scope(truth, scope)
    area = scope if scope not in truth.books and scope in truth.areas else None
    return books, area


def _in_corpus(truth: Truth, provenance: Provenance) -> dict[str, uuid.UUID]:
    """Every book truth knows that this corpus actually holds."""
    found: dict[str, uuid.UUID] = {}
    for slug, record in provenance.books.items():
        book_id = record.get("book_id")
        if slug not in truth.books or not book_id:
            continue
        try:
            found[slug] = uuid.UUID(str(book_id))
        except ValueError:
            # provenance.json is written to be read, so it gets hand-edited.
            continue
    return found


@dataclass(frozen=True)
class _Joined:
    """A book's detected structure and where each row landed in its ToC."""

    book_id: uuid.UUID
    nodes: tuple[Detected, ...]
    locations: LocationMap

    def node_for(self, record_id: str | None) -> str | None:
        return self.locations.node_for(record_id)


def _joined(
    session: Session, slug: str, book_id: uuid.UUID, book: BookTruth, notes: list[str]
) -> _Joined:
    detected = _read_structure(session, book_id)
    located = join(detected, book)
    notes.extend(f"{slug}: {note}" for note in located.unresolved)
    return _Joined(book_id=book_id, nodes=detected, locations=located)


def _chapters(session: Session, book_id: uuid.UUID) -> list[Chapter]:
    """The book's chapters in reading order, sections included."""
    return list(
        session.scalars(
            select(Chapter).where(Chapter.book_id == book_id).order_by(Chapter.position)
        )
    )


def _read_structure(session: Session, book_id: uuid.UUID) -> tuple[Detected, ...]:
    """The book's stored structure, flattened into document order.

    Position is the flattened index rather than each row's own ``position``
    column: a Section numbers itself within its chapter, and both the join's
    tiebreak and the scorer's read position as "which came first in the book".
    """
    rows: list[Detected] = []
    for chapter in _chapters(session, book_id):
        rows.append(
            Detected(
                record_id=str(chapter.id),
                title=chapter.title,
                kind=NODE_KINDS.get(chapter.kind, "chapter"),
                position=len(rows),
            )
        )
        for section in chapter.sections:
            rows.append(
                Detected(
                    record_id=str(section.id),
                    title=section.title,
                    kind="section",
                    position=len(rows),
                    parent_id=str(chapter.id),
                )
            )
    return tuple(rows)


# --- recall -----------------------------------------------------------------


class _SearchCost:
    """What asking the queries cost: the read side's embedding spend."""

    def __init__(self) -> None:
        self.embedding_tokens = 0
        self.seconds = 0.0
        # Calls whose provider reported no usage. Counted rather than assumed
        # zero, so a total that is a lower bound says so.
        self.unreported = 0

    def add(self, results: SearchResults, seconds: float) -> None:
        self.seconds += seconds
        if results.embedding_tokens is None:
            self.unreported += 1
        else:
            self.embedding_tokens += results.embedding_tokens


def _ask_everything(
    session: Session,
    corpus: Corpus,
    truth: Truth,
    scoped: Sequence[BookTruth],
    area: str | None,
    known: Mapping[str, uuid.UUID],
    maps: Mapping[str, _Joined],
    *,
    limit: int,
    cost: _SearchCost,
    notes: list[str],
    report: Callable[[str], None],
) -> list[dict[str, Any]]:
    """Every query the scope covers, asked and located."""
    if corpus.vector_store.verified_model() is None:
        raise BenchError(
            f"Nothing is embedded in this corpus ({corpus.home}); every query would "
            "return nothing and score as a total retrieval failure. Re-run `ingest`."
        )

    asking = _Asking(
        session=session,
        corpus=corpus,
        embedder=build_embedding_provider(corpus.settings),
        maps=maps,
        # Hits come back keyed by book_id; the run file speaks slugs.
        by_book={joined.book_id: slug for slug, joined in maps.items()},
        limit=limit,
        cost=cost,
        notes=notes,
        report=report,
    )

    results: list[dict[str, Any]] = []
    for book in scoped:
        if not book.queries:
            # Truth's own gap rather than the run's, and invisible from the
            # results alone: a book nothing was asked about scores nothing.
            notes.append(f"{book.slug}: no queries authored, so none were asked")
        for query in book.queries:
            # A book's own set asks "does this book's right section come back",
            # so it is filtered to that book: whatever else the corpus happens
            # to hold must not be able to move the score.
            asked = asking.ask(query, book_id=known[book.slug])
            if asked is not None:
                results.append({**asked, "book": book.slug})

    area_queries = truth.areas.get(area or "", ())
    if area is not None and not area_queries:
        notes.append(f"truth/areas/{area}: no cross-book queries authored, so none were asked")
    for query in area_queries:
        # Cross-book queries go to the whole corpus on purpose: which book
        # answers is half of what an area query measures.
        asked = asking.ask(query, book_id=None)
        if asked is not None:
            results.append({**asked, "area": area})

    notes.extend(asking.notes_about_drops())
    return results


@dataclass
class _Asking:
    """Everything asking one query needs, and the tally of what it could not
    place. Bundled because the alternative is an eleven-argument call repeated
    once per query set — and because the drop counters have to survive across
    every query in a run, not per query."""

    session: Session
    corpus: Corpus
    embedder: EmbeddingProvider
    maps: Mapping[str, _Joined]
    by_book: Mapping[uuid.UUID, str]
    limit: int
    cost: _SearchCost
    notes: list[str]
    report: Callable[[str], None]
    # Hits dropped because their record joined to no node, by book.
    unjoined: Counter[str] = field(default_factory=Counter)
    # Hits dropped because they came from a book this truth tree has never
    # heard of — a different miss, and one the operator fixes differently.
    unknown_book: int = 0

    def ask(self, query: Query, *, book_id: uuid.UUID | None) -> dict[str, Any] | None:
        if query.gated:
            self.notes.append(
                f"{query.q!r} is gated on a pipeline feature; not asked, and not scored"
            )
            return None

        self.report(f"? {query.q}")
        started = time.monotonic()
        found = search(
            self.session,
            self.corpus.vector_store,
            self.embedder,
            query.q,
            book_id=book_id,
            record_types=RECORD_TYPES,
            limit=self.limit,
        )
        self.cost.add(found, time.monotonic() - started)
        return {"q": query.q, "hits": self.locate(found)}

    def locate(self, found: SearchResults) -> list[dict[str, Any]]:
        """Hits as locations, in the order search ranked them.

        A hit whose record joins to no node is left out rather than renumbered
        around: the ranks that remain are the ranks the pipeline produced, so
        dropping one cannot promote another. Every drop is counted — an
        invisible one would read downstream as a retrieval miss.
        """
        located: list[dict[str, Any]] = []
        for rank, hit in enumerate(found.hits, start=1):
            slug = self.by_book.get(hit.book_id)
            if slug is None:
                self.unknown_book += 1
                continue
            node = _node_for(hit, self.maps[slug])
            if node is None:
                self.unjoined[slug] += 1
                continue
            located.append(
                {"rank": rank, "book": slug, "loc": node, "record_type": hit.record_type}
            )
        return located

    def notes_about_drops(self) -> list[str]:
        notes = [
            f"{slug}: {count} hit(s) dropped — their record joined to no ToC node"
            for slug, count in sorted(self.unjoined.items())
        ]
        if self.unknown_book:
            notes.append(
                f"{self.unknown_book} hit(s) dropped — they came from a book this "
                "truth tree does not cover"
            )
        if self.cost.unreported:
            notes.append(
                f"{self.cost.unreported} search call(s) reported no embedding usage; "
                "the search cost recorded below is a lower bound"
            )
        return notes


def _node_for(hit: SearchHit, joined: _Joined) -> str | None:
    """The ToC node a hit points at.

    A knowledge object is not a location: it is attached to one by the FKs the
    extraction stage wrote, section first because it is the more precise of the
    two (SPEC §3 scores locations, not records).
    """
    if isinstance(hit.row, KnowledgeObject):
        return joined.node_for(_as_text(hit.row.section_id)) or joined.node_for(
            _as_text(hit.row.chapter_id)
        )
    return joined.node_for(str(hit.record_id))


def _as_text(value: uuid.UUID | None) -> str | None:
    """A nullable FK as the string the location map is keyed by."""
    return None if value is None else str(value)


# --- ingestion --------------------------------------------------------------


def _detected_nodes(nodes: Sequence[Detected]) -> list[dict[str, Any]]:
    return [
        {"title": node.title, "kind": node.kind, "position": node.position} for node in nodes
    ]


def _surfaced(session: Session, book_id: uuid.UUID) -> list[str]:
    """The names extraction actually put in the corpus, for the coverage
    dimension to check the authored inventory against."""
    return [
        row.title
        for row in session.scalars(
            select(KnowledgeObject).where(KnowledgeObject.book_id == book_id)
        )
    ]


def _judge(
    session: Session,
    corpus: Corpus,
    settings: Settings,
    scoped: Sequence[BookTruth],
    known: Mapping[str, uuid.UUID],
    maps: Mapping[str, _Joined],
    *,
    config: JudgeConfig | None,
    provider: LLMProvider | None,
    notes: list[str],
    report: Callable[[str], None],
) -> JudgeReport | None:
    if config is None:
        notes.append(
            "faithfulness not measured: no judge is configured "
            "(BOOKSMART_BENCH_JUDGE_PROVIDER)"
        )
        return None
    summaries: list[SummaryUnderTest] = []
    for book in scoped:
        summaries.extend(
            _summaries_under_test(session, corpus, book.slug, known[book.slug], maps[book.slug], notes)
        )
    if not summaries:
        notes.append("faithfulness not measured: the corpus holds no summaries to judge")
        return None

    judged = judge_summaries(
        provider if provider is not None else build_judge(config, settings),
        summaries,
        on_progress=lambda summary: report(f"judging {summary.book} {summary.loc}"),
    )
    notes.extend(f"judge failure — {failure}" for failure in judged.failures)
    return judged


def _summaries_under_test(
    session: Session,
    corpus: Corpus,
    slug: str,
    book_id: uuid.UUID,
    joined: _Joined,
    notes: list[str],
) -> list[SummaryUnderTest]:
    """Every stored summary paired with the source slice it is judged against.

    Slices are cut from the book's own parsed artifact at the line numbers
    structure detection recorded, so the judge reads the text itself rather than
    a paraphrase of it. A chapter summary is judged against its chapter body —
    the very input the summaries stage was given.

    A *section* summary is judged against the section's own lines, which is
    narrower than what the summariser saw: core writes a chapter's summary and
    all its section summaries in one call over the whole chapter (SPEC §5 asks
    for "section summary vs. section body" anyway). The asymmetry is the point
    of the slice — a section summary that leans on a neighbouring section's
    material is not faithful *to that section* — but it does make the section
    slice the stricter half of the dimension.
    """
    book = session.get(Book, book_id)
    if book is None or book.parsed_path is None:
        notes.append(f"{slug}: no parsed artifact in this corpus; its summaries were not judged")
        return []
    artifact = corpus.storage.resolve(book.parsed_path)
    if not artifact.is_file():
        notes.append(f"{slug}: parsed artifact missing from storage; its summaries were not judged")
        return []

    markdown = artifact.read_text(encoding="utf-8")
    chapters = _chapters(session, book_id)
    boundaries = sorted(
        line
        for chapter in chapters
        for line in (chapter.source_line, *(s.source_line for s in chapter.sections))
        if line is not None
    )

    under_test: list[SummaryUnderTest] = []
    for chapter, body in iter_chapter_bodies(chapters, markdown):
        node = joined.node_for(str(chapter.id))
        if chapter.summary and node:
            under_test.append(SummaryUnderTest(slug, node, chapter.summary, body))
        for section in chapter.sections:
            node = joined.node_for(str(section.id))
            if not (section.summary and node):
                continue
            if section.source_line is None:
                notes.append(
                    f"{slug}: section {section.title!r} has no recorded line, so its "
                    "source slice cannot be rebuilt; not judged"
                )
                continue
            under_test.append(
                SummaryUnderTest(
                    slug,
                    node,
                    section.summary,
                    chapter_body(
                        markdown,
                        section.source_line,
                        _next_boundary(boundaries, section.source_line),
                    ),
                )
            )
    return under_test


def _next_boundary(boundaries: Sequence[int], line: int) -> int | None:
    """The line the next heading starts on — where this slice ends."""
    return next((boundary for boundary in boundaries if boundary > line), None)


# --- cost -------------------------------------------------------------------


def _cost(
    family: str,
    provenance: Provenance,
    scoped: Sequence[BookTruth],
    search_cost: _SearchCost,
    judged: JudgeReport | None,
) -> dict[str, Any]:
    """Everything this measurement cost, per Stage.

    Ingest spend is read back from the corpus provenance (booksmart#65 will
    eventually put it on the Run), and only for the ingestion family: a recall
    run did not build the corpus, and charging it for one would double-count
    the ingest that did.
    """
    per_stage: list[dict[str, Any]] = []
    if family in INGESTION_FAMILIES:
        per_stage.extend(_ingest_stages(provenance, scoped))
    if family in RECALL_FAMILIES:
        per_stage.append(
            {
                "stage": "search",
                "seconds": search_cost.seconds,
                "input_tokens": 0,
                "output_tokens": 0,
                "embedding_tokens": search_cost.embedding_tokens,
            }
        )
    if judged is not None:
        per_stage.append(
            {
                "stage": "judge",
                "seconds": judged.seconds,
                "input_tokens": judged.input_tokens,
                "output_tokens": judged.output_tokens,
                "embedding_tokens": 0,
            }
        )

    return {
        "input_tokens": _total(per_stage, "input_tokens"),
        "output_tokens": _total(per_stage, "output_tokens"),
        "embedding_tokens": _total(per_stage, "embedding_tokens"),
        "wall_seconds": sum(float(entry.get("seconds", 0.0)) for entry in per_stage),
        "per_stage": per_stage,
    }


def _ingest_stages(
    provenance: Provenance, scoped: Sequence[BookTruth]
) -> list[dict[str, Any]]:
    """One row per Stage, summed over the books in scope.

    Summed rather than per book, even though provenance holds the per-book
    breakdown: the report groups these rows by stage name, so two rows both
    called "parse" would silently overwrite each other and under-report the
    scope. The per-book detail stays in the corpus provenance, which is where it
    was written and where it is still readable.
    """
    totals: dict[str, dict[str, Any]] = {}
    for book in scoped:
        for entry in provenance.books.get(book.slug, {}).get("stages", []):
            if not isinstance(entry, dict):
                continue
            row = totals.setdefault(
                str(entry.get("stage", "")),
                {
                    "stage": str(entry.get("stage", "")),
                    "seconds": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "embedding_tokens": 0,
                },
            )
            row["seconds"] += float(entry.get("seconds", 0.0) or 0.0)
            for measure in ("input_tokens", "output_tokens", "embedding_tokens"):
                row[measure] += int(entry.get(measure, 0) or 0)
    return list(totals.values())


def _total(per_stage: Sequence[Mapping[str, Any]], measure: str) -> int:
    return sum(int(entry.get(measure, 0) or 0) for entry in per_stage)


def _run_id(family: str, scope: str) -> str:
    """Sortable, and it says what it is: nobody should have to open two run
    files to find out which is which."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{stamp}-{family}-{scope}"
