"""Run file x truth -> scores, and the verdict over two runs.

Pure by construction: this module imports nothing from the pipeline, and a test
enforces that. The point of the three-artefact split is that scoring can be
trusted to compare two pipelines precisely because it cannot reach either one.
Everything it needs arrives as a run file — resolved locations, never record
ids, because the record->location join is the run writer's job.

Five dimensions, and only four of them score:

    recall         MRR (primary) + hit@5, binary location-level judgements
    structure      P/R/F1 over (chapter, section) nodes vs the authored ToC
    coverage       recall against the concept inventory
    faithfulness   mean claim-support ratio, with a pass bar per summary
    cost           carried, never scored — a cheaper run is not a better one

Two rules shape the output. Per-slice numbers are **diagnostics**: they say
where a difference came from, and they are not what a variant is judged on.
Only the **pooled** aggregate feeds the regression verdict, and there is no
blended headline number across dimensions — an MRR and an F1 do not average
into anything meaningful.

Nothing is ever silently dropped. A gated slice, a query the run never asked,
a book the truth does not cover — each is reported. The failure this guards
against is a truncated run scoring as a catastrophic one, or a gated slice
scoring as a passing one.
"""

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from booksmart_bench.errors import BenchError
from booksmart_bench.truth import BookTruth, Query, Truth

# How far a pooled dimension may fall before it is called a regression.
# Per-dimension because the scales are unrelated: a 0.05 MRR move is ordinary
# query-set noise, while a 0.02 F1 move means structure detection actually
# changed. Improvements are never regressions, however large.
THRESHOLDS: Mapping[str, float] = {
    "recall": 0.05,
    "structure": 0.02,
    "coverage": 0.05,
    "faithfulness": 0.05,
}

# A summary below this claim-support ratio is named individually, not just
# averaged away.
FAITHFULNESS_PASS_BAR = 0.9

HIT_AT_K = 5

# Structure fidelity is defined over the nodes a detector can be expected to
# find. Front and back matter are authored in truth because queries may expect
# them, but a detector that skips the index is not wrong.
SCORED_NODE_KINDS = frozenset({"chapter", "section"})

# Leading numbering a detected heading may carry that the authored title does
# not ("1.2 Grommets" is the node titled "Grommets").
_LEADING_NUMBER = re.compile(r"^\s*\d+(?:[.\-]\d+)*[.)]?\s+")
_NON_WORD = re.compile(r"[^a-z0-9]+")


# --- the run file -----------------------------------------------------------


@dataclass(frozen=True)
class Hit:
    rank: int
    book: str
    loc: str
    record_type: str = ""


@dataclass(frozen=True)
class QueryResult:
    q: str
    hits: tuple[Hit, ...]
    # Which query set this came from: a book's own, or an area's cross-book set.
    book: str | None = None
    area: str | None = None


@dataclass(frozen=True)
class SearchParams:
    mode: str = ""
    limit: int = 0
    record_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class DetectedNode:
    title: str
    kind: str
    position: int


@dataclass(frozen=True)
class BookStructure:
    book: str
    detected: tuple[DetectedNode, ...]


@dataclass(frozen=True)
class BookCoverage:
    book: str
    surfaced: tuple[str, ...]


@dataclass(frozen=True)
class SummaryJudgement:
    book: str
    loc: str
    supported: int
    claims: int

    @property
    def ratio(self) -> float | None:
        """None when the judge found nothing to verify — which is not the same
        as a summary that failed, and must not be averaged in as zero."""
        return None if self.claims == 0 else self.supported / self.claims


@dataclass(frozen=True)
class Cost:
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    wall_seconds: float = 0.0
    per_stage: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class RunFile:
    run_id: str
    snapshot: str
    books: tuple[str, ...]
    search: SearchParams
    results: tuple[QueryResult, ...] = ()
    structure: tuple[BookStructure, ...] = ()
    coverage: tuple[BookCoverage, ...] = ()
    faithfulness: tuple[SummaryJudgement, ...] = ()
    cost: Cost | None = None


def load_run(path: Path) -> RunFile:
    if not path.exists():
        raise BenchError(f"No run file at {path}")
    try:
        document = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise BenchError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise BenchError(f"{path} should contain an object, got {type(document).__name__}")

    corpus = _mapping(document.get("corpus"))
    search = _mapping(document.get("search"))
    return RunFile(
        run_id=str(document.get("run_id", "")),
        snapshot=str(corpus.get("snapshot", "")),
        books=tuple(str(book) for book in _sequence(corpus.get("books"))),
        search=SearchParams(
            mode=str(search.get("mode", "")),
            limit=int(search.get("limit", 0) or 0),
            record_types=tuple(str(t) for t in _sequence(search.get("record_types"))),
        ),
        results=tuple(
            QueryResult(
                q=str(item.get("q", "")),
                book=_optional_str(item.get("book")),
                area=_optional_str(item.get("area")),
                hits=tuple(
                    Hit(
                        rank=int(hit.get("rank", 0) or 0),
                        book=str(hit.get("book", "")),
                        loc=str(hit.get("loc", "")),
                        record_type=str(hit.get("record_type", "")),
                    )
                    for hit in _mappings(item.get("hits"))
                ),
            )
            for item in _mappings(document.get("results"))
        ),
        structure=tuple(
            BookStructure(
                book=str(item.get("book", "")),
                detected=tuple(
                    DetectedNode(
                        title=str(node.get("title", "")),
                        kind=str(node.get("kind", "")),
                        position=int(node.get("position", 0) or 0),
                    )
                    for node in _mappings(item.get("detected"))
                ),
            )
            for item in _mappings(document.get("structure"))
        ),
        coverage=tuple(
            BookCoverage(
                book=str(item.get("book", "")),
                surfaced=tuple(str(name) for name in _sequence(item.get("surfaced"))),
            )
            for item in _mappings(document.get("coverage"))
        ),
        faithfulness=tuple(
            SummaryJudgement(
                book=str(item.get("book", "")),
                loc=str(item.get("loc", "")),
                supported=int(item.get("supported", 0) or 0),
                claims=int(item.get("claims", 0) or 0),
            )
            for item in _mappings(document.get("faithfulness"))
        ),
        cost=_load_cost(document.get("cost")),
    )


def _load_cost(raw: object) -> Cost | None:
    if not isinstance(raw, dict):
        return None
    return Cost(
        input_tokens=int(raw.get("input_tokens", 0) or 0),
        output_tokens=int(raw.get("output_tokens", 0) or 0),
        embedding_tokens=int(raw.get("embedding_tokens", 0) or 0),
        wall_seconds=float(raw.get("wall_seconds", 0.0) or 0.0),
        per_stage=tuple(_mappings(raw.get("per_stage"))),
    )


# --- scores -----------------------------------------------------------------


@dataclass(frozen=True)
class RecallScore:
    mrr: float
    hit_at_5: float
    judged: int
    per_kind: Mapping[str, "RecallScore"] = field(default_factory=dict)
    # Area sets only: was the right book anywhere in the top k, regardless of
    # which node inside it came back.
    attribution: float | None = None


@dataclass(frozen=True)
class StructureScore:
    precision: float
    recall: float
    f1: float
    per_book: Mapping[str, "StructureScore"] = field(default_factory=dict)


@dataclass(frozen=True)
class CoverageScore:
    found: int
    total: int
    per_book: Mapping[str, "CoverageScore"] = field(default_factory=dict)

    @property
    def recall(self) -> float:
        return self.found / self.total if self.total else 0.0


@dataclass(frozen=True)
class FaithfulnessScore:
    mean: float
    judged: int
    below_bar: tuple[SummaryJudgement, ...] = ()


@dataclass(frozen=True)
class Scores:
    run_id: str
    snapshot: str
    recall: RecallScore | None = None
    structure: StructureScore | None = None
    coverage: CoverageScore | None = None
    faithfulness: FaithfulnessScore | None = None
    cost: Cost | None = None
    # Everything deliberately not scored, each with a reason. Never empty
    # silently: a slice that vanished without a note is a slice that passed.
    skipped: tuple[str, ...] = ()
    unasked: tuple[str, ...] = ()

    @property
    def pooled(self) -> Mapping[str, float]:
        """The one number per dimension that a variant is judged on. Cost is
        absent on purpose — it is tracked, not scored."""
        values: dict[str, float] = {}
        if self.recall is not None and self.recall.judged:
            values["recall"] = self.recall.mrr
        if self.structure is not None:
            values["structure"] = self.structure.f1
        if self.coverage is not None and self.coverage.total:
            values["coverage"] = self.coverage.recall
        if self.faithfulness is not None and self.faithfulness.judged:
            values["faithfulness"] = self.faithfulness.mean
        return values

    def as_dict(self) -> dict[str, Any]:
        """A JSON-safe view, for writing scores beside the run they came from."""
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "snapshot": self.snapshot,
            "pooled": dict(self.pooled),
            "skipped": list(self.skipped),
            "unasked": list(self.unasked),
        }
        if self.recall is not None:
            payload["recall"] = {
                "mrr": self.recall.mrr,
                "hit_at_5": self.recall.hit_at_5,
                "judged": self.recall.judged,
                "attribution": self.recall.attribution,
                "per_kind": {
                    kind: {"mrr": slice_.mrr, "hit_at_5": slice_.hit_at_5, "judged": slice_.judged}
                    for kind, slice_ in self.recall.per_kind.items()
                },
            }
        if self.structure is not None:
            payload["structure"] = {
                "precision": self.structure.precision,
                "recall": self.structure.recall,
                "f1": self.structure.f1,
                "per_book": {
                    book: {"precision": s.precision, "recall": s.recall, "f1": s.f1}
                    for book, s in self.structure.per_book.items()
                },
            }
        if self.coverage is not None:
            payload["coverage"] = {
                "found": self.coverage.found,
                "total": self.coverage.total,
                "recall": self.coverage.recall,
            }
        if self.faithfulness is not None:
            payload["faithfulness"] = {
                "mean": self.faithfulness.mean,
                "judged": self.faithfulness.judged,
                "below_bar": [
                    {"book": entry.book, "loc": entry.loc, "ratio": entry.ratio}
                    for entry in self.faithfulness.below_bar
                ],
            }
        if self.cost is not None:
            payload["cost"] = {
                "input_tokens": self.cost.input_tokens,
                "output_tokens": self.cost.output_tokens,
                "embedding_tokens": self.cost.embedding_tokens,
                "wall_seconds": self.cost.wall_seconds,
            }
        return payload


def score(run: RunFile, truth: Truth) -> Scores:
    """Compare one run against truth. Pure; no pipeline, no corpus, no network."""
    skipped: list[str] = []
    unasked: list[str] = []

    recall = _score_recall(run, truth, skipped, unasked)
    structure = _score_structure(run, truth, skipped)
    coverage = _score_coverage(run, truth, skipped)
    faithfulness = _score_faithfulness(run, skipped)

    return Scores(
        run_id=run.run_id,
        snapshot=run.snapshot,
        recall=recall,
        structure=structure,
        coverage=coverage,
        faithfulness=faithfulness,
        cost=run.cost,
        skipped=tuple(skipped),
        unasked=tuple(unasked),
    )


# --- recall -----------------------------------------------------------------


def _score_recall(
    run: RunFile, truth: Truth, skipped: list[str], unasked: list[str]
) -> RecallScore | None:
    asked = {(result.q, result.book, result.area): result for result in run.results}
    judgements: list[tuple[str, float, float, float | None]] = []

    key: tuple[str, str | None, str | None]
    for slug, book in sorted(truth.books.items()):
        for query in book.queries:
            key = (query.q, slug, None)
            judged = _judge(query, asked.get(key), owner=slug, books=truth.books)
            _record(query, judged, key, judgements, skipped, unasked, where=f"truth/{slug}")

    for area, queries in sorted(truth.areas.items()):
        for query in queries:
            key = (query.q, None, area)
            judged = _judge(query, asked.get(key), owner=None, books=truth.books)
            _record(
                query, judged, key, judgements, skipped, unasked, where=f"truth/areas/{area}"
            )

    if not judgements:
        return None

    return _aggregate(judgements)


def _record(
    query: Query,
    judged: tuple[float, float, float | None] | None,
    key: tuple[str, str | None, str | None],
    judgements: list[tuple[str, float, float, float | None]],
    skipped: list[str],
    unasked: list[str],
    *,
    where: str,
) -> None:
    if query.gated:
        skipped.append(f"{where}: {query.q!r} is gated; not scored")
        return
    if judged is None:
        unasked.append(f"{where}: {query.q!r} was never asked by this run")
        return
    reciprocal, hit, attribution = judged
    judgements.append((query.kind, reciprocal, hit, attribution))


def _judge(
    query: Query,
    result: QueryResult | None,
    *,
    owner: str | None,
    books: Mapping[str, BookTruth],
) -> tuple[float, float, float | None] | None:
    """(reciprocal rank, hit@5, attribution) for one query, or None if unasked.

    Binary location-level judgement: a hit counts when its (book, loc) is one
    the query says genuinely answers. Best rank wins — `expects` lists every
    node that answers, not a ranking among them.
    """
    if result is None:
        return None

    wanted = {
        (expectation.book or owner or "", expectation.loc)
        for expectation in query.expects
        if not expectation.is_placeholder
    }
    wanted_books = {book for book, _ in wanted}
    if not wanted:
        return None

    best_rank: int | None = None
    best_book_rank: int | None = None
    for hit in sorted(result.hits, key=lambda h: h.rank):
        if (hit.book, hit.loc) in wanted and best_rank is None:
            best_rank = hit.rank
        if hit.book in wanted_books and best_book_rank is None:
            best_book_rank = hit.rank

    reciprocal = 1.0 / best_rank if best_rank else 0.0
    hit_at_k = 1.0 if best_rank is not None and best_rank <= HIT_AT_K else 0.0
    # Attribution is only meaningful where a query names its book — an
    # area-level question. For a book's own set the answer is always yes.
    attribution: float | None = None
    if owner is None:
        attribution = (
            1.0 if best_book_rank is not None and best_book_rank <= HIT_AT_K else 0.0
        )
    return reciprocal, hit_at_k, attribution


def _aggregate(
    judgements: Sequence[tuple[str, float, float, float | None]],
) -> RecallScore:
    per_kind: dict[str, RecallScore] = {}
    for kind in sorted({kind for kind, _, _, _ in judgements}):
        rows = [row for row in judgements if row[0] == kind]
        per_kind[kind] = RecallScore(
            mrr=_mean(row[1] for row in rows),
            hit_at_5=_mean(row[2] for row in rows),
            judged=len(rows),
        )
    attributions = [row[3] for row in judgements if row[3] is not None]
    return RecallScore(
        mrr=_mean(row[1] for row in judgements),
        hit_at_5=_mean(row[2] for row in judgements),
        judged=len(judgements),
        per_kind=per_kind,
        attribution=_mean(attributions) if attributions else None,
    )


# --- structure fidelity -----------------------------------------------------


def _score_structure(
    run: RunFile, truth: Truth, skipped: list[str]
) -> StructureScore | None:
    per_book: dict[str, StructureScore] = {}
    matched_total = detected_total = expected_total = 0

    for entry in run.structure:
        book = truth.books.get(entry.book)
        if book is None:
            skipped.append(f"structure: run names book {entry.book!r}, absent from truth")
            continue

        expected = [node for node in book.nodes.values() if node.kind in SCORED_NODE_KINDS]
        detected = [node for node in entry.detected if node.kind in SCORED_NODE_KINDS]
        matched = _match_nodes(expected, detected)

        matched_total += matched
        detected_total += len(detected)
        expected_total += len(expected)
        per_book[entry.book] = _prf(matched, len(detected), len(expected))

    if not per_book:
        return None
    pooled = _prf(matched_total, detected_total, expected_total)
    return StructureScore(
        precision=pooled.precision, recall=pooled.recall, f1=pooled.f1, per_book=per_book
    )


def _match_nodes(expected: Sequence[Any], detected: Sequence[DetectedNode]) -> int:
    """How many authored nodes the detector found.

    By normalised title, with position as the tiebreak when a title occurs more
    than once — a book with two sections called "Summary" is ordinary, and
    matching them by name alone would let one detected node satisfy both.
    """
    remaining: dict[tuple[str, str], list[Any]] = {}
    for node in expected:
        remaining.setdefault((node.kind, _normalise(node.title)), []).append(node)

    matched = 0
    for node in sorted(detected, key=lambda n: n.position):
        candidates = remaining.get((node.kind, _normalise(node.title)))
        if candidates:
            candidates.pop(0)
            matched += 1
    return matched


def _prf(matched: int, detected: int, expected: int) -> StructureScore:
    precision = matched / detected if detected else 0.0
    recall = matched / expected if expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return StructureScore(precision=precision, recall=recall, f1=f1)


def _normalise(title: str) -> str:
    """Compare headings the way a reader would: ignore leading numbering, case,
    punctuation and spacing."""
    return _NON_WORD.sub(" ", _LEADING_NUMBER.sub("", title).lower()).strip()


# --- extraction coverage ----------------------------------------------------


def _score_coverage(run: RunFile, truth: Truth, skipped: list[str]) -> CoverageScore | None:
    per_book: dict[str, CoverageScore] = {}
    found_total = expected_total = 0

    for entry in run.coverage:
        book = truth.books.get(entry.book)
        if book is None:
            skipped.append(f"coverage: run names book {entry.book!r}, absent from truth")
            continue

        surfaced = {_normalise(name) for name in entry.surfaced}
        found = sum(
            1
            for concept in book.concepts
            if surfaced & {_normalise(name) for name in (concept.concept, *concept.accept)}
        )
        found_total += found
        expected_total += len(book.concepts)
        per_book[entry.book] = CoverageScore(found=found, total=len(book.concepts))

    if not per_book:
        return None
    return CoverageScore(found=found_total, total=expected_total, per_book=per_book)


# --- summary faithfulness ---------------------------------------------------


def _score_faithfulness(run: RunFile, skipped: list[str]) -> FaithfulnessScore | None:
    ratios: list[float] = []
    below: list[SummaryJudgement] = []
    for judgement in run.faithfulness:
        ratio = judgement.ratio
        if ratio is None:
            skipped.append(
                f"faithfulness: {judgement.book}/{judgement.loc} had no claims to verify"
            )
            continue
        ratios.append(ratio)
        if ratio < FAITHFULNESS_PASS_BAR:
            below.append(judgement)

    if not ratios:
        return None
    return FaithfulnessScore(mean=_mean(ratios), judged=len(ratios), below_bar=tuple(below))


# --- regression -------------------------------------------------------------


@dataclass(frozen=True)
class Change:
    dimension: str
    baseline: float
    candidate: float

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline


@dataclass(frozen=True)
class Verdict:
    changes: tuple[Change, ...]
    regressions: tuple[Change, ...]
    # Dimensions one run scored and the other did not — not a regression, but
    # it does mean the two runs are not fully comparable.
    incomparable: tuple[str, ...] = ()

    @property
    def is_regression(self) -> bool:
        return bool(self.regressions)


def compare(baseline: Scores, candidate: Scores) -> Verdict:
    """Regression verdict over the pooled aggregates only.

    Per-slice numbers say *where* something moved and are deliberately not
    consulted here: a per-area slice is a handful of queries, and letting one
    of them veto a release would make the verdict noise.
    """
    base_pooled, cand_pooled = baseline.pooled, candidate.pooled
    changes: list[Change] = []
    regressions: list[Change] = []

    for dimension in sorted(set(base_pooled) & set(cand_pooled)):
        change = Change(dimension, base_pooled[dimension], cand_pooled[dimension])
        changes.append(change)
        if -change.delta > THRESHOLDS[dimension]:
            regressions.append(change)

    incomparable = sorted(set(base_pooled) ^ set(cand_pooled))
    return Verdict(
        changes=tuple(changes),
        regressions=tuple(regressions),
        incomparable=tuple(incomparable),
    )


# --- helpers ----------------------------------------------------------------


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return math.fsum(collected) / len(collected)


def _mapping(raw: object) -> Mapping[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _sequence(raw: object) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _mappings(raw: object) -> list[Mapping[str, Any]]:
    return [item for item in _sequence(raw) if isinstance(item, dict)]


def _optional_str(raw: object) -> str | None:
    return None if raw is None else str(raw)
