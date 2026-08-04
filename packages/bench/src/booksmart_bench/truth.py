"""Reading hand-authored ground truth, and checking that it says what it claims.

Truth is authored by hand, in a different repository, against books this repo
never sees. That makes two things true at once: it is the only thing anchoring a
benchmark to reality, and it is the easiest thing to get quietly wrong. A `loc:`
that no longer names a real node scores zero forever and looks like a retrieval
regression. A query filed as "conceptual" that happens to share the section's
own vocabulary turns the one slice measuring semantic retrieval into a lexical
one.

So the guards travel with the harness that reads truth, and every verb runs them
before spending anything.

The kind guards are ported from the hybrid-eval fixture set in core's suite,
with one adaptation forced by the setting: there, a query could be checked
against the full text of the record it expected. Here truth points at a ToC node
and the book's text lives in a private checkout, so the comparison is against
the node's *title* — a weaker check, but the one that catches the mistake that
actually gets made, which is writing a paraphrase around the words already in
the heading.

Severity is split deliberately. An **error** is truth that cannot be scored
(a location nothing satisfies, a kind that does not exist). A **warning** is
truth that is incomplete but honest — an unpinned source, a `TBD` placeholder
for a book whose truth has not been authored yet — because area queries are
written before every book in the area exists, and a tree in that state must stay
loadable.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

import yaml

from booksmart_bench.errors import BenchError

QueryKind = Literal["exact-term", "conceptual", "mixed"]
QUERY_KINDS: frozenset[str] = frozenset(get_args(QueryKind))

NodeKind = Literal["front-matter", "chapter", "section", "back-matter"]

Severity = Literal["error", "warning"]

# Below this, a per-kind mean is one lucky query away from moving. Same figure
# the hybrid-eval fixture guard used, and for the same reason.
MIN_QUERIES_PER_KIND = 4

# An exact-term query is a coined phrase, not a sentence. Longer than this and
# the label is almost always wrong.
MAX_EXACT_TERM_WORDS = 6

# The marker an author leaves when a location cannot be filled in yet.
PLACEHOLDER = "TBD"

BOOK_FILES = ("book.yaml", "toc.yaml", "queries.yaml", "concepts.yaml", "index-pairs.yaml")


# --- the shapes -------------------------------------------------------------


@dataclass(frozen=True)
class TocNode:
    id: str
    title: str
    kind: NodeKind
    parent: str | None = None


@dataclass(frozen=True)
class Expectation:
    loc: str
    # Set only on area queries, which must say which book they mean.
    book: str | None = None

    @property
    def is_placeholder(self) -> bool:
        return self.loc.upper().startswith(PLACEHOLDER)


@dataclass(frozen=True)
class Query:
    q: str
    kind: str  # validated by the lint, not the loader, so a typo is reportable
    expects: tuple[Expectation, ...]
    why: str
    # Specced now, scoreable later: a slice waiting on a pipeline feature is
    # written into truth from the start and marked, so the scorer skips and
    # reports it rather than counting a miss it could never have hit.
    gated: bool = False


@dataclass(frozen=True)
class Concept:
    concept: str
    loc: str
    accept: tuple[str, ...] = ()


@dataclass(frozen=True)
class IndexPair:
    term: str
    loc: str
    gated: bool = False


@dataclass(frozen=True)
class BookTruth:
    slug: str
    title: str
    area: str
    source_sha256: str | None
    nodes: dict[str, TocNode]
    # Where the artifact is expected, relative to the assets root. Named in
    # book.yaml rather than derived, because a book's file extension is a
    # property of the artifact, not of the slug.
    source_file: str | None = None
    authors: tuple[str, ...] = ()
    queries: tuple[Query, ...] = ()
    concepts: tuple[Concept, ...] = ()
    index_pairs: tuple[IndexPair, ...] = ()
    # Ids that appeared more than once in toc.yaml; the loader keeps the first
    # and records the collision so the lint can report it.
    duplicate_ids: tuple[str, ...] = ()
    # Entries the loader could not turn into nodes at all, described well enough
    # for the lint to name them.
    malformed_nodes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Truth:
    assets: Path
    books: dict[str, BookTruth] = field(default_factory=dict)
    areas: dict[str, tuple[Query, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Finding:
    severity: Severity
    where: str
    message: str


# --- loading ----------------------------------------------------------------


def load_truth(assets: Path) -> Truth:
    """Read every book and area under ``<assets>/truth``."""
    root = assets / "truth"
    if not root.is_dir():
        raise BenchError(f"No truth/ directory under {assets}")

    books: dict[str, BookTruth] = {}
    areas: dict[str, tuple[Query, ...]] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "areas":
            areas = _load_areas(child)
        else:
            books[child.name] = _load_book(child)
    return Truth(assets=assets, books=books, areas=areas)


def _load_book(directory: Path) -> BookTruth:
    identity = _read_mapping(directory / "book.yaml")
    nodes, duplicates, malformed = _read_toc(directory / "toc.yaml")
    source = identity.get("source") if isinstance(identity.get("source"), dict) else {}
    assert isinstance(source, dict)
    sha256 = source.get("sha256")
    source_file = source.get("file")

    return BookTruth(
        slug=str(identity.get("slug") or directory.name),
        title=str(identity.get("title", "")),
        area=str(identity.get("area", "")),
        source_sha256=None if sha256 is None else str(sha256),
        source_file=None if source_file is None else str(source_file),
        authors=_strings(identity, "authors"),
        nodes=nodes,
        duplicate_ids=duplicates,
        malformed_nodes=malformed,
        queries=_read_queries(directory / "queries.yaml"),
        concepts=tuple(
            Concept(
                concept=str(item.get("concept", "")),
                loc=str(item.get("loc", "")),
                accept=_strings(item, "accept"),
            )
            for item in _read_sequence(directory / "concepts.yaml")
        ),
        index_pairs=tuple(
            IndexPair(
                term=str(item.get("term", "")),
                loc=str(item.get("loc", "")),
                gated=bool(item.get("gated", False)),
            )
            for item in _read_sequence(directory / "index-pairs.yaml")
        ),
    )


def _load_areas(directory: Path) -> dict[str, tuple[Query, ...]]:
    return {
        area.name: _read_queries(area / "queries.yaml")
        for area in sorted(directory.iterdir())
        if area.is_dir()
    }


def _read_toc(path: Path) -> tuple[dict[str, TocNode], tuple[str, ...], tuple[str, ...]]:
    document = _read_mapping(path)
    nodes: dict[str, TocNode] = {}
    duplicates: list[str] = []
    malformed: list[str] = []

    def add(entry: dict[str, object], kind: NodeKind, parent: str | None = None) -> str | None:
        """Record one entry, or note why it could not be recorded.

        An entry with no id is reported rather than raised: it is exactly the
        kind of hand-authoring slip the lint exists to catch, and a traceback
        would say less about it than a finding naming the file and the title.
        """
        title = str(entry.get("title", ""))
        if "id" not in entry:
            malformed.append(f"{kind} entry {title or '<untitled>'!r} has no id")
            return None
        node_id = str(entry["id"])
        if node_id in nodes:
            duplicates.append(node_id)
            return None
        nodes[node_id] = TocNode(id=node_id, title=title, kind=kind, parent=parent)
        return node_id

    for entry in _entries(document, "front_matter"):
        add(entry, "front-matter")
    for chapter in _entries(document, "chapters"):
        chapter_id = add(chapter, "chapter")
        for section in _entries(chapter, "sections"):
            add(section, "section", parent=chapter_id)
    for entry in _entries(document, "back_matter"):
        add(entry, "back-matter")

    return nodes, tuple(duplicates), tuple(malformed)


def _read_queries(path: Path) -> tuple[Query, ...]:
    return tuple(
        Query(
            q=str(item.get("q", "")),
            kind=str(item.get("kind", "")),
            expects=tuple(
                Expectation(
                    loc=str(expectation.get("loc", "")),
                    book=None if expectation.get("book") is None else str(expectation["book"]),
                )
                for expectation in _entries(item, "expects")
            ),
            why=str(item.get("why", "")),
            gated=bool(item.get("gated", False)),
        )
        for item in _read_sequence(path)
    )


def _entries(document: dict[str, object], key: str) -> list[dict[str, object]]:
    """The mappings under ``key``, tolerating an absent, null, or malformed list.

    Structural sloppiness is skipped here rather than raised: the lint reports
    what is missing far more usefully than a loader traceback can.
    """
    value = document.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _strings(document: dict[str, object], key: str) -> tuple[str, ...]:
    value = document.get(key)
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _read_yaml(path: Path, *, required: bool) -> object:
    if not path.exists():
        if required:
            raise BenchError(f"Missing {path.name} in {path.parent}")
        return None
    try:
        return yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise BenchError(f"Could not parse {path}: {exc}") from exc


def _read_mapping(path: Path) -> dict[str, object]:
    document = _read_yaml(path, required=True)
    if not isinstance(document, dict):
        raise BenchError(f"{path} should contain a mapping, got {type(document).__name__}")
    return document


def _read_sequence(path: Path) -> list[dict[str, object]]:
    document = _read_yaml(path, required=False)
    if document is None:
        return []
    if not isinstance(document, list):
        raise BenchError(f"{path} should contain a list, got {type(document).__name__}")
    return [item for item in document if isinstance(item, dict)]


# --- the guards -------------------------------------------------------------


def lint(truth: Truth) -> tuple[Finding, ...]:
    """Every problem in a truth tree, worst kind first by severity at report time."""
    findings: list[Finding] = []
    for slug, book in sorted(truth.books.items()):
        findings.extend(_lint_book(slug, book))
    for area, queries in sorted(truth.areas.items()):
        findings.extend(_lint_area(area, queries, truth.books))
    return tuple(findings)


def errors_in(findings: Iterable[Finding]) -> tuple[Finding, ...]:
    return tuple(finding for finding in findings if finding.severity == "error")


def _lint_book(slug: str, book: BookTruth) -> list[Finding]:
    findings: list[Finding] = []
    where = f"truth/{slug}"

    for duplicate in book.duplicate_ids:
        findings.append(
            Finding("error", f"{where}/toc.yaml", f"duplicate node id {duplicate!r}")
        )
    for malformed in book.malformed_nodes:
        findings.append(Finding("error", f"{where}/toc.yaml", malformed))
    if not book.source_sha256 or book.source_sha256.upper().startswith(PLACEHOLDER):
        findings.append(
            Finding(
                "warning",
                f"{where}/book.yaml",
                "source sha256 is not pinned; truth cannot be tied to an edition yet",
            )
        )

    findings.extend(_lint_queries(f"{where}/queries.yaml", book.queries, book, {slug: book}))
    findings.extend(_lint_kind_coverage(f"{where}/queries.yaml", book.queries))

    for concept in book.concepts:
        if concept.loc not in book.nodes:
            findings.append(
                Finding(
                    "error",
                    f"{where}/concepts.yaml",
                    f"{concept.concept!r} names unknown location {concept.loc!r}",
                )
            )
    for pair in book.index_pairs:
        if pair.loc not in book.nodes:
            findings.append(
                Finding(
                    "error",
                    f"{where}/index-pairs.yaml",
                    f"{pair.term!r} names unknown location {pair.loc!r}",
                )
            )
    findings.extend(
        _lint_gated(f"{where}/index-pairs.yaml", [p.gated for p in book.index_pairs])
    )
    return findings


def _lint_area(area: str, queries: Sequence[Query], books: dict[str, BookTruth]) -> list[Finding]:
    where = f"truth/areas/{area}/queries.yaml"
    findings = _lint_queries(where, queries, None, books)
    findings.extend(_lint_kind_coverage(where, queries))
    return findings


def _lint_queries(
    where: str,
    queries: Sequence[Query],
    owner: BookTruth | None,
    books: dict[str, BookTruth],
) -> list[Finding]:
    findings: list[Finding] = []
    for query in queries:
        if query.kind not in QUERY_KINDS:
            findings.append(
                Finding(
                    "error",
                    where,
                    f"{query.q!r} has unknown kind {query.kind!r}; "
                    f"expected one of {', '.join(sorted(QUERY_KINDS))}",
                )
            )
        if not query.why.strip():
            findings.append(Finding("error", where, f"{query.q!r} has no why-line"))
        if not query.expects:
            findings.append(Finding("error", where, f"{query.q!r} expects nothing"))
        if query.kind == "exact-term" and len(query.q.split()) > MAX_EXACT_TERM_WORDS:
            findings.append(
                Finding(
                    "warning",
                    where,
                    f"{query.q!r} is filed as exact-term but reads as a sentence "
                    f"({len(query.q.split())} words); did it mean mixed?",
                )
            )

        for expectation in query.expects:
            # An area query with no book cannot be resolved at all, and saying
            # so once is clearer than also reporting the location it could never
            # have looked up.
            if owner is None and expectation.book is None:
                findings.append(
                    Finding(
                        "error",
                        where,
                        f"{query.q!r} has an expectation with no book; area queries must "
                        "say which book they mean",
                    )
                )
                continue

            target = _resolve(expectation, owner, books)
            if expectation.is_placeholder:
                findings.append(
                    Finding(
                        "warning",
                        where,
                        f"{query.q!r} has a {PLACEHOLDER} location for "
                        f"{expectation.book or 'this book'}; unscoreable until it is filled in",
                    )
                )
                continue
            if target is None:
                findings.append(
                    Finding(
                        "error",
                        where,
                        f"{query.q!r} names unknown "
                        + (
                            f"book {expectation.book!r}"
                            if expectation.book is not None and expectation.book not in books
                            else f"location {expectation.loc!r}"
                        ),
                    )
                )
                continue
            findings.extend(_lint_kind_claim(where, query, expectation, target))
    return findings


def _lint_kind_claim(
    where: str, query: Query, expectation: Expectation, target: TocNode
) -> list[Finding]:
    """A conceptual query must not reuse the words of the node it expects.

    Checked against the node's title rather than its text: the book itself is
    not in this repo, and the heading is where the shared-vocabulary mistake
    almost always comes from.
    """
    if query.kind != "conceptual":
        return []
    overlap = content_words(query.q) & content_words(target.title)
    if not overlap:
        return []
    return [
        Finding(
            "error",
            where,
            f"conceptual query {query.q!r} shares {', '.join(sorted(overlap))} with the "
            f"title of {expectation.loc!r}; lexical retrieval can find it, so the slice "
            "no longer measures semantic recall",
        )
    ]


def _lint_kind_coverage(where: str, queries: Sequence[Query]) -> list[Finding]:
    """Enough queries of each kind for a per-kind mean to mean something.

    An empty set gets one finding rather than three: a book whose queries are
    not authored yet is a different situation from one whose set is lopsided,
    and reporting it as three separate shortfalls buries that.
    """
    if not queries:
        return [Finding("warning", where, "no queries authored")]

    findings: list[Finding] = []
    findings.extend(_lint_gated(where, [query.gated for query in queries]))
    for kind in sorted(QUERY_KINDS):
        count = sum(1 for query in queries if query.kind == kind)
        if count < MIN_QUERIES_PER_KIND:
            findings.append(
                Finding(
                    "warning",
                    where,
                    f"only {count} {kind} queries (want {MIN_QUERIES_PER_KIND}); "
                    "too few for the per-kind mean to be stable",
                )
            )
    return findings


def _lint_gated(where: str, flags: Sequence[bool]) -> list[Finding]:
    """Report gated entries as a count, not one finding each.

    A gated slice is correct truth waiting on a pipeline feature, so it is not a
    problem to fix — but it does mean the file scores less than it looks like it
    scores, and that has to be visible every run.
    """
    gated = sum(1 for flag in flags if flag)
    if not gated:
        return []
    return [
        Finding(
            "warning",
            where,
            f"{gated} gated entr{'y' if gated == 1 else 'ies'}; "
            "skipped by the scorer until the feature they wait on lands",
        )
    ]


def _resolve(
    expectation: Expectation, owner: BookTruth | None, books: dict[str, BookTruth]
) -> TocNode | None:
    book = owner if expectation.book is None else books.get(expectation.book)
    if book is None:
        return None
    return book.nodes.get(expectation.loc)


# --- lexical comparison -----------------------------------------------------
#
# Ported verbatim in spirit from the hybrid-eval guards: the point is to compare
# words the way a lexical retriever would, so a query cannot claim to share no
# vocabulary while differing only in inflection.

_WORD = re.compile(r"[a-z]+")

# Longest first, so "ing" is tried before "s".
_SUFFIXES = ("ations", "ation", "ings", "ing", "edly", "ers", "er", "ed", "es", "s")


def _stem(word: str) -> str:
    """A crude suffix-stripper, deliberately over-eager.

    Not real stemming. BM25 implementations run a Snowball stemmer, so a query
    saying "grommet" against a heading saying "Grommets" would match lexically
    even though an exact comparison sees two words. Over-stemming only makes the
    guard stricter, which is the safe direction for a check whose whole job is
    to stop a query claiming to be something it is not.
    """
    for suffix in _SUFFIXES:
        if len(word) - len(suffix) >= 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def content_words(text: str) -> set[str]:
    """Stems of the lower-cased words of 4+ letters — a rough stand-in for
    "content word"."""
    return {_stem(word) for word in _WORD.findall(text.lower()) if len(word) >= 4}
