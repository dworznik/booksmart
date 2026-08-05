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
from collections.abc import Iterable, Mapping, Sequence
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

# Everything a toc.yaml entry is allowed to say. Only a chapter takes children —
# a section carrying `sections` is authoring a third level this schema has no
# way to read, which is worth reporting rather than ignoring.
ENTRY_KEYS: frozenset[str] = frozenset({"id", "title"})
CHAPTER_KEYS: frozenset[str] = ENTRY_KEYS | {"sections"}


# --- the shapes -------------------------------------------------------------


@dataclass(frozen=True)
class TocNode:
    id: str
    title: str
    kind: NodeKind
    parent: str | None = None


@dataclass(frozen=True)
class StrayKeys:
    """A node that loaded, and whose entry said something the schema cannot read.

    Deliberately not folded into ``malformed_nodes``, which means an entry that
    never became a node at all. This one did become a node, so nothing downstream
    will hesitate over it — which is what makes it worth a finding of its own.
    """

    node_id: str
    keys: tuple[str, ...]


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
    # Stated in book.yaml so a handover can be checked against it. Edition is the
    # expensive thing to get wrong — chapter numbering moves between editions, so
    # truth authored for one scores the other as a pipeline that lost half the
    # book — and these three are what a file says about itself.
    edition: str = ""
    year: str = ""
    isbn: str = ""

    queries: tuple[Query, ...] = ()
    concepts: tuple[Concept, ...] = ()
    index_pairs: tuple[IndexPair, ...] = ()
    # Ids that appeared more than once in toc.yaml; the loader keeps the first
    # and records the collision so the lint can report it.
    duplicate_ids: tuple[str, ...] = ()
    # Entries the loader could not turn into nodes at all, described well enough
    # for the lint to name them.
    malformed_nodes: tuple[str, ...] = ()
    # Nodes whose entry carried keys the schema does not define. Kept apart from
    # malformed_nodes because these did load.
    stray_keys: tuple[StrayKeys, ...] = ()

    @property
    def is_pinned(self) -> bool:
        """Whether truth is tied to specific bytes yet. Until the source lands,
        book.yaml carries a placeholder where the hash will go."""
        pinned = self.source_sha256
        if pinned is None:
            return False
        return not pinned.upper().startswith(PLACEHOLDER)


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


@dataclass(frozen=True)
class _Toc:
    """One toc.yaml, and everything wrong with it.

    A named carrier rather than a tuple of four, because three of the four are
    problem lists and a caller unpacking them positionally is one edit away from
    reporting duplicates as malformations.
    """

    nodes: dict[str, TocNode]
    duplicate_ids: tuple[str, ...]
    malformed: tuple[str, ...]
    stray_keys: tuple[StrayKeys, ...]


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
    toc = _read_toc(directory / "toc.yaml")
    raw_source = identity.get("source")
    source: Mapping[str, object] = raw_source if isinstance(raw_source, dict) else {}
    sha256 = source.get("sha256")
    source_file = source.get("file")

    return BookTruth(
        slug=str(identity.get("slug") or directory.name),
        title=str(identity.get("title", "")),
        area=str(identity.get("area", "")),
        source_sha256=None if sha256 is None else str(sha256),
        source_file=None if source_file is None else str(source_file),
        authors=_strings(identity, "authors"),
        edition=str(identity.get("edition", "")),
        year=str(identity.get("publication_year", "")),
        isbn=str(identity.get("isbn", "")),
        nodes=toc.nodes,
        duplicate_ids=toc.duplicate_ids,
        malformed_nodes=toc.malformed,
        stray_keys=toc.stray_keys,
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


def _read_toc(path: Path) -> _Toc:
    document = _read_mapping(path)
    nodes: dict[str, TocNode] = {}
    duplicates: list[str] = []
    malformed: list[str] = []
    stray: list[StrayKeys] = []

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
        # Only for entries that became nodes. An id-less or duplicated entry is
        # already reported by something that names it, and two findings for one
        # line says less than one.
        allowed = CHAPTER_KEYS if kind == "chapter" else ENTRY_KEYS
        extra = tuple(sorted(set(entry) - allowed))
        if extra:
            stray.append(StrayKeys(node_id=node_id, keys=extra))
        return node_id

    for entry in _entries(document, "front_matter"):
        add(entry, "front-matter")
    for chapter in _entries(document, "chapters"):
        chapter_id = add(chapter, "chapter")
        for section in _entries(chapter, "sections"):
            add(section, "section", parent=chapter_id)
    for entry in _entries(document, "back_matter"):
        add(entry, "back-matter")

    return _Toc(
        nodes=nodes,
        duplicate_ids=tuple(duplicates),
        malformed=tuple(malformed),
        stray_keys=tuple(stray),
    )


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
    findings.extend(_lint_stray_keys(f"{where}/toc.yaml", book))
    if not book.is_pinned:
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


def _lint_stray_keys(where: str, book: BookTruth) -> list[Finding]:
    """A node whose entry says something the schema cannot read.

    An error rather than a warning, because the likely cause does not merely add
    something unused — it takes something away. Entries are written as YAML flow
    mappings, one per line, where a comma separates entries:

        - { id: "4.1", title: The call, apply, and bind methods }

    That is not one entry with two keys. It is a title of "The call" plus two
    keys named after the rest of the heading, and the node loads with a title
    that is a fragment of the real one. Structure fidelity matches on normalised
    title, so the node can never match the section it describes — it scores a
    false negative and a false positive on every run, and reads as a detection
    problem rather than a quoting mistake. Nothing else here can see it: the
    loader ignores keys it does not know, which is the right default for
    hand-authored files and exactly wrong for this one.

    So the finding names the keys, shows the title as it was recorded, and says
    what to do. Reporting only the stray keys would leave the reader to work out
    why they mattered, which is the part that is hard to see.
    """
    findings: list[Finding] = []
    for stray in book.stray_keys:
        listed = ", ".join(repr(key) for key in stray.keys)
        noun = "key" if len(stray.keys) == 1 else "keys"
        findings.append(
            Finding(
                "error",
                where,
                f"node {stray.node_id!r} has {noun} the schema does not define "
                f"({listed}), and its title is recorded as "
                f"{book.nodes[stray.node_id].title!r}; if that title is cut short, "
                "quote it — an unquoted comma in a flow-mapping title is read as an "
                "entry separator",
            )
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
