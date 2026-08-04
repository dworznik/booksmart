"""The record->location join: what a run writer owes the scorer.

A run file carries ToC node ids, never record ids (SPEC §1). Somebody has to
perform that translation, and it has to be the run writer: the scorer is a pure
comparison of two location sets, and a scorer that resolved record ids would
need the corpus those ids live in — which is exactly the coupling the
three-artefact split exists to prevent.

The join is by normalised title with position as the tiebreak, the rule SPEC §3
sets so truth survives re-ingestion: node ids are the book's own labels, and a
re-parse that renumbers nothing but re-detects everything still lands on the
same nodes. Two refinements the naive version lacks:

- **Sections resolve within their chapter.** A book with "Summary" at the end of
  every chapter is ordinary; matching section titles globally would attribute a
  hit to whichever chapter happened to be authored first.
- **A section under an unresolved chapter falls back to a unique title.** A
  chapter whose heading the parser mangled would otherwise cost every section
  beneath it, and a location that cannot be resolved scores exactly like a
  retrieval miss. The fallback fires only when the title is unambiguous across
  the unclaimed nodes — a guess would be worse than the miss.

Nothing is resolved twice. One detected node claims one authored node, so a
book cannot score above the structure it actually has.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from booksmart_bench.scoring import normalise_title
from booksmart_bench.truth import BookTruth, TocNode

# Truth's node kinds that a detected chapter-level row may claim. The detector
# classifies matter by keyword and the author by hand, so the two disagree
# regularly; the disagreement is the structure dimension's business, and making
# it cost the *location* as well would double-count one mistake.
CHAPTER_LEVEL_KINDS = frozenset({"chapter", "front-matter", "back-matter"})


@dataclass(frozen=True)
class Detected:
    """One node of a book's detected structure, as stored.

    ``position`` is document order across the whole book (not the within-chapter
    index the Section row carries), because both the tiebreak here and the
    scorer's own matching read it as "which came first in the book".
    """

    record_id: str
    title: str
    kind: str  # "chapter" | "section"
    position: int
    # The chapter row a section belongs to; None for a chapter-level node.
    parent_id: str | None = None


@dataclass(frozen=True)
class LocationMap:
    """Which authored node each record turned out to be, and what did not join."""

    by_record: Mapping[str, str] = field(default_factory=dict)
    # One line per detected node that no authored node could claim, said well
    # enough for a run to report it. A silent drop here reads downstream as a
    # retrieval miss, which is the one wrong story this must not tell.
    unresolved: tuple[str, ...] = ()

    def node_for(self, record_id: str | None) -> str | None:
        if record_id is None:
            return None
        return self.by_record.get(record_id)


def join(detected: Sequence[Detected], book: BookTruth) -> LocationMap:
    """Resolve a book's detected structure against its authored ToC."""
    chapter_pool = _pool(
        authored for authored in book.nodes.values() if authored.kind in CHAPTER_LEVEL_KINDS
    )
    section_pool: dict[str | None, dict[str, list[str]]] = {}
    for authored in book.nodes.values():
        if authored.kind == "section":
            section_pool.setdefault(authored.parent, {}).setdefault(
                normalise_title(authored.title), []
            ).append(authored.id)

    ordered = sorted(detected, key=lambda row: row.position)
    resolved: dict[str, str] = {}
    unresolved: list[str] = []

    # Chapters first, in document order, so a section's parent is already known
    # by the time the section is looked at — whatever order the caller passed.
    for row in ordered:
        if row.kind == "section":
            continue
        claimed = _claim(chapter_pool, row.title)
        if claimed is None:
            unresolved.append(f"chapter {row.title!r} matches no authored node")
            continue
        resolved[row.record_id] = claimed

    orphans: list[Detected] = []
    for row in ordered:
        if row.kind != "section":
            continue
        parent_node = resolved.get(row.parent_id) if row.parent_id is not None else None
        if parent_node is None:
            # Its chapter never resolved; the fallback below may still place it.
            orphans.append(row)
            continue
        claimed = _claim(section_pool.get(parent_node, {}), row.title)
        if claimed is None:
            unresolved.append(
                f"section {row.title!r} matches no authored node under {parent_node!r}"
            )
            continue
        resolved[row.record_id] = claimed

    for row in orphans:
        claimed = _claim_unique(section_pool, row.title)
        if claimed is None:
            unresolved.append(
                f"section {row.title!r} matches no authored node "
                "(its chapter did not resolve either)"
            )
            continue
        resolved[row.record_id] = claimed

    return LocationMap(by_record=resolved, unresolved=tuple(unresolved))


def _pool(nodes: Iterable[TocNode]) -> dict[str, list[str]]:
    """Authored nodes by normalised title, each list in ToC order."""
    pool: dict[str, list[str]] = {}
    for node in nodes:
        pool.setdefault(normalise_title(node.title), []).append(node.id)
    return pool


def _claim(pool: dict[str, list[str]], title: str) -> str | None:
    """The first unclaimed node with this title, removed from the pool."""
    candidates = pool.get(normalise_title(title))
    if not candidates:
        return None
    return candidates.pop(0)


def _claim_unique(
    pools: Mapping[str | None, dict[str, list[str]]], title: str
) -> str | None:
    """The one unclaimed section with this title anywhere in the book.

    Only fires for a section whose chapter did not resolve, and only when the
    title is unique among what is left: with two candidates there is no evidence
    to choose between them, and inventing one would put a hit on a location the
    search never returned.
    """
    normalised = normalise_title(title)
    found = [
        (chapter, pool[normalised])
        for chapter, pool in pools.items()
        if pool.get(normalised)
    ]
    if len(found) != 1:
        return None
    _, candidates = found[0]
    if len(candidates) != 1:
        return None
    return candidates.pop(0)
