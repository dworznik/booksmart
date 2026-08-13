"""What the extraction gate measures: one row per book, from a GFM artifact and
the document it came from.

Two halves, because two different things can go wrong.

**Loss** is measured by comparing the artifact to the document's text layer.
The gated number is `longest_dropped_run` — the longest *contiguous* stretch of
source text absent from the output — and not the coverage percentage. That is
deliberate: the pipeline is supposed to drop running headers, footers and page
numbers, so coverage below 100% is the normal case and a target for it would be
a wrong one. Furniture is short and scattered; real loss is contiguous — a
185-line appendix table, a run of preface paragraphs. Gating the longest
unmatched run separates the two with no furniture model at all, and hands back a
quotable excerpt instead of a number nobody can act on.

**Shape** is counted off the artifact: fences, headings, blocks by kind, the
share of characters inside code, and `suspect_headings` — code-shaped ATX titles,
which are the measured symptom of fencing having failed (ADR 0003).

What no reader of the artifact can recover — which rule fenced a block, whether
the extractor declined and why, how many spine documents it skipped — arrives as
the extractor's own testimony in an `ExtractorReport`.

Ratcheting, capping and diffing against a recorded baseline happen in
`booksmart-bench`; nothing here has an opinion about whether a number is good.
"""

import bisect
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import pymupdf

from booksmart_core.parsing.blocks import looks_like_code
from booksmart_core.parsing.contract import ExtractorReport
from booksmart_core.parsing.mupdf import quiet_mupdf
# The one table of characters that are in the text without being in the word.
# This module had its own copy; two copies of a rule about characters nothing
# renders would drift without either side ever looking different.
from booksmart_core.titles import INVISIBLE as _INVISIBLE


# A word split across a line end by a typesetter's hyphen. Joined before
# comparison, because an extractor that de-hyphenates correctly would otherwise
# read as losing a fragment of every wrapped word — noise deep enough to hide a
# real loss under it.
# U+2010 hyphen and U+00AD soft hyphen as escapes, beside the plain one: a
# reader cannot tell the three apart in a diff, and a class that silently
# lost one would stop joining a wrapped word without looking any different.
_HYPHEN_BREAK = re.compile("[-\u2010\u00ad][ \t]*\n[ \t]*")

_WORD = re.compile(r"[0-9a-z]+")

FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _normalise(text: str) -> str:
    """Case-folded, invisibles removed, hyphenated line breaks healed.

    Offsets into the result are what excerpts are quoted from, so the excerpt a
    dropped run reports is lower-case and joined. That is a deliberate trade: a
    quotable-but-normalised excerpt is worth more than an exact one that has to
    be located again by hand.
    """
    return _HYPHEN_BREAK.sub("", text.translate(_INVISIBLE)).lower()


@dataclass(frozen=True)
class _Word:
    text: str
    start: int
    end: int


def _words(normalised: str) -> list[_Word]:
    return [
        _Word(match.group(), match.start(), match.end())
        for match in _WORD.finditer(normalised)
    ]


@dataclass(frozen=True)
class DroppedRun:
    """One contiguous stretch of source text with no counterpart in the output.

    ``excerpt`` is the point. A number says a gate failed; the text says which
    appendix table went missing, which is what somebody can actually fix.
    """

    characters: int
    excerpt: str


# How much of a dropped run to quote. Long enough to recognise the passage,
# short enough that a report of a badly broken book stays readable.
EXCERPT_CHARACTERS = 220


def _anchors(source: Sequence[_Word], output: Sequence[_Word]) -> list[tuple[int, int]]:
    """Word positions that pin the two sequences together, in order.

    Words occurring exactly once on each side are unambiguous joins, so they can
    be paired without any alignment search. The longest increasing subsequence of
    those pairs is the largest set of them that agrees on document order;
    everything else is compared *between* consecutive anchors.

    This is what keeps the comparison local. Matching by multiset alone would
    call a dropped code listing "present" because its identifiers also occur in
    the index sixty pages later — and that is precisely the loss this metric
    exists to catch. Locality without a fitted window, because the anchors come
    from the documents rather than from a constant somebody tuned.
    """
    source_counts = Counter(word.text for word in source)
    output_counts = Counter(word.text for word in output)
    output_index = {word.text: position for position, word in enumerate(output)}

    pairs = [
        (position, output_index[word.text])
        for position, word in enumerate(source)
        if source_counts[word.text] == 1 and output_counts[word.text] == 1
    ]

    # Patience LIS over the output positions: `tails[k]` is the smallest output
    # position ending an increasing run of length k+1, and `previous` threads the
    # chosen pairs back together.
    tails: list[int] = []
    tail_pair: list[int] = []
    previous: list[int] = [-1] * len(pairs)
    for index, (_, output_position) in enumerate(pairs):
        slot = bisect.bisect_left(tails, output_position)
        if slot == len(tails):
            tails.append(output_position)
            tail_pair.append(index)
        else:
            tails[slot] = output_position
            tail_pair[slot] = index
        previous[index] = tail_pair[slot - 1] if slot else -1

    chosen: list[tuple[int, int]] = []
    cursor = tail_pair[-1] if tail_pair else -1
    while cursor != -1:
        chosen.append(pairs[cursor])
        cursor = previous[cursor]
    chosen.reverse()
    return chosen


def _matched_flags(source: Sequence[_Word], output: Sequence[_Word]) -> list[bool]:
    """Which source words the output accounts for, in three passes.

    1. **Anchors** are matched by construction.
    2. **Spans** between consecutive anchors are compared as multisets, so an
       extractor may reorder freely inside one — a footnote moved to the end of
       its section is not loss.
    3. **Whatever is left over anywhere in the output** gets one more chance at
       the source words still unmatched. Without this pass, text that moved
       *across* an anchor — a sidebar re-emitted three paragraphs later — would
       be reported as lost while sitting in plain view in the artifact.

    Pass 3 is what makes the metric order-blind and pass 2 is what makes it
    local, and the order of the two is the whole trick: a word is only credited
    to a distant part of the output once nothing nearby could account for it, so
    the *excerpt* still points at where the text should have been.

    Multiplicity is what stops a dropped listing hiding behind the index:
    matching consumes, so a word the source uses twice and the output once is
    accounted for once.
    """
    matched = [False] * len(source)
    available = Counter(word.text for word in output)
    anchors = _anchors(source, output)

    spans: list[tuple[int, int, int, int]] = []
    source_cursor = 0
    output_cursor = 0
    for source_position, output_position in anchors:
        spans.append((source_cursor, source_position, output_cursor, output_position))
        matched[source_position] = True
        available[source[source_position].text] -= 1
        source_cursor = source_position + 1
        output_cursor = output_position + 1
    spans.append((source_cursor, len(source), output_cursor, len(output)))

    for source_start, source_end, output_start, output_end in spans:
        if source_start >= source_end:
            continue
        nearby = Counter(word.text for word in output[output_start:output_end])
        for position in range(source_start, source_end):
            word = source[position].text
            if nearby[word]:
                nearby[word] -= 1
                available[word] -= 1
                matched[position] = True

    for position, leftover in enumerate(source):
        if not matched[position] and available[leftover.text] > 0:
            available[leftover.text] -= 1
            matched[position] = True
    return matched


@dataclass(frozen=True)
class _Comparison:
    """One alignment of a source against an output, reused by every number that
    reads it.

    Aligning a 1,284-page book is seconds of work, and `dropped_runs` and
    `coverage` answer different questions about the *same* alignment — so it is
    computed once and handed to both.
    """

    normalised_source: str
    words: list[_Word]
    matched: list[bool]


def _compare(source_text: str, markdown: str) -> _Comparison:
    normalised_source = _normalise(source_text)
    source = _words(normalised_source)
    return _Comparison(
        normalised_source=normalised_source,
        words=source,
        matched=_matched_flags(source, _words(_normalise(markdown))) if source else [],
    )


def dropped_runs(source_text: str, markdown: str) -> tuple[DroppedRun, ...]:
    """Every contiguous stretch of ``source_text`` the output does not account
    for, longest first."""
    return _dropped_runs(_compare(source_text, markdown))


def _dropped_runs(comparison: _Comparison) -> tuple[DroppedRun, ...]:
    normalised_source = comparison.normalised_source
    source = comparison.words
    if not source:
        return ()
    matched = comparison.matched

    runs: list[DroppedRun] = []
    start: int | None = None
    end = 0
    for word, is_matched in zip(source, matched, strict=True):
        if is_matched:
            if start is not None:
                runs.append(_run(normalised_source, start, end))
                start = None
        else:
            if start is None:
                start = word.start
            end = word.end
    if start is not None:
        runs.append(_run(normalised_source, start, end))
    return tuple(sorted(runs, key=lambda run: run.characters, reverse=True))


def _run(normalised_source: str, start: int, end: int) -> DroppedRun:
    text = normalised_source[start:end]
    excerpt = " ".join(text[:EXCERPT_CHARACTERS].split())
    if len(text) > EXCERPT_CHARACTERS:
        excerpt += " …"
    return DroppedRun(characters=len(text), excerpt=excerpt)


def coverage(source_text: str, markdown: str) -> float:
    """Percentage of the source's word characters the output accounts for.

    Reported, never gated. A document with no text layer at all is called fully
    covered rather than left undefined — a ZeroDivisionError inside the gate
    would read as a failing book.
    """
    return _coverage(_compare(source_text, markdown))


def _coverage(comparison: _Comparison) -> float:
    total = sum(word.end - word.start for word in comparison.words)
    if not total:
        return 100.0
    accounted = sum(
        word.end - word.start
        for word, is_matched in zip(comparison.words, comparison.matched, strict=True)
        if is_matched
    )
    return round(100.0 * accounted / total, 2)


def suspect_headings(markdown: str) -> tuple[str, ...]:
    """The titles of every code-shaped ATX heading outside a fence.

    Outside a fence, because fenced code is what ADR 0003 buys: counting the
    comments inside a correctly fenced listing would report the success as the
    failure.
    """
    found: list[str] = []
    for kind, lines in _blocks(markdown):
        if kind != "heading":
            continue
        match = HEADING.match(lines[0])
        if match is None:  # pragma: no cover - _blocks only labels matches
            continue
        title = match.group(2)
        if looks_like_code(title):
            found.append(title)
    return tuple(found)


def _blocks(markdown: str) -> list[tuple[str, list[str]]]:
    """The artifact as top-level blocks, labelled by kind.

    Blank-line separated, except that a fence swallows everything up to its
    partner — which is the whole reason this exists rather than a `splitlines`
    comprehension. An unterminated fence is not a block: it is a defect, and
    counting it as code would hide it.
    """
    blocks: list[tuple[str, list[str]]] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            blocks.append((_kind(pending), list(pending)))
            pending.clear()

    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        fence = FENCE.match(line)
        if fence:
            marker = fence.group(1)
            closing = next(
                (
                    candidate
                    for candidate in range(index + 1, len(lines))
                    if lines[candidate].strip().startswith(marker)
                ),
                None,
            )
            if closing is None:
                # Unterminated: treat the opener as ordinary text so the rest of
                # the document is still classified, and report no code block.
                pending.append(line)
                index += 1
                continue
            flush()
            blocks.append(("code", lines[index : closing + 1]))
            index = closing + 1
            continue
        if not line.strip():
            flush()
            index += 1
            continue
        if HEADING.match(line):
            flush()
            blocks.append(("heading", [line]))
            index += 1
            continue
        pending.append(line)
        index += 1
    flush()
    return blocks


def _kind(lines: Sequence[str]) -> str:
    stripped = [line.strip() for line in lines]
    if all(line.startswith("|") for line in stripped):
        return "table"
    if all(re.match(r"([-*+]|\d+[.)])\s", line) for line in stripped):
        return "list"
    return "paragraph"


@dataclass(frozen=True)
class ParseMetrics:
    """One row of the gate: what came out of one document, on one route."""

    route: str
    source_characters: int
    output_characters: int
    coverage_pct: float
    longest_dropped_run: int
    longest_dropped_excerpt: str
    headings: int
    suspect_headings: tuple[str, ...]
    fences: int
    code_characters: int
    code_share: float
    blocks: Mapping[str, int]
    rule_counts: Mapping[str, int]
    declines: tuple[str, ...]
    skipped_documents: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "source_characters": self.source_characters,
            "output_characters": self.output_characters,
            "coverage_pct": self.coverage_pct,
            "longest_dropped_run": self.longest_dropped_run,
            "longest_dropped_excerpt": self.longest_dropped_excerpt,
            "headings": self.headings,
            "suspect_headings": list(self.suspect_headings),
            "fences": self.fences,
            "code_characters": self.code_characters,
            "code_share": self.code_share,
            "blocks": dict(self.blocks),
            "rule_counts": dict(self.rule_counts),
            "declines": list(self.declines),
            "skipped_documents": self.skipped_documents,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Self:
        return cls(
            route=str(raw["route"]),
            source_characters=int(raw["source_characters"]),
            output_characters=int(raw["output_characters"]),
            coverage_pct=float(raw["coverage_pct"]),
            longest_dropped_run=int(raw["longest_dropped_run"]),
            longest_dropped_excerpt=str(raw.get("longest_dropped_excerpt", "")),
            headings=int(raw["headings"]),
            suspect_headings=tuple(str(title) for title in raw["suspect_headings"]),
            fences=int(raw["fences"]),
            code_characters=int(raw["code_characters"]),
            code_share=float(raw["code_share"]),
            blocks={str(kind): int(count) for kind, count in raw["blocks"].items()},
            rule_counts={str(rule): int(count) for rule, count in raw["rule_counts"].items()},
            declines=tuple(str(reason) for reason in raw["declines"]),
            skipped_documents=int(raw["skipped_documents"]),
        )


def measure(markdown: str, *, source_text: str, report: ExtractorReport) -> ParseMetrics:
    """One document's row. Pure — it opens nothing and parses nothing."""
    blocks = _blocks(markdown)
    counts: Counter[str] = Counter(kind for kind, _ in blocks)
    code_characters = sum(
        len("\n".join(lines[1:-1])) + (1 if len(lines) > 2 else 0)
        for kind, lines in blocks
        if kind == "code"
    )
    comparison = _compare(source_text, markdown)
    runs = _dropped_runs(comparison)
    longest = runs[0] if runs else None
    return ParseMetrics(
        route=report.route,
        source_characters=sum(word.end - word.start for word in comparison.words),
        output_characters=len(markdown),
        coverage_pct=_coverage(comparison),
        longest_dropped_run=longest.characters if longest else 0,
        longest_dropped_excerpt=longest.excerpt if longest else "",
        headings=counts["heading"],
        suspect_headings=suspect_headings(markdown),
        fences=counts["code"],
        code_characters=code_characters,
        code_share=round(code_characters / len(markdown), 4) if markdown else 0.0,
        blocks=dict(counts),
        rule_counts=dict(report.rule_counts),
        declines=report.declines,
        skipped_documents=report.skipped_documents,
    )


def source_text_of(path: Path) -> str:
    """The document's text layer, as MuPDF reads it — the reference every loss
    number is measured against.

    Deliberately an *independent* reader rather than the extraction path's own.
    For an EPUB that means MuPDF's reflowed rendering, which shares no code with
    the container reader that produces the artifact; a reference computed by the
    thing under test would report 100% coverage of its own mistakes.
    """
    quiet_mupdf()
    with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
        return "\n".join(page.get_text() for page in document)
