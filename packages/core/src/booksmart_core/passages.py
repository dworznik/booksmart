"""Cutting passages: the book's own words, in windows worth retrieving.

Summaries and knowledge objects are what a model said *about* the text. A
passage is the text — a contiguous run of the parsed markdown, small enough to
be about one thing. It is what lets a search return the sentence that says
something rather than the chapter that discusses it, and the only record whose
words are guaranteed to be the book's: an identifier, an error message or a
coinage that no summary repeated exists nowhere else in the index.

The cutting is structure-aware and deterministic. Each chapter is cut at its
section boundaries first, so no passage straddles two sections; inside a
section, whole paragraphs are packed into a window until the next one would
overflow it. Nothing overlaps: the two rankings of a Hybrid search are fused by
position, and near-duplicate neighbours would each take a rank from a record
that says something different.

No model is involved and none is needed, which is why this is a function the
embeddings Stage calls rather than a Stage of its own (ADR 0005).
"""

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from booksmart_core.models import Chapter, Section
from booksmart_core.structure import FENCE, HEADING

# The size a window is packed toward. It is the `avg_len` of the BM25 Recipe
# the collection locks (ADR 0004): BM25 normalises a term's weight by how long
# its document is relative to that average, so windows of about that length are
# the ones the normalisation is calibrated for.
PASSAGE_WINDOW_TOKENS = 256

# A token here is a whitespace-delimited word. That is an approximation of what
# any real tokenizer counts, chosen on purpose: it needs no model, no vendor and
# no download, and it gives the same answer on every machine — which a passage's
# boundaries have to, since they are written down as character offsets. The
# window is a target to pack toward, not a Limit anything enforces.
_WORD = re.compile(r"\S+")

# A word that ends a sentence: terminal punctuation, then any closing quotes or
# brackets. Good enough to prefer a cut between sentences over one inside a
# sentence; an abbreviation that fools it costs nothing a blind cut would not.
_SENTENCE_CLOSE = re.compile(r"[.!?][\"'”’)\]]*$")


@dataclass(frozen=True)
class PassageSpan:
    """One passage as cut: where it sits in the book, and what it says.

    ``text`` is exactly ``markdown[start_char:end_char]`` — offsets into the
    parsed markdown as a Python string. The span is the durable fact; the text
    is carried so the caller need not slice again.
    """

    chapter: Chapter
    # None for text between a chapter's heading and its first section.
    section: Section | None
    start_char: int
    end_char: int
    text: str


def cut_passages(chapters: Sequence[Chapter], markdown: str) -> list[PassageSpan]:
    """Every chapter's text as passages, in reading order.

    ``chapters`` are the book's, in position order, carrying the 1-based line
    numbers structure detection recorded. Text outside every chapter — whatever
    precedes the first chapter heading — yields nothing, as it does for every
    other Stage that reads a book by its chapters.
    """
    lines = _Lines.of(markdown)
    located = [chapter for chapter in chapters if chapter.source_line is not None]
    passages: list[PassageSpan] = []
    for index, chapter in enumerate(located):
        assert chapter.source_line is not None
        start = min(chapter.source_line - 1, len(lines.text))
        following = located[index + 1].source_line if index + 1 < len(located) else None
        end = min(following - 1, len(lines.text)) if following is not None else len(lines.text)
        for section, first, last in _segments(chapter, start, end):
            for start_char, end_char in _pack(_pieces(lines, first, last)):
                passages.append(
                    PassageSpan(
                        chapter=chapter,
                        section=section,
                        start_char=start_char,
                        end_char=end_char,
                        text=markdown[start_char:end_char],
                    )
                )
    return passages


@dataclass(frozen=True)
class _Lines:
    """The document as lines, each knowing where in the document it starts."""

    text: list[str]
    starts: list[int]

    @classmethod
    def of(cls, markdown: str) -> "_Lines":
        # `splitlines` is what structure detection numbered its lines with, so
        # the two agree on what a line is.
        text = markdown.splitlines(keepends=True)
        starts = [0]
        for line in text[:-1]:
            starts.append(starts[-1] + len(line))
        return cls(text, starts)

    def end_of_text(self, number: int) -> int:
        """Where the line's text ends — before its line break and trailing blanks."""
        return self.starts[number] + len(self.text[number].rstrip())


@dataclass(frozen=True)
class _Piece:
    """The smallest unit packing moves around: a paragraph, or part of one."""

    start: int
    end: int
    tokens: int
    # A heading and nothing else. A window made only of these is not a passage.
    heading: bool = False


def _segments(
    chapter: Chapter, start: int, end: int
) -> Iterator[tuple[Section | None, int, int]]:
    """The chapter's line range cut at its section headings: the text before the
    first section (belonging to no section), then each section's own."""
    bounds: list[tuple[Section | None, int]] = [(None, start)]
    for section in chapter.sections:
        # A section with no recorded line, or one outside the chapter, cannot
        # bound anything; its text stays with whatever precedes it.
        if section.source_line is not None and start < section.source_line - 1 < end:
            bounds.append((section, section.source_line - 1))
    bounds.sort(key=lambda bound: bound[1])
    for position, (owner, first) in enumerate(bounds):
        last = bounds[position + 1][1] if position + 1 < len(bounds) else end
        yield owner, first, last


def _pieces(lines: _Lines, first: int, last: int) -> Iterator[_Piece]:
    """The paragraphs of lines ``first`` up to ``last``, with any paragraph too
    large for one window broken into pieces that fit.

    A paragraph is a run of non-blank lines. A fenced code block is one
    paragraph however many blank lines it contains: ADR 0003 makes a fence the
    statement that its body is code verbatim, and a listing cut at its blank
    lines is several fragments none of which is the listing.
    """
    block: list[int] = []
    fence_marker: str | None = None

    def flush() -> Iterator[_Piece]:
        if block:
            yield from _fit(lines, block)
            block.clear()

    for number in range(first, last):
        line = lines.text[number]
        fence = FENCE.match(line)
        if fence:
            marker = fence.group(1)
            if fence_marker is None:
                # A fence opens a paragraph of its own, even set tight against
                # the prose above it.
                yield from flush()
                fence_marker = marker
                block.append(number)
            elif fence_marker == marker:
                fence_marker = None
                block.append(number)
                yield from flush()
            else:
                block.append(number)
        elif fence_marker is not None:
            block.append(number)
        elif line.strip():
            block.append(number)
        else:
            yield from flush()
    yield from flush()


def _fit(lines: _Lines, block: list[int]) -> Iterator[_Piece]:
    """One paragraph as pieces no larger than a window.

    Almost always one piece: the paragraph. One that overflows a window by
    itself has to be cut somewhere, and where depends on what it is. Prose is
    cut between sentences where it can be and between words where it cannot,
    whatever its line breaks — they are where an Extractor wrapped it, not
    where the author paused. A fenced listing is cut at its line breaks, which
    in code are the units, and inside a line only if one line overflows alone.
    Any of these is better than the alternative, which is a text of unbounded
    length handed to an embedding model with a bounded input.
    """
    counts = [len(_WORD.findall(lines.text[number])) for number in block]
    fenced = FENCE.match(lines.text[block[0]]) is not None
    if sum(counts) <= PASSAGE_WINDOW_TOKENS:
        heading = not fenced and all(HEADING.match(lines.text[number]) for number in block)
        yield _Piece(
            lines.starts[block[0]], lines.end_of_text(block[-1]), sum(counts), heading
        )
    elif not fenced:
        yield from _runs(_words(lines, block))
    else:
        for number, count in zip(block, counts, strict=True):
            if count > PASSAGE_WINDOW_TOKENS:
                yield from _runs(_words(lines, [number]))
            elif count:  # zero is a blank line inside the listing
                yield _Piece(lines.starts[number], lines.end_of_text(number), count)


@dataclass(frozen=True)
class _Word:
    start: int
    end: int
    closes_sentence: bool


def _words(lines: _Lines, numbers: list[int]) -> list[_Word]:
    """Every word on the given lines, located in the document."""
    return [
        _Word(
            lines.starts[number] + match.start(),
            lines.starts[number] + match.end(),
            _SENTENCE_CLOSE.search(match.group()) is not None,
        )
        for number in numbers
        for match in _WORD.finditer(lines.text[number])
    ]


def _runs(words: list[_Word]) -> Iterator[_Piece]:
    """Consecutive runs of at most a window of words, each ending on a sentence
    where one ends in its second half.

    Only the second half is searched, so a cut never leaves less than half a
    window behind it: a long sentence late in the run costs a mid-sentence cut
    rather than a fragment.
    """
    first = 0
    while first < len(words):
        last = min(first + PASSAGE_WINDOW_TOKENS, len(words))
        if last < len(words):
            for index in range(last - 1, first + (last - first) // 2 - 1, -1):
                if words[index].closes_sentence:
                    last = index + 1
                    break
        yield _Piece(words[first].start, words[last - 1].end, last - first)
        first = last


def _pack(pieces: Iterator[_Piece]) -> Iterator[tuple[int, int]]:
    """Greedily pack consecutive pieces into windows; yield each window's span.

    A window closes when the next piece would overflow it, so every window holds
    whole pieces and no text is in two windows. A window holding nothing but
    headings is dropped: a heading alone says nothing its chapter and section
    records do not, and as the shortest document in the collection it would
    outrank every real passage on the words it shares with them.
    """
    window: list[_Piece] = []
    tokens = 0

    def close() -> Iterator[tuple[int, int]]:
        if window and not all(piece.heading for piece in window):
            yield window[0].start, window[-1].end
        window.clear()

    for piece in pieces:
        if window and tokens + piece.tokens > PASSAGE_WINDOW_TOKENS:
            yield from close()
            tokens = 0
        window.append(piece)
        tokens += piece.tokens
    yield from close()
