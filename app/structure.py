"""Heading-tree structure detection over parsed Markdown.

The chapter level is the minimum ATX heading level present in the document
(parsers map the largest font to the smallest level, but not every book starts
at '#'), with one refinement: a single lone heading at that level followed by
deeper ones is treated as the book title, not a chapter. The section level is
the next level down. Deeper headings are ignored, as are headings inside
fenced code blocks and section headings that appear before the first chapter.
"""

import re
from dataclasses import dataclass, field

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^(```|~~~)")

# Emphasis spans to unwrap in heading titles: double markers anywhere, single
# markers only when they don't touch word characters on the outside, so
# intra-word underscores (snake_case) survive.
EMPHASIS_SPANS = (
    re.compile(r"\*\*(.+?)\*\*"),
    re.compile(r"__(.+?)__"),
    re.compile(r"(?<!\w)\*(.+?)\*(?!\w)"),
    re.compile(r"(?<!\w)_(.+?)_(?!\w)"),
)


def _strip_emphasis(title: str) -> str:
    """Remove Markdown emphasis markers (pymupdf4llm emits bold headings for
    some EPUBs) so persisted titles are plain text."""
    previous = None
    while previous != title:
        previous = title
        for span in EMPHASIS_SPANS:
            title = span.sub(r"\1", title)
    return title.strip()


@dataclass(frozen=True)
class Heading:
    level: int
    title: str
    line: int  # 1-based line number in the parsed markdown


@dataclass(frozen=True)
class DetectedSection:
    title: str
    line: int


@dataclass
class DetectedChapter:
    title: str
    line: int
    kind: str = "chapter"  # "front_matter" | "chapter" | "back_matter"
    sections: list[DetectedSection] = field(default_factory=list)


# Titles that are only ever peripheral matter when they make up the entire
# heading (so "Notes on Distributed Systems" stays a chapter), plus prefixes
# that stay unambiguous with a subtitle ("Preface to the Second Edition",
# "Appendix A: Notation").
FRONT_MATTER = re.compile(
    r"^(cover|half title|title page|copyright( page)?|contents|table of contents"
    r"|dedication|epigraph|foreword)$"
    r"|^(preface|list of (figures|tables|illustrations))\b",
    re.IGNORECASE,
)
BACK_MATTER = re.compile(
    r"^(index|bibliography|references|glossary|notes|epilogue|afterword"
    r"|further reading|credits|colophon|errata)$"
    r"|^appendix\b",
    re.IGNORECASE,
)
# Legitimately appears on either side of the body; resolved by position.
EITHER_MATTER = re.compile(r"^(acknowledg|about the )", re.IGNORECASE)


def _classify_matter(chapters: list[DetectedChapter]) -> None:
    """Mark front and back matter so downstream stages can skip or weight it;
    marking beats dropping because provenance may still point into a preface."""
    body_seen = False
    for chapter in chapters:
        if FRONT_MATTER.search(chapter.title):
            chapter.kind = "front_matter"
        elif BACK_MATTER.search(chapter.title):
            chapter.kind = "back_matter"
        elif EITHER_MATTER.search(chapter.title):
            chapter.kind = "back_matter" if body_seen else "front_matter"
        else:
            body_seen = True
    # A title page carries the book's own name, which no keyword list can know;
    # an unrecognized first heading directly followed by front matter is one.
    if (
        len(chapters) > 1
        and chapters[0].kind == "chapter"
        and chapters[1].kind == "front_matter"
    ):
        chapters[0].kind = "front_matter"


def _headings(markdown: str) -> list[Heading]:
    """Every ATX heading outside fenced code blocks."""
    found: list[Heading] = []
    fence_marker: str | None = None
    for number, raw in enumerate(markdown.splitlines(), start=1):
        fence = FENCE.match(raw)
        if fence:
            marker = fence.group(1)
            if fence_marker is None:
                fence_marker = marker
            elif fence_marker == marker:
                fence_marker = None
            continue
        if fence_marker is not None:
            continue
        match = HEADING.match(raw)
        if match and match.group(2):
            title = _strip_emphasis(match.group(2))
            if title:
                found.append(Heading(level=len(match.group(1)), title=title, line=number))
    return found


# A heading that is only a chapter/part designator ("Chapter 4", "Part II"),
# with no title of its own.
CHAPTER_NUMBER = re.compile(r"^(chapter|part)\s+(\d+|[ivxlcdm]+)\.?$", re.IGNORECASE)


# A number/title pair sits on adjacent lines, at most a blank line or two
# apart; a larger gap means the bare number heading owns a real chapter body
# and the next heading starts a different chapter.
MERGE_MAX_LINE_GAP = 3


def _merge_number_title_pairs(headings: list[Heading], chapter_level: int) -> list[Heading]:
    """Books often typeset the chapter number and its title as two consecutive
    same-level headings. Merge such a pair into one heading ("Chapter 4:
    Modules Should Be Deep") anchored at the number heading's line."""
    merged: list[Heading] = []
    index = 0
    while index < len(headings):
        heading = headings[index]
        following = headings[index + 1] if index + 1 < len(headings) else None
        if (
            heading.level == chapter_level
            and CHAPTER_NUMBER.match(heading.title)
            and following is not None
            and following.level == chapter_level
            and not CHAPTER_NUMBER.match(following.title)
            and following.line - heading.line <= MERGE_MAX_LINE_GAP
        ):
            merged.append(
                Heading(
                    level=chapter_level,
                    title=f"{heading.title}: {following.title}",
                    line=heading.line,
                )
            )
            index += 2
            continue
        merged.append(heading)
        index += 1
    return merged


def detect_structure(markdown: str) -> list[DetectedChapter]:
    headings = _headings(markdown)
    if not headings:
        return []

    levels = sorted({heading.level for heading in headings})
    # A lone top-level heading above deeper ones is the book title, not a chapter.
    if len(levels) > 1 and sum(1 for h in headings if h.level == levels[0]) == 1:
        levels = levels[1:]
        headings = [h for h in headings if h.level >= levels[0]]

    chapter_level = levels[0]
    section_level = levels[1] if len(levels) > 1 else None
    headings = _merge_number_title_pairs(headings, chapter_level)

    chapters: list[DetectedChapter] = []
    for heading in headings:
        if heading.level == chapter_level:
            chapters.append(DetectedChapter(title=heading.title, line=heading.line))
        elif heading.level == section_level and chapters:
            chapters[-1].sections.append(
                DetectedSection(title=heading.title, line=heading.line)
            )
    _classify_matter(chapters)
    return chapters
