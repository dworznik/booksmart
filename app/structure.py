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
    sections: list[DetectedSection] = field(default_factory=list)


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
            found.append(Heading(level=len(match.group(1)), title=match.group(2), line=number))
    return found


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

    chapters: list[DetectedChapter] = []
    for heading in headings:
        if heading.level == chapter_level:
            chapters.append(DetectedChapter(title=heading.title, line=heading.line))
        elif heading.level == section_level and chapters:
            chapters[-1].sections.append(
                DetectedSection(title=heading.title, line=heading.line)
            )
    return chapters
