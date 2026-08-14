"""The `pdf` route: read the page's own spans.

`pymupdf` directly, with no `pymupdf4llm` above it. That library hard-pins
`pymupdf`, `pymupdf_layout` (Polyform Noncommercial) and `tabulate`; dropping it
takes a noncommercial term out of the dependency tree of an MIT-declared published
package and unpins pymupdf, which has zero dependencies of its own.
Its layout helpers are deliberately *not* vendored — copying
AGPL source into an MIT repo is a worse position than depending on it.

## Code: a tiered composite over font families

The signal is which *family* a line is set in, because that is what a typesetter
actually varies to mark a listing. Measured over the pinned PDFs:

1. Dominant family by character mass is prose. The body's median line length,
   median left margin and modal size come from lines in that family.
2. **Tier 1** — any non-dominant family holding at least 2% of lines and at least
   80% monospaced is a code family. Monospace is decided by MuPDF's flag, then a
   family-name regex, then constant glyph advance. Five of the eight pinned PDFs
   are caught here, one of them only by the name regex: it carries 156k characters
   of Courier that MuPDF does not flag.
3. **Tier 2**, only if tier 1 selected nothing — a non-dominant family whose lines
   are markedly shorter than the body's *and* which is either indented well past
   the body margin or set smaller than the body. The disjunction is load-bearing
   and each leg is carried by a different book: one is caught by size contrast
   alone (11.3pt against a 15.0pt body, its indent under the threshold), the other
   by indentation alone (+72pt at exactly the body size).
4. Otherwise **decline**. One pinned book is 1,284 pages of one family, its Eiffel
   differentiated only by bold keywords and italic identifiers. Per ADR 0003 that
   declines rather than guesses: indentation reconstructed wrongly reads as
   authoritative, and nothing downstream can tell it from indentation that is
   right.

Tier 2's thresholds are fitted to eight books, and `booksmart-core` is published.
They are kept because the probe reproduces (see the PR), and they are the reason
tier 2 fires *only* when tier 1 found nothing — a fitted rule that never runs on
the books a generic rule already handles.

## Headings: size, but not size alone

Size alone is what `pymupdf4llm` did and it over-promotes. A heading is a **line**,
not a paragraph, and it is never inside a code run. Heading detection **declines
independently of code detection**: the two rest on different evidence — family
separates code from prose, size separates headings from body text — so a
single-family document still typesets its chapter openers larger, and the book
that declines its code keeps its chapters.
"""

import re
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from booksmart_core.parsing.blocks import Block, to_gfm
from booksmart_core.parsing.contract import ExtractorReport, ParseResult
from booksmart_core.parsing.mupdf import quiet_mupdf

MONO_FLAG = 1 << 3
ITALIC_FLAG = 1 << 1

MONO_NAME = re.compile(
    r"(?i)mono|courier|consolas|menlo|inconsolata|source ?code|fira ?code"
    r"|jetbrains|dejavu ?sans ?mono|pt ?mono|nimbus ?mono|letter ?gothic"
    r"|lucida ?console|andale"
)

# Style tokens count as style only when a separator precedes them, so
# "NewCenturySchlbk-Roman" folds to "NewCenturySchlbk" while "TimesNewRoman" is
# left alone. Folding more eagerly merges genuinely distinct families, and the
# whole rule rests on telling families apart.
STYLE_SUFFIX = re.compile(
    r"[-,_ ](?:bold|italic|oblique|regular|roman|light|medium|semibold|book|black"
    r"|heavy|condensed|cond|demi|it|bolditalic|semibolditalic|typnarr)+$",
    re.IGNORECASE,
)

# --- the tier thresholds, all three of them --------------------------------
#
# Named and gathered here because they are the fitted part of this module, and a
# reader deciding whether to trust it on a book outside the pinned set should be
# able to see all of them at once.

# A family below this share of the document's lines is a caption, a marginal note
# or a symbol font, not the way this book sets its listings.
MIN_FAMILY_SHARE = 0.02
# How much of a family's mass has to read as monospaced before the family does.
MONO_SHARE = 0.80
# Tier 2: lines this much shorter than the body's median are not prose.
SHORT_LINE_SHARE = 0.60
# Tier 2, first leg: indented this many ems past the body's left margin.
INDENT_EMS = 1.5
# Tier 2, second leg: set below this share of the body size.
SMALL_SIZE_SHARE = 0.90
# A lone contrasting line is an inline term or a caption. A listing is a run.
MIN_CODE_RUN = 2
# How far short of its block's measure a line may fall and still read as wrapped
# prose. About one short word: a ragged-right setting varies by a few ems, while a
# table row or a ToC entry stops far shorter than that, line after line.
RAGGED_EMS = 4.0


def family_of(fontname: str) -> str:
    """The family a font name belongs to, with subset prefix and style stripped."""
    name = fontname.split("+")[-1]
    previous = None
    while previous != name:
        previous = name
        name = STYLE_SUFFIX.sub("", name)
    return name


@dataclass
class Line:
    """One line of a page, with what the rules need to judge it."""

    text: str
    page: int
    x0: float
    x1: float
    top: float
    size: float
    block: int
    families: Counter[str] = field(default_factory=Counter)
    mono_chars: int = 0
    italic_chars: int = 0
    nonspace_chars: int = 0

    @property
    def family(self) -> str:
        return self.families.most_common(1)[0][0] if self.families else ""

    @property
    def italic_ratio(self) -> float:
        return self.italic_chars / self.nonspace_chars if self.nonspace_chars else 0.0


def _glyph_advance_uniform(widths: Sequence[float]) -> bool:
    """Constant glyph advance means a monospaced face.

    The third tier of the monospace test, and the one that catches a font whose
    name says nothing and whose flags are unset — a Type 1 subset, typically.
    """
    if len(widths) < 6:
        return False
    mean = statistics.fmean(widths)
    if mean <= 0:
        return False
    return (statistics.pstdev(widths) / mean) < 0.02


def read_lines(document: pymupdf.Document) -> list[Line]:
    """Every line of the document, in pymupdf's reading order.

    ``rawdict`` rather than ``dict`` because glyph boxes are what the third
    monospace tier reads, and a font that neither declares nor names itself
    monospaced can only be recognised from them.
    """
    lines: list[Line] = []
    for number in range(document.page_count):
        page = document[number]
        page_text = page.get_text("rawdict")  # type: ignore[no-untyped-call]
        for block_index, block in enumerate(page_text["blocks"]):
            if block.get("type") != 0:  # an image
                continue
            for raw in block["lines"]:
                line = _line_of(raw, page=number, block=block_index)
                if line is not None:
                    lines.append(line)
    return lines


def _line_of(raw: dict[str, object], *, page: int, block: int) -> Line | None:
    bbox = raw["bbox"]
    assert isinstance(bbox, Sequence)
    line = Line(
        text="",
        page=page,
        x0=float(bbox[0]),
        top=float(bbox[1]),
        x1=float(bbox[2]),
        size=0.0,
        block=block,
    )
    parts: list[str] = []
    # Size by character mass, not the maximum. A drop cap or a decorative initial
    # is one span many times the body size, and taking the max would judge the
    # whole paragraph's first line at it — promoting it to the top heading level
    # and pushing every real chapter title down one.
    sizes: Counter[float] = Counter()
    spans = raw["spans"]
    assert isinstance(spans, Iterable)
    for span in spans:
        chars = span.get("chars", [])
        text = "".join(str(character["c"]) for character in chars)
        parts.append(text)
        nonspace = sum(1 for character in text if not character.isspace())
        if not nonspace:
            continue
        sizes[round(float(span["size"]), 1)] += nonspace
        family = family_of(str(span["font"]))
        line.families[family] += nonspace
        line.nonspace_chars += nonspace
        if int(span["flags"]) & ITALIC_FLAG:
            line.italic_chars += nonspace
        widths = [
            character["bbox"][2] - character["bbox"][0]
            for character in chars
            if str(character["c"]).strip()
        ]
        if (
            int(span["flags"]) & MONO_FLAG
            or MONO_NAME.search(str(span["font"]))
            or _glyph_advance_uniform(widths)
        ):
            line.mono_chars += nonspace
    line.text = "".join(parts)
    line.size = sizes.most_common(1)[0][0] if sizes else 0.0
    return line if line.nonspace_chars else None


@dataclass(frozen=True)
class FontProfile:
    """What the document's typography says, and which rule said it."""

    dominant: str
    body_size: float
    body_length: float  # median line length in characters
    body_x0: float  # median left margin
    code_families: frozenset[str] = frozenset()
    tier: str = ""  # "mono" | "contrast" — which tier selected the families
    code_decline: str = ""
    heading_sizes: tuple[float, ...] = ()  # distinct heading sizes, largest first
    heading_decline: str = ""


def read_typography(lines: Sequence[Line]) -> FontProfile:
    """Read the document's typography once, for the whole document.

    Document-wide rather than per-page on purpose: a page of pure listing has no
    prose to be dominant, and judged alone its code family *is* the body.
    """
    mass: Counter[str] = Counter()
    for line in lines:
        mass.update(line.families)
    if not mass:
        return FontProfile(
            dominant="",
            body_size=0.0,
            body_length=0.0,
            body_x0=0.0,
            code_decline="no text spans in the document",
            heading_decline="no text spans in the document",
        )
    dominant = mass.most_common(1)[0][0]
    body = [line for line in lines if line.family == dominant]
    body_size = _modal_size(body or lines)
    body_length = statistics.median([len(line.text) for line in body or lines])
    body_x0 = statistics.median([line.x0 for line in body or lines])

    code_families, tier = _code_families(lines, dominant, body_size, body_length, body_x0)
    heading_sizes = _heading_sizes(lines, body_size, float(body_length), code_families)
    return FontProfile(
        dominant=dominant,
        body_size=body_size,
        body_length=float(body_length),
        body_x0=float(body_x0),
        code_families=code_families,
        tier=tier,
        code_decline=""
        if code_families
        else "no font signal: no family separates code from prose in this document",
        heading_sizes=heading_sizes,
        heading_decline=""
        if heading_sizes
        else "no size contrast: nothing is set larger than the body text",
    )


def _modal_size(lines: Sequence[Line]) -> float:
    """The most common size by character mass, not by line count.

    By mass because a book with many short captions and few long paragraphs has
    a modal *line* size that is the caption's.
    """
    mass: Counter[float] = Counter()
    for line in lines:
        mass[round(line.size, 1)] += line.nonspace_chars
    return mass.most_common(1)[0][0] if mass else 0.0


def _code_families(
    lines: Sequence[Line],
    dominant: str,
    body_size: float,
    body_length: float,
    body_x0: float,
) -> tuple[frozenset[str], str]:
    counts: Counter[str] = Counter(line.family for line in lines)
    minimum = max(1, int(len(lines) * MIN_FAMILY_SHARE))
    candidates = [
        name
        for name, count in counts.items()
        if name and name != dominant and count >= minimum
    ]

    mono = frozenset(
        name for name in candidates if _mono_share(lines, name) >= MONO_SHARE
    )
    if mono:
        return mono, "mono"

    contrasting = frozenset(
        name
        for name in candidates
        if _looks_like_a_listing(lines, name, body_size, body_length, body_x0)
    )
    if contrasting:
        return contrasting, "contrast"
    return frozenset(), ""


def _mono_share(lines: Sequence[Line], name: str) -> float:
    mono = total = 0
    for line in lines:
        if line.family != name:
            continue
        mono += line.mono_chars
        total += line.nonspace_chars
    return mono / total if total else 0.0


def _looks_like_a_listing(
    lines: Sequence[Line],
    name: str,
    body_size: float,
    body_length: float,
    body_x0: float,
) -> bool:
    """Tier 2. Short lines, plus either an indent or a smaller size."""
    family_lines = [line for line in lines if line.family == name]
    if not family_lines:
        return False
    if statistics.median([len(line.text) for line in family_lines]) >= body_length * SHORT_LINE_SHARE:
        return False
    indented = (
        statistics.median([line.x0 for line in family_lines])
        > body_x0 + INDENT_EMS * body_size
    )
    smaller = statistics.median([line.size for line in family_lines]) < body_size * SMALL_SIZE_SHARE
    return indented or smaller


def _heading_sizes(
    lines: Sequence[Line],
    body_size: float,
    body_length: float,
    code_families: frozenset[str],
) -> tuple[float, ...]:
    """Distinct sizes that head a *line*, largest first, capped at six levels.

    The short-line test is applied here and not only when a level is assigned. A
    pull quote or an epigraph set larger than the body is a paragraph, and letting
    its size into this list spends one of the six levels on it — which pushes
    every real heading down a rank and can push the deepest one off the end.
    """
    sizes = {
        round(line.size, 1)
        for line in lines
        if _could_be_a_heading(line, body_size, code_families)
        and len(line.text.strip()) < body_length
    }
    return tuple(sorted(sizes, reverse=True)[:6])


def _could_be_a_heading(
    line: Line, body_size: float, code_families: frozenset[str]
) -> bool:
    """Larger than the body, short enough to be a line rather than a paragraph,
    and not part of a listing.

    "Short" is measured against the body's own median line length rather than a
    constant: a paragraph line runs to the column's width, and a heading does
    not.
    """
    return (
        round(line.size, 1) > round(body_size, 1)
        and line.family not in code_families
        # A title says something. Bullet glyphs, rules and arrows are set large
        # and short, and without this every list marker in a book whose bullets
        # outsize its body becomes a chapter — which, since detect_structure
        # takes the minimum heading level present, demotes the real ones.
        and any(character.isalnum() for character in line.text)
    )


# --- assembling the document ----------------------------------------------

_HYPHEN_END = re.compile(r"(\w)[-‐­]$")


@dataclass(frozen=True)
class _Run:
    """Consecutive lines that belong together — one listing, or one paragraph."""

    lines: tuple[Line, ...]
    is_code: bool


def _runs(lines: Sequence[Line], typography: FontProfile) -> list[_Run]:
    """Split the document into code runs and everything else.

    A code line is only code inside a maximal run of at least two consecutive
    lines in a code family. A lone contrasting line is an inline term, a caption
    or a variable name set in the code face mid-sentence.
    """
    flags = [line.family in typography.code_families for line in lines]
    # Demote runs shorter than the minimum before grouping, so a single
    # contrasting line does not split the paragraph around it.
    index = 0
    while index < len(flags):
        if not flags[index]:
            index += 1
            continue
        end = index
        while end < len(flags) and flags[end]:
            end += 1
        if end - index < MIN_CODE_RUN:
            for position in range(index, end):
                flags[position] = False
        index = end

    runs: list[_Run] = []
    current: list[Line] = []
    current_code = False
    for line, is_code in zip(lines, flags, strict=True):
        if current and is_code == current_code:
            current.append(line)
            continue
        if current:
            runs.append(_Run(lines=tuple(current), is_code=current_code))
        current = [line]
        current_code = is_code
    if current:
        runs.append(_Run(lines=tuple(current), is_code=current_code))
    return runs


def _code_block(run: _Run, typography: FontProfile) -> Block:
    """A listing, with its indentation reconstructed from left margins.

    The x0 deltas are what indentation *is* in a PDF: there are no leading spaces
    in the text stream, so a listing read naively comes out flush left and its
    nesting is gone. Widths come from the run's own glyph boxes rather than a
    constant, because the unit is one character of the code face.
    """
    left = min(line.x0 for line in run.lines)
    character = _character_width(run)
    body: list[str] = []
    for line in run.lines:
        indent = int(round((line.x0 - left) / character)) if character else 0
        body.append(" " * max(0, indent) + line.text.rstrip())
    return Block(
        kind="code",
        text="\n".join(body),
        rule=f"pdf:{typography.tier}-family",
    )


def _character_width(run: _Run) -> float:
    """One character of the run's face, from the lines' own geometry."""
    widths = [
        (line.x1 - line.x0) / len(line.text.rstrip())
        for line in run.lines
        if line.text.strip() and line.x1 > line.x0
    ]
    return statistics.median(widths) if widths else 0.0


def _prose_blocks(run: _Run, typography: FontProfile) -> list[Block]:
    """Headings and paragraphs, with lines joined and hyphenation healed."""
    blocks: list[Block] = []
    paragraph: list[Line] = []
    heading: list[Line] = []
    heading_level = 0

    def flush_paragraph() -> None:
        if not paragraph:
            return
        blocks.append(
            Block(kind="paragraph", text=_join(paragraph, typography.body_size))
        )
        paragraph.clear()

    def flush_heading() -> None:
        if not heading:
            return
        blocks.append(
            Block(kind="heading", text=_join(heading, typography.body_size), level=heading_level)
        )
        heading.clear()

    for line in run.lines:
        level = _heading_level(line, typography)
        if level is not None:
            # A title too long for its measure wraps, and the second line is the
            # same heading — "Chapter 2" / "Composing Small Parts" is
            # one chapter, not two. Only a line directly beneath the previous one,
            # in the same text block and at the same size, joins: the labels of a
            # state diagram are also large, short and adjacent in the artifact,
            # and they are separate blocks a page apart on the paper.
            if heading and _continues_the_heading(line, heading[-1], level, heading_level):
                heading.append(line)
                continue
            flush_heading()
            flush_paragraph()
            heading.append(line)
            heading_level = level
            continue
        flush_heading()
        if paragraph and _starts_a_paragraph(line, paragraph[-1], typography):
            flush_paragraph()
        paragraph.append(line)
    flush_heading()
    flush_paragraph()
    return blocks


def _continues_the_heading(line: Line, previous: Line, level: int, previous_level: int) -> bool:
    return (
        level == previous_level
        and line.block == previous.block
        and line.page == previous.page
        and 0 <= line.top - previous.top <= line.size * 2.0
    )


def _heading_level(line: Line, typography: FontProfile) -> int | None:
    """Rank among the document's distinct heading sizes, largest first.

    The "short line" test is what keeps this from being size alone, which is what
    over-promoted: a paragraph set in a larger face — a pull quote, an epigraph —
    is not six headings.
    """
    if not typography.heading_sizes:
        return None
    if not _could_be_a_heading(line, typography.body_size, typography.code_families):
        return None
    if len(line.text.strip()) >= typography.body_length:
        return None
    size = round(line.size, 1)
    if size not in typography.heading_sizes:
        return None
    return typography.heading_sizes.index(size) + 1


def _starts_a_paragraph(line: Line, previous: Line, typography: FontProfile) -> bool:
    """Whether this line opens a new paragraph rather than continuing one.

    A new text block, a first-line indent, or a vertical gap wider than a line.
    Kept minimal on purpose: getting layout wrong shows up as a
    dropped run in the gate, and the elaborate version measured earlier failed
    that same test by losing a 185-line appendix table.
    """
    if line.block != previous.block or line.page != previous.page:
        return True
    if line.x0 > previous.x0 + typography.body_size * 0.5:
        return True
    return line.top - previous.top > typography.body_size * 2.0


def _join(lines: Sequence[Line], body_size: float) -> str:
    """One block's lines, joined where they wrap and kept apart where they do not.

    Wrapped prose fills its measure: every line but the last reaches the right
    margin, because that is what wrapping *is*. A table row, a ToC entry and a
    centred line stop well short of it, line after line — so joining on a blank
    "is this the same block" test runs a table's rows together into one run-on
    sentence, which is how a table gets passed through as a paragraph instead of
    as a block.

    The measure is the block's own widest line, so nothing here is calibrated to
    a corpus; the tolerance is in ems of the body size, which is the width of
    roughly one short word — a ragged-right setting still reads as wrapped.

    Hyphenation is healed at a real wrap: a word broken across a line end by the
    typesetter is one word, and leaving the hyphen in makes it two, both wrong.
    """
    if not lines:
        return ""
    measure = max(line.x1 for line in lines)
    joined = lines[0].text.strip()
    for previous, line in zip(lines, lines[1:], strict=False):
        text = line.text.strip()
        if measure - previous.x1 > body_size * RAGGED_EMS:
            joined = f"{joined}\n{text}"
        elif _HYPHEN_END.search(joined):
            joined = _HYPHEN_END.sub(r"\1", joined) + text
        else:
            joined = f"{joined} {text}"
    return joined


class PdfExtractor:
    route = "pdf"

    def extract(self, path: Path, log: Callable[[str], None]) -> ParseResult:
        quiet_mupdf()
        with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
            log(f"pdf: reading {document.page_count} page(s)")
            lines = read_lines(document)
        typography = read_typography(lines)
        log(
            f"pdf: {len(lines)} lines, body {typography.dominant} at "
            f"{typography.body_size}pt"
        )
        if typography.code_families:
            log(
                f"pdf: code families ({typography.tier} tier): "
                f"{', '.join(sorted(typography.code_families))}"
            )

        blocks: list[Block] = []
        fences = 0
        for run in _runs(lines, typography):
            if run.is_code:
                blocks.append(_code_block(run, typography))
                fences += 1
            else:
                blocks.extend(_prose_blocks(run, typography))

        rule_counts = {f"pdf:{typography.tier}-family": fences} if fences else {}
        declines = tuple(
            reason for reason in (typography.code_decline, typography.heading_decline) if reason
        )
        log(f"pdf: {fences} fence(s), {sum(1 for b in blocks if b.kind == 'heading')} heading(s)")
        return ParseResult(
            markdown=to_gfm(blocks),
            report=ExtractorReport(
                route=self.route, rule_counts=rule_counts, declines=declines
            ),
        )
