"""Throwaway: adjudicate rule A (mono ladder) vs rule B (dominant-font contrast).

Ticket: dworznik/booksmart#88. Not repo code — deleted once the decision lands.

Rule A (the incoming spec): per span, mono flag -> mono name regex -> glyph
advance uniformity (CV < 0.02). A line is code when mono chars are >= 80% of
its non-whitespace chars.

Rule B (issue #83, measured): the page-dominant body font family is prose; a
maximal run of consecutive lines set in a non-dominant family is code.
Guard rails: size-outliers (headings/captions) excluded, italic variants of the
body family are emphasis not code.
"""

import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.TOOLS.mupdf_display_warnings(False)

MONO_NAME = re.compile(
    r"(?i)mono|courier|consolas|menlo|inconsolata|source ?code|fira ?code"
    r"|jetbrains|dejavu ?sans ?mono|pt ?mono|nimbus ?mono|letter ?gothic"
    r"|lucida ?console|andale"
)
MONO_FLAG = 1 << 3
ITALIC_FLAG = 1 << 1

# Style tokens only count as style when a separator precedes them, so
# "NewCenturySchlbk-Roman" folds to "NewCenturySchlbk" while "TimesNewRoman"
# is left alone. Folding too eagerly merges genuinely distinct families.
STYLE_SUFFIX = re.compile(
    r"[-,_ ](?:bold|italic|oblique|regular|roman|light|medium|semibold|book|black"
    r"|heavy|condensed|cond|demi|it|bolditalic|semibolditalic|typnarr)+$",
    re.IGNORECASE,
)


def family(fontname: str) -> str:
    name = fontname.split("+")[-1]
    previous = None
    while previous != name:
        previous = name
        name = STYLE_SUFFIX.sub("", name)
    return name


@dataclass
class Line:
    text: str
    y: float
    x0: float
    page: int
    size: float
    families: Counter = field(default_factory=Counter)   # family -> char count
    mono_chars: int = 0
    nonspace_chars: int = 0
    italic_chars: int = 0

    @property
    def top_family(self) -> str:
        return self.families.most_common(1)[0][0] if self.families else ""

    @property
    def mono_ratio(self) -> float:
        return self.mono_chars / self.nonspace_chars if self.nonspace_chars else 0.0

    @property
    def italic_ratio(self) -> float:
        return self.italic_chars / self.nonspace_chars if self.nonspace_chars else 0.0


def glyph_advance_uniform(chars: list[dict]) -> bool:
    """Tier 3 of rule A: constant glyph advance means a monospaced face."""
    widths = [c["bbox"][2] - c["bbox"][0] for c in chars if c["c"].strip()]
    if len(widths) < 6:
        return False
    mean = statistics.fmean(widths)
    if mean <= 0:
        return False
    return (statistics.pstdev(widths) / mean) < 0.02


def read_lines(doc, page_numbers: list[int]) -> list[Line]:
    lines: list[Line] = []
    for number in page_numbers:
        page = doc[number]
        data = page.get_text("rawdict")
        for block in data["blocks"]:
            if block.get("type") != 0:
                continue
            for raw in block["lines"]:
                text_parts, sizes = [], []
                line = Line(text="", y=raw["bbox"][1], x0=raw["bbox"][0],
                            page=number, size=0.0)
                for span in raw["spans"]:
                    chars = span.get("chars", [])
                    text = "".join(c["c"] for c in chars)
                    text_parts.append(text)
                    nonspace = sum(1 for c in text if not c.isspace())
                    if not nonspace:
                        continue
                    sizes.append(span["size"])
                    fam = family(span["font"])
                    line.families[fam] += nonspace
                    line.nonspace_chars += nonspace
                    if span["flags"] & ITALIC_FLAG:
                        line.italic_chars += nonspace
                    is_mono = bool(span["flags"] & MONO_FLAG) or bool(
                        MONO_NAME.search(span["font"])
                    ) or glyph_advance_uniform(chars)
                    if is_mono:
                        line.mono_chars += nonspace
                line.text = "".join(text_parts)
                line.size = max(sizes) if sizes else 0.0
                if line.nonspace_chars:
                    lines.append(line)
    return lines


def rule_a(lines: list[Line]) -> list[bool]:
    return [line.mono_ratio >= 0.80 for line in lines]


def rule_b(lines: list[Line], min_run: int = 2) -> tuple[list[bool], str, float]:
    """Dominant family by character mass; body size as the modal line size."""
    mass: Counter = Counter()
    for line in lines:
        mass.update(line.families)
    dominant = mass.most_common(1)[0][0] if mass else ""

    sizes = Counter(round(line.size, 1) for line in lines)
    body_size = sizes.most_common(1)[0][0] if sizes else 0.0

    candidate = []
    for line in lines:
        contrasting = line.top_family != dominant
        # Guard rails: display type is a heading, not a listing; an italic run
        # in the body family is emphasis.
        oversized = line.size > body_size * 1.15
        emphasis = line.italic_ratio >= 0.5 and line.top_family == dominant
        candidate.append(contrasting and not oversized and not emphasis)

    # Only maximal runs of >= min_run consecutive contrasting lines survive: a
    # lone contrasting line is an inline term or a caption, not a listing.
    flags = [False] * len(candidate)
    start = None
    for index, value in enumerate([*candidate, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_run:
                for j in range(start, index):
                    flags[j] = True
            start = None
    return flags, dominant, body_size


def runs(flags: list[bool]) -> int:
    return sum(1 for i, v in enumerate(flags) if v and (i == 0 or not flags[i - 1]))


def sample_pages(total: int, count: int, fraction: float = 0.40) -> list[int]:
    start = int(total * fraction)
    return list(range(start, min(start + count, total)))


def main() -> None:
    directory = Path("/home/patryk/booksmart-bench/sources")
    books = sorted(p for p in directory.glob("*.pdf"))
    pages = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    dump = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"{'book':38} {'lines':>6} {'A':>6} {'B':>6} {'both':>6} "
          f"{'A-only':>7} {'B-only':>7} {'Bruns':>6}  dominant/body")
    for path in books:
        doc = pymupdf.open(path)
        lines = read_lines(doc, sample_pages(doc.page_count, pages))
        a = rule_a(lines)
        b, dominant, body_size = rule_b(lines)
        both = sum(1 for x, y in zip(a, b) if x and y)
        print(
            f"{path.stem:38} {len(lines):6} {sum(a):6} {sum(b):6} {both:6} "
            f"{sum(a) - both:7} {sum(b) - both:7} {runs(b):6}  "
            f"{dominant}/{body_size}"
        )
        if dump and dump in path.stem:
            print(f"\n--- {path.stem}: A=mono-ladder B=font-contrast ---")
            for line, av, bv in zip(lines, a, b):
                mark = f"{'A' if av else '.'}{'B' if bv else '.'}"
                fam = line.top_family[:18]
                print(f"  {mark} p{line.page:<4} x{line.x0:6.1f} {fam:18} "
                      f"{line.text[:78]!r}")
            print("--- end dump ---\n")
        doc.close()


if __name__ == "__main__":
    main()
