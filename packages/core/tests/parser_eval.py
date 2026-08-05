"""Machinery for comparing the parsers in the chain on the same document.

Not a test module (no ``test_`` prefix, so pytest does not collect it): this is
what ``test_parser_eval.py`` drives.

The chain prefers ``marker`` over ``pymupdf`` for PDFs, but ``marker-pdf`` is not
a declared dependency and is not installed in CI, so in practice every PDF this
project has ever parsed went through ``pymupdf4llm`` and ``MarkerParser`` has
never run. Before adopting marker — a multi-gigabyte ML dependency — or removing
the branch of the chain that prefers it, the difference has to be measured on
the kind of document this project ingests.

What is measured, and why these five numbers:

- **headings** — what structure detection has to work from. The benchmark scores
  structure fidelity against a hand-authored table of contents, so a parser that
  loses headings loses that dimension directly.
- **fenced code blocks** — programming books are substantially code, and a
  listing flattened into prose is both wrong and actively misleading to a
  summariser.
- **page furniture** — running headers and bare page numbers left in the text
  stream end up in summaries and embeddings, where they are pure noise.
- **letter-spaced runs** — display type set with tracking ("S M A L L T A L K")
  extracts as garbage tokens.
- **characters per page** — the crude check that any text came out at all.

None of these is quality. They are proxies chosen because they are countable
without a human reading the output, and they should be read alongside a sample
of the markdown rather than instead of one.

A caution learned from running this: ``pymupdf4llm``'s code detection is
font-dependent rather than absent. It fences code set in a face it recognises as
monospace (70 fences per 20 pages of the GNU Bash manual) and misses code set in
a face it does not (0 per 40 pages of one typeset programming book). A single
document therefore cannot answer the question — run this over both a
well-behaved manual and an awkward book before concluding anything.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from booksmart_core.parsing import ParserChain, build_default_chain

# Five or more single letters separated by spaces: the signature of display type
# set with letter spacing, which extracts as one token per letter.
_LETTER_SPACED = re.compile(r"(?:[A-Za-z] ){4,}[A-Za-z]")


def _quiet_mupdf() -> None:
    """MuPDF narrates recoverable defects — a missing stylesheet in an EPUB is
    reported once per chapter and then ignored, which reads as a hang when it
    scrolls past a silent twenty-minute marker run. The defects are recoverable,
    so the narration is noise; anything genuinely fatal still raises."""
    pymupdf.TOOLS.mupdf_display_errors(False)  # type: ignore[no-untyped-call]
    pymupdf.TOOLS.mupdf_display_warnings(False)  # type: ignore[no-untyped-call]


@dataclass(frozen=True)
class Metrics:
    """Countable properties of one parser's markdown for one document."""

    parser: str
    pages: int
    characters: int
    headings: int
    code_fences: int
    letter_spaced_runs: int
    page_number_lines: int
    seconds: float

    @property
    def characters_per_page(self) -> int:
        return self.characters // self.pages if self.pages else 0

    def as_row(self) -> str:
        return (
            f"{self.parser:22} {self.pages:6} {self.characters_per_page:8} "
            f"{self.headings:9} {self.code_fences:7} {self.letter_spaced_runs:9} "
            f"{self.page_number_lines:8} {self.seconds:8.1f}"
        )

    @staticmethod
    def header() -> str:
        return (
            f"{'parser':22} {'pages':>6} {'chars/pg':>8} {'headings':>9} "
            f"{'fences':>7} {'spaced':>9} {'pagenos':>8} {'secs':>8}"
        )


def measure(markdown: str, *, parser: str, pages: int, seconds: float) -> Metrics:
    """Count the five proxies over one parser's output. Pure."""
    lines = markdown.splitlines()
    return Metrics(
        parser=parser,
        pages=pages,
        characters=len(markdown),
        headings=sum(1 for line in lines if line.lstrip().startswith("#")),
        # Fences come in pairs; an odd count means an unterminated block, which
        # is worth not rounding away, hence the explicit division.
        code_fences=markdown.count("```") // 2,
        letter_spaced_runs=sum(1 for line in lines if _LETTER_SPACED.search(line)),
        page_number_lines=sum(1 for line in lines if line.strip().isdigit()),
        seconds=seconds,
    )


def page_count(path: Path) -> int:
    _quiet_mupdf()
    doc = pymupdf.open(path)  # type: ignore[no-untyped-call]
    try:
        return int(doc.page_count)
    finally:
        doc.close()


def available_parsers(chain: ParserChain | None = None) -> list[str]:
    """Which parsers in the chain could actually run on a PDF here.

    Availability is environmental — marker is an undeclared optional import and
    OCR needs a system binary — so a comparison has to say which parsers it was
    able to include rather than assume three.
    """
    resolved = chain or build_default_chain()
    names: list[str] = []
    for parser in resolved._parsers:  # noqa: SLF001 - the chain exposes no accessor
        if "pdf" not in parser.supported_formats:
            continue
        names.append(parser.name)
    return names


def marker_is_installed() -> bool:
    """Whether ``MarkerParser`` would do anything but raise ParserUnavailable."""
    try:
        import marker.converters.pdf  # noqa: F401
    except ImportError:
        return False
    return True


def build_probe_pdf(path: Path, *, pages: int = 3) -> Path:
    """A small PDF carrying, deliberately, each thing the metrics count.

    Generated rather than committed: a fixture nobody can read the licence of is
    a fixture nobody can ship, and this one is reproducible from four lines of
    intent — a heading, a code listing whose lines must not be run together,
    a letter-spaced running header, and a bare page number.
    """
    listing = [
        "def extract(path):",
        "    return chain.extract(path)",
        "",
        "class Parser:",
        "    name = 'probe'",
    ]
    doc = pymupdf.open()  # type: ignore[no-untyped-call]
    for number in range(1, pages + 1):
        page = doc.new_page()
        page.insert_text((72, 40), "P R O B E   D O C U M E N T", fontsize=8)
        page.insert_text((72, 90), f"Chapter {number}: Extracting Text", fontsize=18)
        page.insert_textbox(
            pymupdf.Rect(72, 110, 520, 260),
            "Prose before the listing, long enough that the page carries real text "
            "and not only furniture. " * 4,
            fontsize=10,
        )
        page.insert_textbox(
            pymupdf.Rect(72, 270, 520, 400),
            "\n".join(listing),
            fontsize=9,
            fontname="cour",
        )
        page.insert_text((300, 760), str(number), fontsize=9)
    doc.save(path)
    doc.close()
    return path


def first_pages(path: Path, count: int, destination: Path, *, start: int = 0) -> Path:
    """A copy of the first ``count`` pages, for comparing parsers affordably.

    marker is an ML pipeline and orders of magnitude slower than pymupdf4llm, so
    running both over a whole manual is a poor first move — long enough that it
    is hard to tell a slow parse from a hung one. Both parsers see the identical
    truncated document, so the comparison stays fair; only its scope narrows.

    ``start`` exists because of where books keep their code. The first twenty
    pages of a technical manual are already full of listings; the first twenty
    pages of a book are cover, contents and preface — a slice from the front of
    one corpus book contained two lines of code, which silently turned a
    comparison of code handling into a comparison of front matter.
    """
    _quiet_mupdf()
    doc = pymupdf.open(path)  # type: ignore[no-untyped-call]
    if not doc.is_pdf:
        # select() is PDF-only, and a *ranged* convert_to_pdf fails outright on
        # reflowed documents — so convert the whole book and slice the result.
        # The comparison is unaffected: MuPDF's rendered layout is what every
        # parser here reads anyway, and span fonts keep their monospace flags
        # through the conversion (verified against a real EPUB).
        rendered = doc.convert_to_pdf()  # type: ignore[no-untyped-call]
        doc.close()
        doc = pymupdf.open("pdf", rendered)  # type: ignore[no-untyped-call]
        destination = destination.with_suffix(".pdf")
    try:
        first = min(start, max(doc.page_count - 1, 0))
        doc.select(range(first, min(first + count, doc.page_count)))
        # Repair while saving, unconditionally. Real books arrive with syntax
        # MuPDF tolerates on read and refuses to re-serialize — one corpus book
        # has a dict entry whose key is not a name, and a plain save dies on it
        # with FzErrorSyntax: invalid key in dict. Both parsers see the same
        # repaired file, so the comparison is unaffected; and unlike rebuilding
        # via insert_pdf, this keeps the outline, which pymupdf4llm reads for
        # headings.
        doc.save(destination, garbage=4, clean=True)
    finally:
        doc.close()
    return destination


def compare(
    path: Path,
    parsers: Sequence[str] | None = None,
    dump_to: Path | None = None,
) -> list[Metrics]:
    """Run each named parser over one document and measure its output.

    Each parser is driven through a one-element ``ParserChain`` rather than the
    default chain, because the default stops at the first success and the point
    here is to see them all.
    """
    import time

    _quiet_mupdf()
    wanted = set(parsers) if parsers else set(available_parsers())
    results: list[Metrics] = []
    total_pages = page_count(path)

    for parser in build_default_chain()._parsers:  # noqa: SLF001
        if parser.name not in wanted or "pdf" not in parser.supported_formats:
            continue
        started = time.perf_counter()
        file_format = "epub" if path.suffix.lower() == ".epub" else "pdf"
        # Narrate progress. A first marker run downloads models and then works
        # in silence for many minutes; without these lines that is
        # indistinguishable from a hang, and has been mistaken for one twice.
        print(f"  {parser.name}: parsing {total_pages} page(s)…", flush=True)
        try:
            markdown = ParserChain([parser]).extract(path, file_format, lambda _: None).markdown
        except Exception as exc:  # noqa: BLE001 - an unavailable parser is a result
            results.append(
                Metrics(
                    parser=f"{parser.name} ({type(exc).__name__})",
                    pages=total_pages, characters=0, headings=0, code_fences=0,
                    letter_spaced_runs=0, page_number_lines=0,
                    seconds=time.perf_counter() - started,
                )
            )
            continue
        print(
            f"  {parser.name}: done in {time.perf_counter() - started:.0f}s", flush=True
        )
        if dump_to is not None:
            # A marker run costs twenty minutes; throwing its output away and
            # keeping five integers is a poor trade. The counts say *whether*
            # the parsers differ; only the markdown says *how*.
            dump_to.mkdir(parents=True, exist_ok=True)
            (dump_to / f"{path.stem}.{parser.name}.md").write_text(markdown)
        results.append(
            measure(
                markdown,
                parser=parser.name,
                pages=total_pages,
                seconds=time.perf_counter() - started,
            )
        )
    return results


def render(results: Sequence[Metrics]) -> str:
    return "\n".join([Metrics.header(), *(result.as_row() for result in results)])
