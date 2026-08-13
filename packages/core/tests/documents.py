"""Documents to test extraction against, built rather than committed.

Not a test module (no ``test_`` prefix, so pytest does not collect it).

Generated rather than checked in, for two reasons. A fixture nobody can read the
licence of is a fixture nobody can ship — no book text ever lands in this repo —
and each of these is reproducible from a few lines of stated intent, which is
worth more to the next reader than a binary blob.

This began life as ``parser_eval.py``, the harness for comparing marker against
pymupdf4llm over one document. That comparison is answered and its
loser is gone: marker measured strictly worse on every axis, and there is now one
extractor per route with nothing to compare it against in-process. What survives
is the part that was never about the comparison — the document builders — because
the extraction gate needs a document whose answer is known before anything reads
it. The five countable proxies that harness measured are superseded by
``booksmart_core.parse_metrics``, which measures a superset of them and measures
them against the source rather than in the abstract.
"""

from pathlib import Path

import pymupdf


def quiet_mupdf() -> None:
    """MuPDF narrates recoverable defects — a missing stylesheet in an EPUB is
    reported once per chapter and then ignored. The defects are recoverable, so
    the narration is noise; anything genuinely fatal still raises."""
    pymupdf.TOOLS.mupdf_display_errors(False)  # type: ignore[no-untyped-call]
    pymupdf.TOOLS.mupdf_display_warnings(False)  # type: ignore[no-untyped-call]


def page_count(path: Path) -> int:
    quiet_mupdf()
    doc = pymupdf.open(path)  # type: ignore[no-untyped-call]
    try:
        return int(doc.page_count)
    finally:
        doc.close()


def build_probe_pdf(path: Path, *, pages: int = 3) -> Path:
    """A small PDF carrying, deliberately, each thing extraction has to handle:
    a heading, a code listing whose lines must not be run together, a
    letter-spaced running header, a paragraph of real prose, and a bare page
    number."""
    listing = [
        "def extract(path):",
        "    return router.extract(path)",
        "",
        "class Extractor:",
        "    route = 'probe'",
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
    """A copy of ``count`` pages from ``start``.

    Used by the gate's one deliberate-omission test: three pages in, two pages
    out, and the missing page has to come back as a long dropped run. A truncation
    whose size is known is the only way to know a loss metric can see one.

    ``start`` exists because of where books keep their code. The first twenty
    pages of a manual are already full of listings; the first twenty pages of a
    book are cover, contents and preface — a slice from the front of one corpus
    book contained two lines of code, which silently turned a comparison of code
    handling into a comparison of front matter.
    """
    quiet_mupdf()
    doc = pymupdf.open(path)  # type: ignore[no-untyped-call]
    if not doc.is_pdf:
        # select() is PDF-only, and a *ranged* convert_to_pdf fails outright on
        # reflowed documents — so convert the whole book and slice the result.
        rendered = doc.convert_to_pdf()  # type: ignore[no-untyped-call]
        doc.close()
        doc = pymupdf.open("pdf", rendered)  # type: ignore[no-untyped-call]
        destination = destination.with_suffix(".pdf")
    try:
        # Refused rather than clamped. `parse-gate.yaml` names a start page per
        # book by hand, so a start past the end is a typo — and clamping it to the
        # last page produced a one-page slice that gated the index instead of the
        # chapter, reporting numbers that looked like a real measurement.
        if start >= doc.page_count:
            raise ValueError(
                f"{path.name} has {doc.page_count} page(s), so a slice starting at "
                f"page {start} is past the end of the book"
            )
        doc.select(range(start, min(start + count, doc.page_count)))
        # Repair while saving, unconditionally. Real books arrive with syntax
        # MuPDF tolerates on read and refuses to re-serialize — one corpus book
        # has a dict entry whose key is not a name, and a plain save dies on it
        # with FzErrorSyntax: invalid key in dict. And unlike rebuilding via
        # insert_pdf, this keeps the outline.
        doc.save(destination, garbage=4, clean=True)
    finally:
        doc.close()
    return destination
