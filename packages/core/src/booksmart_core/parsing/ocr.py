"""The OCR route: a PDF whose text layer is not carrying the book.

Promoted from the old chain's last fallthrough to a route entered on evidence —
a chars-per-page probe the router runs before anything is extracted. That change
is the point: as a fallthrough it was reached by accident, and a scan and a
merely-awkward book were indistinguishable in the artifact afterwards.

Two things it does *not* do.

It declines code detection outright. OCR returns recognised glyphs and no font
information at all, so every signal the other routes separate code from prose
with — family, size, glyph advance, indentation — is simply absent. Per ADR 0003
a signal that is not reliable declines and reports, rather than guessing.

And it no longer emits `## Page N` headings. Page numbers are furniture, and
furniture entering as *structure* is the expensive kind: `detect_structure` takes
the minimum heading level present, so a book of two hundred `## Page N` headings
had two hundred chapters and no real ones.
"""

import os
import shutil
from collections.abc import Callable, Sequence
from glob import glob
from pathlib import Path

import pymupdf

from booksmart_core.parsing.blocks import Block, to_gfm
from booksmart_core.parsing.contract import ExtractorReport, ParseResult, ExtractorUnavailable
from booksmart_core.parsing.mupdf import quiet_mupdf

TESSDATA_CANDIDATES = (
    "/opt/homebrew/share/tessdata",
    "/usr/local/share/tessdata",
    "/usr/share/tessdata",
    "/usr/share/tesseract-ocr/*/tessdata",
)

# Rendering resolution handed to tesseract. High enough that body text at a
# typical book size resolves; higher costs time and returns nothing.
OCR_DPI = 300


def find_tessdata() -> str | None:
    configured = os.environ.get("TESSDATA_PREFIX")
    if configured and Path(configured).is_dir():
        return configured
    for pattern in TESSDATA_CANDIDATES:
        for match in sorted(glob(pattern)):
            if Path(match).is_dir():
                return match
    return None


def ocr_markdown(pages: Sequence[str]) -> str:
    """Recognised pages as GFM, with the book's own typography kept out of it.

    Every page goes through the block serialiser, which escapes the line-leading
    characters GFM reads as structure. Joining the raw recognised text instead
    let a page beginning `# 3` — a page number under a hash-like mark — become an
    h1, and a bulleted list become a real list. Structure detection reads this
    Markdown, so those inventions moved chapter boundaries nothing in the book
    supports.

    A separate function from ``extract`` because the escaping is the part worth
    testing and the tesseract binary is the part that is not always installed.
    """
    return to_gfm(Block(kind="paragraph", text=page) for page in pages)


class OcrExtractor:
    route = "pdf-ocr"

    def extract(self, path: Path, log: Callable[[str], None]) -> ParseResult:
        if shutil.which("tesseract") is None:
            raise ExtractorUnavailable("tesseract binary not found")
        tessdata = find_tessdata()
        if tessdata is None:
            raise ExtractorUnavailable("tessdata directory not found; set TESSDATA_PREFIX")

        quiet_mupdf()
        pages: list[str] = []
        recognised = 0
        with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
            for number, page in enumerate(document, start=1):
                textpage = page.get_textpage_ocr(full=True, dpi=OCR_DPI, tessdata=tessdata)
                text = page.get_text(textpage=textpage).strip()
                if text:
                    pages.append(text)
                    recognised += 1
                if number % 25 == 0:
                    log(f"pdf-ocr: {number} of {document.page_count} pages")
        log(f"pdf-ocr: recognised text on {recognised} page(s)")
        return ParseResult(
            # No page heading and no rule fired, so the report carries no rule
            # counts: an artifact from this route is prose all the way down, by
            # construction. `ocr_markdown` is what keeps the book's own
            # typography from becoming Markdown structure.
            markdown=ocr_markdown(pages),
            report=ExtractorReport(
                route=self.route,
                declines=(
                    "no font signal: OCR returns glyphs without font information",
                    # Said independently of the font decline above. They are two
                    # different absences — one is why no code was fenced, this is
                    # why no chapter tree was claimed — and a reader of the report
                    # cannot infer the second from the first.
                    "no heading signal: OCR returns no type size, so no structure is claimed",
                ),
            ),
        )
