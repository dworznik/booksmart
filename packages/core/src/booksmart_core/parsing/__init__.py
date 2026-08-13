"""Turning a book file into one GFM string, by format.

The public surface is small on purpose, and it is the whole of what leaves this
package (ADR 0003): an `ExtractorRouter` that picks exactly one route per document,
the `ParseResult` it returns, and the two failures a caller can see. Any block IR
an extractor builds internally stays internal — downstream stages read Markdown
and nothing else, so a change to how a format is read cannot reach them.

Three routes, no fallback:

    epub                      -> the OCF container, read directly
    pdf, healthy text layer   -> the page's own spans
    pdf, text layer absent    -> OCR

There is deliberately no second attempt. The previous chain's real failure mode
was measured as *silently dropping text* rather than raising — so
exception-triggered fallback catches almost nothing, while guaranteeing that
nobody can tell afterwards which extractor produced an artifact.
"""

from booksmart_core.parsing.contract import (
    EXTRACTION_VERSION,
    Extractor,
    ExtractorReport,
    ExtractorUnavailable,
    ParseFailure,
    ParseResult,
    Route,
)
from booksmart_core.parsing.epub import EpubExtractor
from booksmart_core.parsing.mupdf import quiet_mupdf
from booksmart_core.parsing.ocr import OcrExtractor, ocr_markdown
from booksmart_core.parsing.pdf import PdfExtractor
from booksmart_core.parsing.router import (
    MIN_CHARS_PER_PAGE,
    ExtractorRouter,
    build_default_router,
    text_layer_density,
)

__all__ = [
    "EXTRACTION_VERSION",
    "MIN_CHARS_PER_PAGE",
    "EpubExtractor",
    "Extractor",
    "ExtractorReport",
    "ExtractorRouter",
    "ExtractorUnavailable",
    "OcrExtractor",
    "ocr_markdown",
    "ParseFailure",
    "ParseResult",
    "PdfExtractor",
    "Route",
    "build_default_router",
    "quiet_mupdf",
    "text_layer_density",
]
