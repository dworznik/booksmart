"""One document, one route, no fallback.

Routing is by format, and for PDFs by whether the text layer is actually carrying
the book. That second question is asked *before* anything is extracted, because
the alternative — try the better extractor, fall back if it fails — does not work here:
the failure mode was measured as silently dropping text, not raising. A
fallback triggered by exceptions would have caught none of the cases it was
written for, while making an artifact's provenance unknowable afterwards.

So a failing route is a failing parse. The `ParseFailure` names the route, which
is the first thing anybody debugging a bad artifact needs.
"""

from collections.abc import Callable
from pathlib import Path

import pymupdf

from booksmart_core.parsing.contract import (
    Extractor,
    ParseFailure,
    ParseResult,
    ExtractorUnavailable,
    Route,
)
from booksmart_core.parsing.epub import EpubExtractor
from booksmart_core.parsing.mupdf import quiet_mupdf
from booksmart_core.parsing.ocr import OcrExtractor
from booksmart_core.parsing.pdf import PdfExtractor

# Below this many characters per page, averaged over pages spread through the
# document, the text layer is not carrying the book and OCR is the only route that
# will read it. `booksmart-bench`'s `sources` verb reads the same constant from
# here to warn about a scan arriving where a native PDF was expected — the warning
# and the routing decision have to *be* the same decision, or the handover check
# passes a file the pipeline then routes somewhere else.
MIN_CHARS_PER_PAGE = 200

# How many pages the probe reads. Spread evenly rather than taken from the front:
# the first pages of a book are cover, half title and copyright, several of which
# are images in a perfectly healthy PDF.
PROBE_PAGES = 24


def text_layer_density(path: Path) -> int:
    """Characters of extractable text per page, over a spread sample.

    A scan has near-zero everywhere and a native PDF has thousands everywhere, so
    the question does not need every page — which matters, because this runs on
    every ingest and reading a 1,284-page book in full costs seconds.

    A file MuPDF cannot open at all is a ``ParseFailure`` rather than whatever
    MuPDF chose to raise. This runs *before* the route is chosen, so an unwrapped
    error here is the one failure that escapes without naming a route — and a
    corrupt download is exactly the case the caller is trying to route.
    """
    quiet_mupdf()
    try:
        with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
            pages = document.page_count
            if not pages:
                return 0
            step = max(1, pages // PROBE_PAGES)
            sampled = range(0, pages, step)
            return int(
                sum(len(document[index].get_text()) for index in sampled) / len(sampled)
            )
    except ParseFailure:
        raise
    except Exception as exc:  # noqa: BLE001 — MuPDF raises its own hierarchy
        raise ParseFailure(
            f"{path.name} could not be opened as a PDF, so no route can be chosen ({exc})"
        ) from exc


class ExtractorRouter:
    """Picks the one extractor for a document, and holds it to its result."""

    def __init__(self, *, epub: Extractor, pdf: Extractor, ocr: Extractor) -> None:
        self._by_route: dict[Route, Extractor] = {"epub": epub, "pdf": pdf, "pdf-ocr": ocr}

    def route_for(self, path: Path, file_format: str, log: Callable[[str], None]) -> Route:
        """Which route this document takes, and why, in the log."""
        if file_format == "epub":
            log("route: epub (read from the container)")
            return "epub"
        if file_format == "pdf":
            density = text_layer_density(path)
            if density >= MIN_CHARS_PER_PAGE:
                log(f"route: pdf (~{density} characters per page of text layer)")
                return "pdf"
            log(
                f"route: pdf-ocr (~{density} characters per page, under "
                f"{MIN_CHARS_PER_PAGE} — the text layer is not carrying this book)"
            )
            return "pdf-ocr"
        raise ParseFailure(
            f"no route for {file_format!r} files; this pipeline reads pdf and epub"
        )

    def extract(self, path: Path, file_format: str, log: Callable[[str], None]) -> ParseResult:
        route = self.route_for(path, file_format, log)
        extractor = self._by_route[route]
        try:
            result = extractor.extract(path, log)
        except ExtractorUnavailable as exc:
            raise ParseFailure(f"the {route} route is unavailable — {exc}") from exc
        except ParseFailure as exc:
            # Re-raised with the route in front but its own words intact. An
            # extractor that has already said something precise — a spine item
            # absent from the archive — should not have that buried under a
            # generic wrapper; but "which route produced this" is the first
            # question a bad artifact raises, so the message has to answer it.
            raise ParseFailure(f"the {route} route failed — {exc}") from exc
        except Exception as exc:
            raise ParseFailure(f"the {route} route failed — {type(exc).__name__}: {exc}") from exc
        if not any(character.isalnum() for character in result.markdown):
            # A parse that silently produced nothing is far worse than one that
            # raises: an empty book ingests, and then scores zero on everything.
            raise ParseFailure(f"the {route} route produced no text content")
        for reason in result.report.declines:
            log(f"{route}: declined — {reason}")
        log(f"{route}: succeeded")
        return result


def build_default_router() -> ExtractorRouter:
    """The router every consumer gets unless it injects its own.

    Built per call rather than held at module level. The module-level chain this
    replaces existed only to amortise marker's model load; every remaining
    extractor is stateless, so a shared instance buys nothing and costs the
    ability to substitute one in a test.
    """
    return ExtractorRouter(
        epub=EpubExtractor(),
        pdf=PdfExtractor(),
        ocr=OcrExtractor(),
    )
