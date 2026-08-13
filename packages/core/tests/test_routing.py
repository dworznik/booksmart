"""Routing through the parse stage, against real files and a real database.

What is checked here and not in `test_router.py` is the part a stub cannot
tell you: that `Book.parser_used` ends up holding a *route* name, that the run log
records the routing decision, and that a scan really does come back as text.
`parser_used` keeps its column and its schema — there is no migration — and
`EXTRACTION_VERSION` is what says which vocabulary a row is written in:

    "1"  marker | pymupdf | ocr
    "2"  epub | pdf | pdf-ocr
    "3"  epub | pdf | pdf-ocr, with the heading set taken from the container

The OCR tests need tesseract on the machine (present in CI and the image).
"""

import io
import shutil
import uuid
import zipfile
from collections.abc import Callable
from pathlib import Path

import pymupdf
import pytest
from sqlalchemy.orm import Session, sessionmaker

from booksmart_core.config import Settings
from booksmart_core.parsing.contract import ExtractorReport
from booksmart_core.parsing import (
    EXTRACTION_VERSION,
    OcrExtractor,
    ocr_markdown,
    ParseFailure,
    ParseResult,
    ExtractorRouter,
    build_default_router,
)
from booksmart_core.runner import execute_run
from booksmart_core.storage import BookStorage

from .conftest import get_run, run_scope
from .test_ingestion_api import CORRUPT_PDF_BYTES, make_pdf_bytes, register_book

OCR_TEXT = "SCANNED BOOK FALLBACK"


def make_scanned_pdf_bytes(text: str = OCR_TEXT) -> bytes:
    """A PDF containing only a rasterized image of text — no text layer at all."""
    source = pymupdf.open()
    page = source.new_page()
    page.insert_text((72, 150), text, fontsize=36)
    pixmap = page.get_pixmap(dpi=200)
    source.close()

    scanned = pymupdf.open()
    image_page = scanned.new_page()
    image_page.insert_image(image_page.rect, pixmap=pixmap)
    data: bytes = scanned.tobytes()
    scanned.close()
    return data


def make_epub_bytes(text: str = "Deep modules hide complexity behind simple interfaces") -> bytes:
    """A minimal but valid EPUB with one chapter of real text."""
    container = (
        '<?xml version="1.0"?>'
        '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>'
        "</rootfiles></container>"
    )
    opf = (
        '<?xml version="1.0"?>'
        '<package version="2.0" xmlns="http://www.idpf.org/2007/opf" unique-identifier="id">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        '<dc:title>Test Book</dc:title><dc:identifier id="id">test-book</dc:identifier>'
        "<dc:language>en</dc:language></metadata>"
        '<manifest><item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/></manifest>'
        '<spine><itemref idref="ch1"/></spine></package>'
    )
    chapter = (
        '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml">'
        f"<head><title>Chapter 1</title></head><body><h1>Chapter 1</h1><p>{text}</p></body></html>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("content.opf", opf)
        archive.writestr("ch1.xhtml", chapter)
    return buffer.getvalue()


def run_job(
    session_factory: sessionmaker[Session], settings: Settings, book_id: str
) -> dict[str, object]:
    """Run a full ingest synchronously and return the finished Run."""
    return run_scope(session_factory, settings, book_id, "full")


class TestWhatParserUsedHolds:
    def test_a_text_pdf_records_the_pdf_route(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage)

        run = run_job(session_factory, settings, book_id)

        assert run["status"] == "succeeded"
        assert run["parser_used"] == "pdf"

    def test_an_epub_records_the_epub_route(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(
            session_factory, storage, filename="apod.epub", content=make_epub_bytes()
        )

        run = run_job(session_factory, settings, book_id)

        assert run["status"] == "succeeded"
        assert run["parser_used"] == "epub"
        parsed = storage.resolve(str(run["output_path"]))
        assert "Deep modules hide complexity" in parsed.read_text(encoding="utf-8")

    def test_the_route_vocabulary_outlives_the_stamp_that_introduced_it(self) -> None:
        """A row's `parser_used` is only readable beside its version stamp. Rows
        written under "1" say `pymupdf`, and that is still a true statement about
        what produced them — which is why there is no migration. "3" keeps "2"'s
        vocabulary and changes what a heading is, so the stamp moves and the route
        names do not."""
        assert EXTRACTION_VERSION == "3"


class TestParseLogs:
    def test_a_successful_run_records_the_routing_decision(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage)

        run = run_job(session_factory, settings, book_id)

        log_file = Path(settings.storage_root) / "logs" / f"{run['id']}.log"
        content = log_file.read_text(encoding="utf-8")
        assert "route: pdf" in content
        assert "succeeded" in content
        assert "marker" not in content, "there is no marker route to attempt"

    def test_a_failed_run_also_writes_a_log(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage, content=CORRUPT_PDF_BYTES)

        run = run_job(session_factory, settings, book_id)

        assert run["status"] == "failed"
        log_file = Path(settings.storage_root) / "logs" / f"{run['id']}.log"
        assert log_file.exists()


class TestAFailingRouteFailsTheRun:
    def test_there_is_no_second_attempt(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        """Under the old chain this run succeeded via a fallback. It now fails,
        which is the change: an artifact whose provenance is unknowable is worse
        than a run that stopped and said which route broke."""
        book_id = register_book(session_factory, storage)
        default = build_default_router()
        router = ExtractorRouter(
            epub=BrokenExtractor("epub"),
            pdf=BrokenExtractor("pdf"),
            ocr=default._by_route["pdf-ocr"],  # noqa: SLF001
        )

        run_id = execute_run(
            session_factory, settings.storage_root, uuid.UUID(book_id), "full", router=router
        )

        run = get_run(session_factory, str(run_id))
        assert run is not None
        assert run["status"] == "failed"
        assert "pdf route failed" in str(run["error"])


class BrokenExtractor:
    """A route whose extractor cannot read the file it was handed."""

    def __init__(self, route: str) -> None:
        self.route = route

    def extract(self, path: Path, log: Callable[[str], None]) -> ParseResult:
        raise RuntimeError("simulated extractor failure")


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
class TestTheOcrRoute:
    def test_it_reads_a_scanned_pdf(self, tmp_path: Path) -> None:
        scan = tmp_path / "scan.pdf"
        scan.write_bytes(make_scanned_pdf_bytes())

        result = OcrExtractor().extract(scan, lambda _: None)

        assert "SCANNED" in result.markdown.upper()

    def test_it_emits_no_page_headings(self, tmp_path: Path) -> None:
        """`## Page N` was furniture entering as *structure*: `detect_structure`
        takes the minimum heading level present, so a book of two hundred of
        these had two hundred chapters and none of its real ones."""
        scan = tmp_path / "scan.pdf"
        scan.write_bytes(make_scanned_pdf_bytes())

        result = OcrExtractor().extract(scan, lambda _: None)

        assert "## Page" not in result.markdown
        assert not any(line.startswith("#") for line in result.markdown.splitlines())

    def test_it_declines_code_detection_and_says_why(self, tmp_path: Path) -> None:
        """OCR returns recognised glyphs and no font information, so every signal
        the other routes separate code from prose with is simply absent (ADR
        0003)."""
        scan = tmp_path / "scan.pdf"
        scan.write_bytes(make_scanned_pdf_bytes())

        result = OcrExtractor().extract(scan, lambda _: None)

        assert result.report.declines
        assert "no font signal" in result.report.declines[0]
        assert "```" not in result.markdown

    def test_a_scanned_pdf_ingests_and_records_the_ocr_route(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        """Under the old chain either the pymupdf step or the OCR step could have
        won, and the artifact did not say which. The probe decides now, before
        anything is extracted."""
        book_id = register_book(
            session_factory, storage, filename="scan.pdf", content=make_scanned_pdf_bytes()
        )

        run = run_job(session_factory, settings, book_id)

        assert run["status"] == "succeeded"
        assert run["parser_used"] == "pdf-ocr"
        parsed = storage.resolve(str(run["output_path"])).read_text(encoding="utf-8")
        assert "SCANNED" in parsed.upper()

    def test_a_text_pdf_never_reaches_ocr(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage, content=make_pdf_bytes())

        run = run_job(session_factory, settings, book_id)

        assert run["parser_used"] == "pdf"


class TestTheReportReachesTheStage:
    def test_a_declining_route_still_produces_an_artifact(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        """A decline is not a failure. The book comes out as prose, and the reason
        is recorded — which is the whole of ADR 0003's third constraint."""
        book_id = register_book(session_factory, storage)
        declining = DecliningExtractor()
        router = ExtractorRouter(epub=declining, pdf=declining, ocr=declining)

        run_id = execute_run(
            session_factory, settings.storage_root, uuid.UUID(book_id), "full", router=router
        )

        run = get_run(session_factory, str(run_id))
        assert run is not None
        assert run["status"] == "succeeded"
        log = (Path(settings.storage_root) / "logs" / f"{run_id}.log").read_text(encoding="utf-8")
        assert "indentation is CSS-only" in log


class DecliningExtractor:
    route = "pdf"

    def extract(self, path: Path, log: Callable[[str], None]) -> ParseResult:
        return ParseResult(
            markdown="# A Chapter\n\nProse only, with no fences claimed.",
            report=ExtractorReport(route=self.route, declines=("indentation is CSS-only",)),
        )


def test_parse_failure_is_importable_from_the_package_surface() -> None:
    """A consumer catches this; it must not move into a submodule path."""
    assert issubclass(ParseFailure, Exception)


class TestOcrTextIsNotReadAsMarkdown:
    """OCR returns the book's own typography as plain text, and the book's
    typography is full of characters GFM reads as structure.

    Tested through `ocr_markdown` rather than through `extract`, because the
    escaping is the part worth pinning and tesseract is the part that is not
    always installed.
    """

    @pytest.mark.parametrize(
        "recognised",
        [
            "# 3",  # a page number under a hash-like mark
            "- item one",  # a bulleted list
            "> quoted line",
            "1. numbered",
            "```",
        ],
    )
    def test_a_page_never_becomes_markdown_structure(self, recognised: str) -> None:
        rendered = ocr_markdown([recognised])

        assert not rendered.lstrip().startswith(("#", "-", ">", "```"))

    def test_the_route_still_claims_no_structure_at_all(self) -> None:
        """`detect_structure` takes the minimum heading level present, so a
        single invented heading is not a small error — it becomes the book's
        chapter level and every real one is pushed below it."""
        rendered = ocr_markdown(["# 3", "## Chapter", "ordinary prose"])

        assert not any(line.startswith("#") for line in rendered.splitlines())

    def test_ordinary_prose_survives_unharmed(self) -> None:
        rendered = ocr_markdown(["Widgets mesh with sprockets.", "Grommets seal the joint."])

        assert "Widgets mesh with sprockets." in rendered
        assert "Grommets seal the joint." in rendered
