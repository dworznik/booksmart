"""Integration tests for the parser preference chain inside the parse stage.

Marker is an optional dependency and not installed in dev/CI, so the chain's
first attempt is always logged as unavailable and extraction falls through to
PyMuPDF; the OCR tests need tesseract on the machine (present in CI and the
image).
"""

import io
import shutil
import uuid
import zipfile
from pathlib import Path

import pymupdf
import pytest
from sqlalchemy.orm import Session, sessionmaker

from booksmart_core.config import Settings
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


class TestParserRecording:
    def test_text_pdf_records_pymupdf_as_parser_used(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage)

        run = run_job(session_factory, settings, book_id)

        assert run["status"] == "succeeded"
        assert run["parser_used"] == "pymupdf"


class TestParseLogs:
    def test_successful_run_writes_log_with_all_attempts(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage)

        run = run_job(session_factory, settings, book_id)

        log_file = Path(settings.storage_root) / "logs" / f"{run['id']}.log"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "marker" in content  # attempted first, unavailable
        assert "pymupdf" in content and "succeeded" in content

    def test_failed_run_also_writes_log(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage, content=CORRUPT_PDF_BYTES)

        run = run_job(session_factory, settings, book_id)

        assert run["status"] == "failed"
        log_file = Path(settings.storage_root) / "logs" / f"{run['id']}.log"
        assert log_file.exists()
        assert "pymupdf" in log_file.read_text(encoding="utf-8")


class TestParserChainFallback:
    def test_run_succeeds_via_fallback_when_preferred_parser_fails(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        from booksmart_core.parsing import ParserChain, PyMuPDFParser

        book_id = register_book(session_factory, storage)
        chain = ParserChain([FailingParser(), PyMuPDFParser()])

        run_id = execute_run(
            session_factory, settings.storage_root, uuid.UUID(book_id), "full", chain=chain
        )

        run = get_run(session_factory, str(run_id))
        assert run is not None
        assert run["status"] == "succeeded"
        assert run["parser_used"] == "pymupdf"
        log = (Path(settings.storage_root) / "logs" / f"{run_id}.log").read_text(encoding="utf-8")
        assert "always-fails" in log and "failed" in log


class TestEpubExtraction:
    def test_valid_epub_extracts_to_markdown(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage, filename="apod.epub", content=make_epub_bytes())

        run = run_job(session_factory, settings, book_id)

        assert run["status"] == "succeeded"
        assert run["parser_used"] == "pymupdf"
        parsed = Path(str(run["output_path"]))
        assert "Deep modules hide complexity" in parsed.read_text(encoding="utf-8")


class FailingParser:
    """Stands in for an earlier chain stage that cannot handle the file."""

    name = "always-fails"
    supported_formats = frozenset({"pdf"})

    def parse(self, path: Path) -> str:
        raise RuntimeError("simulated parser failure")


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
class TestOcrFallback:
    """The standalone OCR stage. Note: pymupdf4llm has its own integrated OCR for
    image-only pages when tesseract is present, so with real parsers a scanned PDF
    usually succeeds at the pymupdf step; the OCR stage is the safety net behind it."""

    def test_ocr_parser_reads_scanned_pdf(self, tmp_path: Path) -> None:
        from booksmart_core.parsing import OcrParser

        scan = tmp_path / "scan.pdf"
        scan.write_bytes(make_scanned_pdf_bytes())

        markdown = OcrParser().parse(scan)

        assert "SCANNED" in markdown.upper()

    def test_chain_falls_back_to_ocr_when_earlier_parsers_fail(self, tmp_path: Path) -> None:
        from booksmart_core.parsing import OcrParser, ParserChain

        scan = tmp_path / "scan.pdf"
        scan.write_bytes(make_scanned_pdf_bytes())
        log: list[str] = []

        result = ParserChain([FailingParser(), OcrParser()]).extract(scan, "pdf", log.append)

        assert result.parser == "ocr"
        assert "SCANNED" in result.markdown.upper()
        assert any("always-fails" in line and "failed" in line for line in log)

    def test_scanned_pdf_ingests_successfully(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(
            session_factory, storage, filename="scan.pdf", content=make_scanned_pdf_bytes()
        )

        run = run_job(session_factory, settings, book_id)

        assert run["status"] == "succeeded"
        # pymupdf4llm OCRs image-only pages itself when tesseract is present,
        # so either stage may have won — but the text must be there.
        assert run["parser_used"] in ("pymupdf", "ocr")
        parsed = Path(str(run["output_path"])).read_text(encoding="utf-8")
        assert "SCANNED" in parsed.upper()

    def test_text_pdf_does_not_reach_ocr(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage, content=make_pdf_bytes())

        run = run_job(session_factory, settings, book_id)

        assert run["parser_used"] == "pymupdf"
