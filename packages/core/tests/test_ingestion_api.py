"""Ingestion end to end: run the pipeline over a stored book -> parsed Markdown.

The HTTP server is gone (booksmart-core is a library); the Runner's
``execute_run`` runs the whole pipeline synchronously and records a Run.
"""

import io
import zipfile
from pathlib import Path

import pymupdf
from sqlalchemy.orm import Session, sessionmaker

from booksmart_core.config import Settings
from booksmart_core.storage import BookStorage

from .conftest import run_scope, runs_for_book, store_book

EXTRACT_TEXT = "Ubiquitous Language rules the domain"

# Valid PDF magic but no usable structure: stores fine, fails at parse.
CORRUPT_PDF_BYTES = (
    b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"
)


def make_pdf_bytes(text: str = EXTRACT_TEXT) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data: bytes = doc.tobytes()
    doc.close()
    return data


def make_fake_epub_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", "<container/>")
    return buffer.getvalue()


def register_book(
    session_factory: sessionmaker[Session],
    storage: BookStorage,
    *,
    filename: str = "clean-code.pdf",
    content: bytes | None = None,
) -> str:
    """Store a book and its original file, returning the book id."""
    return store_book(
        session_factory,
        storage,
        title="Clean Code",
        author="Robert C. Martin",
        filename=filename,
        content=content if content is not None else make_pdf_bytes(),
    )


def full_run(
    session_factory: sessionmaker[Session], settings: Settings, book_id: str
) -> dict[str, object]:
    return run_scope(session_factory, settings, book_id, "full")


class TestFullIngest:
    def test_ingest_runs_synchronously_and_records_a_succeeded_run(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage)

        run = full_run(session_factory, settings, book_id)

        assert run["book_id"] == book_id
        assert run["status"] == "succeeded"
        assert run["scope"] == "full"
        assert run["created_at"]
        assert run["finished_at"]
        assert run["error"] is None

    def test_parsed_markdown_written_to_storage(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage)

        run = full_run(session_factory, settings, book_id)

        parsed = Path(settings.storage_root) / "parsed" / book_id / "parsed.md"
        assert parsed.exists()
        assert EXTRACT_TEXT in parsed.read_text(encoding="utf-8")
        # output_path is persisted relative to the storage root (portability).
        assert run["output_path"] == str(Path("parsed") / book_id / "parsed.md")

    def test_repeated_ingests_accumulate_history(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage)

        first = full_run(session_factory, settings, book_id)
        second = full_run(session_factory, settings, book_id)

        assert first["id"] != second["id"]
        history = runs_for_book(session_factory, book_id)
        assert {run["id"] for run in history} == {first["id"], second["id"]}


class TestIngestFailures:
    def test_corrupt_pdf_marks_run_failed_with_error(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage, content=CORRUPT_PDF_BYTES)

        run = full_run(session_factory, settings, book_id)

        assert run["status"] == "failed"
        assert run["error"]
        assert run["finished_at"] is not None
        assert run["output_path"] is None
        assert not (Path(settings.storage_root) / "parsed" / book_id).exists()

    def test_unparseable_epub_marks_run_failed_with_clear_error(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(
            session_factory, storage, filename="ddd.epub", content=make_fake_epub_bytes()
        )

        run = full_run(session_factory, settings, book_id)

        assert run["status"] == "failed"
        assert "epub" in str(run["error"]).lower()
