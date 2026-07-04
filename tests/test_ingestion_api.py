"""End-to-end tests for ingestion: trigger job -> worker extracts -> parsed Markdown."""

import io
import zipfile
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.worker import process_one_job

EXTRACT_TEXT = "Ubiquitous Language rules the domain"

# Valid PDF magic but no usable structure: registers fine, fails extraction.
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
    client: TestClient, filename: str = "clean-code.pdf", content: bytes | None = None
) -> str:
    response = client.post(
        "/books",
        data={"title": "Clean Code", "author": "Robert C. Martin"},
        files={"file": (filename, content or make_pdf_bytes(), "application/octet-stream")},
    )
    assert response.status_code == 201
    book_id: str = response.json()["id"]
    return book_id


class TestTriggerIngestion:
    def test_ingest_returns_202_with_queued_job(self, client: TestClient) -> None:
        book_id = register_book(client)

        response = client.post(f"/books/{book_id}/ingest")

        assert response.status_code == 202
        job = response.json()
        assert job["id"]
        assert job["book_id"] == book_id
        assert job["status"] == "queued"
        assert job["created_at"]
        assert job["started_at"] is None
        assert job["finished_at"] is None
        assert job["error"] is None

    def test_ingest_unknown_book_returns_404(self, client: TestClient) -> None:
        response = client.post("/books/00000000-0000-0000-0000-000000000000/ingest")

        assert response.status_code == 404

    def test_repeated_ingests_accumulate_history(self, client: TestClient) -> None:
        book_id = register_book(client)

        first = client.post(f"/books/{book_id}/ingest").json()
        second = client.post(f"/books/{book_id}/ingest").json()

        assert first["id"] != second["id"]
        assert client.get(f"/jobs/{first['id']}").status_code == 200
        assert client.get(f"/jobs/{second['id']}").status_code == 200


class TestJobStatus:
    def test_queued_job_is_reported(self, client: TestClient) -> None:
        book_id = register_book(client)
        created = client.post(f"/books/{book_id}/ingest").json()

        response = client.get(f"/jobs/{created['id']}")

        assert response.status_code == 200
        assert response.json() == created

    def test_unknown_job_returns_404(self, client: TestClient) -> None:
        response = client.get("/jobs/00000000-0000-0000-0000-000000000000")

        assert response.status_code == 404


class TestWorkerExtraction:
    def test_empty_queue_processes_nothing(
        self, client: TestClient, session_factory: sessionmaker[Session], settings: Settings
    ) -> None:
        assert process_one_job(session_factory, settings.storage_root) is False

    def test_successful_extraction_marks_job_succeeded(
        self, client: TestClient, session_factory: sessionmaker[Session], settings: Settings
    ) -> None:
        book_id = register_book(client)
        job_id = client.post(f"/books/{book_id}/ingest").json()["id"]

        assert process_one_job(session_factory, settings.storage_root) is True

        job = client.get(f"/jobs/{job_id}").json()
        assert job["status"] == "succeeded"
        assert job["started_at"] is not None
        assert job["finished_at"] is not None
        assert job["error"] is None

    def test_parsed_markdown_written_to_storage(
        self, client: TestClient, session_factory: sessionmaker[Session], settings: Settings
    ) -> None:
        book_id = register_book(client)
        job_id = client.post(f"/books/{book_id}/ingest").json()["id"]

        process_one_job(session_factory, settings.storage_root)

        parsed = Path(settings.storage_root) / "parsed" / book_id / f"{job_id}.md"
        assert parsed.exists()
        assert EXTRACT_TEXT in parsed.read_text(encoding="utf-8")
        assert client.get(f"/jobs/{job_id}").json()["output_path"] == str(parsed)

    def test_corrupt_pdf_marks_job_failed_with_error(
        self, client: TestClient, session_factory: sessionmaker[Session], settings: Settings
    ) -> None:
        book_id = register_book(client, content=CORRUPT_PDF_BYTES)
        job_id = client.post(f"/books/{book_id}/ingest").json()["id"]

        assert process_one_job(session_factory, settings.storage_root) is True

        job = client.get(f"/jobs/{job_id}").json()
        assert job["status"] == "failed"
        assert job["error"]
        assert job["finished_at"] is not None
        assert job["output_path"] is None
        assert not (Path(settings.storage_root) / "parsed" / book_id).exists()

    def test_unparseable_epub_marks_job_failed_with_clear_error(
        self, client: TestClient, session_factory: sessionmaker[Session], settings: Settings
    ) -> None:
        book_id = register_book(client, filename="ddd.epub", content=make_fake_epub_bytes())
        job_id = client.post(f"/books/{book_id}/ingest").json()["id"]

        process_one_job(session_factory, settings.storage_root)

        job = client.get(f"/jobs/{job_id}").json()
        assert job["status"] == "failed"
        assert "epub" in job["error"].lower()

    def test_oldest_queued_job_is_claimed_first(
        self, client: TestClient, session_factory: sessionmaker[Session], settings: Settings
    ) -> None:
        book_id = register_book(client)
        first_id = client.post(f"/books/{book_id}/ingest").json()["id"]
        second_id = client.post(f"/books/{book_id}/ingest").json()["id"]

        process_one_job(session_factory, settings.storage_root)

        assert client.get(f"/jobs/{first_id}").json()["status"] == "succeeded"
        assert client.get(f"/jobs/{second_id}").json()["status"] == "queued"
