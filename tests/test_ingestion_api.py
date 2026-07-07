"""End-to-end tests for ingestion: trigger a run -> pipeline extracts -> parsed Markdown.

With the polling worker gone (ADR 0002), POST /ingest runs the whole pipeline
synchronously and returns the finished Run — there is no queued state."""

import io
import zipfile
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

from app.config import Settings

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
    def test_ingest_runs_synchronously_and_returns_succeeded_run(
        self, client: TestClient
    ) -> None:
        book_id = register_book(client)

        response = client.post(f"/books/{book_id}/ingest")

        assert response.status_code == 200
        run = response.json()
        assert run["id"]
        assert run["book_id"] == book_id
        assert run["status"] == "succeeded"
        assert run["scope"] == "full"
        assert run["created_at"]
        assert run["finished_at"]
        assert run["error"] is None

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


class TestRunStatus:
    def test_run_is_retrievable_after_ingest(self, client: TestClient) -> None:
        book_id = register_book(client)
        created = client.post(f"/books/{book_id}/ingest").json()

        response = client.get(f"/jobs/{created['id']}")

        assert response.status_code == 200
        assert response.json() == created

    def test_unknown_run_returns_404(self, client: TestClient) -> None:
        response = client.get("/jobs/00000000-0000-0000-0000-000000000000")

        assert response.status_code == 404


class TestPipelineExtraction:
    def test_successful_extraction_marks_run_succeeded(self, client: TestClient) -> None:
        book_id = register_book(client)

        run = client.post(f"/books/{book_id}/ingest").json()

        assert run["status"] == "succeeded"
        assert run["finished_at"] is not None
        assert run["error"] is None

    def test_parsed_markdown_written_to_storage(
        self, client: TestClient, settings: Settings
    ) -> None:
        book_id = register_book(client)

        run = client.post(f"/books/{book_id}/ingest").json()

        parsed = Path(settings.storage_root) / "parsed" / book_id / "parsed.md"
        assert parsed.exists()
        assert EXTRACT_TEXT in parsed.read_text(encoding="utf-8")
        assert run["output_path"] == str(parsed)

    def test_corrupt_pdf_marks_run_failed_with_error(
        self, client: TestClient, settings: Settings
    ) -> None:
        book_id = register_book(client, content=CORRUPT_PDF_BYTES)

        run = client.post(f"/books/{book_id}/ingest").json()

        assert run["status"] == "failed"
        assert run["error"]
        assert run["finished_at"] is not None
        assert run["output_path"] is None
        assert not (Path(settings.storage_root) / "parsed" / book_id).exists()

    def test_unparseable_epub_marks_run_failed_with_clear_error(
        self, client: TestClient
    ) -> None:
        book_id = register_book(client, filename="ddd.epub", content=make_fake_epub_bytes())

        run = client.post(f"/books/{book_id}/ingest").json()

        assert run["status"] == "failed"
        assert "epub" in run["error"].lower()
