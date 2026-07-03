"""End-to-end tests for book registration: upload -> persist -> fetch."""

import hashlib
import io
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.db import get_db
from app.main import create_app

PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"


def make_epub_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", "<container/>")
    return buffer.getvalue()


def upload_book(client: TestClient, filename: str = "clean-code.pdf", content: bytes = PDF_BYTES, **fields: str):
    data = {"title": "Clean Code", "author": "Robert C. Martin", **fields}
    return client.post("/books", data=data, files={"file": (filename, content, "application/octet-stream")})


class TestRegisterBook:
    def test_pdf_upload_returns_created_book(self, client: TestClient) -> None:
        response = upload_book(client)

        assert response.status_code == 201
        body = response.json()
        assert body["title"] == "Clean Code"
        assert body["author"] == "Robert C. Martin"
        assert body["file_format"] == "pdf"
        assert body["original_filename"] == "clean-code.pdf"
        assert body["id"]
        assert body["uploaded_at"]

    def test_checksum_and_hash_computed_automatically(self, client: TestClient) -> None:
        response = upload_book(client)

        body = response.json()
        assert body["checksum"] == hashlib.md5(PDF_BYTES).hexdigest()
        assert body["file_hash"] == hashlib.sha256(PDF_BYTES).hexdigest()

    def test_original_file_preserved_unmodified(self, client: TestClient, settings: Settings) -> None:
        response = upload_book(client)

        book_id = response.json()["id"]
        stored = list((Path(settings.storage_root) / "books" / book_id).iterdir())
        assert len(stored) == 1
        assert stored[0].name == "clean-code.pdf"
        assert stored[0].read_bytes() == PDF_BYTES

    def test_epub_upload_accepted(self, client: TestClient) -> None:
        response = upload_book(client, filename="ddd.epub", content=make_epub_bytes())

        assert response.status_code == 201
        assert response.json()["file_format"] == "epub"

    def test_optional_metadata_roundtrips(self, client: TestClient) -> None:
        response = upload_book(
            client, edition="2nd", publication_year="2008", isbn="9780132350884"
        )

        body = response.json()
        assert body["edition"] == "2nd"
        assert body["publication_year"] == 2008
        assert body["isbn"] == "9780132350884"

    def test_optional_metadata_defaults_to_null(self, client: TestClient) -> None:
        body = upload_book(client).json()

        assert body["edition"] is None
        assert body["publication_year"] is None
        assert body["isbn"] is None

    def test_unsupported_extension_rejected(self, client: TestClient) -> None:
        response = upload_book(client, filename="notes.txt", content=b"just text")

        assert response.status_code == 415

    def test_pdf_extension_with_non_pdf_content_rejected(self, client: TestClient) -> None:
        response = upload_book(client, filename="fake.pdf", content=b"not a pdf at all")

        assert response.status_code == 415

    def test_epub_extension_with_non_zip_content_rejected(self, client: TestClient) -> None:
        response = upload_book(client, filename="fake.epub", content=b"not a zip")

        assert response.status_code == 415

    def test_rejected_upload_stores_nothing(self, client: TestClient, settings: Settings) -> None:
        upload_book(client, filename="notes.txt", content=b"just text")

        books_dir = Path(settings.storage_root) / "books"
        assert not books_dir.exists() or not any(books_dir.iterdir())
        assert client.get("/books").json() == []

    def test_db_failure_discards_stored_file(self, settings: Settings) -> None:
        app = create_app(settings)

        class ExplodingSession:
            def add(self, obj: object) -> None:
                pass

            def commit(self) -> None:
                raise RuntimeError("db down")

            def rollback(self) -> None:
                pass

        app.dependency_overrides[get_db] = ExplodingSession
        with TestClient(app, raise_server_exceptions=False) as failing_client:
            response = upload_book(failing_client)

        assert response.status_code == 500
        books_dir = Path(settings.storage_root) / "books"
        assert not books_dir.exists() or not any(books_dir.iterdir())

    def test_missing_title_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/books",
            data={"author": "Anonymous"},
            files={"file": ("a.pdf", PDF_BYTES, "application/octet-stream")},
        )

        assert response.status_code == 422


class TestListAndFetchBooks:
    def test_empty_list(self, client: TestClient) -> None:
        response = client.get("/books")

        assert response.status_code == 200
        assert response.json() == []

    def test_uploaded_books_appear_in_list(self, client: TestClient) -> None:
        first = upload_book(client).json()
        second = upload_book(client, title="The Pragmatic Programmer", author="Hunt & Thomas").json()

        listed = client.get("/books").json()
        assert {book["id"] for book in listed} == {first["id"], second["id"]}

    def test_fetch_single_book(self, client: TestClient) -> None:
        created = upload_book(client).json()

        response = client.get(f"/books/{created['id']}")

        assert response.status_code == 200
        assert response.json() == created

    def test_fetch_unknown_book_returns_404(self, client: TestClient) -> None:
        response = client.get("/books/00000000-0000-0000-0000-000000000000")

        assert response.status_code == 404
