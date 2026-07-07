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

HINT_FIELDS = (
    "primary_topic",
    "language",
    "framework",
    "methodology",
    "notes",
    "trust_level",
    "intended_use",
)


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


class TestRegisterBookWithHints:
    def test_hints_roundtrip(self, client: TestClient) -> None:
        response = upload_book(
            client,
            primary_topic="software craftsmanship",
            language="Python",
            framework="FastAPI",
            methodology="TDD",
            notes="Read chapters 1-3 first",
            trust_level="high",
            intended_use="team onboarding",
        )

        assert response.status_code == 201
        body = response.json()
        assert body["primary_topic"] == "software craftsmanship"
        assert body["language"] == "Python"
        assert body["framework"] == "FastAPI"
        assert body["methodology"] == "TDD"
        assert body["notes"] == "Read chapters 1-3 first"
        assert body["trust_level"] == "high"
        assert body["intended_use"] == "team onboarding"

    def test_hints_default_to_null(self, client: TestClient) -> None:
        body = upload_book(client).json()

        for field in HINT_FIELDS:
            assert body[field] is None


class TestUpdateBook:
    def test_patch_updates_metadata_and_hints(self, client: TestClient) -> None:
        created = upload_book(client).json()

        response = client.patch(
            f"/books/{created['id']}",
            json={
                "title": "Clean Code (annotated)",
                "edition": "3rd",
                "publication_year": 2010,
                "primary_topic": "refactoring",
                "trust_level": "medium",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["title"] == "Clean Code (annotated)"
        assert body["edition"] == "3rd"
        assert body["publication_year"] == 2010
        assert body["primary_topic"] == "refactoring"
        assert body["trust_level"] == "medium"

    def test_patch_is_partial_absent_fields_unchanged(self, client: TestClient) -> None:
        created = upload_book(
            client, edition="2nd", isbn="9780132350884", notes="original notes"
        ).json()

        body = client.patch(f"/books/{created['id']}", json={"author": "R. C. Martin"}).json()

        assert body["author"] == "R. C. Martin"
        assert body["edition"] == "2nd"
        assert body["isbn"] == "9780132350884"
        assert body["notes"] == "original notes"

    def test_patch_explicit_null_clears_field(self, client: TestClient) -> None:
        created = upload_book(client, edition="2nd", notes="scratch this").json()

        body = client.patch(
            f"/books/{created['id']}", json={"edition": None, "notes": None}
        ).json()

        assert body["edition"] is None
        assert body["notes"] is None

    def test_patch_persists_across_fetches(self, client: TestClient) -> None:
        created = upload_book(client).json()

        client.patch(f"/books/{created['id']}", json={"methodology": "DDD"})

        fetched = client.get(f"/books/{created['id']}").json()
        assert fetched["methodology"] == "DDD"

    def test_patch_after_upload_succeeds(self, client: TestClient) -> None:
        # Ingestion states arrive in later slices; a fully uploaded book is the
        # closest thing to "ingested" today. Metadata must stay editable in any
        # state, so this test pins that down for the post-upload state.
        created = upload_book(client).json()
        assert created["checksum"]  # upload fully completed

        response = client.patch(
            f"/books/{created['id']}",
            json={"title": "Updated After Upload", "intended_use": "reference"},
        )

        assert response.status_code == 200
        assert response.json()["title"] == "Updated After Upload"
        assert response.json()["intended_use"] == "reference"

    def test_patch_unknown_book_returns_404(self, client: TestClient) -> None:
        response = client.patch(
            "/books/00000000-0000-0000-0000-000000000000", json={"title": "Ghost"}
        )

        assert response.status_code == 404

    def test_patch_file_fields_rejected(self, client: TestClient) -> None:
        created = upload_book(client).json()

        for field in ("checksum", "file_hash", "original_filename", "file_format", "storage_path", "uploaded_at"):
            response = client.patch(f"/books/{created['id']}", json={field: "tampered"})
            assert response.status_code == 422, field

        fetched = client.get(f"/books/{created['id']}").json()
        assert fetched["checksum"] == created["checksum"]

    def test_patch_null_title_rejected(self, client: TestClient) -> None:
        created = upload_book(client).json()

        response = client.patch(f"/books/{created['id']}", json={"title": None})

        assert response.status_code == 422

    def test_patch_null_author_rejected(self, client: TestClient) -> None:
        created = upload_book(client).json()

        response = client.patch(f"/books/{created['id']}", json={"author": None})

        assert response.status_code == 422


class TestDuplicateUpload:
    def test_duplicate_content_returns_409_with_existing_book_id(
        self, client: TestClient
    ) -> None:
        original = upload_book(client).json()

        response = upload_book(client, filename="same-bytes.pdf", title="Different Title")

        assert response.status_code == 409
        assert response.json()["detail"]["existing_book_id"] == original["id"]

    def test_rejected_duplicate_creates_no_row_and_no_file(
        self, client: TestClient, settings: Settings
    ) -> None:
        original = upload_book(client).json()

        upload_book(client)

        books = client.get("/books").json()
        assert [book["id"] for book in books] == [original["id"]]
        book_dirs = list((Path(settings.storage_root) / "books").iterdir())
        assert [d.name for d in book_dirs] == [original["id"]]

    def test_original_book_unchanged_after_rejected_duplicate(
        self, client: TestClient
    ) -> None:
        original = upload_book(client).json()

        upload_book(client, title="Sneaky Overwrite", author="Somebody Else")

        assert client.get(f"/books/{original['id']}").json() == original

    def test_same_metadata_different_content_accepted(self, client: TestClient) -> None:
        first = upload_book(client)
        second = upload_book(client, content=PDF_BYTES + b"% different trailing bytes\n")

        assert first.status_code == 201
        assert second.status_code == 201
        assert first.json()["id"] != second.json()["id"]


class TestListAndFetchBooks:
    def test_empty_list(self, client: TestClient) -> None:
        response = client.get("/books")

        assert response.status_code == 200
        assert response.json() == []

    def test_uploaded_books_appear_in_list(self, client: TestClient) -> None:
        first = upload_book(client).json()
        second = upload_book(
            client,
            title="The Pragmatic Programmer",
            author="Hunt & Thomas",
            content=PDF_BYTES + b"% pragprog\n",
        ).json()

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
