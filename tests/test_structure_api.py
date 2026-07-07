"""Integration tests: ingestion detects structure, persists it, and serves the tree."""

from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings

from .test_ingestion_api import register_book

BOOK_OUTLINE = [
    ("Chapter One: Modules", ["Deep Modules", "Shallow Modules"]),
    ("Chapter Two: Complexity", ["Symptoms"]),
]


def make_structured_pdf_bytes() -> bytes:
    """Font sizes chosen so pymupdf4llm emits '#' for chapters and '##' for sections."""
    doc = pymupdf.open()
    for chapter, sections in BOOK_OUTLINE:
        page = doc.new_page()
        y = 80.0
        page.insert_text((72, y), chapter, fontsize=24)
        y += 50
        for section in sections:
            page.insert_text((72, y), section, fontsize=16)
            y += 30
            page.insert_text((72, y), "Body text explaining the idea in plain words.", fontsize=11)
            y += 40
    data: bytes = doc.tobytes()
    doc.close()
    return data


def ingest(client: TestClient, session_factory: sessionmaker[Session], settings: Settings, book_id: str) -> dict[str, object]:
    """Trigger a full ingest. With the polling worker gone (ADR 0002), the
    request runs the whole pipeline synchronously and returns the finished Run.

    session_factory / settings are unused now — the endpoint owns execution —
    but kept in the signature so the many call sites need no change."""
    response = client.post(f"/books/{book_id}/ingest")
    assert response.status_code == 200, response.text
    job: dict[str, object] = response.json()
    return job


def structure_outline(client: TestClient, book_id: str) -> list[tuple[str, list[str]]]:
    chapters = client.get(f"/books/{book_id}/structure").json()
    return [
        (chapter["title"], [section["title"] for section in chapter["sections"]])
        for chapter in chapters
    ]


class TestStructureDetectionStage:
    def test_known_book_yields_expected_outline(
        self, client: TestClient, session_factory: sessionmaker[Session], settings: Settings
    ) -> None:
        book_id = register_book(client, content=make_structured_pdf_bytes())

        job = ingest(client, session_factory, settings, book_id)

        assert job["status"] == "succeeded"
        assert structure_outline(client, book_id) == BOOK_OUTLINE

    def test_chapters_carry_ordering_and_source_locations(
        self, client: TestClient, session_factory: sessionmaker[Session], settings: Settings
    ) -> None:
        book_id = register_book(client, content=make_structured_pdf_bytes())
        ingest(client, session_factory, settings, book_id)

        chapters = client.get(f"/books/{book_id}/structure").json()

        assert [chapter["position"] for chapter in chapters] == [0, 1]
        assert all(isinstance(chapter["source_line"], int) for chapter in chapters)
        first_sections = chapters[0]["sections"]
        assert [section["position"] for section in first_sections] == [0, 1]
        assert first_sections[0]["source_line"] < first_sections[1]["source_line"]

    def test_stage_is_visible_in_parse_log(
        self, client: TestClient, session_factory: sessionmaker[Session], settings: Settings
    ) -> None:
        book_id = register_book(client, content=make_structured_pdf_bytes())

        job = ingest(client, session_factory, settings, book_id)

        log = (Path(settings.storage_root) / "logs" / f"{job['id']}.log").read_text(encoding="utf-8")
        assert "structure" in log
        assert "2 chapters" in log and "3 sections" in log

    def test_failed_rerun_preserves_previous_structure(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        book_id = register_book(client, content=make_structured_pdf_bytes())
        ingest(client, session_factory, settings, book_id)
        assert structure_outline(client, book_id) == BOOK_OUTLINE

        def explode(markdown: str) -> object:
            raise RuntimeError("structure stage blew up")

        monkeypatch.setattr("app.stages.detect_structure", explode)
        job = ingest(client, session_factory, settings, book_id)

        assert job["status"] == "failed"
        assert "structure" in str(job["error"])
        assert structure_outline(client, book_id) == BOOK_OUTLINE

    def test_reingest_replaces_structure_without_duplication(
        self, client: TestClient, session_factory: sessionmaker[Session], settings: Settings
    ) -> None:
        book_id = register_book(client, content=make_structured_pdf_bytes())

        ingest(client, session_factory, settings, book_id)
        first = structure_outline(client, book_id)
        ingest(client, session_factory, settings, book_id)
        second = structure_outline(client, book_id)

        assert first == second == BOOK_OUTLINE


class TestStructureKinds:
    def test_front_and_back_matter_distinguishable_in_response(
        self, client: TestClient, session_factory: sessionmaker[Session], settings: Settings
    ) -> None:
        doc = pymupdf.open()
        for heading in ("Contents", "Preface", "Chapter One: Modules", "Index"):
            page = doc.new_page()
            page.insert_text((72, 80), heading, fontsize=24)
            page.insert_text((72, 130), "Body text in regular size.", fontsize=11)
        pdf: bytes = doc.tobytes()
        doc.close()
        book_id = register_book(client, content=pdf)

        job = ingest(client, session_factory, settings, book_id)

        assert job["status"] == "succeeded"
        chapters = client.get(f"/books/{book_id}/structure").json()
        assert [(chapter["title"], chapter["kind"]) for chapter in chapters] == [
            ("Contents", "front_matter"),
            ("Preface", "front_matter"),
            ("Chapter One: Modules", "chapter"),
            ("Index", "back_matter"),
        ]

    def test_body_chapters_carry_chapter_kind(
        self, client: TestClient, session_factory: sessionmaker[Session], settings: Settings
    ) -> None:
        book_id = register_book(client, content=make_structured_pdf_bytes())
        ingest(client, session_factory, settings, book_id)

        chapters = client.get(f"/books/{book_id}/structure").json()

        assert {chapter["kind"] for chapter in chapters} == {"chapter"}


class TestStructureEndpoint:
    def test_book_without_ingestion_has_empty_structure(self, client: TestClient) -> None:
        book_id = register_book(client)

        response = client.get(f"/books/{book_id}/structure")

        assert response.status_code == 200
        assert response.json() == []

    def test_unknown_book_returns_404(self, client: TestClient) -> None:
        response = client.get("/books/00000000-0000-0000-0000-000000000000/structure")

        assert response.status_code == 404
