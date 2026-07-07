"""Integration tests: ingestion detects structure and persists the chapter tree."""

from pathlib import Path

import pymupdf
import pytest
from sqlalchemy.orm import Session, sessionmaker

from booksmart_core.config import Settings
from booksmart_core.storage import BookStorage

from .conftest import book_structure, run_scope
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


def ingest(
    session_factory: sessionmaker[Session], settings: Settings, book_id: str
) -> dict[str, object]:
    """Run the whole pipeline synchronously (full scope) and return the Run."""
    return run_scope(session_factory, settings, book_id, "full")


def structure_outline(
    session_factory: sessionmaker[Session], book_id: str
) -> list[tuple[str, list[str]]]:
    return [
        (chapter["title"], [section["title"] for section in chapter["sections"]])  # type: ignore[index]
        for chapter in book_structure(session_factory, book_id)
    ]


class TestStructureDetectionStage:
    def test_known_book_yields_expected_outline(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage, content=make_structured_pdf_bytes())

        run = ingest(session_factory, settings, book_id)

        assert run["status"] == "succeeded"
        assert structure_outline(session_factory, book_id) == BOOK_OUTLINE

    def test_chapters_carry_ordering_and_source_locations(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage, content=make_structured_pdf_bytes())
        ingest(session_factory, settings, book_id)

        chapters = book_structure(session_factory, book_id)

        assert [chapter["position"] for chapter in chapters] == [0, 1]
        assert all(isinstance(chapter["source_line"], int) for chapter in chapters)
        first_sections = chapters[0]["sections"]
        assert [section["position"] for section in first_sections] == [0, 1]  # type: ignore[index,union-attr]
        assert first_sections[0]["source_line"] < first_sections[1]["source_line"]  # type: ignore[index,operator]

    def test_stage_is_visible_in_parse_log(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage, content=make_structured_pdf_bytes())

        run = ingest(session_factory, settings, book_id)

        log = (Path(settings.storage_root) / "logs" / f"{run['id']}.log").read_text(encoding="utf-8")
        assert "structure" in log
        assert "2 chapters" in log and "3 sections" in log

    def test_failed_rerun_preserves_previous_structure(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        book_id = register_book(session_factory, storage, content=make_structured_pdf_bytes())
        ingest(session_factory, settings, book_id)
        assert structure_outline(session_factory, book_id) == BOOK_OUTLINE

        def explode(markdown: str) -> object:
            raise RuntimeError("structure stage blew up")

        monkeypatch.setattr("booksmart_core.stages.detect_structure", explode)
        run = ingest(session_factory, settings, book_id)

        assert run["status"] == "failed"
        assert "structure" in str(run["error"])
        assert structure_outline(session_factory, book_id) == BOOK_OUTLINE

    def test_reingest_replaces_structure_without_duplication(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage, content=make_structured_pdf_bytes())

        ingest(session_factory, settings, book_id)
        first = structure_outline(session_factory, book_id)
        ingest(session_factory, settings, book_id)
        second = structure_outline(session_factory, book_id)

        assert first == second == BOOK_OUTLINE


class TestStructureKinds:
    def test_front_and_back_matter_distinguishable(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        doc = pymupdf.open()
        for heading in ("Contents", "Preface", "Chapter One: Modules", "Index"):
            page = doc.new_page()
            page.insert_text((72, 80), heading, fontsize=24)
            page.insert_text((72, 130), "Body text in regular size.", fontsize=11)
        pdf: bytes = doc.tobytes()
        doc.close()
        book_id = register_book(session_factory, storage, content=pdf)

        run = ingest(session_factory, settings, book_id)

        assert run["status"] == "succeeded"
        chapters = book_structure(session_factory, book_id)
        assert [(chapter["title"], chapter["kind"]) for chapter in chapters] == [
            ("Contents", "front_matter"),
            ("Preface", "front_matter"),
            ("Chapter One: Modules", "chapter"),
            ("Index", "back_matter"),
        ]

    def test_body_chapters_carry_chapter_kind(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage, content=make_structured_pdf_bytes())
        ingest(session_factory, settings, book_id)

        chapters = book_structure(session_factory, book_id)

        assert {chapter["kind"] for chapter in chapters} == {"chapter"}


class TestStructureReadModel:
    def test_book_without_ingestion_has_empty_structure(
        self, session_factory: sessionmaker[Session], storage: BookStorage
    ) -> None:
        book_id = register_book(session_factory, storage)

        assert book_structure(session_factory, book_id) == []
