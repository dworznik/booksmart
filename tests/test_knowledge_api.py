"""Integration tests: knowledge extraction persists typed objects with provenance."""

import json
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.extraction import EXTRACTION_SYSTEM_PROMPT

from .conftest import StubLLMProvider
from .test_ingestion_api import register_book
from .test_profile_api import register_book_with_hints
from .test_structure_api import ingest

CHAPTER_ONE_OBJECTS = [
    {
        "type": "Principle",
        "title": "Deep modules",
        "content": "Modules should be deep.",
        "summary": "Prefer deep modules.",
        "confidence": 0.9,
        "section_index": 0,
        "page": 4,
        "paragraph": 2,
    },
    {
        "type": "Definition",
        "title": "Shallow module",
        "content": "A module whose interface is complex relative to its functionality.",
        "summary": "Interface cost outweighs functionality.",
        "confidence": 0.75,
        "section_index": None,
        "page": None,
        "paragraph": None,
    },
]

CHAPTER_TWO_OBJECTS = [
    {
        "type": "Smell",
        "title": "Change amplification",
        "content": "A simple change requires edits in many places.",
        "summary": "Symptom of complexity.",
        "confidence": 0.8,
        "section_index": 0,
        "page": 11,
        "paragraph": 1,
    },
]


def prime_extraction(stub: StubLLMProvider) -> None:
    stub.queue(
        EXTRACTION_SYSTEM_PROMPT, json.dumps(CHAPTER_ONE_OBJECTS), json.dumps(CHAPTER_TWO_OBJECTS)
    )


class TestKnowledgeExtractionStage:
    def test_objects_persisted_with_full_provenance(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(client)
        prime_extraction(stub_llm)

        job = ingest(client, session_factory, settings, book_id)

        assert job["status"] == "succeeded"
        objects = client.get(f"/books/{book_id}/knowledge-objects").json()
        assert len(objects) == 3

        chapters = client.get(f"/books/{book_id}/structure").json()
        principle = next(o for o in objects if o["type"] == "Principle")
        assert principle["title"] == "Deep modules"
        assert principle["summary"] == "Prefer deep modules."
        assert principle["confidence"] == 0.9
        assert principle["book_id"] == book_id
        assert principle["chapter_id"] == chapters[0]["id"]
        assert principle["section_id"] == chapters[0]["sections"][0]["id"]
        assert principle["edition"] == "2nd"  # frozen at extraction time
        assert principle["page"] == 4
        assert principle["paragraph"] == 2
        assert principle["extraction_model"] == "stub-llm-1"
        assert principle["extraction_prompt_version"] == "1"
        assert principle["created_at"]
        assert "Chapter One: Modules" in principle["source_location"]
        assert "Deep Modules" in principle["source_location"]

        definition = next(o for o in objects if o["type"] == "Definition")
        assert definition["section_id"] is None
        assert definition["chapter_id"] == chapters[0]["id"]

        smell = next(o for o in objects if o["type"] == "Smell")
        assert smell["chapter_id"] == chapters[1]["id"]
        assert smell["section_id"] == chapters[1]["sections"][0]["id"]

    def test_extraction_prompts_carry_chapter_text_and_sections(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(client)
        prime_extraction(stub_llm)

        ingest(client, session_factory, settings, book_id)

        extraction_calls = [p for p, s in stub_llm.calls if s == EXTRACTION_SYSTEM_PROMPT]
        assert len(extraction_calls) == 2  # one per chapter
        assert "Chapter One: Modules" in extraction_calls[0]
        assert "Deep Modules" in extraction_calls[0]  # section listed
        assert "Body text explaining the idea" in extraction_calls[0]  # chapter body
        assert "Chapter Two: Complexity" in extraction_calls[1]
        assert "Chapter One: Modules" not in extraction_calls[1]  # sliced, not whole book

    def test_reingest_replaces_objects_wholesale(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(client)
        prime_extraction(stub_llm)
        ingest(client, session_factory, settings, book_id)
        first_ids = {o["id"] for o in client.get(f"/books/{book_id}/knowledge-objects").json()}

        prime_extraction(stub_llm)
        ingest(client, session_factory, settings, book_id)

        objects = client.get(f"/books/{book_id}/knowledge-objects").json()
        assert len(objects) == 3  # replaced, not accumulated
        assert first_ids.isdisjoint({o["id"] for o in objects})

    def test_invalid_response_fails_job_and_preserves_previous_objects(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(client)
        prime_extraction(stub_llm)
        ingest(client, session_factory, settings, book_id)

        stub_llm.queue(EXTRACTION_SYSTEM_PROMPT, "this is not JSON")
        job = ingest(client, session_factory, settings, book_id)

        assert job["status"] == "failed"
        assert "extraction" in str(job["error"])
        assert len(client.get(f"/books/{book_id}/knowledge-objects").json()) == 3

    def test_book_without_detected_chapters_extracts_nothing_but_succeeds(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
    ) -> None:
        # A plain-text PDF with no headings yields no chapters, so no LLM calls.
        book_id = register_book(client)

        job = ingest(client, session_factory, settings, book_id)

        assert job["status"] == "succeeded"
        assert client.get(f"/books/{book_id}/knowledge-objects").json() == []
        log = (Path(settings.storage_root) / "logs" / f"{job['id']}.log").read_text(
            encoding="utf-8"
        )
        assert "no chapters detected" in log

    def test_out_of_range_section_index_keeps_chapter_link_only(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(client)
        rogue = dict(CHAPTER_ONE_OBJECTS[0], section_index=7)
        stub_llm.queue(EXTRACTION_SYSTEM_PROMPT, json.dumps([rogue]), "[]")

        job = ingest(client, session_factory, settings, book_id)

        assert job["status"] == "succeeded"
        objects = client.get(f"/books/{book_id}/knowledge-objects").json()
        assert len(objects) == 1
        assert objects[0]["chapter_id"] is not None
        assert objects[0]["section_id"] is None
        log = (Path(settings.storage_root) / "logs" / f"{job['id']}.log").read_text(
            encoding="utf-8"
        )
        assert "out of range" in log

    def test_stage_is_visible_in_parse_log(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(client)
        prime_extraction(stub_llm)

        job = ingest(client, session_factory, settings, book_id)

        log = (Path(settings.storage_root) / "logs" / f"{job['id']}.log").read_text(
            encoding="utf-8"
        )
        assert "extraction" in log
        assert "3 knowledge objects" in log


class TestKnowledgeObjectEndpoints:
    def test_list_filterable_by_type(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(client)
        prime_extraction(stub_llm)
        ingest(client, session_factory, settings, book_id)

        principles = client.get(f"/books/{book_id}/knowledge-objects?type=Principle").json()
        assert [o["type"] for o in principles] == ["Principle"]
        checklists = client.get(f"/books/{book_id}/knowledge-objects?type=Checklist").json()
        assert checklists == []

    def test_invalid_type_filter_is_rejected(self, client: TestClient) -> None:
        book_id = register_book_with_hints(client)

        response = client.get(f"/books/{book_id}/knowledge-objects?type=Vibe")

        assert response.status_code == 422

    def test_object_fetchable_by_id(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(client)
        prime_extraction(stub_llm)
        ingest(client, session_factory, settings, book_id)
        listed = client.get(f"/books/{book_id}/knowledge-objects").json()[0]

        response = client.get(f"/knowledge-objects/{listed['id']}")

        assert response.status_code == 200
        assert response.json() == listed

    def test_unknown_object_returns_404(self, client: TestClient) -> None:
        response = client.get("/knowledge-objects/00000000-0000-0000-0000-000000000000")

        assert response.status_code == 404

    def test_unknown_book_returns_404(self, client: TestClient) -> None:
        response = client.get("/books/00000000-0000-0000-0000-000000000000/knowledge-objects")

        assert response.status_code == 404

    def test_book_without_extraction_has_empty_list(self, client: TestClient) -> None:
        book_id = register_book_with_hints(client)

        response = client.get(f"/books/{book_id}/knowledge-objects")

        assert response.status_code == 200
        assert response.json() == []
