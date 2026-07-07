"""Integration tests: knowledge extraction persists typed objects with provenance."""

import json
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from booksmart_core.config import Settings
from booksmart_core.extraction import EXTRACTION_SYSTEM_PROMPT
from booksmart_core.storage import BookStorage

from .conftest import StubLLMProvider, book_structure, knowledge_objects
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
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)
        prime_extraction(stub_llm)

        run = ingest(session_factory, settings, book_id)

        assert run["status"] == "succeeded"
        objects = knowledge_objects(session_factory, book_id)
        assert len(objects) == 3

        chapters = book_structure(session_factory, book_id)
        principle = next(o for o in objects if o["type"] == "Principle")
        assert principle["title"] == "Deep modules"
        assert principle["summary"] == "Prefer deep modules."
        assert principle["confidence"] == 0.9
        assert principle["book_id"] == book_id
        assert principle["chapter_id"] == chapters[0]["id"]
        assert principle["section_id"] == chapters[0]["sections"][0]["id"]  # type: ignore[index]
        assert principle["edition"] == "2nd"  # frozen at extraction time
        assert principle["page"] == 4
        assert principle["paragraph"] == 2
        assert principle["extraction_model"] == "stub-llm-1"
        assert principle["extraction_prompt_version"] == "2"
        assert "Chapter One: Modules" in str(principle["source_location"])
        assert "Deep Modules" in str(principle["source_location"])

        definition = next(o for o in objects if o["type"] == "Definition")
        assert definition["section_id"] is None
        assert definition["chapter_id"] == chapters[0]["id"]

        smell = next(o for o in objects if o["type"] == "Smell")
        assert smell["chapter_id"] == chapters[1]["id"]
        assert smell["section_id"] == chapters[1]["sections"][0]["id"]  # type: ignore[index]

    def test_extraction_prompts_carry_chapter_text_and_sections(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)
        prime_extraction(stub_llm)

        ingest(session_factory, settings, book_id)

        extraction_calls = [p for p, s in stub_llm.calls if s == EXTRACTION_SYSTEM_PROMPT]
        assert len(extraction_calls) == 2  # one per chapter
        assert "Chapter One: Modules" in extraction_calls[0]
        assert "Deep Modules" in extraction_calls[0]  # section listed
        assert "Body text explaining the idea" in extraction_calls[0]  # chapter body
        assert "Chapter Two: Complexity" in extraction_calls[1]
        assert "Chapter One: Modules" not in extraction_calls[1]  # sliced, not whole book

    def test_reingest_replaces_objects_wholesale(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)
        prime_extraction(stub_llm)
        ingest(session_factory, settings, book_id)
        first_ids = {o["id"] for o in knowledge_objects(session_factory, book_id)}

        prime_extraction(stub_llm)
        ingest(session_factory, settings, book_id)

        objects = knowledge_objects(session_factory, book_id)
        assert len(objects) == 3  # replaced, not accumulated
        assert first_ids.isdisjoint({o["id"] for o in objects})

    def test_invalid_response_fails_run_and_preserves_previous_objects(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)
        prime_extraction(stub_llm)
        ingest(session_factory, settings, book_id)

        # A chapter's call is retried once, so persistent failure needs two
        # bad responses for the same chapter.
        stub_llm.queue(EXTRACTION_SYSTEM_PROMPT, "this is not JSON", "still not JSON")
        run = ingest(session_factory, settings, book_id)

        assert run["status"] == "failed"
        assert "extraction" in str(run["error"])
        assert len(knowledge_objects(session_factory, book_id)) == 3

    def test_element_with_unsupported_type_is_dropped_not_fatal(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)
        # Chapter 1: one valid object plus one in the book's own vocabulary.
        red_flag = dict(CHAPTER_ONE_OBJECTS[0], type="Red Flag", title="Shallow Module")
        stub_llm.queue(
            EXTRACTION_SYSTEM_PROMPT,
            json.dumps([CHAPTER_ONE_OBJECTS[0], red_flag]),
            json.dumps(CHAPTER_TWO_OBJECTS),
        )

        run = ingest(session_factory, settings, book_id)

        assert run["status"] == "succeeded"
        objects = knowledge_objects(session_factory, book_id)
        assert "Red Flag" not in {o["type"] for o in objects}
        log = (Path(settings.storage_root) / "logs" / f"{run['id']}.log").read_text(
            encoding="utf-8"
        )
        assert "Red Flag" in log and "dropped" in log

    def test_transient_invalid_response_is_retried_and_run_succeeds(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)
        # Chapter 1 answers garbage once, then valid objects on the retry;
        # chapter 2 answers valid objects directly.
        stub_llm.queue(
            EXTRACTION_SYSTEM_PROMPT,
            "this is not JSON",
            json.dumps(CHAPTER_ONE_OBJECTS),
            json.dumps(CHAPTER_TWO_OBJECTS),
        )

        run = ingest(session_factory, settings, book_id)

        assert run["status"] == "succeeded"
        assert len(knowledge_objects(session_factory, book_id)) == 3
        log = (Path(settings.storage_root) / "logs" / f"{run['id']}.log").read_text(
            encoding="utf-8"
        )
        assert "retrying" in log

    def test_book_without_detected_chapters_extracts_nothing_but_succeeds(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        # A plain-text PDF with no headings yields no chapters, so no LLM calls.
        book_id = register_book(session_factory, storage)

        run = ingest(session_factory, settings, book_id)

        assert run["status"] == "succeeded"
        assert knowledge_objects(session_factory, book_id) == []
        log = (Path(settings.storage_root) / "logs" / f"{run['id']}.log").read_text(
            encoding="utf-8"
        )
        assert "no chapters detected" in log

    def test_out_of_range_section_index_keeps_chapter_link_only(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)
        rogue = dict(CHAPTER_ONE_OBJECTS[0], section_index=7)
        stub_llm.queue(EXTRACTION_SYSTEM_PROMPT, json.dumps([rogue]), "[]")

        run = ingest(session_factory, settings, book_id)

        assert run["status"] == "succeeded"
        objects = knowledge_objects(session_factory, book_id)
        assert len(objects) == 1
        assert objects[0]["chapter_id"] is not None
        assert objects[0]["section_id"] is None
        log = (Path(settings.storage_root) / "logs" / f"{run['id']}.log").read_text(
            encoding="utf-8"
        )
        assert "out of range" in log

    def test_stage_is_visible_in_run_log(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)
        prime_extraction(stub_llm)

        run = ingest(session_factory, settings, book_id)

        log = (Path(settings.storage_root) / "logs" / f"{run['id']}.log").read_text(
            encoding="utf-8"
        )
        assert "extraction" in log
        assert "3 knowledge objects" in log


class TestKnowledgeReadModel:
    def test_list_filterable_by_type(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)
        prime_extraction(stub_llm)
        ingest(session_factory, settings, book_id)

        principles = knowledge_objects(session_factory, book_id, "Principle")
        assert [o["type"] for o in principles] == ["Principle"]
        assert knowledge_objects(session_factory, book_id, "Checklist") == []

    def test_book_without_extraction_has_empty_list(
        self, session_factory: sessionmaker[Session], storage: BookStorage
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)

        assert knowledge_objects(session_factory, book_id) == []
