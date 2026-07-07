"""Integration tests: the profile stage generates, versions, and persists book profiles."""

import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from booksmart_core.config import Settings
from booksmart_core.llm import LLMResponse
from booksmart_core.models import BookProfile
from booksmart_core.profile import PROFILE_SYSTEM_PROMPT
from booksmart_core.runner import execute_run
from booksmart_core.storage import BookStorage

from .conftest import StubLLMProvider, get_run, latest_profile, store_book
from .test_structure_api import ingest, make_structured_pdf_bytes

BOOK_FIELDS: dict[str, object] = {
    "title": "A Philosophy of Software Design",
    "author": "John Ousterhout",
    "edition": "2nd",
    "publication_year": 2021,
    "primary_topic": "software design",
    "notes": "Focus on the deep-modules argument",
    "trust_level": "high",
}


def register_book_with_hints(
    session_factory: sessionmaker[Session], storage: BookStorage
) -> str:
    return store_book(
        session_factory,
        storage,
        filename="apod.pdf",
        content=make_structured_pdf_bytes(),
        **BOOK_FIELDS,
    )


class ExplodingLLM:
    model = "exploding-llm"

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        raise RuntimeError("provider went down")


class TestProfileGenerationStage:
    def test_profile_is_generated_and_persisted(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)

        run = ingest(session_factory, settings, book_id)

        assert run["status"] == "succeeded"
        profile = latest_profile(session_factory, book_id)
        assert profile is not None
        assert profile["book_id"] == book_id
        assert profile["content"] == "A stubbed book profile."
        assert profile["model"] == "stub-llm-1"
        assert profile["prompt_version"] == "1"
        assert profile["created_at"]

    def test_prompt_carries_metadata_hints_and_outline(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)

        ingest(session_factory, settings, book_id)

        profile_calls = [(p, s) for p, s in stub_llm.calls if s == PROFILE_SYSTEM_PROMPT]
        assert len(profile_calls) == 1
        prompt, system = profile_calls[0]
        assert system  # the stage identifies itself to the model
        assert "A Philosophy of Software Design" in prompt
        assert "John Ousterhout" in prompt
        assert "software design" in prompt  # hint
        assert "Focus on the deep-modules argument" in prompt  # hint
        assert "Chapter One: Modules" in prompt  # detected structure
        assert "Deep Modules" in prompt  # detected section

    def test_reingest_appends_new_version_and_read_model_returns_latest(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)

        ingest(session_factory, settings, book_id)
        stub_llm.text = "A better profile after reprocessing."
        ingest(session_factory, settings, book_id)

        profile = latest_profile(session_factory, book_id)
        assert profile is not None
        assert profile["content"] == "A better profile after reprocessing."
        with session_factory() as session:
            stored = session.scalars(
                select(BookProfile).where(BookProfile.book_id == uuid.UUID(book_id))
            ).all()
        assert len(stored) == 2  # history is preserved, not overwritten

    def test_llm_failure_fails_run_and_preserves_previous_profile(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)
        ingest(session_factory, settings, book_id)

        run_id = execute_run(
            session_factory, settings.storage_root, uuid.UUID(book_id), "full", llm=ExplodingLLM()
        )

        run = get_run(session_factory, str(run_id))
        assert run is not None
        assert run["status"] == "failed"
        assert "profile" in str(run["error"])
        profile = latest_profile(session_factory, book_id)
        assert profile is not None
        assert profile["content"] == "A stubbed book profile."

    def test_stage_is_visible_in_run_log(
        self, session_factory: sessionmaker[Session], settings: Settings, storage: BookStorage
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)

        run = ingest(session_factory, settings, book_id)

        log = (Path(settings.storage_root) / "logs" / f"{run['id']}.log").read_text(
            encoding="utf-8"
        )
        assert "profile" in log
        assert "stub-llm-1" in log


class TestProfileReadModel:
    def test_book_without_profile_has_none(
        self, session_factory: sessionmaker[Session], storage: BookStorage
    ) -> None:
        book_id = register_book_with_hints(session_factory, storage)

        assert latest_profile(session_factory, book_id) is None
