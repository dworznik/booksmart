"""Integration tests: the profile stage generates, versions, and serves book profiles."""

import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.llm import LLMResponse
from app.models import BookProfile
from app.profile import PROFILE_SYSTEM_PROMPT
from app.worker import process_one_job

from .conftest import StubLLMProvider
from .test_structure_api import ingest, make_structured_pdf_bytes

BOOK_FIELDS = {
    "title": "A Philosophy of Software Design",
    "author": "John Ousterhout",
    "edition": "2nd",
    "publication_year": "2021",
    "primary_topic": "software design",
    "notes": "Focus on the deep-modules argument",
    "trust_level": "high",
}


def register_book_with_hints(client: TestClient) -> str:
    response = client.post(
        "/books",
        data=BOOK_FIELDS,
        files={"file": ("apod.pdf", make_structured_pdf_bytes(), "application/octet-stream")},
    )
    assert response.status_code == 201
    book_id: str = response.json()["id"]
    return book_id


class ExplodingLLM:
    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        raise RuntimeError("provider went down")


class TestProfileGenerationStage:
    def test_profile_is_generated_and_served_via_api(
        self, client: TestClient, session_factory: sessionmaker[Session], settings: Settings
    ) -> None:
        book_id = register_book_with_hints(client)

        job = ingest(client, session_factory, settings, book_id)

        assert job["status"] == "succeeded"
        response = client.get(f"/books/{book_id}/profile")
        assert response.status_code == 200
        profile = response.json()
        assert profile["book_id"] == book_id
        assert profile["content"] == "A stubbed book profile."
        assert profile["model"] == "stub-llm-1"
        assert profile["prompt_version"] == "1"
        assert profile["created_at"]

    def test_prompt_carries_metadata_hints_and_outline(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(client)

        ingest(client, session_factory, settings, book_id)

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

    def test_reingest_appends_new_version_and_api_returns_latest(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        settings: Settings,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = register_book_with_hints(client)

        ingest(client, session_factory, settings, book_id)
        stub_llm.text = "A better profile after reprocessing."
        ingest(client, session_factory, settings, book_id)

        assert client.get(f"/books/{book_id}/profile").json()["content"] == (
            "A better profile after reprocessing."
        )
        with session_factory() as session:
            stored = session.scalars(
                select(BookProfile).where(BookProfile.book_id == uuid.UUID(book_id))
            ).all()
        assert len(stored) == 2  # history is preserved, not overwritten

    def test_llm_failure_fails_job_and_preserves_previous_profile(
        self, client: TestClient, session_factory: sessionmaker[Session], settings: Settings
    ) -> None:
        book_id = register_book_with_hints(client)
        ingest(client, session_factory, settings, book_id)

        job_id = client.post(f"/books/{book_id}/ingest").json()["id"]
        assert process_one_job(session_factory, settings.storage_root, llm=ExplodingLLM()) is True

        job = client.get(f"/jobs/{job_id}").json()
        assert job["status"] == "failed"
        assert "profile" in str(job["error"])
        assert client.get(f"/books/{book_id}/profile").json()["content"] == (
            "A stubbed book profile."
        )

    def test_stage_is_visible_in_parse_log(
        self, client: TestClient, session_factory: sessionmaker[Session], settings: Settings
    ) -> None:
        book_id = register_book_with_hints(client)

        job = ingest(client, session_factory, settings, book_id)

        log = (Path(settings.storage_root) / "logs" / f"{job['id']}.log").read_text(
            encoding="utf-8"
        )
        assert "profile" in log
        assert "stub-llm-1" in log


class TestProfileEndpoint:
    def test_book_without_profile_returns_404(self, client: TestClient) -> None:
        book_id = register_book_with_hints(client)

        response = client.get(f"/books/{book_id}/profile")

        assert response.status_code == 404

    def test_unknown_book_returns_404(self, client: TestClient) -> None:
        response = client.get("/books/00000000-0000-0000-0000-000000000000/profile")

        assert response.status_code == 404
