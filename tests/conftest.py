"""Integration test fixtures.

Tests run against a real PostgreSQL instance (docker compose up -d postgres).
A dedicated test database is created per session and migrated with Alembic.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.extraction import EXTRACTION_SYSTEM_PROMPT
from app.llm import LLMResponse
from app.main import create_app
from app.summaries import SUMMARY_SYSTEM_PROMPT
from app.vectors import VectorStore

TEST_DATABASE_URL = os.environ.get(
    "BOOKSMART_TEST_DATABASE_URL",
    "postgresql+psycopg://booksmart:booksmart@localhost:5432/booksmart_test",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def database_url() -> str:
    admin_url = TEST_DATABASE_URL.rsplit("/", 1)[0] + "/postgres"
    test_db_name = TEST_DATABASE_URL.rsplit("/", 1)[1]
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"),
            {"name": test_db_name},
        ).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{test_db_name}"'))
    admin_engine.dispose()

    alembic_cfg = AlembicConfig(str(PROJECT_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    command.upgrade(alembic_cfg, "head")
    return TEST_DATABASE_URL


@pytest.fixture()
def settings(database_url: str, tmp_path: Path) -> Settings:
    return Settings(database_url=database_url, storage_root=tmp_path / "storage")


@pytest.fixture()
def client(settings: Settings, database_url: str) -> Iterator[TestClient]:
    engine = create_engine(database_url)
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE books CASCADE"))
        conn.commit()
    engine.dispose()

    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def db_engine(database_url: str) -> Iterator[Engine]:
    engine = create_engine(database_url)
    yield engine
    engine.dispose()


@pytest.fixture()
def session_factory(db_engine: Engine) -> sessionmaker[Session]:
    """Session factory for driving the worker synchronously in tests."""
    return sessionmaker(bind=db_engine)


class StubLLMProvider:
    """Canned-response LLM provider that records every prompt it receives.

    Stages are told apart by their system prompt: `queue()` enqueues responses
    for one stage, consumed in call order. Without queued responses a stage
    falls back to its entry in `defaults` (extraction needs valid JSON to keep
    unrelated tests ingesting cleanly), and otherwise to `text`.
    """

    defaults: dict[str, str] = {
        EXTRACTION_SYSTEM_PROMPT: "[]",
        SUMMARY_SYSTEM_PROMPT: (
            '{"chapter_summary": "A stubbed chapter summary.", "section_summaries": []}'
        ),
    }

    def __init__(self, text: str = "A stubbed book profile.", model: str = "stub-llm-1") -> None:
        self.text = text
        self.model = model
        self.queues: dict[str, list[str]] = {}
        self.calls: list[tuple[str, str | None]] = []

    def queue(self, system: str, *responses: str) -> None:
        self.queues.setdefault(system, []).extend(responses)

    # Fixed per-call usage so tests can assert exact accumulated totals.
    INPUT_TOKENS_PER_CALL = 100
    OUTPUT_TOKENS_PER_CALL = 10

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        self.calls.append((prompt, system))
        queued = self.queues.get(system or "")
        if queued:
            text = queued.pop(0)
        elif system in self.defaults:
            text = self.defaults[system]
        else:
            text = self.text
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=self.INPUT_TOKENS_PER_CALL,
            output_tokens=self.OUTPUT_TOKENS_PER_CALL,
        )


@pytest.fixture()
def stub_llm() -> StubLLMProvider:
    return StubLLMProvider()


class StubEmbeddingProvider:
    """Deterministic tiny vectors; records every batch of texts it embeds."""

    model = "stub-embed-1"

    def __init__(self) -> None:
        self.max_batch = 100  # a Limit in the real providers; overridable per test
        self.batches: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(text) % 5 + 1), 1.0, 0.5] for text in texts]


@pytest.fixture()
def stub_embedder() -> StubEmbeddingProvider:
    return StubEmbeddingProvider()


@pytest.fixture()
def vector_store() -> VectorStore:
    """A fresh in-memory Qdrant per test."""
    return VectorStore(QdrantClient(":memory:"))


@pytest.fixture(autouse=True)
def _never_call_real_llm(
    monkeypatch: pytest.MonkeyPatch,
    stub_llm: StubLLMProvider,
    stub_embedder: StubEmbeddingProvider,
    vector_store: VectorStore,
) -> None:
    """The worker builds real providers when none are injected; tests never do that.

    The same instances as the `stub_llm` / `stub_embedder` / `vector_store`
    fixtures are installed, so tests can inspect prompts, embedded texts, and
    stored vectors without passing the stubs explicitly.
    """
    monkeypatch.setattr("app.worker.build_default_llm", lambda: stub_llm)
    monkeypatch.setattr("app.worker.build_default_embedder", lambda: stub_embedder)
    monkeypatch.setattr("app.worker.build_default_vector_store", lambda: vector_store)
