"""Integration test fixtures and read-model helpers.

The suite is dialect-neutral and runs against whichever database
``BOOKSMART_TEST_DATABASE_URL`` names. Unset, it defaults to a service-free
SQLite file (the CLI's dialect and the local/default CI path); CI also runs the
whole suite a second time against a Postgres service container, so a Postgres-ism
cannot land silently. A dedicated database is migrated once per session with the
same single Alembic history any consumer uses.

The HTTP server is gone (booksmart-core is a library now), so tests drive the
pipeline through the public Runner / Stage functions and read results straight
from the database. The helpers at the bottom rebuild the read shapes the old GET
routers served (a book's structure, its latest profile, its knowledge objects,
its runs) so ported scenario tests can keep asserting on the same shapes.
"""

import io
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from qdrant_client import QdrantClient
from sqlalchemy import Engine, create_engine, event, select, text
from sqlalchemy.orm import Session, sessionmaker

from booksmart_core import MIGRATIONS_PATH
from booksmart_core.config import Settings
from booksmart_core.extraction import EXTRACTION_SYSTEM_PROMPT
from booksmart_core.llm import LLMResponse
from booksmart_core.models import Base, Book, BookProfile, Chapter, KnowledgeObject, Run
from booksmart_core.runner import execute_run
from booksmart_core.storage import BookStorage, hash_stream
from booksmart_core.summaries import SUMMARY_SYSTEM_PROMPT
from booksmart_core.vectors import VectorStore

# Unset -> a portable, service-free SQLite database resolved per session below.
TEST_DATABASE_URL = os.environ.get("BOOKSMART_TEST_DATABASE_URL", "")


def enable_sqlite_foreign_keys(engine: Engine) -> None:
    """Make a SQLite engine enforce foreign keys on every connection.

    SQLite honours FKs (and thus the ON DELETE CASCADE the stages' bulk deletes
    rely on) only when asked, per connection. Shared by the session fixture and
    the standalone engines the portability test builds."""

    @event.listens_for(engine, "connect")
    def _enable_fk(dbapi_conn: object, _: object) -> None:
        cursor = dbapi_conn.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def _ensure_postgres_database(url: str) -> None:
    """Create the Postgres test database if it does not exist yet."""
    admin_url = url.rsplit("/", 1)[0] + "/postgres"
    test_db_name = url.rsplit("/", 1)[1]
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"),
            {"name": test_db_name},
        ).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{test_db_name}"'))
    admin_engine.dispose()


@pytest.fixture(scope="session")
def database_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    url = TEST_DATABASE_URL
    if not url:
        db_path = tmp_path_factory.mktemp("db") / "booksmart_test.db"
        url = f"sqlite:///{db_path}"
    if url.startswith("postgresql"):
        _ensure_postgres_database(url)

    # Resolve the migration history from the installed package — the same path a
    # consumer (CLI or server) uses — rather than a source-tree layout.
    alembic_cfg = AlembicConfig()
    alembic_cfg.set_main_option("script_location", str(MIGRATIONS_PATH))
    alembic_cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(alembic_cfg, "head")
    return url


@pytest.fixture()
def settings(database_url: str, tmp_path: Path) -> Settings:
    return Settings(database_url=database_url, storage_root=tmp_path / "storage")


@pytest.fixture(scope="session")
def db_engine(database_url: str) -> Iterator[Engine]:
    engine = create_engine(database_url)
    if engine.dialect.name == "sqlite":
        enable_sqlite_foreign_keys(engine)
    yield engine
    engine.dispose()


def _reset_database(engine: Engine) -> None:
    """Empty every table, giving each test a clean slate (the role the old
    TestClient fixture played). Postgres truncates books and cascades; SQLite
    has no TRUNCATE, so delete each table children-first with FKs enforced."""
    with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            conn.execute(text("TRUNCATE TABLE books CASCADE"))
        else:
            for table in reversed(Base.metadata.sorted_tables):
                conn.execute(table.delete())


@pytest.fixture()
def session_factory(db_engine: Engine) -> sessionmaker[Session]:
    """A clean-slate session factory for driving the pipeline in tests."""
    _reset_database(db_engine)
    return sessionmaker(bind=db_engine)


@pytest.fixture()
def storage(settings: Settings) -> BookStorage:
    return BookStorage(settings.storage_root)


# --- book registration ------------------------------------------------------
#
# The upload router that validated and stored books is gone (documented in
# docs/api-notes/); tests create books directly from core primitives.


def store_book(
    session_factory: sessionmaker[Session],
    storage: BookStorage,
    *,
    title: str,
    author: str,
    filename: str,
    content: bytes,
    **fields: object,
) -> str:
    """Persist a book and its original file the way the old upload endpoint did,
    minus the HTTP concerns (multipart, magic-byte validation, dedup — those
    are documented for consumers to reimplement). Returns the book id."""
    file_format = str(fields.pop("file_format", None) or Path(filename).suffix.lstrip(".").lower())
    book_id = uuid.uuid4()
    stream = io.BytesIO(content)
    file_hash = hash_stream(stream)
    stored = storage.save_original(book_id, filename, stream, file_hash)
    with session_factory() as session:
        session.add(
            Book(
                id=book_id,
                title=title,
                author=author,
                original_filename=stored.path.name,
                file_format=file_format,
                storage_path=str(stored.path),
                checksum=stored.checksum,
                file_hash=stored.file_hash,
                **fields,
            )
        )
        session.commit()
    return str(book_id)


# --- read models (rebuild the removed GET routers' shapes) ------------------


def run_dict(run: Run) -> dict[str, object]:
    return {
        "id": str(run.id),
        "book_id": str(run.book_id),
        "scope": run.scope,
        "status": run.status,
        "error": run.error,
        "output_path": run.output_path,
        "parser_used": run.parser_used,
        "extraction_version": run.extraction_version,
        "model_version": run.model_version,
        "prompt_version": run.prompt_version,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


def get_run(session_factory: sessionmaker[Session], run_id: str) -> dict[str, object] | None:
    with session_factory() as session:
        run = session.get(Run, uuid.UUID(run_id))
        return run_dict(run) if run is not None else None


def run_scope(
    session_factory: sessionmaker[Session],
    settings: Settings,
    book_id: str,
    scope: str = "full",
) -> dict[str, object]:
    """Run one scope to completion and return its Run as a read-model dict — the
    library-level equivalent of the old POST /ingest / POST /reprocess."""
    run_id = execute_run(session_factory, settings.storage_root, uuid.UUID(book_id), scope)
    run = get_run(session_factory, str(run_id))
    assert run is not None
    return run


def runs_for_book(session_factory: sessionmaker[Session], book_id: str) -> list[dict[str, object]]:
    """A book's run history, oldest first (the old GET /books/{id}/jobs)."""
    with session_factory() as session:
        return [
            run_dict(run)
            for run in session.scalars(
                select(Run)
                .where(Run.book_id == uuid.UUID(book_id))
                .order_by(Run.created_at, Run.id)
            )
        ]


def book_structure(session_factory: sessionmaker[Session], book_id: str) -> list[dict[str, object]]:
    """The chapter/section tree (the old GET /books/{id}/structure)."""
    with session_factory() as session:
        chapters = session.scalars(
            select(Chapter)
            .where(Chapter.book_id == uuid.UUID(book_id))
            .order_by(Chapter.position)
        )
        return [
            {
                "id": str(chapter.id),
                "position": chapter.position,
                "title": chapter.title,
                "kind": chapter.kind,
                "source_line": chapter.source_line,
                "sections": [
                    {
                        "id": str(section.id),
                        "position": section.position,
                        "title": section.title,
                        "source_line": section.source_line,
                    }
                    for section in chapter.sections
                ],
            }
            for chapter in chapters
        ]


def latest_profile(
    session_factory: sessionmaker[Session], book_id: str
) -> dict[str, object] | None:
    """The newest generated profile (the old GET /books/{id}/profile)."""
    with session_factory() as session:
        profile = session.scalars(
            select(BookProfile)
            .where(BookProfile.book_id == uuid.UUID(book_id))
            .order_by(BookProfile.created_at.desc(), BookProfile.id.desc())
            .limit(1)
        ).first()
        if profile is None:
            return None
        return {
            "id": str(profile.id),
            "book_id": str(profile.book_id),
            "content": profile.content,
            "model": profile.model,
            "prompt_version": profile.prompt_version,
            "created_at": profile.created_at.isoformat(),
        }


def knowledge_objects(
    session_factory: sessionmaker[Session], book_id: str, type_filter: str | None = None
) -> list[dict[str, object]]:
    """A book's knowledge objects (the old GET /books/{id}/knowledge-objects)."""
    with session_factory() as session:
        query = (
            select(KnowledgeObject)
            .where(KnowledgeObject.book_id == uuid.UUID(book_id))
            .order_by(KnowledgeObject.created_at, KnowledgeObject.id)
        )
        if type_filter is not None:
            query = query.where(KnowledgeObject.type == type_filter)
        return [
            {
                "id": str(ko.id),
                "book_id": str(ko.book_id),
                "chapter_id": str(ko.chapter_id) if ko.chapter_id else None,
                "section_id": str(ko.section_id) if ko.section_id else None,
                "type": ko.type,
                "title": ko.title,
                "content": ko.content,
                "summary": ko.summary,
                "source_location": ko.source_location,
                "confidence": ko.confidence,
                "edition": ko.edition,
                "page": ko.page,
                "paragraph": ko.paragraph,
                "extraction_model": ko.extraction_model,
                "extraction_prompt_version": ko.extraction_prompt_version,
            }
            for ko in session.scalars(query)
        ]


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
    """The runner builds real providers when none are injected; tests never do that.

    The same instances as the `stub_llm` / `stub_embedder` / `vector_store`
    fixtures are installed, so tests can inspect prompts, embedded texts, and
    stored vectors without passing the stubs explicitly.
    """
    monkeypatch.setattr("booksmart_core.runner.build_default_llm", lambda: stub_llm)
    monkeypatch.setattr("booksmart_core.runner.build_default_embedder", lambda: stub_embedder)
    monkeypatch.setattr("booksmart_core.runner.build_default_vector_store", lambda: vector_store)
