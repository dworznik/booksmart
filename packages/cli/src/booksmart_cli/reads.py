"""Read queries behind the display commands — the shapes the removed GET routers
served (see docs/api-notes/http-surface.md and the conftest read-model helpers).

Each opens a short session and returns detached ORM rows (or plain dicts for the
chapter/section tree, whose relationship would otherwise lazy-load after the
session closes), so command code can render without holding a session open.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from booksmart_core.extraction import KNOWLEDGE_OBJECT_TYPES
from booksmart_core.llm import build_embedding_provider
from booksmart_core.models import Book, BookProfile, Chapter, KnowledgeObject, Run
from booksmart_core.search import SearchResults
from booksmart_core.search import search as core_search
from booksmart_core.vectors import (
    RECORD_TYPES,
    RecordType,
    build_vector_store,
    unknown_record_types,
)

from booksmart_cli.errors import (
    BookNotFoundError,
    CliError,
    KnowledgeNotFoundError,
    NoProfileError,
    RunNotFoundError,
)
from booksmart_cli.runtime import Runtime


@dataclass(frozen=True)
class SectionView:
    position: int
    title: str


@dataclass(frozen=True)
class ChapterView:
    position: int
    title: str
    kind: str
    sections: list[SectionView]


def _require_book(session: Session, book_id: uuid.UUID) -> Book:
    book = session.get(Book, book_id)
    if book is None:
        raise BookNotFoundError(f"No book with id {book_id}")
    return book


def list_books(runtime: Runtime) -> list[Book]:
    """Every book, oldest first (the old GET /books)."""
    with runtime.session_factory() as session:
        books = list(session.scalars(select(Book).order_by(Book.uploaded_at, Book.id)))
        session.expunge_all()
    return books


def get_book(runtime: Runtime, book_id: uuid.UUID) -> Book:
    with runtime.session_factory() as session:
        book = _require_book(session, book_id)
        session.expunge(book)
    return book


def book_structure(runtime: Runtime, book_id: uuid.UUID) -> list[ChapterView]:
    """The chapter/section tree (the old GET /books/{id}/structure)."""
    with runtime.session_factory() as session:
        _require_book(session, book_id)
        chapters = session.scalars(
            select(Chapter).where(Chapter.book_id == book_id).order_by(Chapter.position)
        )
        return [
            ChapterView(
                position=chapter.position,
                title=chapter.title,
                kind=chapter.kind,
                sections=[
                    SectionView(position=section.position, title=section.title)
                    for section in chapter.sections
                ],
            )
            for chapter in chapters
        ]


def latest_profile(runtime: Runtime, book_id: uuid.UUID) -> BookProfile:
    """The newest generated profile (the old GET /books/{id}/profile)."""
    with runtime.session_factory() as session:
        _require_book(session, book_id)
        profile = session.scalars(
            select(BookProfile)
            .where(BookProfile.book_id == book_id)
            .order_by(BookProfile.created_at.desc(), BookProfile.id.desc())
            .limit(1)
        ).first()
        if profile is None:
            raise NoProfileError(f"Book {book_id} has no profile yet; run an ingest first")
        session.expunge(profile)
    return profile


def list_runs(runtime: Runtime, book_id: uuid.UUID) -> list[Run]:
    """A book's run history, oldest first (the old GET /books/{id}/jobs)."""
    with runtime.session_factory() as session:
        _require_book(session, book_id)
        runs = list(
            session.scalars(
                select(Run).where(Run.book_id == book_id).order_by(Run.created_at, Run.id)
            )
        )
        session.expunge_all()
    return runs


def get_run(runtime: Runtime, run_id: uuid.UUID) -> Run:
    with runtime.session_factory() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise RunNotFoundError(f"No run with id {run_id}")
        session.expunge(run)
    return run


def list_knowledge(
    runtime: Runtime, book_id: uuid.UUID, type_filter: str | None = None
) -> list[KnowledgeObject]:
    """A book's knowledge objects (the old GET /books/{id}/knowledge-objects)."""
    if type_filter is not None and type_filter not in KNOWLEDGE_OBJECT_TYPES:
        raise CliError(
            f"Unknown knowledge type {type_filter!r}; expected one of "
            f"{', '.join(sorted(KNOWLEDGE_OBJECT_TYPES))}"
        )
    with runtime.session_factory() as session:
        _require_book(session, book_id)
        query = (
            select(KnowledgeObject)
            .where(KnowledgeObject.book_id == book_id)
            .order_by(KnowledgeObject.created_at, KnowledgeObject.id)
        )
        if type_filter is not None:
            query = query.where(KnowledgeObject.type == type_filter)
        objects = list(session.scalars(query))
        session.expunge_all()
    return objects


def get_knowledge(runtime: Runtime, object_id: uuid.UUID) -> KnowledgeObject:
    with runtime.session_factory() as session:
        obj = session.get(KnowledgeObject, object_id)
        if obj is None:
            raise KnowledgeNotFoundError(f"No knowledge object with id {object_id}")
        session.expunge(obj)
    return obj


def semantic_search(
    runtime: Runtime,
    query: str,
    *,
    book_id: uuid.UUID | None = None,
    record_types: Sequence[str] | None = None,
    limit: int = 10,
    score_threshold: float | None = None,
) -> SearchResults:
    """Rank embedded records against a natural-language query (no HTTP ancestor —
    this is the first post-split feature, issue #30).

    Validates the user's input before building any provider, so a typo'd book id
    or record type never demands an embedding API key. The embedded Qdrant client
    is closed on the way out: it holds a single-process lock on the on-disk
    directory, and the next command has to be able to open it.

    Core's ``SearchResults`` is passed through whole rather than unwrapped to
    ``.hits``: the `search` command renders hits only, but this is the read
    layer, and dropping the query's usage here would put it out of reach of
    anything else built on reads.py."""
    if not query.strip():
        raise CliError("Search query must not be empty")
    if limit < 1:
        raise CliError(f"--limit must be at least 1, got {limit}")
    unknown = unknown_record_types(record_types or ())
    if unknown:
        raise CliError(
            f"Unknown record type {', '.join(repr(name) for name in unknown)}; "
            f"expected one of {', '.join(RECORD_TYPES)}"
        )
    if book_id is not None:
        with runtime.session_factory() as session:
            _require_book(session, book_id)

    embedder = build_embedding_provider(runtime.settings)
    vector_store = build_vector_store(runtime.settings)
    try:
        with runtime.session_factory() as session:
            return core_search(
                session,
                vector_store,
                embedder,
                query,
                book_id=book_id,
                # Validated against RECORD_TYPES above.
                record_types=(
                    cast("list[RecordType]", list(record_types)) if record_types else None
                ),
                limit=limit,
                score_threshold=score_threshold,
            )
    finally:
        vector_store.close()
