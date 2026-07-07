"""Read queries behind the display commands — the shapes the removed GET routers
served (see docs/api-notes/http-surface.md and the conftest read-model helpers).

Each opens a short session and returns detached ORM rows (or plain dicts for the
chapter/section tree, whose relationship would otherwise lazy-load after the
session closes), so command code can render without holding a session open.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from booksmart_core.extraction import KNOWLEDGE_OBJECT_TYPES
from booksmart_core.models import Book, BookProfile, Chapter, KnowledgeObject, Run

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
