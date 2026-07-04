import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Book, BookProfile, Chapter
from app.schemas import BookOut, BookProfileOut, BookUpdate, ChapterOut
from app.storage import BookStorage, hash_stream

router = APIRouter(prefix="/books", tags=["books"])


@dataclass(frozen=True)
class SupportedFormat:
    name: str
    magic: bytes


SUPPORTED_FORMATS = {
    ".pdf": SupportedFormat("pdf", b"%PDF"),
    # EPUB is a ZIP container, so its magic is the ZIP local-file-header signature.
    ".epub": SupportedFormat("epub", b"PK\x03\x04"),
}


def _validated_format(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix.lower()
    supported = SUPPORTED_FORMATS.get(suffix)
    if supported is None:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type {suffix or '(none)'}; expected .pdf or .epub",
        )
    header = file.file.read(len(supported.magic))
    file.file.seek(0)
    if not header.startswith(supported.magic):
        raise HTTPException(
            status_code=415,
            detail=f"File content does not look like a {supported.name.upper()} file",
        )
    return supported.name


def _storage(request: Request) -> BookStorage:
    return request.app.state.storage  # type: ignore[no-any-return]


@router.post("", status_code=201, response_model=BookOut)
def register_book(
    file: UploadFile,
    title: Annotated[str, Form(min_length=1)],
    author: Annotated[str, Form(min_length=1)],
    edition: Annotated[str | None, Form()] = None,
    publication_year: Annotated[int | None, Form()] = None,
    isbn: Annotated[str | None, Form()] = None,
    primary_topic: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    framework: Annotated[str | None, Form()] = None,
    methodology: Annotated[str | None, Form()] = None,
    notes: Annotated[str | None, Form()] = None,
    trust_level: Annotated[str | None, Form()] = None,
    intended_use: Annotated[str | None, Form()] = None,
    db: Session = Depends(get_db),
    storage: BookStorage = Depends(_storage),
) -> Book:
    file_format = _validated_format(file)

    # Byte-identical content means duplicate, whatever the metadata says.
    file_hash = hash_stream(file.file)
    existing = db.scalars(select(Book).where(Book.file_hash == file_hash)).first()
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "A book with identical file content is already registered",
                "existing_book_id": str(existing.id),
            },
        )

    book_id = uuid.uuid4()
    stored = storage.save_original(book_id, file.filename or f"book.{file_format}", file.file)
    book = Book(
        id=book_id,
        title=title,
        author=author,
        edition=edition,
        publication_year=publication_year,
        isbn=isbn,
        primary_topic=primary_topic,
        language=language,
        framework=framework,
        methodology=methodology,
        notes=notes,
        trust_level=trust_level,
        intended_use=intended_use,
        original_filename=stored.path.name,
        file_format=file_format,
        storage_path=str(stored.path),
        checksum=stored.checksum,
        file_hash=stored.file_hash,
    )
    try:
        db.add(book)
        db.commit()
    except Exception:
        db.rollback()
        storage.discard(book_id)
        raise
    db.refresh(book)
    return book


@router.get("", response_model=list[BookOut])
def list_books(db: Session = Depends(get_db)) -> list[Book]:
    return list(db.scalars(select(Book).order_by(Book.uploaded_at, Book.id)))


@router.get("/{book_id}", response_model=BookOut)
def get_book(book_id: uuid.UUID, db: Session = Depends(get_db)) -> Book:
    book = db.get(Book, book_id)
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    return book


@router.get("/{book_id}/structure", response_model=list[ChapterOut])
def get_book_structure(book_id: uuid.UUID, db: Session = Depends(get_db)) -> list[Chapter]:
    if db.get(Book, book_id) is None:
        raise HTTPException(status_code=404, detail="Book not found")
    return list(
        db.scalars(
            select(Chapter).where(Chapter.book_id == book_id).order_by(Chapter.position)
        )
    )


@router.get("/{book_id}/profile", response_model=BookProfileOut)
def get_book_profile(book_id: uuid.UUID, db: Session = Depends(get_db)) -> BookProfile:
    """The latest generated profile; older versions stay in the table as history."""
    if db.get(Book, book_id) is None:
        raise HTTPException(status_code=404, detail="Book not found")
    profile = db.scalars(
        select(BookProfile)
        .where(BookProfile.book_id == book_id)
        .order_by(BookProfile.created_at.desc(), BookProfile.id.desc())
        .limit(1)
    ).first()
    if profile is None:
        raise HTTPException(status_code=404, detail="No profile generated for this book yet")
    return profile


@router.patch("/{book_id}", response_model=BookOut)
def update_book(
    book_id: uuid.UUID, payload: BookUpdate, db: Session = Depends(get_db)
) -> Book:
    """Update metadata and hints. Books stay editable in any state, forever."""
    book = db.get(Book, book_id)
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    for field, value in payload.changes().items():
        setattr(book, field, value)
    db.commit()
    db.refresh(book)
    return book
