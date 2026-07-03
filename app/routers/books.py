import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Book
from app.schemas import BookOut
from app.storage import BookStorage

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
    db: Session = Depends(get_db),
    storage: BookStorage = Depends(_storage),
) -> Book:
    file_format = _validated_format(file)

    book_id = uuid.uuid4()
    stored = storage.save_original(book_id, file.filename or f"book.{file_format}", file.file)
    book = Book(
        id=book_id,
        title=title,
        author=author,
        edition=edition,
        publication_year=publication_year,
        isbn=isbn,
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
