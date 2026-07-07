"""Book registration: the validation, dedup, and storage the removed upload
endpoint owned (docs/api-notes/upload-validation.md), ported for a local file
path instead of a multipart stream.
"""

import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from booksmart_core.models import Book
from booksmart_core.storage import hash_stream

from booksmart_cli.errors import (
    BookNotFoundError,
    CliError,
    DuplicateBookError,
    UnsupportedFileError,
)
from booksmart_cli.runtime import Runtime


@dataclass(frozen=True)
class SupportedFormat:
    name: str
    magic: bytes


# Suffix must match AND the leading bytes must match — a .pdf of HTML is
# rejected. EPUB is a ZIP container, so its magic is the ZIP local-file header.
SUPPORTED_FORMATS = {
    ".pdf": SupportedFormat("pdf", b"%PDF"),
    ".epub": SupportedFormat("epub", b"PK\x03\x04"),
}

# The optional bibliographic metadata + hints a user may set on `add`/`update`
# — the single source of truth for what is editable and what `books show`
# prints. File-provenance columns (storage_path, checksum, file_hash, …) are
# never here: the old PATCH rejected them 422, and the CLI never exposes them.
METADATA_FIELDS = (
    "edition",
    "publication_year",
    "isbn",
    "primary_topic",
    "language",
    "framework",
    "methodology",
    "notes",
    "trust_level",
    "intended_use",
)


def validated_format(path: Path) -> str:
    """Format name if ``path``'s suffix and magic bytes agree, else raise."""
    supported = SUPPORTED_FORMATS.get(path.suffix.lower())
    if supported is None:
        raise UnsupportedFileError(
            f"Unsupported file type {path.suffix or '(none)'}; expected .pdf or .epub"
        )
    with path.open("rb") as stream:
        header = stream.read(len(supported.magic))
    if not header.startswith(supported.magic):
        raise UnsupportedFileError(
            f"File content does not look like a {supported.name.upper()} file"
        )
    return supported.name


def register_book(
    runtime: Runtime,
    path: Path,
    *,
    title: str,
    author: str,
    metadata: dict[str, object] | None = None,
) -> Book:
    """Validate the file, reject a byte-identical duplicate, store the original,
    and persist the Book row — rolling the stored file back if the insert fails.
    Returns the persisted (detached) Book."""
    if not path.is_file():
        raise UnsupportedFileError(f"No such file: {path}")
    file_format = validated_format(path)

    with path.open("rb") as stream:
        file_hash = hash_stream(stream)

    with runtime.session_factory() as session:
        existing = session.scalars(
            select(Book).where(Book.file_hash == file_hash)
        ).first()
        if existing is not None:
            raise DuplicateBookError(str(existing.id))

    book_id = uuid.uuid4()
    with path.open("rb") as stream:
        stored = runtime.storage.save_original(book_id, path.name, stream, file_hash)

    book = Book(
        id=book_id,
        title=title,
        author=author,
        original_filename=stored.path.name,
        file_format=file_format,
        storage_path=str(stored.path),
        checksum=stored.checksum,
        file_hash=stored.file_hash,
        **{k: v for k, v in (metadata or {}).items() if v is not None},
    )
    try:
        with runtime.session_factory() as session:
            session.add(book)
            session.commit()
            session.refresh(book)
            session.expunge(book)
    except Exception:
        # Store-then-rollback guard: drop the orphaned original if the DB
        # insert fails, so a failed registration leaves nothing behind.
        runtime.storage.discard(book_id)
        raise
    return book


def update_book(runtime: Runtime, book_id: uuid.UUID, changes: dict[str, object]) -> Book:
    """Apply metadata changes to a book. ``title``/``author`` may be changed but
    not cleared; any editable field may be set or cleared (``None``). Absent keys
    are untouched. Returns the updated (detached) Book."""
    allowed = {"title", "author", *METADATA_FIELDS}
    unknown = set(changes) - allowed
    if unknown:
        # Guards against a caller reaching a file-provenance column; the old
        # PATCH answered 422 for these, the CLI never exposes them.
        raise CliError(f"Fields are not editable: {', '.join(sorted(unknown))}")
    with runtime.session_factory() as session:
        book = session.get(Book, book_id)
        if book is None:
            raise BookNotFoundError(f"No book with id {book_id}")
        for field, value in changes.items():
            setattr(book, field, value)
        session.commit()
        session.refresh(book)
        session.expunge(book)
    return book
