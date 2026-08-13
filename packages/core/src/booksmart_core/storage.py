"""Filesystem object storage (v1).

Originals live under storage/books/<book_id>/; parsed Markdown under
storage/parsed/<book_id>/parsed.md; run logs under storage/logs/<run_id>.log.

Persisted paths (``Book.storage_path`` / ``Book.parsed_path`` / ``Run.output_path``)
are stored *relative* to the storage root and resolved at read time. That keeps
the whole data dir (sqlite file + storage/ + local Qdrant) a portable unit: move
it under a different absolute root and every artifact still resolves.
"""

import hashlib
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class StoredFile:
    # Relative to the storage root (see module docstring); resolve via
    # BookStorage.resolve before touching the filesystem.
    path: Path
    checksum: str
    file_hash: str


def hash_stream(stream: BinaryIO) -> str:
    """SHA-256 of the stream's content, leaving the stream rewound for reuse."""
    sha256 = hashlib.sha256()
    while chunk := stream.read(CHUNK_SIZE):
        sha256.update(chunk)
    stream.seek(0)
    return sha256.hexdigest()


class BookStorage:
    def __init__(self, storage_root: Path) -> None:
        self._root = Path(storage_root)
        self._books_dir = self._root / "books"
        self._parsed_dir = self._root / "parsed"
        self._logs_dir = self._root / "logs"

    def resolve(self, stored_path: str | Path) -> Path:
        """Resolve a persisted root-relative path to an absolute filesystem path.

        Idempotent for already-absolute inputs so callers holding a legacy
        absolute value still work."""
        path = Path(stored_path)
        return path if path.is_absolute() else self._root / path

    def save_original(
        self, book_id: uuid.UUID, filename: str, stream: BinaryIO, file_hash: str
    ) -> StoredFile:
        """Stream the upload to disk, computing the MD5 checksum on the way.

        The SHA-256 hash comes from the caller (hash_stream), which needed it
        before deciding to store anything at all. The returned path is relative
        to the storage root.
        """
        # Uploads may carry path separators in the client-supplied name; keep only the basename.
        relative = Path("books") / str(book_id) / Path(filename).name
        target = self._root / relative
        target.parent.mkdir(parents=True, exist_ok=True)

        md5 = hashlib.md5()
        try:
            with target.open("wb") as out:
                while chunk := stream.read(CHUNK_SIZE):
                    md5.update(chunk)
                    out.write(chunk)
        except Exception:
            shutil.rmtree(target.parent, ignore_errors=True)
            raise

        return StoredFile(path=relative, checksum=md5.hexdigest(), file_hash=file_hash)

    def save_parsed(self, book_id: uuid.UUID, markdown: str) -> Path:
        """Write the book's current parsed markdown, replacing any earlier one.

        One artifact per book (the parse stage replaces its output wholesale
        and the pointer lives on Book.parsed_path), so the filename is stable
        rather than per-run — a run never learns its own id. Returns the path
        relative to the storage root."""
        relative = Path("parsed") / str(book_id) / "parsed.md"
        target = self._root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown, encoding="utf-8")
        return relative

    def save_log(self, run_id: uuid.UUID, content: str) -> Path:
        """Persist a run's log. Not referenced from the database, so it stays an
        absolute path rather than a relocatable pointer."""
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        target = self._logs_dir / f"{run_id}.log"
        target.write_text(content, encoding="utf-8")
        return target

    def discard(self, book_id: uuid.UUID) -> None:
        shutil.rmtree(self._books_dir / str(book_id), ignore_errors=True)
