"""Filesystem object storage (v1).

Originals live under storage/books/<book_id>/; parsed Markdown under
storage/parsed/<book_id>/<job_id>.md.
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
        self._books_dir = storage_root / "books"
        self._parsed_dir = storage_root / "parsed"
        self._logs_dir = storage_root / "logs"

    def save_original(self, book_id: uuid.UUID, filename: str, stream: BinaryIO) -> StoredFile:
        """Stream the upload to disk, computing MD5 checksum and SHA-256 hash on the way."""
        book_dir = self._books_dir / str(book_id)
        book_dir.mkdir(parents=True, exist_ok=True)
        # Uploads may carry path separators in the client-supplied name; keep only the basename.
        target = book_dir / Path(filename).name

        md5 = hashlib.md5()
        sha256 = hashlib.sha256()
        try:
            with target.open("wb") as out:
                while chunk := stream.read(CHUNK_SIZE):
                    md5.update(chunk)
                    sha256.update(chunk)
                    out.write(chunk)
        except Exception:
            shutil.rmtree(book_dir, ignore_errors=True)
            raise

        return StoredFile(path=target, checksum=md5.hexdigest(), file_hash=sha256.hexdigest())

    def save_parsed(self, book_id: uuid.UUID, job_id: uuid.UUID, markdown: str) -> Path:
        parsed_dir = self._parsed_dir / str(book_id)
        parsed_dir.mkdir(parents=True, exist_ok=True)
        target = parsed_dir / f"{job_id}.md"
        target.write_text(markdown, encoding="utf-8")
        return target

    def save_log(self, job_id: uuid.UUID, content: str) -> Path:
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        target = self._logs_dir / f"{job_id}.log"
        target.write_text(content, encoding="utf-8")
        return target

    def discard(self, book_id: uuid.UUID) -> None:
        shutil.rmtree(self._books_dir / str(book_id), ignore_errors=True)
