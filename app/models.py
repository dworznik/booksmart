import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Book(Base):
    __tablename__ = "books"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[str]
    author: Mapped[str]
    edition: Mapped[str | None]
    publication_year: Mapped[int | None]
    isbn: Mapped[str | None]

    primary_topic: Mapped[str | None]
    language: Mapped[str | None]
    framework: Mapped[str | None]
    methodology: Mapped[str | None]
    notes: Mapped[str | None]
    trust_level: Mapped[str | None]
    intended_use: Mapped[str | None]

    original_filename: Mapped[str]
    file_format: Mapped[str]
    storage_path: Mapped[str]
    checksum: Mapped[str]
    file_hash: Mapped[str]
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class IngestionJob(Base):
    """One ingestion run for a book. Rows are never deleted; they form the history."""

    __tablename__ = "ingestion_jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    book_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("books.id"))
    status: Mapped[str] = mapped_column(default="queued")
    error: Mapped[str | None] = mapped_column(Text)
    output_path: Mapped[str | None]
    parser_used: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
