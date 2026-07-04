import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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


class Chapter(Base):
    """A detected top-level unit of a book's logical structure. Replaced wholesale
    on each successful ingestion run."""

    __tablename__ = "chapters"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    book_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    position: Mapped[int]
    title: Mapped[str]
    source_line: Mapped[int | None]
    summary: Mapped[str | None] = mapped_column(Text)
    embedding_id: Mapped[uuid.UUID | None]
    embedding_model: Mapped[str | None]
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    sections: Mapped[list["Section"]] = relationship(
        back_populates="chapter",
        cascade="all, delete-orphan",
        order_by="Section.position",
    )


class Section(Base):
    __tablename__ = "sections"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    chapter_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chapters.id", ondelete="CASCADE")
    )
    position: Mapped[int]
    title: Mapped[str]
    source_line: Mapped[int | None]
    summary: Mapped[str | None] = mapped_column(Text)
    embedding_id: Mapped[uuid.UUID | None]
    embedding_model: Mapped[str | None]
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    chapter: Mapped[Chapter] = relationship(back_populates="sections")


class BookProfile(Base):
    """An LLM-generated summary of what a book covers. Rows are never deleted;
    each ingestion run appends a new version and the API serves the latest."""

    __tablename__ = "book_profiles"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    book_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    content: Mapped[str] = mapped_column(Text)
    model: Mapped[str]
    prompt_version: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class KnowledgeObject(Base):
    """A typed candidate knowledge object extracted from a book. Replaced
    wholesale per book on each successful extraction run. Provenance fields
    (edition, extraction model, prompt version) are frozen at extraction time
    even though the book's own metadata stays editable."""

    __tablename__ = "knowledge_objects"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    book_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    chapter_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("chapters.id", ondelete="SET NULL")
    )
    section_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("sections.id", ondelete="SET NULL")
    )

    type: Mapped[str]
    title: Mapped[str]
    content: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)
    source_location: Mapped[str]
    confidence: Mapped[float]

    edition: Mapped[str | None]
    page: Mapped[int | None]
    paragraph: Mapped[int | None]
    extraction_model: Mapped[str]
    extraction_prompt_version: Mapped[str]
    embedding_id: Mapped[uuid.UUID | None]
    embedding_model: Mapped[str | None]
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
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
