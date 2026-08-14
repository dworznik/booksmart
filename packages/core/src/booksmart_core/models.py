import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    """Timestamp default computed client-side rather than via a server ``now()``.

    The baseline migration is dialect-neutral (sqlite and Postgres share one
    history), and ``now()`` is a Postgres-ism; generating the value in Python
    keeps inserts portable across both."""
    return datetime.now(UTC)


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
        DateTime(timezone=True), default=_utcnow
    )

    # The current parsed-markdown artifact and the parser that produced it,
    # written by the parse stage and replaced wholesale on every re-parse.
    # Downstream stages resolve their input from here instead of querying past
    # runs; NULL until the book has been parsed at least once.
    parsed_path: Mapped[str | None]
    parser_used: Mapped[str | None]


class Chapter(Base):
    """A detected top-level unit of a book's logical structure. Replaced wholesale
    on each successful ingestion run."""

    __tablename__ = "chapters"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    book_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    position: Mapped[int]
    title: Mapped[str]
    # "front_matter" | "chapter" | "back_matter"; lets downstream stages skip
    # or weight peripheral matter while provenance can still point into it.
    kind: Mapped[str] = mapped_column(default="chapter", server_default="chapter")
    source_line: Mapped[int | None]
    summary: Mapped[str | None] = mapped_column(Text)
    summary_model: Mapped[str | None]
    summary_prompt_version: Mapped[str | None]
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
    summary_model: Mapped[str | None]
    summary_prompt_version: Mapped[str | None]
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
        DateTime(timezone=True), default=_utcnow
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
        DateTime(timezone=True), default=_utcnow
    )


class Run(Base):
    """The record of one pipeline execution over a book (CONTEXT.md: Run).

    Runner-owned provenance: its Scope, outcome, version stamps and token
    spend. Created the moment execution starts — there is no queued state, so
    status is only ``running`` | ``succeeded`` | ``failed``. Rows are never
    deleted; they form the history. Stages never see this row."""

    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    book_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("books.id"))
    # Which stages this run covers: "full" | "profile" | "extraction" | "embeddings".
    scope: Mapped[str] = mapped_column(default="full", server_default="full")
    status: Mapped[str] = mapped_column(default="running")
    error: Mapped[str | None] = mapped_column(Text)
    # The parsed artifact this run produced and the parser it used; set only on
    # runs whose scope includes the parse stage, NULL on incremental runs that
    # reused the book's existing parsed markdown.
    output_path: Mapped[str | None]
    parser_used: Mapped[str | None]
    # Stamped when the run executes, so history records exactly what produced it.
    extraction_version: Mapped[str | None]
    model_version: Mapped[str | None]
    prompt_version: Mapped[str | None]
    # Summed provider-reported LLM usage across the run's calls; NULL when the
    # scope made no LLM calls.
    input_tokens: Mapped[int | None]
    output_tokens: Mapped[int | None]
    # Embedding usage, summed the same way and kept apart: it is billed at a
    # different rate, so folding it into `input_tokens` would produce a total
    # nobody can cost from. NULL when the scope embedded nothing.
    embedding_tokens: Mapped[int | None]
    # created_at is the execution start (there is no queued state before it).
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Eagerly loaded, because a Run is routinely read and then detached — the
    # CLI expunges it before rendering — and a lazy relationship there raises
    # instead of answering.
    stages: Mapped[list["RunStage"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="RunStage.position",
        lazy="selectin",
    )


class RunStage(Base):
    """What one Stage of a Run did: its spend, its throughput, its clock.

    The Runner already receives a `StageReport` per Stage and summed it into the
    Run row; this is that report kept rather than discarded. A run-level total
    can say what a book cost and never which Stage to move to a cheaper model,
    nor which Stage is the slow one.

    Stages stay Run-blind (ADR 0002): no Stage writes this, and none can see it.
    The Runner owns the Run record, and these rows are part of it — they are
    written once when the Run is finalized, and never updated."""

    __tablename__ = "run_stages"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    # Execution order within the run, so history reads back in the order it
    # happened even where two stages share a timestamp.
    position: Mapped[int]
    stage: Mapped[str]
    # Zero rather than NULL: a Stage that ran and called no provider spent
    # nothing, which is a measurement. The Run-level totals keep the NULLs,
    # where the distinction is "no LLM work" against "LLM work reporting zero".
    input_tokens: Mapped[int] = mapped_column(default=0)
    output_tokens: Mapped[int] = mapped_column(default=0)
    embedding_tokens: Mapped[int] = mapped_column(default=0)
    # What the Stage got through — `{"chapters": 12}` and the like. A cost is
    # only readable beside a throughput: five hundred tokens over two chapters
    # is a different fact from five hundred over fifty.
    counts: Mapped[dict[str, int]] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    run: Mapped[Run] = relationship(back_populates="stages")

    @property
    def seconds(self) -> float | None:
        """How long the Stage took, or nothing if it never reported a clock."""
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()
