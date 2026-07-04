import uuid
from datetime import datetime

from sqlalchemy import DateTime, func
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
