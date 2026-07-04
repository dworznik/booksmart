import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BookOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    author: str
    edition: str | None
    publication_year: int | None
    isbn: str | None
    primary_topic: str | None
    language: str | None
    framework: str | None
    methodology: str | None
    notes: str | None
    trust_level: str | None
    intended_use: str | None
    original_filename: str
    file_format: str
    checksum: str
    file_hash: str
    uploaded_at: datetime


class BookUpdate(BaseModel):
    """Partial update: only fields present in the request body are applied.

    File fields (checksum, file_hash, original_filename, file_format,
    storage_path, uploaded_at) are never editable, so extra fields are
    rejected outright.
    """

    model_config = ConfigDict(extra="forbid")

    title: Annotated[str, Field(min_length=1)] | None = None
    author: Annotated[str, Field(min_length=1)] | None = None
    edition: str | None = None
    publication_year: int | None = None
    isbn: str | None = None
    primary_topic: str | None = None
    language: str | None = None
    framework: str | None = None
    methodology: str | None = None
    notes: str | None = None
    trust_level: str | None = None
    intended_use: str | None = None

    @model_validator(mode="after")
    def _reject_null_required_fields(self) -> "BookUpdate":
        for field in ("title", "author"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null")
        return self

    def changes(self) -> dict[str, object]:
        """Only the fields explicitly provided in the request body."""
        return self.model_dump(exclude_unset=True)


class SectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    position: int
    title: str
    source_line: int | None


class ChapterOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    position: int
    title: str
    source_line: int | None
    sections: list[SectionOut]


class BookProfileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: uuid.UUID
    book_id: uuid.UUID
    content: str
    model: str
    prompt_version: str
    created_at: datetime


class KnowledgeObjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    book_id: uuid.UUID
    chapter_id: uuid.UUID | None
    section_id: uuid.UUID | None
    type: str
    title: str
    content: str
    summary: str
    source_location: str
    confidence: float
    edition: str | None
    page: int | None
    paragraph: int | None
    extraction_model: str
    extraction_prompt_version: str
    created_at: datetime


class IngestionJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    book_id: uuid.UUID
    status: str
    error: str | None
    output_path: str | None
    parser_used: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
