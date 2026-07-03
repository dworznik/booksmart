import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class BookOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    author: str
    edition: str | None
    publication_year: int | None
    isbn: str | None
    original_filename: str
    file_format: str
    checksum: str
    file_hash: str
    uploaded_at: datetime
