"""Qdrant vector storage.

One collection holds every embedded record; payloads link each point back to
its relational row (record_type + record_id) and its book. Postgres and the
filesystem remain the source of truth - Qdrant only ever stores summaries and
extracted content, never the sole copy of raw book text.
"""

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

COLLECTION_NAME = "booksmart"


@dataclass(frozen=True)
class VectorRecord:
    id: str  # UUID string; stored on the relational row as embedding_id
    vector: list[float]
    payload: dict[str, Any]


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str = COLLECTION_NAME) -> None:
        self.client = client
        self.collection = collection

    def _ensure_collection(self, vector_size: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=qmodels.VectorParams(
                    size=vector_size, distance=qmodels.Distance.COSINE
                ),
            )

    def replace_book_points(self, book_id: str, records: list[VectorRecord]) -> None:
        """Upsert this run's points, then drop the book's stale points from
        earlier runs. Upsert-first so a mid-replace failure leaves the previous
        vectors in place rather than none."""
        book_filter = qmodels.FieldCondition(
            key="book_id", match=qmodels.MatchValue(value=book_id)
        )
        if records:
            self._ensure_collection(vector_size=len(records[0].vector))
            self.client.upsert(
                self.collection,
                points=[
                    qmodels.PointStruct(id=record.id, vector=record.vector, payload=record.payload)
                    for record in records
                ],
            )
            stale = qmodels.Filter(
                must=[book_filter],
                must_not=[qmodels.HasIdCondition(has_id=[record.id for record in records])],
            )
        elif self.client.collection_exists(self.collection):
            stale = qmodels.Filter(must=[book_filter])
        else:
            return
        self.client.delete(self.collection, points_selector=qmodels.FilterSelector(filter=stale))
