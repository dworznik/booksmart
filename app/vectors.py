"""Qdrant vector storage.

One collection holds every embedded record; payloads link each point back to
its relational row (record_type + record_id) and its book. Postgres and the
filesystem remain the source of truth - Qdrant only ever stores summaries and
extracted content, never the sole copy of raw book text.

The collection is locked to one embedding model (ADR 0001): collection
metadata records the model it was created for, and writes from any other
model are rejected even at matching dimensions — vectors from different
models are incomparable, and same-dimension mixing degrades search in a way
no operator can diagnose from symptoms. Switching models is an explicit
migration: drop the collection and reprocess embeddings for every book.
"""

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from app.llm import ProviderConfigError

COLLECTION_NAME = "booksmart"

EMBEDDING_MODEL_KEY = "embedding_model"


@dataclass(frozen=True)
class VectorRecord:
    id: str  # UUID string; stored on the relational row as embedding_id
    vector: list[float]
    payload: dict[str, Any]


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str = COLLECTION_NAME) -> None:
        self.client = client
        self.collection = collection

    def _ensure_collection(self, vector_size: int, embedding_model: str) -> None:
        """Create the collection for this model, or verify the model lock."""
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=qmodels.VectorParams(
                    size=vector_size, distance=qmodels.Distance.COSINE
                ),
                metadata={EMBEDDING_MODEL_KEY: embedding_model},
            )
            return
        metadata = self.client.get_collection(self.collection).config.metadata or {}
        locked_model = metadata.get(EMBEDDING_MODEL_KEY)
        if locked_model is None:
            # A collection created before model locking records no model, so
            # the lock cannot be verified — stamping the configured model onto
            # vectors that may come from another model is exactly the silent
            # mixing ADR 0001 forbids.
            raise ProviderConfigError(
                f"vector collection {self.collection!r} predates model locking and "
                f"records no embedding model; drop the collection and reprocess "
                f"embeddings to adopt model-locked storage (ADR 0001)"
            )
        if locked_model != embedding_model:
            raise ProviderConfigError(
                f"vector collection {self.collection!r} is locked to embedding model "
                f"{locked_model!r} but the configured embedder is {embedding_model!r}; "
                f"switching models requires dropping the collection and reprocessing "
                f"embeddings for every book (ADR 0001)"
            )

    def replace_book_points(
        self, book_id: str, records: list[VectorRecord], embedding_model: str
    ) -> None:
        """Upsert this run's points, then drop the book's stale points from
        earlier runs. Upsert-first so a mid-replace failure leaves the previous
        vectors in place rather than none."""
        book_filter = qmodels.FieldCondition(
            key="book_id", match=qmodels.MatchValue(value=book_id)
        )
        if records:
            self._ensure_collection(len(records[0].vector), embedding_model)
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
            # Deleting a book's points is safe under any model; only writes
            # are model-locked.
            stale = qmodels.Filter(must=[book_filter])
        else:
            return
        self.client.delete(self.collection, points_selector=qmodels.FilterSelector(filter=stale))
