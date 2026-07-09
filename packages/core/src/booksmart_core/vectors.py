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

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, get_args

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from booksmart_core.config import Settings
from booksmart_core.errors import ProviderConfigError

COLLECTION_NAME = "booksmart"

EMBEDDING_MODEL_KEY = "embedding_model"

# The relational row a point was embedded from, recorded in its payload as
# record_type + record_id. The payload is the only link back from a vector to
# the source of truth, so this literal is the collection's contract — writers
# (the embedding stage) and readers (search) both spell it from here.
RecordType = Literal["chapter", "section", "knowledge_object"]

RECORD_TYPES: tuple[RecordType, ...] = get_args(RecordType)


def unknown_record_types(names: Iterable[str]) -> list[str]:
    """Which of ``names`` are not record types, sorted. Callers phrase the error
    in their own taxonomy — core raises for a caller bug, the CLI for a typo."""
    return sorted(set(names) - set(RECORD_TYPES))


@dataclass(frozen=True)
class VectorRecord:
    id: str  # UUID string; stored on the relational row as embedding_id
    vector: list[float]
    payload: dict[str, Any]


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str = COLLECTION_NAME) -> None:
        self.client = client
        self.collection = collection

    def close(self) -> None:
        """Release the store. Embedded on-disk Qdrant holds a single-process lock
        on its directory, so a command that does not close it locks out the next
        one; against a server this just drops the connection."""
        self.client.close()

    def locked_model(self) -> str | None:
        """The embedding model this collection is locked to, or ``None`` when the
        collection does not exist yet (nothing has ever been embedded).

        Raises if the collection exists but records no model: a collection
        created before model locking cannot have its lock verified, and treating
        its vectors as the configured model's is exactly the silent mixing ADR
        0001 forbids. Both readers and writers need the lock, so both go through
        here."""
        if not self.client.collection_exists(self.collection):
            return None
        metadata = self.client.get_collection(self.collection).config.metadata or {}
        locked_model = metadata.get(EMBEDDING_MODEL_KEY)
        if locked_model is None:
            raise ProviderConfigError(
                f"vector collection {self.collection!r} predates model locking and "
                f"records no embedding model; drop the collection and reprocess "
                f"embeddings to adopt model-locked storage (ADR 0001)"
            )
        return str(locked_model)

    def _ensure_collection(self, vector_size: int, embedding_model: str) -> None:
        """Create the collection for this model, or verify the model lock."""
        locked_model = self.locked_model()
        if locked_model is None:
            self.client.create_collection(
                self.collection,
                vectors_config=qmodels.VectorParams(
                    size=vector_size, distance=qmodels.Distance.COSINE
                ),
                metadata={EMBEDDING_MODEL_KEY: embedding_model},
            )
            return
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

    def search(
        self,
        vector: list[float],
        *,
        query_filter: qmodels.Filter | None = None,
        limit: int = 10,
        offset: int = 0,
        score_threshold: float | None = None,
    ) -> list[qmodels.ScoredPoint]:
        """Nearest points to ``vector``, best first (COSINE: higher is closer).

        Takes an already-embedded vector, keeping the store free of any embedder
        dependency — verifying the model lock and embedding the query belong to
        the caller (``booksmart_core.search``)."""
        response = self.client.query_points(
            self.collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            offset=offset,
            score_threshold=score_threshold,
        )
        return response.points


def build_vector_store(settings: Settings) -> VectorStore:
    """Connect to Qdrant the way ``settings`` asks: embedded on-disk when
    ``qdrant_path`` is set (the CLI's no-service default, where the on-disk
    format is pinned to the qdrant-client version), else the server at
    ``qdrant_url`` (booksmart-api's shape)."""
    if settings.qdrant_path is not None:
        return VectorStore(QdrantClient(path=str(settings.qdrant_path)))
    return VectorStore(QdrantClient(url=settings.qdrant_url))
