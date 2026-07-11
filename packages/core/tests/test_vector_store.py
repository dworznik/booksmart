"""Unit tests for the model-locked vector collection (ADR 0001).

The collection records the embedding model it was created for and rejects
writes from any other model, even at matching dimensions — same-dimension
mixing silently corrupts search.
"""

import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from booksmart_core.llm import ProviderConfigError
from booksmart_core.vectors import VectorRecord, VectorStore


def make_records(count: int = 2, size: int = 3) -> list[VectorRecord]:
    return [
        VectorRecord(
            id=str(uuid.uuid4()),
            vector=[1.0] * size,
            payload={"book_id": "book-1"},
        )
        for _ in range(count)
    ]


@pytest.fixture()
def store() -> VectorStore:
    return VectorStore(QdrantClient(":memory:"))


class TestModelLockedCollection:
    def test_first_write_records_the_embedding_model(self, store: VectorStore) -> None:
        store.replace_book_points("book-1", make_records(), embedding_model="embed-a")

        info = store.client.get_collection(store.collection)
        assert info.config.metadata == {"embedding_model": "embed-a"}

    def test_same_model_writes_are_accepted(self, store: VectorStore) -> None:
        store.replace_book_points("book-1", make_records(), embedding_model="embed-a")
        store.replace_book_points("book-1", make_records(), embedding_model="embed-a")

    def test_mismatched_model_write_fails_with_actionable_message(
        self, store: VectorStore
    ) -> None:
        store.replace_book_points("book-1", make_records(), embedding_model="embed-a")

        with pytest.raises(ProviderConfigError) as excinfo:
            store.replace_book_points("book-1", make_records(), embedding_model="embed-b")

        message = str(excinfo.value)
        assert "embed-a" in message
        assert "embed-b" in message
        assert "drop" in message.lower()
        assert "reprocess" in message.lower()

    def test_mismatch_rejected_even_at_matching_dimensions(self, store: VectorStore) -> None:
        # The dangerous case from ADR 0001: a dimension check alone would pass.
        store.replace_book_points("book-1", make_records(size=3), embedding_model="embed-a")

        with pytest.raises(ProviderConfigError):
            store.replace_book_points("book-1", make_records(size=3), embedding_model="embed-b")

    def test_legacy_collection_without_metadata_is_rejected(self, store: VectorStore) -> None:
        # A pre-lock collection records no model, so the lock cannot be
        # verified; stamping the configured model onto it would be the silent
        # mixing ADR 0001 forbids. The operator must migrate explicitly.
        store.client.create_collection(
            store.collection,
            vectors_config=qmodels.VectorParams(size=3, distance=qmodels.Distance.COSINE),
        )

        with pytest.raises(ProviderConfigError) as excinfo:
            store.replace_book_points("book-1", make_records(), embedding_model="embed-a")

        message = str(excinfo.value)
        assert "predates model locking" in message
        assert "drop" in message.lower()
        assert "reprocess" in message.lower()

    def test_legacy_unnamed_vector_schema_is_rejected(self, store: VectorStore) -> None:
        # A collection created before vectors were named cannot be written or
        # queried under the named schema; like a model switch, adopting it is
        # an explicit migration (ADR 0001), not a silent read.
        store.client.create_collection(
            store.collection,
            vectors_config=qmodels.VectorParams(size=3, distance=qmodels.Distance.COSINE),
            metadata={"embedding_model": "embed-a"},
        )

        with pytest.raises(ProviderConfigError) as excinfo:
            store.replace_book_points("book-1", make_records(), embedding_model="embed-a")

        message = str(excinfo.value)
        assert "predates named vectors" in message
        assert "drop" in message.lower()
        assert "reprocess" in message.lower()

        # Readers verify the lock through the same gate, so search is refused too.
        with pytest.raises(ProviderConfigError):
            store.locked_model()

    def test_points_live_under_the_named_dense_vector(self, store: VectorStore) -> None:
        # The named schema is the collection's contract (issue #37): sparse
        # vectors can later be added beside "dense" without a schema pivot.
        records = make_records(count=1)
        store.replace_book_points("book-1", records, embedding_model="embed-a")

        vectors_config = store.client.get_collection(store.collection).config.params.vectors
        assert isinstance(vectors_config, dict)
        assert set(vectors_config) == {"dense"}

        hits = store.search([1.0, 1.0, 1.0], limit=1)
        assert [hit.id for hit in hits] == [records[0].id]

    def test_empty_replace_needs_no_collection(self, store: VectorStore) -> None:
        store.replace_book_points("book-1", [], embedding_model="embed-a")

        assert not store.client.collection_exists(store.collection)

    def test_locked_model_reports_no_lock_before_anything_is_embedded(
        self, store: VectorStore
    ) -> None:
        assert store.locked_model() is None

        store.replace_book_points("book-1", make_records(), embedding_model="embed-a")

        assert store.locked_model() == "embed-a"


def test_close_releases_the_store(store: VectorStore) -> None:
    # Embedded on-disk Qdrant locks its directory until closed; readers of a
    # closed store are the caller's problem, but closing must not raise.
    store.replace_book_points("book-1", make_records(), embedding_model="embed-a")

    store.close()
