"""Unit tests for the model-locked vector collection (ADR 0001).

The collection records the embedding model it was created for and rejects
writes from any other model, even at matching dimensions — same-dimension
mixing silently corrupts search.
"""

import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from app.llm import ProviderConfigError
from app.vectors import VectorRecord, VectorStore


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

    def test_empty_replace_needs_no_collection(self, store: VectorStore) -> None:
        store.replace_book_points("book-1", [], embedding_model="embed-a")

        assert not store.client.collection_exists(store.collection)
