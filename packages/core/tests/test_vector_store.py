"""Unit tests for the model-locked vector collection (ADRs 0001 and 0003).

The collection records what it was created for — a dense embedding model and a
sparse recipe — and rejects writes from any other, even at matching dimensions:
same-dimension mixing silently corrupts search, and BM25 parameter drift
silently degrades recall. Both are the diagnose-from-symptoms failure the two
ADRs exist to prevent.
"""

import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from booksmart_core.llm import ProviderConfigError
from booksmart_core.sparse import SparseVector
from booksmart_core.vectors import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    CollectionLock,
    VectorRecord,
    VectorStore,
)

RECIPE = "sparse-a(k=1.2,b=0.75)"
OTHER_RECIPE = "sparse-a(k=1.5,b=0.75)"


def make_records(count: int = 2, size: int = 3) -> list[VectorRecord]:
    return [
        VectorRecord(
            id=str(uuid.uuid4()),
            vector=[1.0] * size,
            sparse=SparseVector(indices=[1, 2], values=[0.5, 0.5]),
            payload={"book_id": "book-1"},
        )
        for _ in range(count)
    ]


def write(store: VectorStore, records: list[VectorRecord], *, model: str = "embed-a") -> None:
    store.replace_book_points("book-1", records, model, RECIPE)


@pytest.fixture()
def store() -> VectorStore:
    return VectorStore(QdrantClient(":memory:"))


class TestModelLockedCollection:
    def test_first_write_records_both_models(self, store: VectorStore) -> None:
        write(store, make_records())

        info = store.client.get_collection(store.collection)
        assert info.config.metadata == {
            "embedding_model": "embed-a",
            "sparse_recipe": RECIPE,
        }

    def test_same_models_writes_are_accepted(self, store: VectorStore) -> None:
        write(store, make_records())
        write(store, make_records())

    def test_mismatched_model_write_fails_with_actionable_message(
        self, store: VectorStore
    ) -> None:
        write(store, make_records())

        with pytest.raises(ProviderConfigError) as excinfo:
            write(store, make_records(), model="embed-b")

        message = str(excinfo.value)
        assert "embed-a" in message
        assert "embed-b" in message
        assert "drop" in message.lower()
        assert "reprocess" in message.lower()

    def test_mismatch_rejected_even_at_matching_dimensions(self, store: VectorStore) -> None:
        # The dangerous case from ADR 0001: a dimension check alone would pass.
        write(store, make_records(size=3))

        with pytest.raises(ProviderConfigError):
            write(store, make_records(size=3), model="embed-b")

    def test_legacy_collection_without_metadata_is_rejected(self, store: VectorStore) -> None:
        # A pre-lock collection records no model, so the lock cannot be
        # verified; stamping the configured model onto it would be the silent
        # mixing ADR 0001 forbids. The operator must migrate explicitly.
        store.client.create_collection(
            store.collection,
            vectors_config=qmodels.VectorParams(size=3, distance=qmodels.Distance.COSINE),
        )

        with pytest.raises(ProviderConfigError) as excinfo:
            write(store, make_records())

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
            write(store, make_records())

        message = str(excinfo.value)
        assert "predates named vectors" in message
        assert "drop" in message.lower()
        assert "reprocess" in message.lower()

        # Readers verify the lock through the same gate, so search is refused too.
        with pytest.raises(ProviderConfigError):
            store.verified_lock()

    def test_named_schema_without_a_dense_vector_is_rejected(self, store: VectorStore) -> None:
        # A collection whose named vectors do not include "dense" cannot serve
        # this code either. Without the gate, the miss surfaces as a raw
        # "Not existing vector name" ValueError from the client, with none of
        # the migration guidance ADR 0001 promises.
        store.client.create_collection(
            store.collection,
            vectors_config={
                "embedding": qmodels.VectorParams(size=3, distance=qmodels.Distance.COSINE)
            },
            metadata={"embedding_model": "embed-a"},
        )

        with pytest.raises(ProviderConfigError) as excinfo:
            write(store, make_records())

        message = str(excinfo.value)
        assert "dense" in message
        assert "embedding" in message  # names what the collection does define
        assert "drop" in message.lower()
        assert "reprocess" in message.lower()

    def test_empty_replace_needs_no_collection(self, store: VectorStore) -> None:
        store.replace_book_points("book-1", [], "embed-a", RECIPE)

        assert not store.client.collection_exists(store.collection)

    def test_verified_lock_reports_no_lock_before_anything_is_embedded(
        self, store: VectorStore
    ) -> None:
        assert store.verified_lock() is None

        write(store, make_records())

        assert store.verified_lock() == CollectionLock(
            embedding_model="embed-a", sparse_recipe=RECIPE
        )


class TestSparseModelLock:
    """The sparse half of the lock (issue #38).

    BM25 parameters shape every term weight, so the collection locks against the
    whole recipe rather than the model name — drift in k, b, avg_len or language
    degrades recall with no symptom an operator could trace back to it.
    """

    def test_mismatched_recipe_write_fails_with_actionable_message(
        self, store: VectorStore
    ) -> None:
        write(store, make_records())

        with pytest.raises(ProviderConfigError) as excinfo:
            store.replace_book_points("book-1", make_records(), "embed-a", OTHER_RECIPE)

        message = str(excinfo.value)
        assert RECIPE in message
        assert OTHER_RECIPE in message
        assert "drop" in message.lower()
        assert "reprocess" in message.lower()

    def test_parameter_drift_alone_is_enough_to_refuse(self, store: VectorStore) -> None:
        # Same model name, one parameter apart: the case a name-only lock misses.
        write(store, make_records())

        with pytest.raises(ProviderConfigError):
            store.replace_book_points("book-1", make_records(), "embed-a", OTHER_RECIPE)

    def test_collection_predating_hybrid_is_rejected(self, store: VectorStore) -> None:
        # Dense-only collection from before #38: it records no sparse recipe, so
        # the sparse lock cannot be verified and its points carry no sparse
        # vector. Adopting it would leave half the corpus lexically invisible.
        store.client.create_collection(
            store.collection,
            vectors_config={
                DENSE_VECTOR_NAME: qmodels.VectorParams(size=3, distance=qmodels.Distance.COSINE)
            },
            metadata={"embedding_model": "embed-a"},
        )

        with pytest.raises(ProviderConfigError) as excinfo:
            write(store, make_records())

        message = str(excinfo.value)
        assert "predates hybrid retrieval" in message
        assert "drop" in message.lower()
        assert "reprocess" in message.lower()

        # Readers are refused through the same gate.
        with pytest.raises(ProviderConfigError):
            store.verified_lock()

    def test_collection_without_the_sparse_vector_is_rejected(self, store: VectorStore) -> None:
        # Locked to a sparse recipe but with nowhere to put the vectors: the
        # miss would otherwise surface as an opaque client error mid-upsert.
        store.client.create_collection(
            store.collection,
            vectors_config={
                DENSE_VECTOR_NAME: qmodels.VectorParams(size=3, distance=qmodels.Distance.COSINE)
            },
            metadata={"embedding_model": "embed-a", "sparse_recipe": RECIPE},
        )

        with pytest.raises(ProviderConfigError) as excinfo:
            write(store, make_records())

        message = str(excinfo.value)
        assert SPARSE_VECTOR_NAME in message
        assert "drop" in message.lower()
        assert "reprocess" in message.lower()

    def test_new_collections_ask_qdrant_for_idf(self, store: VectorStore) -> None:
        # BM25 splits across the two sides: we weight the document terms, Qdrant
        # supplies the IDF at query time from collection statistics. Without the
        # modifier every term is weighted alike and BM25 stops being BM25.
        write(store, make_records())

        sparse_config = store.client.get_collection(store.collection).config.params.sparse_vectors
        assert sparse_config is not None
        assert sparse_config[SPARSE_VECTOR_NAME].modifier == qmodels.Modifier.IDF


class TestBothVectorsAlways:
    """Upserting a point with a missing named vector *deletes* that vector.

    Verified against embedded Qdrant, and the reason a VectorRecord cannot be
    constructed without both halves: any write path that could emit a
    dense-only point would silently strip the sparse half off every record it
    touched, and the corpus would degrade to dense-only one reprocess at a time.
    """

    def test_every_written_point_carries_both_vectors(self, store: VectorStore) -> None:
        records = make_records(count=3)
        write(store, records)

        points = store.client.retrieve(
            store.collection, ids=[record.id for record in records], with_vectors=True
        )
        assert len(points) == 3
        for point in points:
            assert isinstance(point.vector, dict)
            assert set(point.vector) == {DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME}

    def test_rewriting_a_book_keeps_both_vectors(self, store: VectorStore) -> None:
        # The trap in its natural habitat: a reprocess re-upserts existing point
        # ids, so a dense-only rewrite here is what would strip the sparse half.
        records = make_records(count=2)
        write(store, records)
        write(store, records)

        points = store.client.retrieve(
            store.collection, ids=[record.id for record in records], with_vectors=True
        )
        for point in points:
            assert isinstance(point.vector, dict)
            assert set(point.vector) == {DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME}

    def test_a_record_cannot_be_built_without_a_sparse_vector(self) -> None:
        with pytest.raises(TypeError):
            VectorRecord(  # type: ignore[call-arg]
                id=str(uuid.uuid4()), vector=[1.0], payload={}
            )

    def test_stale_points_are_still_dropped(self, store: VectorStore) -> None:
        write(store, make_records(count=3))
        kept = make_records(count=1)
        write(store, kept)

        remaining, _ = store.client.scroll(store.collection, limit=10)
        assert [str(point.id) for point in remaining] == [kept[0].id]


class TestDenseSearch:
    def test_points_are_queryable_under_the_named_dense_vector(
        self, store: VectorStore
    ) -> None:
        records = make_records(count=1)
        write(store, records)

        vectors_config = store.client.get_collection(store.collection).config.params.vectors
        assert isinstance(vectors_config, dict)
        assert set(vectors_config) == {DENSE_VECTOR_NAME}

        hits = store.search([1.0, 1.0, 1.0], limit=1)
        assert [hit.id for hit in hits] == [records[0].id]


def test_close_releases_the_store(store: VectorStore) -> None:
    # Embedded on-disk Qdrant locks its directory until closed; readers of a
    # closed store are the caller's problem, but closing must not raise.
    write(store, make_records())

    store.close()
