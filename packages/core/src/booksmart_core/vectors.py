"""Qdrant vector storage.

One collection holds every embedded record; payloads link each point back to
its relational row (record_type + record_id) and its book. Postgres and the
filesystem remain the source of truth - Qdrant only ever stores summaries and
extracted content, never the sole copy of raw book text.

Every point carries two vectors: a dense one from the embedding provider and a
sparse (BM25) one computed client-side, the two halves hybrid retrieval fuses.
They are always written together — Qdrant treats an upsert as the full set of
named vectors for that point, so writing one alone *deletes* the other, and a
dense-only rewrite would quietly strip the corpus of its lexical half one
reprocess at a time. ``VectorRecord`` therefore cannot be built with only one.

The collection is locked to both models (ADR 0001): collection metadata records
the dense embedding model and the sparse *recipe* it was created for, and writes
from any other are rejected even at matching dimensions. Vectors from different
dense models are incomparable, and same-dimension mixing degrades search in a
way no operator can diagnose from symptoms; BM25 parameter drift does the same
to recall. Switching either is an explicit migration: drop the collection and
reprocess embeddings for every book.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, get_args

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from booksmart_core.config import Settings
from booksmart_core.errors import ProviderConfigError
from booksmart_core.sparse import SparseVector

COLLECTION_NAME = "booksmart"

EMBEDDING_MODEL_KEY = "embedding_model"
# The sparse side records a full recipe (model plus BM25 parameters), not a bare
# model name — see booksmart_core.sparse. The key stays short and legible.
SPARSE_MODEL_KEY = "sparse_model"

# The collection's one dense vector, stored under an explicit name so that
# adding further named vectors (e.g. a sparse one for hybrid retrieval) is
# additive rather than a schema pivot.
DENSE_VECTOR_NAME = "dense"

# The sparse companion. Named for what it holds rather than for BM25, so
# swapping the sparse model (miniCOIL, SPLADE) is a recipe change and a
# reprocess rather than another schema pivot.
SPARSE_VECTOR_NAME = "text-sparse"

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
    """One point: both of its vectors, and the payload linking it to its row.

    ``sparse`` has no default on purpose. A caller that could omit it would emit
    a point Qdrant reads as "this record has no sparse vector", silently deleting
    the one already stored (see the module docstring); making it required turns
    that invariant into something the type checker enforces rather than something
    every future write path has to remember.
    """

    id: str  # UUID string; stored on the relational row as embedding_id
    vector: list[float]
    sparse: SparseVector
    payload: dict[str, Any]


@dataclass(frozen=True)
class CollectionLock:
    """What a collection is locked to — both halves of the retrieval recipe.

    Readers and writers both need the pair, and both must check it, so it comes
    back as one value from one gate (``verified_lock``) rather than as two
    lookups a caller could do only half of.
    """

    embedding_model: str
    sparse_recipe: str


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str = COLLECTION_NAME) -> None:
        self.client = client
        self.collection = collection

    def close(self) -> None:
        """Release the store. Embedded on-disk Qdrant holds a single-process lock
        on its directory, so a command that does not close it locks out the next
        one; against a server this just drops the connection."""
        self.client.close()

    def verified_lock(self) -> CollectionLock | None:
        """Verify the collection matches this code's storage contract, and report
        the models it is locked to — ``None`` when the collection does not exist
        yet (nothing has ever been embedded).

        The contract has a part per vector: the collection records the dense
        model and the sparse recipe it was created for (ADR 0001), and it stores
        each under the name this code reads and writes (``dense``,
        ``text-sparse``). A collection that breaks any part is refused, not
        adopted — its vectors cannot be verified or even addressed as this code
        expects, and reading them anyway is exactly the silent mixing ADR 0001
        forbids. Both readers and writers need the models, so both come through
        here and get the check for free."""
        if not self.client.collection_exists(self.collection):
            return None
        config = self.client.get_collection(self.collection).config
        metadata = config.metadata or {}

        locked_model = metadata.get(EMBEDDING_MODEL_KEY)
        if locked_model is None:
            raise ProviderConfigError(
                f"vector collection {self.collection!r} predates model locking and "
                f"records no embedding model; drop the collection and reprocess "
                f"embeddings to adopt model-locked storage (ADR 0001)"
            )
        vectors = config.params.vectors
        if not isinstance(vectors, dict):
            raise ProviderConfigError(
                f"vector collection {self.collection!r} predates named vectors and "
                f"stores its embeddings under the old unnamed schema; drop the "
                f"collection and reprocess embeddings to adopt named-vector "
                f"storage (ADR 0001)"
            )
        if DENSE_VECTOR_NAME not in vectors:
            # Named, but not with the vector this code reads and writes; without
            # this the miss surfaces as an opaque "not existing vector name"
            # error from the client, deep inside a write or a query.
            defined = ", ".join(repr(name) for name in sorted(vectors)) or "none"
            raise ProviderConfigError(
                f"vector collection {self.collection!r} defines no {DENSE_VECTOR_NAME!r} "
                f"vector (it defines {defined}); drop the collection and reprocess "
                f"embeddings to adopt named-vector storage (ADR 0001)"
            )

        locked_recipe = metadata.get(SPARSE_MODEL_KEY)
        if locked_recipe is None:
            raise ProviderConfigError(
                f"vector collection {self.collection!r} predates hybrid retrieval and "
                f"records no sparse model; its points carry no sparse vector, so "
                f"adopting it would leave the corpus lexically invisible. Drop the "
                f"collection and reprocess embeddings (ADR 0001)"
            )
        sparse_vectors = config.params.sparse_vectors or {}
        if SPARSE_VECTOR_NAME not in sparse_vectors:
            defined = ", ".join(repr(name) for name in sorted(sparse_vectors)) or "none"
            raise ProviderConfigError(
                f"vector collection {self.collection!r} defines no {SPARSE_VECTOR_NAME!r} "
                f"sparse vector (it defines {defined}); drop the collection and "
                f"reprocess embeddings to adopt hybrid storage (ADR 0001)"
            )
        return CollectionLock(
            embedding_model=str(locked_model), sparse_recipe=str(locked_recipe)
        )

    def _ensure_collection(
        self, vector_size: int, embedding_model: str, sparse_recipe: str
    ) -> None:
        """Create the collection for these models, or verify both locks."""
        lock = self.verified_lock()
        if lock is None:
            self.client.create_collection(
                self.collection,
                vectors_config={
                    DENSE_VECTOR_NAME: qmodels.VectorParams(
                        size=vector_size, distance=qmodels.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    # BM25 is split across the two sides: we weight the document
                    # terms, Qdrant supplies the IDF at query time from the
                    # collection's own statistics. Without this modifier every
                    # term counts alike and BM25 stops being BM25.
                    SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(
                        modifier=qmodels.Modifier.IDF
                    )
                },
                metadata={
                    EMBEDDING_MODEL_KEY: embedding_model,
                    SPARSE_MODEL_KEY: sparse_recipe,
                },
            )
            return
        if lock.embedding_model != embedding_model:
            raise ProviderConfigError(
                f"vector collection {self.collection!r} is locked to embedding model "
                f"{lock.embedding_model!r} but the configured embedder is "
                f"{embedding_model!r}; switching models requires dropping the "
                f"collection and reprocessing embeddings for every book (ADR 0001)"
            )
        if lock.sparse_recipe != sparse_recipe:
            # The recipe, not the model name: a parameter apart is still a
            # different weighting of every term in the corpus, and the resulting
            # recall loss has no symptom that points back here.
            raise ProviderConfigError(
                f"vector collection {self.collection!r} is locked to sparse model "
                f"{lock.sparse_recipe!r} but the configured sparse embedder is "
                f"{sparse_recipe!r}; switching models or their parameters requires "
                f"dropping the collection and reprocessing embeddings for every book "
                f"(ADR 0001)"
            )

    def replace_book_points(
        self,
        book_id: str,
        records: list[VectorRecord],
        embedding_model: str,
        sparse_recipe: str,
    ) -> None:
        """Upsert this run's points, then drop the book's stale points from
        earlier runs. Upsert-first so a mid-replace failure leaves the previous
        vectors in place rather than none.

        Both named vectors go in the same PointStruct. Qdrant reads an upsert as
        the complete set of named vectors for that point, so splitting them into
        two writes would leave a window where one is missing — and a rewrite that
        named only one would delete the other outright."""
        book_filter = qmodels.FieldCondition(
            key="book_id", match=qmodels.MatchValue(value=book_id)
        )
        if records:
            self._ensure_collection(len(records[0].vector), embedding_model, sparse_recipe)
            self.client.upsert(
                self.collection,
                points=[
                    qmodels.PointStruct(
                        id=record.id,
                        vector={
                            DENSE_VECTOR_NAME: record.vector,
                            SPARSE_VECTOR_NAME: _sparse(record.sparse),
                        },
                        payload=record.payload,
                    )
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

        Dense-only, and scored as it always was; hybrid retrieval is
        ``hybrid_search``. Takes an already-embedded vector, keeping the store
        free of any embedder dependency — verifying the model lock and embedding
        the query belong to the caller (``booksmart_core.search``)."""
        response = self.client.query_points(
            self.collection,
            query=vector,
            using=DENSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=limit,
            offset=offset,
            score_threshold=score_threshold,
        )
        return response.points

    def hybrid_search(
        self,
        vector: list[float],
        sparse: SparseVector,
        *,
        query_filter: qmodels.Filter | None = None,
        limit: int = 10,
        offset: int = 0,
        score_threshold: float | None = None,
    ) -> list[qmodels.ScoredPoint]:
        """Dense and sparse retrieval in one query, fused by Reciprocal Rank
        Fusion, best first.

        Scores are **not** similarities. RRF sums 1/(rank + k) across the
        branches a point appeared in, so the number describes agreement between
        two rankings and is comparable only within one result set. Callers that
        display it have to say so; ``score_threshold`` therefore cannot apply to
        the fused score, and bounds the dense branch instead — where it is still
        a cosine floor. A record the floor excludes can still rank on its lexical
        match alone, which is the point of hybrid retrieval.

        Two constraints here are not obvious and are load-bearing, both verified
        against embedded Qdrant:

        The filter is duplicated onto each prefetch branch rather than set once
        at the root. Embedded Qdrant *silently ignores* a root-level filter on a
        fusion query — it returns other books' records with no error — while the
        server honours it. Per-branch filters behave identically on both, so
        that is the only spelling that is correct everywhere.

        Each branch fetches ``limit + offset`` candidates. A branch that fetched
        only ``limit`` would run out before fusion reached the offset, and later
        pages would come back short for no visible reason.
        """
        candidates = limit + offset
        response = self.client.query_points(
            self.collection,
            prefetch=[
                qmodels.Prefetch(
                    query=vector,
                    using=DENSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=candidates,
                    score_threshold=score_threshold,
                ),
                qmodels.Prefetch(
                    query=_sparse(sparse),
                    using=SPARSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=candidates,
                ),
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=limit,
            offset=offset,
        )
        return response.points


def _sparse(vector: SparseVector) -> qmodels.SparseVector:
    """Core's sparse vector as the client's. The seam in ``sparse.py`` is
    deliberately store-agnostic, so the conversion lives here, at the boundary."""
    return qmodels.SparseVector(indices=vector.indices, values=vector.values)


def build_vector_store(settings: Settings) -> VectorStore:
    """Connect to Qdrant the way ``settings`` asks: embedded on-disk when
    ``qdrant_path`` is set (the CLI's no-service default, where the on-disk
    format is pinned to the qdrant-client version), else the server at
    ``qdrant_url`` (a server consumer's shape)."""
    if settings.qdrant_path is not None:
        return VectorStore(QdrantClient(path=str(settings.qdrant_path)))
    return VectorStore(QdrantClient(url=settings.qdrant_url))
