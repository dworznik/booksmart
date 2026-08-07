"""Unit tests for semantic search — the read side of the embedding pipeline.

Vectors are seeded by hand (rather than by running the embedding stage) so the
geometry is exact: the query embeds to [1, 0], and each seeded point sits at a
known angle from it, making the expected COSINE ranking arithmetic rather than
a property of some provider's model.
"""

import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from sqlalchemy.orm import Session, sessionmaker

from booksmart_core.errors import ProviderConfigError
from booksmart_core.fakes import FakeSparseEmbeddingProvider
from booksmart_core.llm import EmbeddingResponse
from booksmart_core.models import Book, Chapter, KnowledgeObject, Section
from booksmart_core.search import SearchHit, search
from booksmart_core.vectors import RecordType, VectorRecord, VectorStore

from .conftest import store_book

SPARSE_MODEL = "stub-sparse-1"

# The query vector, and points at 0°, 45° and 90° from it.
QUERY_VECTOR = [1.0, 0.0]
EXACT = [1.0, 0.0]
DIAGONAL = [1.0, 1.0]
ORTHOGONAL = [0.0, 1.0]


class QueryEmbedder:
    """Embeds any query to the same fixed vector; records what it was asked."""

    model = "stub-embed-1"
    max_batch = 100

    # Fixed per-call usage so tests can assert an exact number.
    INPUT_TOKENS_PER_CALL = 9

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or QUERY_VECTOR
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> EmbeddingResponse:
        self.calls.append(list(texts))
        return EmbeddingResponse(
            vectors=[list(self.vector) for _ in texts],
            input_tokens=self.INPUT_TOKENS_PER_CALL,
        )


class SilentQueryEmbedder:
    """Same vectors as QueryEmbedder, but reports no usage at all."""

    model = "stub-embed-1"
    max_batch = 100

    def embed(self, texts: list[str]) -> EmbeddingResponse:
        return EmbeddingResponse(vectors=[list(QUERY_VECTOR) for _ in texts])


@pytest.fixture()
def embedder() -> QueryEmbedder:
    return QueryEmbedder()


@pytest.fixture()
def store() -> VectorStore:
    return VectorStore(QdrantClient(":memory:"))


@pytest.fixture()
def book_id(session_factory: sessionmaker[Session], storage: object) -> uuid.UUID:
    return uuid.UUID(
        store_book(
            session_factory,
            storage,  # type: ignore[arg-type]
            title="A Philosophy of Software Design",
            author="Ousterhout",
            filename="apsd.pdf",
            content=b"%PDF-1.4 fake",
        )
    )


def seed(
    session: Session,
    store: VectorStore,
    book_id: uuid.UUID,
    *,
    embedding_model: str = "stub-embed-1",
) -> dict[str, uuid.UUID]:
    """One chapter, one section and one knowledge object, each with a vector at a
    known angle from the query: chapter exact, section 45°, object orthogonal."""
    chapter = Chapter(book_id=book_id, position=0, title="Deep Modules", summary="Deep modules.")
    session.add(chapter)
    session.flush()
    section = Section(chapter_id=chapter.id, position=0, title="Interfaces", summary="Interfaces.")
    knowledge = KnowledgeObject(
        book_id=book_id,
        type="Principle",
        title="Depth",
        content="Deep modules have simple interfaces.",
        summary="Depth beats breadth.",
        source_location="ch1",
        confidence=1.0,
        extraction_model="stub-llm-1",
        extraction_prompt_version="1",
    )
    session.add_all([section, knowledge])
    session.commit()

    rows: list[tuple[RecordType, uuid.UUID, list[float], str]] = [
        ("chapter", chapter.id, EXACT, "Deep modules."),
        ("section", section.id, DIAGONAL, "Interfaces."),
        ("knowledge_object", knowledge.id, ORTHOGONAL, "Depth beats breadth."),
    ]
    # Sparse vectors come from the fake BM25 provider over the same text the
    # dense vector stands for, so the two halves describe one record — hybrid
    # tests need the lexical side to mean something, and dense-only tests are
    # unaffected by its presence.
    sparse_provider = FakeSparseEmbeddingProvider(model=SPARSE_MODEL)
    sparse_vectors = sparse_provider.embed_documents([text for *_, text in rows])
    store.replace_book_points(
        str(book_id),
        [
            VectorRecord(
                id=str(uuid.uuid4()),
                vector=vector,
                sparse=sparse_vector,
                payload={
                    "record_type": record_type,
                    "record_id": str(record_id),
                    "book_id": str(book_id),
                    "text": text,
                },
            )
            for (record_type, record_id, vector, text), sparse_vector in zip(
                rows, sparse_vectors, strict=True
            )
        ],
        embedding_model,
        sparse_provider.recipe,
    )
    return {"chapter": chapter.id, "section": section.id, "knowledge_object": knowledge.id}


class TestRanking:
    def test_hits_come_back_best_first_with_scores_and_payload(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        with session_factory() as session:
            ids = seed(session, store, book_id)
            hits = search(session, store, embedder, "deep modules", mode="dense").hits

        assert [hit.record_type for hit in hits] == ["chapter", "section", "knowledge_object"]
        assert [hit.record_id for hit in hits] == [
            ids["chapter"],
            ids["section"],
            ids["knowledge_object"],
        ]
        assert hits[0].score == pytest.approx(1.0)
        assert hits[1].score == pytest.approx(0.7071, abs=1e-3)
        assert hits[2].score == pytest.approx(0.0, abs=1e-6)
        assert hits[0].text == "Deep modules."
        assert hits[0].book_id == book_id
        assert embedder.calls == [["deep modules"]]

    def test_hits_carry_the_resolved_row(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        with session_factory() as session:
            seed(session, store, book_id)
            hits = search(session, store, embedder, "deep modules", mode="dense").hits

        chapter = hits[0].row
        assert isinstance(chapter, Chapter)
        assert chapter.title == "Deep Modules"
        assert isinstance(hits[1].row, Section)
        assert isinstance(hits[2].row, KnowledgeObject)

    def test_rows_are_detached_and_usable_after_the_session_closes(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        with session_factory() as session:
            seed(session, store, book_id)
            hits = search(session, store, embedder, "deep modules", mode="dense").hits
            # A later commit expires every row the session still holds; a hit that
            # was never detached would then try to refresh itself once the session
            # is gone, and raise DetachedInstanceError instead of rendering.
            session.commit()

        assert [hit.title for hit in hits] == ["Deep Modules", "Interfaces", "Depth"]

    def test_stale_points_whose_row_is_gone_are_skipped(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        with session_factory() as session:
            ids = seed(session, store, book_id)
            session.delete(session.get(Chapter, ids["chapter"]))
            session.commit()
            hits = search(session, store, embedder, "deep modules", mode="dense").hits

        # The chapter (and its cascaded section) are gone; only the object remains.
        assert [hit.record_type for hit in hits] == ["knowledge_object"]


class TestFilters:
    def test_record_types_narrows_the_search(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        with session_factory() as session:
            seed(session, store, book_id)
            hits = search(
                session,
                store,
                embedder,
                "deep modules",
                mode="dense",
                record_types=["section", "chapter"],
            ).hits

        assert [hit.record_type for hit in hits] == ["chapter", "section"]

    def test_book_id_narrows_the_search(
        self,
        session_factory: sessionmaker[Session],
        storage: object,
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        other_id = uuid.UUID(
            store_book(
                session_factory,
                storage,  # type: ignore[arg-type]
                title="Other",
                author="Someone",
                filename="other.pdf",
                content=b"%PDF-1.4 other",
            )
        )
        with session_factory() as session:
            seed(session, store, book_id)
            seed(session, store, other_id)
            hits = search(
                session, store, embedder, "deep modules", mode="dense", book_id=other_id
            ).hits

        assert len(hits) == 3
        assert {hit.book_id for hit in hits} == {other_id}

    def test_score_threshold_drops_weak_hits(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        with session_factory() as session:
            seed(session, store, book_id)
            hits = search(
                session, store, embedder, "deep modules", mode="dense", score_threshold=0.5
            ).hits

        assert [hit.record_type for hit in hits] == ["chapter", "section"]

    def test_limit_and_offset_page_through_the_ranking(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        with session_factory() as session:
            seed(session, store, book_id)
            first = search(
                session, store, embedder, "deep modules", mode="dense", limit=1
            ).hits
            second = search(
                session, store, embedder, "deep modules", mode="dense", limit=1, offset=1
            ).hits

        assert [hit.record_type for hit in first] == ["chapter"]
        assert [hit.record_type for hit in second] == ["section"]

    def test_unknown_record_type_is_rejected(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        with session_factory() as session:
            seed(session, store, book_id)
            bad_types: list[RecordType] = ["paragraph"]  # type: ignore[list-item]
            # A caller bug, not a misconfiguration: a plain ValueError, and never
            # a silent empty result from filtering on a type nothing can carry.
            with pytest.raises(ValueError) as excinfo:
                search(
                    session, store, embedder, "deep modules", mode="dense",
                    record_types=bad_types,
                )

        assert not isinstance(excinfo.value, ProviderConfigError)
        assert "paragraph" in str(excinfo.value)


class TestModelLock:
    def test_mismatched_model_is_rejected_before_embedding(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        with session_factory() as session:
            seed(session, store, book_id, embedding_model="embed-a")
            with pytest.raises(ProviderConfigError) as excinfo:
                search(session, store, embedder, "deep modules", mode="dense")

        message = str(excinfo.value)
        assert "embed-a" in message
        assert "stub-embed-1" in message
        # Reading the lock first means no query is ever embedded with the wrong model.
        assert embedder.calls == []

    def test_legacy_collection_without_a_recorded_model_is_rejected(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
    ) -> None:
        store.client.create_collection(
            store.collection,
            vectors_config=qmodels.VectorParams(size=2, distance=qmodels.Distance.COSINE),
        )
        with session_factory() as session:
            with pytest.raises(ProviderConfigError) as excinfo:
                search(session, store, embedder, "deep modules", mode="dense")

        assert "predates model locking" in str(excinfo.value)


class TestEmptyStore:
    def test_missing_collection_returns_no_hits(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
    ) -> None:
        with session_factory() as session:
            hits = search(session, store, embedder, "deep modules", mode="dense").hits

        assert hits == []
        # Nothing to search: the query is never embedded.
        assert embedder.calls == []

    def test_book_with_no_embeddings_returns_no_hits(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        other = uuid.uuid4()
        with session_factory() as session:
            seed(session, store, book_id)
            results = search(
                session, store, embedder, "deep modules", mode="dense", book_id=other
            )

        assert results.hits == []
        # No hits, but the query was still embedded and still cost something —
        # the empty result here must not be confused with the empty result of a
        # search that never embedded anything (see TestQueryEmbeddingUsage).
        assert results.embedding_tokens == QueryEmbedder.INPUT_TOKENS_PER_CALL


class TestQueryEmbeddingUsage:
    """A search costs exactly one embedding call, and says what it cost."""

    def test_results_report_the_usage_the_embedder_reported(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        with session_factory() as session:
            seed(session, store, book_id)
            results = search(session, store, embedder, "deep modules", mode="dense")

        assert results.hits
        assert results.embedding_tokens == QueryEmbedder.INPUT_TOKENS_PER_CALL

    def test_usage_the_provider_withholds_stays_unknown(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        book_id: uuid.UUID,
    ) -> None:
        with session_factory() as session:
            seed(session, store, book_id)
            results = search(
                session, store, SilentQueryEmbedder(), "deep modules", mode="dense"
            )

        assert results.hits
        assert results.embedding_tokens is None

    def test_a_search_that_never_embeds_spent_nothing(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
    ) -> None:
        # No collection: search returns early without embedding. Zero is the
        # honest answer — nothing was billed — and it is not the None that
        # means "we called and the provider would not say".
        with session_factory() as session:
            results = search(session, store, embedder, "deep modules", mode="dense")

        assert results.hits == []
        assert results.embedding_tokens == 0
        assert embedder.calls == []


def test_search_hit_is_immutable(
    session_factory: sessionmaker[Session],
    store: VectorStore,
    embedder: QueryEmbedder,
    book_id: uuid.UUID,
) -> None:
    with session_factory() as session:
        seed(session, store, book_id)
        hit = search(session, store, embedder, "deep modules", mode="dense").hits[0]

    assert isinstance(hit, SearchHit)
    with pytest.raises(AttributeError):
        hit.score = 0.0  # type: ignore[misc]


def sparse_query_embedder() -> FakeSparseEmbeddingProvider:
    """The query side of the same fake BM25 `seed` weighted the documents with,
    so its recipe matches the lock the collection was created under."""
    return FakeSparseEmbeddingProvider(model=SPARSE_MODEL)


class TestHybridRanking:
    """Issue #39: dense and sparse retrieval fused with Reciprocal Rank Fusion.

    The dense side is fixed by construction — `QueryEmbedder` returns the same
    vector for every query, so dense ranking is always chapter, section, object.
    That makes the sparse side the only thing that can move a result, which is
    exactly what these tests need to observe.
    """

    def test_a_lexical_match_outranks_a_closer_dense_neighbour(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        # The knowledge object is dense-orthogonal to the query — dead last on
        # similarity — but its text *is* the query. This is the case hybrid
        # exists for, and the demo issue #39 asks for.
        with session_factory() as session:
            seed(session, store, book_id)
            dense_only = search(
                session, store, embedder, "depth beats breadth", mode="dense"
            ).hits
            hybrid = search(
                session,
                store,
                embedder,
                "depth beats breadth",
                sparse_embedder=sparse_query_embedder(),
                mode="hybrid",
            ).hits

        assert [hit.record_type for hit in dense_only][-1] == "knowledge_object"
        assert hybrid[0].record_type == "knowledge_object"

    def test_dense_only_ranking_is_unchanged_by_this_slice(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        with session_factory() as session:
            seed(session, store, book_id)
            results = search(session, store, embedder, "deep modules", mode="dense")

        assert [hit.record_type for hit in results.hits] == [
            "chapter",
            "section",
            "knowledge_object",
        ]
        # Still cosine similarities, to the same numbers as before fusion existed.
        assert results.hits[0].score == pytest.approx(1.0)
        assert results.hits[1].score == pytest.approx(0.7071, abs=1e-3)
        assert results.mode == "dense"

    def test_hits_are_ranked_from_one(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        with session_factory() as session:
            seed(session, store, book_id)
            hits = search(
                session,
                store,
                embedder,
                "deep modules",
                sparse_embedder=sparse_query_embedder(),
            ).hits

        assert [hit.rank for hit in hits] == [1, 2, 3]

    def test_rank_counts_from_the_offset_not_from_the_page(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        # A rank that restarted at 1 on every page would be a position in the
        # page, which is not a thing anyone wants to know.
        with session_factory() as session:
            seed(session, store, book_id)
            hits = search(
                session,
                store,
                embedder,
                "deep modules",
                sparse_embedder=sparse_query_embedder(),
                limit=2,
                offset=1,
            ).hits

        assert [hit.rank for hit in hits] == [2, 3]

    def test_fused_scores_are_not_similarities(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        # The knowledge object is orthogonal to the query: cosine 0.0, the
        # weakest similarity there is. Under fusion it scores well above zero,
        # because a rank score says "this placed somewhere", not "this is close".
        # Same record, same query, two numbers that mean different things — which
        # is why the mode has to travel with the score to anything that renders it.
        with session_factory() as session:
            seed(session, store, book_id)
            dense = search(session, store, embedder, "deep modules", mode="dense")
            hybrid = search(
                session,
                store,
                embedder,
                "deep modules",
                sparse_embedder=sparse_query_embedder(),
            )

        assert dense.mode == "dense"
        assert hybrid.mode == "hybrid"
        weakest_dense = {hit.record_type: hit.score for hit in dense.hits}["knowledge_object"]
        weakest_hybrid = {hit.record_type: hit.score for hit in hybrid.hits}["knowledge_object"]
        assert weakest_dense == pytest.approx(0.0, abs=1e-6)
        assert weakest_hybrid > 0.1

    def test_hybrid_is_the_default_mode(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        with session_factory() as session:
            seed(session, store, book_id)
            results = search(
                session,
                store,
                embedder,
                "deep modules",
                sparse_embedder=sparse_query_embedder(),
            )

        assert results.mode == "hybrid"

    def test_hybrid_without_a_sparse_embedder_is_a_caller_bug(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        # Silently degrading to dense-only would be the worst of both: the
        # caller believes it is searching hybrid and gets fewer results.
        with session_factory() as session:
            seed(session, store, book_id)
            with pytest.raises(ValueError) as excinfo:
                search(session, store, embedder, "deep modules", mode="hybrid")

        assert "sparse_embedder" in str(excinfo.value)

    def test_a_query_with_no_indexable_terms_still_returns_dense_hits(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        # BM25 drops stopwords, so a query can embed to an empty sparse vector.
        # The sparse branch then contributes nothing and fusion falls back to
        # the dense ranking — it must not error, and must not return nothing.
        with session_factory() as session:
            seed(session, store, book_id)
            hits = search(
                session,
                store,
                embedder,
                "!!! ???",
                sparse_embedder=sparse_query_embedder(),
            ).hits

        assert [hit.record_type for hit in hits] == ["chapter", "section", "knowledge_object"]


class TestHybridFilters:
    def test_book_filter_is_applied_to_every_prefetch_branch(
        self,
        session_factory: sessionmaker[Session],
        storage: object,
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        # The regression this exists for: embedded Qdrant *silently ignores* a
        # root-level filter on a fusion query. A filter applied only at the root
        # returns other books' records with no error at all, so this test is the
        # only thing standing between that behaviour and production.
        other_id = uuid.UUID(
            store_book(
                session_factory,
                storage,  # type: ignore[arg-type]
                title="Other",
                author="Someone",
                filename="other.pdf",
                content=b"%PDF-1.4 other",
            )
        )
        with session_factory() as session:
            seed(session, store, book_id)
            seed(session, store, other_id)
            hits = search(
                session,
                store,
                embedder,
                "deep modules",
                sparse_embedder=sparse_query_embedder(),
                book_id=other_id,
            ).hits

        assert len(hits) == 3
        assert {hit.book_id for hit in hits} == {other_id}

    def test_record_type_filter_is_applied_to_every_prefetch_branch(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        with session_factory() as session:
            seed(session, store, book_id)
            hits = search(
                session,
                store,
                embedder,
                "depth beats breadth",
                sparse_embedder=sparse_query_embedder(),
                record_types=["chapter", "section"],
            ).hits

        # Without a per-branch filter the sparse branch would surface the
        # knowledge object, whose text matches the query exactly.
        assert {hit.record_type for hit in hits} == {"chapter", "section"}

    def test_score_threshold_bounds_the_dense_branch_only(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        # A cosine floor cannot apply to a fused score, so under hybrid it bounds
        # the dense branch. A record the floor excludes can still rank on the
        # strength of its lexical match — which is the point of hybrid, and why
        # the flag is not simply refused here.
        with session_factory() as session:
            seed(session, store, book_id)
            hits = search(
                session,
                store,
                embedder,
                "depth beats breadth",
                sparse_embedder=sparse_query_embedder(),
                score_threshold=0.5,
            ).hits

        # The orthogonal knowledge object (cosine 0.0) is below the floor, so the
        # dense branch drops it — but it is an exact lexical match, so it stays.
        assert "knowledge_object" in {hit.record_type for hit in hits}

    def test_pagination_returns_whole_pages(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        # Each prefetch branch must fetch at least limit+offset candidates, or a
        # later page comes back short: the branches would have run out before
        # fusion reached the offset.
        with session_factory() as session:
            seed(session, store, book_id)
            pages = [
                search(
                    session,
                    store,
                    embedder,
                    "deep modules",
                    sparse_embedder=sparse_query_embedder(),
                    limit=1,
                    offset=offset,
                ).hits
                for offset in range(3)
            ]

        assert [len(page) for page in pages] == [1, 1, 1]
        # And the three pages are the three distinct records, in ranked order.
        assert len({page[0].record_id for page in pages}) == 3
        assert [page[0].rank for page in pages] == [1, 2, 3]


class TestHybridModelLock:
    def test_a_drifted_sparse_recipe_is_refused_before_embedding(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        # Reading the corpus with a differently-parameterised BM25 is the read
        # side of the drift ADR 0001 refuses on write: the query's term weights
        # would not be comparable with the stored ones.
        with session_factory() as session:
            seed(session, store, book_id)
            drifted = FakeSparseEmbeddingProvider(model="stub-sparse-2")
            with pytest.raises(ProviderConfigError) as excinfo:
                search(
                    session, store, embedder, "deep modules", sparse_embedder=drifted
                )

        message = str(excinfo.value)
        assert "stub-sparse-2" in message
        assert SPARSE_MODEL in message
        assert "drop" in message.lower()
        assert "reprocess" in message.lower()
        # The lock is read first, so no query is embedded against a corpus it
        # cannot be compared with.
        assert embedder.calls == []

    def test_dense_only_search_needs_no_sparse_provider_at_all(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        embedder: QueryEmbedder,
        book_id: uuid.UUID,
    ) -> None:
        # Dense-only never touches the sparse vectors, so demanding a sparse
        # provider (whose construction downloads a model) would be a cost for
        # nothing — and a recipe mismatch is not a reason to refuse it.
        with session_factory() as session:
            seed(session, store, book_id)
            hits = search(session, store, embedder, "deep modules", mode="dense").hits

        assert len(hits) == 3
