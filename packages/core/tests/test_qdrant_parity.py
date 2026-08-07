"""Parity between embedded local-mode Qdrant and a Qdrant server (issue #33).

The CLI searches an embedded local store; a server consumer searches a Qdrant
server — two implementations of the same query contract. What semantic search
rests on is that they agree on which points each filter admits (``book_id`` →
``MatchValue``, ``record_types`` → ``MatchAny``) and on COSINE score
semantics (``score_threshold`` keeping ``>= threshold``, higher is closer) —
asserted here by seeding identical vectors into both backends and running the
same ``search(...)`` calls against each.

At this corpus size the server, like local mode, answers by exact scan — it
builds HNSW only past its indexing threshold, so the approximate regime is
not reached here. The full ranked order is still deliberately not compared:
over a real corpus HNSW is entitled to permute near-ties, and the parity
contract must not promise more than the backends share. The admitted set,
the top hit, and each point's score are not allowed to differ.

The module needs a running Qdrant server, so it is skipped unless
``BOOKSMART_TEST_QDRANT_URL`` names one (the ``BOOKSMART_TEST_DATABASE_URL``
pattern) — the default ``pytest`` run stays service-free.
"""

import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session, sessionmaker

from booksmart_core.models import Chapter, KnowledgeObject, Section
from booksmart_core.search import SearchResults, search
from booksmart_core.storage import BookStorage
from booksmart_core.vectors import RecordType, VectorRecord, VectorStore

from .conftest import QueryEmbedder, store_book

QDRANT_URL = os.environ.get("BOOKSMART_TEST_QDRANT_URL", "")

pytestmark = pytest.mark.skipif(
    not QDRANT_URL,
    reason="parity needs a Qdrant server; set BOOKSMART_TEST_QDRANT_URL to run",
)

# A collection of its own, so a parity run can never adopt (or corrupt) data a
# real consumer keeps on the same server.
PARITY_COLLECTION = "booksmart_parity_test"

# The query embeds to [1, 0] (the test_search geometry); every seeded point
# sits at a known angle from it, so each expected COSINE score is arithmetic,
# and the two books spread the scores so that every filter shape and the 0.5
# threshold each split the corpus somewhere interesting.
BOOK_A_VECTORS: dict[RecordType, list[float]] = {
    "chapter": [1.0, 0.0],  # 1.0 — the unambiguous top hit
    "section": [1.0, 1.0],  # ≈0.7071
    "knowledge_object": [0.0, 1.0],  # 0.0
}
BOOK_B_VECTORS: dict[RecordType, list[float]] = {
    "chapter": [0.9, 0.1],  # ≈0.9939 — near the top hit, but no tie
    "section": [1.0, 2.0],  # ≈0.4472 — just under the 0.5 threshold
    "knowledge_object": [-1.0, 0.5],  # ≈-0.8944 — negative similarity
}


@pytest.fixture()
def local_store() -> VectorStore:
    return VectorStore(QdrantClient(":memory:"))


@pytest.fixture()
def server_store() -> Iterator[VectorStore]:
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(PARITY_COLLECTION):
        client.delete_collection(PARITY_COLLECTION)  # a crashed run's leftovers
    yield VectorStore(client, collection=PARITY_COLLECTION)
    client.delete_collection(PARITY_COLLECTION)
    client.close()


def seed_book(
    session: Session, book_id: uuid.UUID, vectors: dict[RecordType, list[float]]
) -> tuple[dict[RecordType, uuid.UUID], list[VectorRecord]]:
    """One chapter, section and knowledge object for ``book_id`` with the given
    vectors; returns their row ids and the points to write into each backend."""
    chapter = Chapter(book_id=book_id, position=0, title="Chapter", summary="Chapter.")
    session.add(chapter)
    session.flush()
    section = Section(chapter_id=chapter.id, position=0, title="Section", summary="Section.")
    knowledge = KnowledgeObject(
        book_id=book_id,
        type="Principle",
        title="Idea",
        content="An idea.",
        summary="An idea.",
        source_location="ch1",
        confidence=1.0,
        extraction_model="stub-llm-1",
        extraction_prompt_version="1",
    )
    session.add_all([section, knowledge])
    session.commit()

    ids: dict[RecordType, uuid.UUID] = {
        "chapter": chapter.id,
        "section": section.id,
        "knowledge_object": knowledge.id,
    }
    records = [
        VectorRecord(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "record_type": record_type,
                "record_id": str(ids[record_type]),
                "book_id": str(book_id),
                "text": f"{record_type} text",
            },
        )
        for record_type, vector in vectors.items()
    ]
    return ids, records


@dataclass(frozen=True)
class Corpus:
    """Two seeded books, their rows in the database, and the two backends
    holding the *same* points."""

    local: VectorStore
    server: VectorStore
    book_a: uuid.UUID
    book_b: uuid.UUID
    ids_a: dict[RecordType, uuid.UUID]
    ids_b: dict[RecordType, uuid.UUID]


def points(
    ids: dict[RecordType, uuid.UUID], *types: RecordType
) -> set[tuple[RecordType, uuid.UUID]]:
    return {(record_type, ids[record_type]) for record_type in types}


@pytest.fixture()
def corpus(
    session_factory: sessionmaker[Session],
    storage: BookStorage,
    local_store: VectorStore,
    server_store: VectorStore,
) -> Corpus:
    book_a = uuid.UUID(
        store_book(
            session_factory,
            storage,
            title="A Philosophy of Software Design",
            author="Ousterhout",
            filename="apsd.pdf",
            content=b"%PDF-1.4 a",
        )
    )
    book_b = uuid.UUID(
        store_book(
            session_factory,
            storage,
            title="Domain-Driven Design",
            author="Evans",
            filename="ddd.pdf",
            content=b"%PDF-1.4 b",
        )
    )
    with session_factory() as session:
        ids_a, records_a = seed_book(session, book_a, BOOK_A_VECTORS)
        ids_b, records_b = seed_book(session, book_b, BOOK_B_VECTORS)
    for store in (local_store, server_store):
        store.replace_book_points(str(book_a), records_a, embedding_model=QueryEmbedder.model)
        store.replace_book_points(str(book_b), records_b, embedding_model=QueryEmbedder.model)
    return Corpus(local_store, server_store, book_a, book_b, ids_a, ids_b)


def search_both(
    session: Session, corpus: Corpus, **kwargs: Any
) -> tuple[SearchResults, SearchResults]:
    """The same ``search(...)`` call against each backend."""
    local = search(session, corpus.local, QueryEmbedder(), "deep modules", **kwargs)
    server = search(session, corpus.server, QueryEmbedder(), "deep modules", **kwargs)
    return local, server


def admitted(results: SearchResults) -> set[tuple[RecordType, uuid.UUID]]:
    return {(hit.record_type, hit.record_id) for hit in results.hits}


def assert_parity(local: SearchResults, server: SearchResults) -> None:
    """The parity contract: same admitted set, same top hit, same score per
    point — but not the full ranked order, which HNSW may permute among
    near-ties."""
    assert admitted(local) == admitted(server)
    if local.hits:
        assert (local.hits[0].record_type, local.hits[0].record_id) == (
            server.hits[0].record_type,
            server.hits[0].record_id,
        )
    server_scores = {(hit.record_type, hit.record_id): hit.score for hit in server.hits}
    for hit in local.hits:
        assert hit.score == pytest.approx(
            server_scores[(hit.record_type, hit.record_id)], abs=1e-6
        )


class TestSearchParity:
    """Each case pins the local result to the geometry (the semantics the unit
    tests in test_search already establish) *and* asserts the server agrees —
    two equal-but-wrong results cannot pass."""

    def test_unfiltered_search_agrees_on_all_points_and_the_top_hit(
        self, session_factory: sessionmaker[Session], corpus: Corpus
    ) -> None:
        with session_factory() as session:
            local, server = search_both(session, corpus)

        assert admitted(local) == points(
            corpus.ids_a, "chapter", "section", "knowledge_object"
        ) | points(corpus.ids_b, "chapter", "section", "knowledge_object")
        assert local.hits[0].record_id == corpus.ids_a["chapter"]
        assert local.hits[0].score == pytest.approx(1.0)
        assert_parity(local, server)

    def test_book_id_filter_admits_the_same_points(
        self, session_factory: sessionmaker[Session], corpus: Corpus
    ) -> None:
        with session_factory() as session:
            local, server = search_both(session, corpus, book_id=corpus.book_b)

        assert admitted(local) == points(
            corpus.ids_b, "chapter", "section", "knowledge_object"
        )
        assert_parity(local, server)

    def test_record_types_filter_admits_the_same_points(
        self, session_factory: sessionmaker[Session], corpus: Corpus
    ) -> None:
        with session_factory() as session:
            local, server = search_both(
                session, corpus, record_types=["chapter", "section"]
            )

        assert admitted(local) == points(corpus.ids_a, "chapter", "section") | points(
            corpus.ids_b, "chapter", "section"
        )
        assert_parity(local, server)

    def test_combined_filters_admit_the_same_single_point(
        self, session_factory: sessionmaker[Session], corpus: Corpus
    ) -> None:
        with session_factory() as session:
            local, server = search_both(
                session,
                corpus,
                book_id=corpus.book_a,
                record_types=["knowledge_object"],
            )

        assert admitted(local) == points(corpus.ids_a, "knowledge_object")
        assert_parity(local, server)

    def test_score_threshold_keeps_the_same_points(
        self, session_factory: sessionmaker[Session], corpus: Corpus
    ) -> None:
        # 0.5 sits between book B's section (≈0.4472) and book A's section
        # (≈0.7071): both backends must draw the >= threshold line there.
        with session_factory() as session:
            local, server = search_both(session, corpus, score_threshold=0.5)

        assert admitted(local) == points(corpus.ids_a, "chapter", "section") | points(
            corpus.ids_b, "chapter"
        )
        assert_parity(local, server)
