"""The hybrid-vs-dense retrieval eval (issue #40), and the guards that keep it honest.

Three things live here, and only one of them costs money:

``TestFixtureSet`` checks the query set says what it claims — an "exact-term"
query whose term is not actually in the corpus, or a "conceptual" query that
shares vocabulary with its target, would quietly turn the experiment into a
measurement of nothing. These run always and need no providers.

``TestHarness`` runs the whole comparison against the fake providers. It proves
nothing about retrieval quality (the fake embedder derives vectors from text
length) and is not meant to: it keeps the harness from rotting between the rare
occasions the real eval is run.

``TestRealEval`` is the eval. It runs against the embedding provider booksmart
actually ships, because the question — should hybrid be the default? — is about
this product's retrieval, and a stand-in model would answer a different
question. It is skipped unless explicitly asked for, and writes its report to
stdout:

    BOOKSMART_EVAL_EMBEDDING_PROVIDER=gemini \\
    BOOKSMART_EVAL_API_KEY=... \\
    uv run pytest packages/core/tests/test_hybrid_eval.py -k RealEval -s

Its findings are written up in docs/research/hybrid-search-eval.md. Re-run it
when the default embedding model changes, or when the sparse model is
reconsidered (BM25 vs miniCOIL vs SPLADE) — its job is to inform those choices,
not to gate a build.
"""

import os
import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session, sessionmaker

from booksmart_core.config import Settings
from booksmart_core.fakes import FakeEmbeddingProvider, FakeSparseEmbeddingProvider
from booksmart_core.llm import EmbeddingProvider, build_embedding_provider
from booksmart_core.sparse import SparseEmbeddingProvider, build_sparse_embedding_provider
from booksmart_core.storage import BookStorage
from booksmart_core.vectors import VectorStore

from . import hybrid_eval
from .conftest import store_book

EVAL_PROVIDER = os.environ.get("BOOKSMART_EVAL_EMBEDDING_PROVIDER", "")
EVAL_API_KEY = os.environ.get("BOOKSMART_EVAL_API_KEY", "")
EVAL_MODEL = os.environ.get("BOOKSMART_EVAL_EMBEDDING_MODEL") or None
# Where to write the generated tables. Unset, the report is printed and nothing
# is written: a test that overwrote a checked-in document as a side effect would
# make `pytest` a thing you cannot run safely on a dirty tree.
EVAL_REPORT = os.environ.get("BOOKSMART_EVAL_REPORT", "")

LIMIT = 10


@pytest.fixture()
def store() -> VectorStore:
    return VectorStore(QdrantClient(":memory:"))


@pytest.fixture()
def eval_book(session_factory: sessionmaker[Session], storage: BookStorage) -> uuid.UUID:
    return uuid.UUID(
        store_book(
            session_factory,
            storage,
            title="A Philosophy of Software Design",
            author="Ousterhout",
            filename="apsd.pdf",
            content=b"%PDF-1.4 fixture",
        )
    )


class TestFixtureSet:
    """The query set's own claims, checked. A fixture that lies about its kind
    makes the eval measure something other than what the report will say."""

    def test_every_expectation_names_a_real_record(self) -> None:
        for fixture in hybrid_eval.QUERIES:
            for key in fixture.expects:
                assert key in hybrid_eval.BY_KEY, f"{fixture.query!r} expects unknown {key!r}"

    def test_record_keys_are_unique(self) -> None:
        keys = [record.key for record in hybrid_eval.CORPUS]
        assert len(keys) == len(set(keys))

    def test_all_three_query_kinds_are_represented(self) -> None:
        kinds = {fixture.kind for fixture in hybrid_eval.QUERIES}
        assert kinds == {"exact-term", "conceptual", "mixed"}
        for kind in kinds:
            count = sum(1 for f in hybrid_eval.QUERIES if f.kind == kind)
            assert count >= 4, f"only {count} {kind} queries; too few to average over"

    def test_exact_term_queries_appear_verbatim_in_their_targets(self) -> None:
        # If the phrase is not literally there, the sparse branch cannot match it
        # and the query is not testing what it says it tests.
        for fixture in hybrid_eval.QUERIES:
            if fixture.kind != "exact-term":
                continue
            for key in fixture.expects:
                haystack = hybrid_eval.BY_KEY[key].text.lower()
                needle = fixture.query.lower()
                assert needle in haystack, f"{needle!r} is not verbatim in {key!r}"

    def test_exact_term_queries_are_not_verbatim_anywhere_else(self) -> None:
        # Otherwise a "win" could come from retrieving an unexpected record that
        # also contains the phrase.
        for fixture in hybrid_eval.QUERIES:
            if fixture.kind != "exact-term":
                continue
            elsewhere = [
                record.key
                for record in hybrid_eval.CORPUS
                if record.key not in fixture.expects
                and fixture.query.lower() in record.text.lower()
            ]
            assert not elsewhere, f"{fixture.query!r} also appears verbatim in {elsewhere}"

    def test_conceptual_queries_share_no_content_word_with_their_targets(self) -> None:
        # The defining property of the slice: sparse retrieval must be unable to
        # help, so that fusion is being tested for harm rather than for benefit.
        for fixture in hybrid_eval.QUERIES:
            if fixture.kind != "conceptual":
                continue
            asked = hybrid_eval.content_words(fixture.query)
            for key in fixture.expects:
                overlap = asked & hybrid_eval.content_words(hybrid_eval.BY_KEY[key].text)
                assert not overlap, (
                    f"conceptual query {fixture.query!r} shares {sorted(overlap)} with {key!r}"
                )

    def test_every_query_explains_itself(self) -> None:
        for fixture in hybrid_eval.QUERIES:
            assert fixture.why.strip(), f"{fixture.query!r} has no rationale"


class TestHarness:
    """The comparison runs end to end. Says nothing about quality — the fake
    embedder has no notion of meaning — only that the machinery still works."""

    def test_the_comparison_runs_and_scores_every_query(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        eval_book: uuid.UUID,
    ) -> None:
        embedder = FakeEmbeddingProvider()
        sparse = FakeSparseEmbeddingProvider()
        with session_factory() as session:
            corpus = hybrid_eval.populate(session, store, eval_book, embedder, sparse)
            pairs = hybrid_eval.compare(session, store, corpus, embedder, sparse, limit=LIMIT)

        assert len(pairs) == len(hybrid_eval.QUERIES)
        for hybrid, dense in pairs:
            assert hybrid.mode == "hybrid"
            assert dense.mode == "dense"
            assert set(hybrid.ranks) == set(hybrid.query.expects)

    def test_exact_terms_are_found_by_the_sparse_branch_alone(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        eval_book: uuid.UUID,
    ) -> None:
        """The one quality claim the fakes *can* support, and it is worth having:
        with a dense embedder that encodes nothing but text length, every
        exact-term win must come from BM25. If this fails, the sparse branch is
        not contributing to fusion at all."""
        embedder = FakeEmbeddingProvider()
        sparse = FakeSparseEmbeddingProvider()
        with session_factory() as session:
            corpus = hybrid_eval.populate(session, store, eval_book, embedder, sparse)
            pairs = hybrid_eval.compare(session, store, corpus, embedder, sparse, limit=LIMIT)

        exact = [pair for pair in pairs if pair[0].query.kind == "exact-term"]
        hybrid_hits = sum(outcome.hit_at(5) for outcome, _ in exact)
        assert hybrid_hits == len(exact), "the sparse branch is not reaching fusion"

    def test_the_report_renders(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        eval_book: uuid.UUID,
    ) -> None:
        embedder = FakeEmbeddingProvider()
        sparse = FakeSparseEmbeddingProvider()
        with session_factory() as session:
            corpus = hybrid_eval.populate(session, store, eval_book, embedder, sparse)
            pairs = hybrid_eval.compare(session, store, corpus, embedder, sparse, limit=LIMIT)

        report = hybrid_eval.render_report(
            pairs,
            embedding_model=embedder.model,
            sparse_recipe=sparse.recipe,
            limit=LIMIT,
        )

        assert "## Per-query ranks" in report
        assert "## Aggregates" in report
        for fixture in hybrid_eval.QUERIES:
            assert fixture.query in report


@pytest.mark.skipif(
    not (EVAL_PROVIDER and EVAL_API_KEY),
    reason=(
        "the real eval costs an embedding call per record and per query; set "
        "BOOKSMART_EVAL_EMBEDDING_PROVIDER and BOOKSMART_EVAL_API_KEY to run it"
    ),
)
class TestRealEval:
    def test_hybrid_versus_dense_on_the_shipped_embedder(
        self,
        session_factory: sessionmaker[Session],
        store: VectorStore,
        eval_book: uuid.UUID,
    ) -> None:
        embedder = _real_embedder()
        sparse = _real_sparse()
        with session_factory() as session:
            corpus = hybrid_eval.populate(session, store, eval_book, embedder, sparse)
            pairs = hybrid_eval.compare(session, store, corpus, embedder, sparse, limit=LIMIT)

        report = hybrid_eval.render_report(
            pairs,
            embedding_model=embedder.model,
            sparse_recipe=sparse.recipe,
            limit=LIMIT,
        )
        print("\n" + report)
        if EVAL_REPORT:
            Path(EVAL_REPORT).write_text(report, encoding="utf-8")

        # Not a quality gate — the eval informs a decision, it does not enforce
        # one. Asserting that hybrid wins would have been a way of deciding the
        # answer in advance, and the answer turned out to be interesting (see
        # docs/research/hybrid-search-eval.md). What is asserted is only that the
        # run produced usable data.
        assert len(pairs) == len(hybrid_eval.QUERIES)
        assert any(outcome.best_rank is not None for outcome, _ in pairs)


def _real_embedder() -> EmbeddingProvider:
    settings = Settings(
        embedding_provider=EVAL_PROVIDER,
        embedding_model=EVAL_MODEL,
        **{f"{EVAL_PROVIDER}_api_key": EVAL_API_KEY},
    )
    return build_embedding_provider(settings)


def _real_sparse() -> SparseEmbeddingProvider:
    return build_sparse_embedding_provider(Settings(sparse_provider="fastembed"))


