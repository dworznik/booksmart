"""Unit tests for the sparse (BM25) embedding seam.

The sparse provider is selected through Settings like the dense one, but it is a
different kind of thing: no vendor, no key, no network, no usage — just local
term weighting. What it must get right is the *recipe* (the value the collection
locks against) and the document/query asymmetry BM25-over-Qdrant depends on.

The fastembed tests are skipped when fastembed is not installed: it is an
optional dependency of core (``booksmart-core[sparse]``), and constructing the
provider downloads the model, so they need a populated cache or a network.
"""

import pytest

from booksmart_core.config import Settings
from booksmart_core.errors import ProviderConfigError
from booksmart_core.sparse import (
    DEFAULT_SPARSE_MODELS,
    SparseEmbeddingProvider,
    SparseVector,
    build_sparse_embedding_provider,
    recipe_of,
)


class TestSparseVector:
    def test_is_immutable(self) -> None:
        vector = SparseVector(indices=[1, 2], values=[0.5, 0.5])

        with pytest.raises(AttributeError):
            vector.indices = [3]  # type: ignore[misc]

    def test_rejects_mismatched_lengths(self) -> None:
        # Qdrant pairs indices with values positionally; a mismatch is a caller
        # bug that would otherwise surface as an opaque error deep in an upsert.
        with pytest.raises(ValueError):
            SparseVector(indices=[1, 2], values=[0.5])


class TestProviderSelection:
    def test_unknown_provider_is_rejected_by_name(self) -> None:
        with pytest.raises(ProviderConfigError) as excinfo:
            build_sparse_embedding_provider(Settings(sparse_provider="bm42"))

        assert "bm42" in str(excinfo.value)
        assert "fastembed" in str(excinfo.value)

    def test_fake_provider_needs_no_dependency(self) -> None:
        provider = build_sparse_embedding_provider(Settings(sparse_provider="fake"))

        assert provider.recipe.startswith(DEFAULT_SPARSE_MODELS["fake"])

    def test_explicit_model_overrides_the_provider_default(self) -> None:
        provider = build_sparse_embedding_provider(
            Settings(sparse_provider="fake", sparse_model="fake-sparse-2")
        )

        assert provider.recipe.startswith("fake-sparse-2")


@pytest.fixture()
def fake_sparse() -> SparseEmbeddingProvider:
    return build_sparse_embedding_provider(Settings(sparse_provider="fake"))


class TestFakeSparseProvider:
    """The fake stands in for BM25 wherever a real one would need a download."""

    def test_documents_embed_deterministically(
        self, fake_sparse: SparseEmbeddingProvider
    ) -> None:
        first = fake_sparse.embed_documents(["deep modules", "shallow modules"])
        again = fake_sparse.embed_documents(["deep modules", "shallow modules"])

        assert [v.indices for v in first] == [v.indices for v in again]
        assert [v.values for v in first] == [v.values for v in again]

    def test_shared_terms_share_indices(self, fake_sparse: SparseEmbeddingProvider) -> None:
        deep, shallow = fake_sparse.embed_documents(["deep modules", "shallow modules"])

        # "modules" is common to both, so the two vectors must overlap — the
        # property every lexical match in hybrid retrieval rests on.
        assert set(deep.indices) & set(shallow.indices)

    def test_query_and_document_agree_on_terms(
        self, fake_sparse: SparseEmbeddingProvider
    ) -> None:
        document = fake_sparse.embed_documents(["deep modules"])[0]
        query = fake_sparse.embed_query("deep modules")

        assert set(query.indices) <= set(document.indices)

    def test_query_weights_are_flat(self, fake_sparse: SparseEmbeddingProvider) -> None:
        # Query-side weights are 1 per term: the IDF comes from Qdrant's `idf`
        # modifier, computed from collection statistics at query time. Weighting
        # the query as though it were a document double-counts term frequency.
        query = fake_sparse.embed_query("deep modules interfaces")

        assert set(query.values) == {1.0}

    def test_recipe_is_stable_across_instances(self) -> None:
        first = build_sparse_embedding_provider(Settings(sparse_provider="fake"))
        second = build_sparse_embedding_provider(Settings(sparse_provider="fake"))

        assert first.recipe == second.recipe


@pytest.fixture(scope="module")
def bm25() -> SparseEmbeddingProvider:
    pytest.importorskip("fastembed", reason="core's optional [sparse] extra")
    from booksmart_core.sparse import FastEmbedBM25Provider

    return FastEmbedBM25Provider()


class TestFastEmbedBM25:
    def test_recipe_names_the_model_and_every_bm25_parameter(
        self, bm25: SparseEmbeddingProvider
    ) -> None:
        assert bm25.recipe.startswith("Qdrant/bm25")
        # Drift in any of these silently changes term weights, so all four are
        # part of what the collection locks against (issue #38).
        for parameter in ("avg_len", "b", "k", "language"):
            assert f"{parameter}=" in bm25.recipe

    def test_documents_are_weighted_and_queries_are_not(
        self, bm25: SparseEmbeddingProvider
    ) -> None:
        document = bm25.embed_documents(["deep modules have simple interfaces"])[0]
        query = bm25.embed_query("deep modules")

        assert set(document.values) != {1.0}  # BM25 saturation applied
        assert set(query.values) == {1.0}  # IDF left to Qdrant

    def test_indices_and_values_are_plain_python_numbers(
        self, bm25: SparseEmbeddingProvider
    ) -> None:
        # fastembed returns numpy arrays; anything reaching the Qdrant client has
        # to be JSON-serialisable or the upsert fails on the wire.
        document = bm25.embed_documents(["deep modules"])[0]

        assert all(type(index) is int for index in document.indices)
        assert all(type(value) is float for value in document.values)

    def test_batches_come_back_in_order(self, bm25: SparseEmbeddingProvider) -> None:
        texts = ["alpha term", "beta term", "gamma term"]
        one_by_one = [bm25.embed_documents([text])[0] for text in texts]
        batched = bm25.embed_documents(texts)

        assert [v.indices for v in batched] == [v.indices for v in one_by_one]

    def test_stopwords_alone_embed_to_nothing(self, bm25: SparseEmbeddingProvider) -> None:
        # BM25 drops stopwords, so a query of nothing but stopwords yields an
        # empty vector. Qdrant scores that as no match rather than raising, but
        # the search path has to know it is a real possibility.
        assert bm25.embed_query("the and of").indices == []


class TestRecipeStrictness:
    """A partial recipe is refused rather than silently shortened.

    The recipe is read off the constructed model so that drift in the *library's*
    defaults trips the collection lock. The same indirection means a library that
    renamed or moved those attributes would leave us locking collections against a
    bare model name — at which point parameter drift stops being detectable, which
    is the one thing the recipe exists to do.
    """

    def test_a_model_missing_a_parameter_cannot_be_locked(self) -> None:
        class HalfDescribed:
            k = 1.2
            b = 0.75
            # no avg_len, no language

        with pytest.raises(ProviderConfigError) as excinfo:
            recipe_of("Qdrant/bm25", HalfDescribed())

        message = str(excinfo.value)
        assert "avg_len" in message
        assert "language" in message
        assert "k=" not in message  # names what is missing, not what was found

    def test_a_model_describing_nothing_cannot_be_locked(self) -> None:
        with pytest.raises(ProviderConfigError):
            recipe_of("prithivida/Splade_PP_en_v1", object())

    def test_a_fully_described_model_produces_a_stable_recipe(self) -> None:
        class Described:
            k = 1.2
            b = 0.75
            avg_len = 256.0
            language = "english"

        assert recipe_of("Qdrant/bm25", Described()) == (
            "Qdrant/bm25(k=1.2,b=0.75,avg_len=256.0,language=english)"
        )

    def test_parameter_order_is_fixed_not_incidental(self) -> None:
        # The recipe is compared as a string, so a reordering would read as drift
        # and force a needless reprocess of every book.
        class Described:
            language = "english"
            avg_len = 256.0
            b = 0.75
            k = 1.2

        assert recipe_of("m", Described()).index("k=") < recipe_of("m", Described()).index("b=")
