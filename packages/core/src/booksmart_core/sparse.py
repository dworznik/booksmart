"""Client-side sparse (BM25) embeddings — the lexical half of hybrid retrieval.

Nothing here calls a vendor API. BM25 term weights are computed locally from the
text, so a sparse provider has no key, no network at query time, and no usage to
report; it is a seam for the same reason the dense one is (the model is a choice,
selected through ``Settings``), not because a vendor sits behind it. That is why
it lives beside ``llm.py`` rather than inside it — ``llm.py`` is where vendor API
facts and keys live, and a sparse provider has neither.

What a sparse provider has instead of a bare model name is a **recipe**: the
model plus the parameters it actually ran with. BM25's k, b and avg_len shape
every term weight, and its language selects the stopword list and stemmer. Drift
in any of them changes retrieval without changing anything an operator can
observe — the exact failure mode ADR 0001 exists to prevent — so the collection
locks against the whole recipe, not just the name (see ``vectors.py``). Reading
the parameters back off the constructed model, rather than off our own config,
means a change in the library's own defaults is caught too.

Documents and queries are deliberately embedded by different calls. BM25 weights
a document's terms with the saturation parameters; the query side emits a flat
weight per term and leaves the IDF to Qdrant, which computes it at query time
from collection statistics via the ``idf`` modifier. Embedding a query as though
it were a document double-counts term frequency and skews the ranking.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from booksmart_core.config import Settings
from booksmart_core.errors import ProviderConfigError

# The BM25 recipe's parameters, in the order they are written into it. Read off
# the constructed model by these names; a model that does not define one simply
# leaves it out (SPLADE and miniCOIL are not parameterised this way).
BM25_RECIPE_PARAMETERS = ("k", "b", "avg_len", "language")

DEFAULT_SPARSE_MODELS = {
    "fastembed": "Qdrant/bm25",
    # Deterministic local term hashing, no dependency and no download (CI,
    # local dev) — the sparse counterpart of the fake dense embedder.
    "fake": "fake-sparse-1",
}


@dataclass(frozen=True)
class SparseVector:
    """Term ids and their weights — one document's or one query's lexical shape.

    Core's own type rather than the Qdrant client's, so the seam a consumer
    implements does not drag a vector-store dependency in with it. ``vectors.py``
    converts at the boundary.
    """

    indices: list[int]
    values: list[float]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"sparse vector has {len(self.indices)} indices but "
                f"{len(self.values)} values; they are paired positionally"
            )


class SparseEmbeddingProvider(Protocol):
    # The full recipe (model plus parameters), which is what the collection is
    # locked to — not the model name alone. See the module docstring.
    recipe: str

    def embed_documents(self, texts: list[str]) -> list[SparseVector]: ...

    def embed_query(self, text: str) -> SparseVector: ...


def _recipe(model: str, parameters: object) -> str:
    """``model(k=1.2,b=0.75,…)`` — stable, human-readable, and diffable in an
    error message, which is the only place an operator ever sees it."""
    parts = []
    for name in BM25_RECIPE_PARAMETERS:
        value = getattr(parameters, name, None)
        if value is not None:
            parts.append(f"{name}={value}")
    return f"{model}({','.join(parts)})" if parts else model


class FastEmbedBM25Provider:
    """BM25 term weights from fastembed — the default sparse implementation.

    fastembed is an optional dependency (``booksmart-core[sparse]``): core stays
    installable without it for consumers that bring their own sparse provider or
    do not use hybrid retrieval, so the import is local to the constructor and
    its absence is a configuration error with an install hint, not an ImportError
    from deep inside a stage.

    Constructing this downloads the model on first use (a few MB of stopwords and
    stemmer data, cached thereafter). It is therefore built once per process, at
    the point a run or a search actually needs it.
    """

    def __init__(self, model: str = DEFAULT_SPARSE_MODELS["fastembed"]) -> None:
        try:
            from fastembed import SparseTextEmbedding
        except ModuleNotFoundError as exc:
            raise ProviderConfigError(
                f"sparse provider 'fastembed' needs the fastembed package to embed "
                f"with {model!r}; install booksmart-core[sparse], or select another "
                f"sparse_provider"
            ) from exc
        self.model = model
        self._embedder = SparseTextEmbedding(model_name=model)
        # Off the constructed model, so a change in fastembed's own defaults
        # trips the collection lock instead of silently re-weighting the corpus.
        self.recipe = _recipe(model, getattr(self._embedder, "model", None))

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [_from_fastembed(vector) for vector in self._embedder.embed(texts)]

    def embed_query(self, text: str) -> SparseVector:
        return _from_fastembed(next(iter(self._embedder.query_embed(text))))


def _from_fastembed(vector: Any) -> SparseVector:
    """fastembed answers in numpy arrays of numpy scalars; the Qdrant client has
    to serialise these to JSON, which numpy types do not survive."""
    return SparseVector(
        indices=[int(index) for index in vector.indices],
        values=[float(value) for value in vector.values],
    )


def build_sparse_embedding_provider(settings: Settings) -> SparseEmbeddingProvider:
    if settings.sparse_provider not in DEFAULT_SPARSE_MODELS:
        raise ProviderConfigError(
            f"Unknown sparse provider {settings.sparse_provider!r}; "
            f"expected one of {sorted(DEFAULT_SPARSE_MODELS)}"
        )
    model = settings.sparse_model or DEFAULT_SPARSE_MODELS[settings.sparse_provider]
    if settings.sparse_provider == "fake":
        # Lazy for the same import-cycle reason as in build_llm_provider.
        from booksmart_core.fakes import FakeSparseEmbeddingProvider

        return FakeSparseEmbeddingProvider(model=model)
    return FastEmbedBM25Provider(model=model)
