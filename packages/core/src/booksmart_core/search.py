"""Search over the embedded vectors — the read side of the pipeline.

Embed the query, search Qdrant, and resolve each hit's payload (``record_type``
+ ``record_id``) back to the relational row it was embedded from. Pure read: no
writes, no Stage, no Runner.

Retrieval is hybrid by default: the query is embedded twice, densely by the
configured embedding provider and lexically by the sparse one, and the two
rankings are fused with Reciprocal Rank Fusion. Dense retrieval alone matches
meaning and misses words — a proper noun or an exact term the model never
learned ranks on nothing in particular — and the sparse half is what covers it.
``mode="dense"`` keeps the pre-hybrid behaviour, and with it cosine scores that
are comparable across queries; fused scores are not.

The model lock (ADR 0001) is verified *before* the query is embedded. A query
embedded by one model and compared against another model's vectors returns
plausible, silently wrong rankings — the failure has no symptom an operator can
diagnose, so it is refused rather than served.

This is the seam a server's HTTP search endpoint would reuse: it takes a session
and an already-built vector store and embedder, and returns detached rows plus
the query embedding's usage — one unbatched call per query, so a consumer that
costs its search traffic cannot get that number anywhere else.
"""

import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from qdrant_client import models as qmodels
from sqlalchemy import select
from sqlalchemy.orm import Session

from booksmart_core.errors import ProviderConfigError
from booksmart_core.llm import EmbeddingProvider
from booksmart_core.models import Chapter, KnowledgeObject, Section
from booksmart_core.sparse import SparseEmbeddingProvider
from booksmart_core.vectors import RECORD_TYPES, RecordType, VectorStore, unknown_record_types

Record = Chapter | Section | KnowledgeObject

# How to retrieve. "hybrid" fuses dense and sparse retrieval with Reciprocal
# Rank Fusion and is the default: it finds the paraphrases dense search is good
# at *and* the exact terms it is bad at. "dense" is the pre-hybrid behaviour,
# kept because its scores are cosine similarities — comparable across queries,
# which fused rank scores are not.
SearchMode = Literal["hybrid", "dense"]

_RECORD_MODELS: dict[RecordType, type[Record]] = {
    "chapter": Chapter,
    "section": Section,
    "knowledge_object": KnowledgeObject,
}


@dataclass(frozen=True)
class SearchHit:
    """One ranked result: where it placed, what it scored, and the row.

    ``row`` is detached from the session that loaded it, so callers can render
    after closing it (the ``reads.py`` pattern). Its column values are loaded;
    its relationships are not, and touching them after the session closes raises.
    """

    # 1-based position in the overall ranking, counted from the offset rather
    # than from the page — a rank that restarted at 1 on every page would be a
    # position within the page, which is not a thing anyone wants to know.
    # Ranks skip where a point's row has since been deleted (see `_resolve`):
    # this is the position Qdrant gave the record, not a renumbering of what
    # survived.
    rank: int
    # What the score *means* depends on the mode, and only ``SearchResults.mode``
    # says which: a COSINE similarity under "dense", a fused rank score under
    # "hybrid" (see ``VectorStore.hybrid_search``). Anything rendering this has
    # to carry the mode with it.
    score: float
    record_type: RecordType
    record_id: uuid.UUID
    book_id: uuid.UUID
    # The exact text that was embedded, from the point's payload.
    text: str
    row: Record

    @property
    def title(self) -> str:
        return self.row.title


@dataclass(frozen=True)
class SearchResults:
    """The ranked hits, plus what embedding the query cost.

    A search is exactly one embedding call, unbatched, so a consumer costing
    search traffic needs the number this carries — the read-side counterpart of
    ``StageReport.embedding_tokens``.

    ``embedding_tokens`` is what the query cost: the provider's count, or 0 when
    no call was made at all (nothing is embedded yet, so no query was sent).
    Those two are not distinguishable, and needn't be — both cost nothing.
    ``None`` is the state that matters: the provider was asked and would not say.

    It is ``int | None`` where ``StageReport.embedding_tokens`` is a plain
    ``int``, deliberately. A Stage sums many batches, so one silent provider
    among them can only ever make the total a lower bound; a search has exactly
    one call to report, so "unknown" survives here instead of being rounded into
    a number a consumer would then trust.
    """

    hits: list[SearchHit]
    embedding_tokens: int | None
    # Which retrieval produced these hits, and therefore what ``SearchHit.score``
    # means. Reported rather than assumed: the two scores are different kinds of
    # number on different scales, and a renderer that guessed would eventually
    # label one as the other.
    mode: SearchMode = "hybrid"


def search(
    session: Session,
    vector_store: VectorStore,
    embedder: EmbeddingProvider,
    query: str,
    *,
    sparse_embedder: SparseEmbeddingProvider | None = None,
    mode: SearchMode = "hybrid",
    book_id: uuid.UUID | None = None,
    record_types: Sequence[RecordType] | None = None,
    limit: int = 10,
    offset: int = 0,
    score_threshold: float | None = None,
) -> SearchResults:
    """The embedded records that best answer ``query``, best first, and what
    embedding the query cost.

    ``mode`` decides what "best" means, and what a score is:

    - **hybrid** (default) fuses dense and sparse retrieval with Reciprocal Rank
      Fusion. Scores are rank-derived — comparable within one result set and
      meaningless across queries — and ``score_threshold`` bounds the dense
      branch, where it is still a cosine floor. Needs ``sparse_embedder``.
    - **dense** is the pre-hybrid behaviour: scores are COSINE similarities and
      ``score_threshold`` keeps only hits scoring at least that much.

    Nothing embedded yet (no collection), or nothing matching the filters, is an
    empty result rather than an error — an un-ingested book is a normal state,
    not a failure.

    Points whose relational row has since been deleted are skipped, so ``hits``
    may be shorter than ``limit``: Qdrant is a derived index, and the database is
    the source of truth.
    """
    _validate_record_types(record_types)
    if mode == "hybrid" and sparse_embedder is None:
        # Falling back to dense-only would be the worst outcome available: the
        # caller believes it is searching hybrid and quietly gets less.
        raise ValueError(
            "hybrid search needs a sparse_embedder to embed the query's terms; "
            "pass one, or search with mode='dense'"
        )

    lock = vector_store.verified_lock()
    if lock is None:
        # Nothing embedded yet; not worth embedding the query. No call was
        # made, so the query cost 0 — not the None that means "we asked".
        return SearchResults(hits=[], embedding_tokens=0, mode=mode)
    if lock.embedding_model != embedder.model:
        raise ProviderConfigError(
            f"vector collection {vector_store.collection!r} is locked to embedding model "
            f"{lock.embedding_model!r} but the configured embedder is {embedder.model!r}; "
            f"a query embedded by a different model cannot be compared against these "
            f"vectors (ADR 0001)"
        )
    # The sparse half of the lock is only checked where the sparse vectors are
    # actually read. A dense-only search does not compare term weights with
    # anything, so a drifted recipe is not a reason to refuse it — and demanding
    # a sparse provider just to verify the lock would cost a model download for
    # a search that will never use it.
    if sparse_embedder is not None and mode == "hybrid":
        if lock.sparse_recipe != sparse_embedder.recipe:
            raise ProviderConfigError(
                f"vector collection {vector_store.collection!r} is locked to sparse model "
                f"{lock.sparse_recipe!r} but the configured sparse embedder is "
                f"{sparse_embedder.recipe!r}; a query weighted by different BM25 "
                f"parameters cannot be compared against these vectors. Drop the "
                f"collection and reprocess embeddings for every book (ADR 0001)"
            )

    embedded = embedder.embed([query])
    query_filter = _build_filter(book_id, record_types)
    if mode == "hybrid":
        assert sparse_embedder is not None  # guarded above
        points = vector_store.hybrid_search(
            embedded.vectors[0],
            sparse_embedder.embed_query(query),
            query_filter=query_filter,
            limit=limit,
            offset=offset,
            score_threshold=score_threshold,
        )
    else:
        points = vector_store.search(
            embedded.vectors[0],
            query_filter=query_filter,
            limit=limit,
            offset=offset,
            score_threshold=score_threshold,
        )
    return SearchResults(
        hits=_resolve(session, points, first_rank=offset + 1),
        embedding_tokens=embedded.input_tokens,
        mode=mode,
    )


def _validate_record_types(record_types: Sequence[RecordType] | None) -> None:
    """A record type outside the collection's contract is a caller bug, not a
    configuration mistake — so a plain ValueError, not a taxonomy error. Filtering
    on it would otherwise just return nothing, which reads as "no matches"."""
    unknown = unknown_record_types(record_types or ())
    if unknown:
        raise ValueError(
            f"Unknown record type(s) {', '.join(repr(name) for name in unknown)}; "
            f"expected one of {', '.join(RECORD_TYPES)}"
        )


def _build_filter(
    book_id: uuid.UUID | None, record_types: Sequence[RecordType] | None
) -> qmodels.Filter | None:
    conditions: list[qmodels.Condition] = []
    if book_id is not None:
        conditions.append(
            qmodels.FieldCondition(key="book_id", match=qmodels.MatchValue(value=str(book_id)))
        )
    if record_types:
        conditions.append(
            qmodels.FieldCondition(
                key="record_type", match=qmodels.MatchAny(any=[str(name) for name in record_types])
            )
        )
    return qmodels.Filter(must=conditions) if conditions else None


@dataclass(frozen=True)
class _Ref:
    """What a point's payload says it was embedded from."""

    record_type: RecordType
    record_id: uuid.UUID
    book_id: uuid.UUID


def _resolve(
    session: Session, points: Sequence[qmodels.ScoredPoint], *, first_rank: int = 1
) -> list[SearchHit]:
    """Load each point's row, preserving Qdrant's ranking, in one query per type.

    ``first_rank`` is the rank of the first point on this page, so ranks read as
    positions in the whole ranking rather than in the page."""
    refs = [_ref(point.payload or {}) for point in points]
    wanted: dict[RecordType, set[uuid.UUID]] = defaultdict(set)
    for ref in refs:
        if ref is not None:
            wanted[ref.record_type].add(ref.record_id)

    rows: dict[tuple[RecordType, uuid.UUID], Record] = {}
    for record_type, ids in wanted.items():
        # Untyped because mypy joins the three mapped classes to their declarative
        # Base, which declares no `id`; _RECORD_MODELS is the checked contract.
        model: Any = _RECORD_MODELS[record_type]
        for row in session.scalars(select(model).where(model.id.in_(ids))):
            rows[(record_type, row.id)] = row
            # Detach: the caller renders hits after its session closes.
            session.expunge(row)

    hits: list[SearchHit] = []
    for position, (point, ref) in enumerate(zip(points, refs, strict=True)):
        if ref is None:
            continue  # a payload this schema does not understand
        row = rows.get((ref.record_type, ref.record_id))
        if row is None:
            continue  # a stale point: its row has been deleted
        hits.append(
            SearchHit(
                rank=first_rank + position,
                score=point.score,
                record_type=ref.record_type,
                record_id=ref.record_id,
                book_id=ref.book_id,
                text=str((point.payload or {}).get("text", "")),
                row=row,
            )
        )
    return hits


def _ref(payload: dict[str, object]) -> _Ref | None:
    record_type = payload.get("record_type")
    if not isinstance(record_type, str) or record_type not in RECORD_TYPES:
        return None
    try:
        record_id = uuid.UUID(str(payload.get("record_id")))
        book_id = uuid.UUID(str(payload.get("book_id")))
    except ValueError:
        return None
    return _Ref(record_type, record_id, book_id)
