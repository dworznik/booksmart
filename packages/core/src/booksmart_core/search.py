"""Semantic search over the embedded vectors — the read side of the pipeline.

Embed the query with the collection's locked embedding model, ANN-search Qdrant,
and resolve each hit's payload (``record_type`` + ``record_id``) back to the
relational row it was embedded from. Pure read: no writes, no Stage, no Runner.

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
from typing import Any

from qdrant_client import models as qmodels
from sqlalchemy import select
from sqlalchemy.orm import Session

from booksmart_core.errors import ProviderConfigError
from booksmart_core.llm import EmbeddingProvider
from booksmart_core.models import Chapter, KnowledgeObject, Section
from booksmart_core.vectors import RECORD_TYPES, RecordType, VectorStore, unknown_record_types

Record = Chapter | Section | KnowledgeObject

_RECORD_MODELS: dict[RecordType, type[Record]] = {
    "chapter": Chapter,
    "section": Section,
    "knowledge_object": KnowledgeObject,
}


@dataclass(frozen=True)
class SearchHit:
    """One ranked result: the vector's similarity, what it points at, and the row.

    ``row`` is detached from the session that loaded it, so callers can render
    after closing it (the ``reads.py`` pattern). Its column values are loaded;
    its relationships are not, and touching them after the session closes raises.
    """

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


def search(
    session: Session,
    vector_store: VectorStore,
    embedder: EmbeddingProvider,
    query: str,
    *,
    book_id: uuid.UUID | None = None,
    record_types: Sequence[RecordType] | None = None,
    limit: int = 10,
    offset: int = 0,
    score_threshold: float | None = None,
) -> SearchResults:
    """The most similar embedded records to ``query``, best first, and what
    embedding the query cost.

    Scores are COSINE similarities: higher is closer, and ``score_threshold``
    keeps only hits scoring at least that much. Nothing embedded yet (no
    collection), or nothing matching the filters, is an empty result rather than
    an error — an un-ingested book is a normal state, not a failure.

    Points whose relational row has since been deleted are skipped, so
    ``hits`` may be shorter than ``limit``: Qdrant is a derived index, and the
    database is the source of truth.
    """
    _validate_record_types(record_types)

    locked_model = vector_store.verified_model()
    if locked_model is None:
        # Nothing embedded yet; not worth embedding the query. No call was
        # made, so the query cost 0 — not the None that means "we asked".
        return SearchResults(hits=[], embedding_tokens=0)
    if locked_model != embedder.model:
        raise ProviderConfigError(
            f"vector collection {vector_store.collection!r} is locked to embedding model "
            f"{locked_model!r} but the configured embedder is {embedder.model!r}; a query "
            f"embedded by a different model cannot be compared against these vectors "
            f"(ADR 0001)"
        )

    embedded = embedder.embed([query])
    points = vector_store.search(
        embedded.vectors[0],
        query_filter=_build_filter(book_id, record_types),
        limit=limit,
        offset=offset,
        score_threshold=score_threshold,
    )
    return SearchResults(
        hits=_resolve(session, points), embedding_tokens=embedded.input_tokens
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


def _resolve(session: Session, points: Sequence[qmodels.ScoredPoint]) -> list[SearchHit]:
    """Load each point's row, preserving Qdrant's ranking, in one query per type."""
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
    for point, ref in zip(points, refs, strict=True):
        if ref is None:
            continue  # a payload this schema does not understand
        row = rows.get((ref.record_type, ref.record_id))
        if row is None:
            continue  # a stale point: its row has been deleted
        hits.append(
            SearchHit(
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
