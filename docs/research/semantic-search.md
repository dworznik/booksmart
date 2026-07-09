# Semantic search over the embedded vectors (issue #30) — findings

> This is a new notes location, `docs/research/`, sitting alongside
> `docs/adr/`, `docs/api-notes/`, and `docs/prds/`. It holds pre-implementation
> research: primary-source facts and a recommended shape, not a decision record.

## Summary

- **Query API:** In the pinned `qdrant-client>=1.18.0,<1.19.0`, the search call
  is **`QdrantClient.query_points(...)`**. The old `search()` / `search_batch()` /
  `search_groups()` methods are **gone from the public client** in 1.18 (not just
  deprecated) — only `query_points`, `query_batch_points`, `query_points_groups`,
  `scroll`, and `retrieve` remain in the query family. Pass a raw query vector as
  `query=`. It returns a `QueryResponse` with `.points: list[ScoredPoint]`; each
  `ScoredPoint` has `.id`, `.score`, `.payload`, `.version`, `.vector`.
- **Local-mode parity:** `QdrantClient(path=...)` supports `query_points` with the
  **same** `query_filter`, `score_threshold`, `limit`, `offset`, `with_payload`,
  and `with_vectors`. Local mode is brute-force (no HNSW), single-process, and
  recommended for small collections (~20k points), but the filtering and cosine
  scoring semantics match server mode. This is what the CLI already uses.
- **Model-lock read check:** the locked model is stored in collection metadata
  under `EMBEDDING_MODEL_KEY` and read back via
  `client.get_collection(name).config.metadata` — a real, populated field in 1.18
  (`CollectionConfig.metadata`). The search path must read it, compare it to the
  configured `embedder.model`, and fail loudly on mismatch before embedding the
  query, mirroring the write-side lock in `vectors.py:_ensure_collection`.

Every claim below is cited to a primary source: an inline path into the installed
`qdrant-client==1.18.0` source (the exact pinned version, the most authoritative
description of its own behavior), the Qdrant API reference, or a `path:symbol`
into this repo. Where a claim could not be verified against a primary source it
says so.

---

## 1. Query API in qdrant-client 1.18 — `query_points`, and `search` is gone

`query_points` is the universal search entry point. Exact signature (verbatim
from the installed source):

```python
def query_points(
    self,
    collection_name: str,
    query: types.PointId | list[float] | list[list[float]] | types.SparseVector
        | types.Query | types.NumpyArray | types.Document | types.Image
        | types.InferenceObject | None = None,
    using: str | None = None,
    prefetch: types.Prefetch | list[types.Prefetch] | None = None,
    query_filter: types.Filter | None = None,
    search_params: types.SearchParams | None = None,
    limit: int = 10,
    offset: int | None = None,
    with_payload: bool | Sequence[str] | types.PayloadSelector = True,
    with_vectors: bool | Sequence[str] = False,
    score_threshold: float | None = None,
    lookup_from: types.LookupLocation | None = None,
    consistency: types.ReadConsistency | None = None,
    shard_key_selector: types.ShardKeySelector | None = None,
    timeout: int | None = None,
    **kwargs: Any,
) -> types.QueryResponse:
```
Source: `.venv/lib/python3.13/site-packages/qdrant_client/qdrant_client.py:269-296`
(installed `qdrant-client==1.18.0`, confirmed via
`qdrant_client-1.18.0.dist-info`).

- Passing a **raw dense vector** as `query=[...]` runs nearest-neighbour (ANN)
  search: the docstring says *"If `list[float]` - use as a dense vector for
  nearest search."* (`qdrant_client.py:305`). This is exactly our case — we
  supply `embedder.embed([query])[0]`.
- The relevant params for search are `collection_name`, `query` (the vector),
  `query_filter`, `limit`, `offset`, `with_payload`, `with_vectors`,
  `score_threshold`. There is **no** separate `query_vector`/`vector` kwarg — the
  vector goes in `query`.

**`search()` is removed, not just deprecated.** The public `QdrantClient` in 1.18
exposes only these query-family methods (grep of `qdrant_client.py`):
`query_batch_points` (225), `query_points` (269), `query_points_groups` (442),
`scroll` (705), `retrieve` (1078) — plus `search_matrix_pairs`/`search_matrix_offsets`
(distance-matrix helpers, unrelated). There is **no `def search(`** on the client
class; `search` survives only internally in the local engine
(`qdrant_client/local/local_collection.py:535`, `local/qdrant_local.py`), which
`query_points` delegates to. So new code must call `query_points`; there is no
`search` to reach for.
Source: `.venv/.../qdrant_client/qdrant_client.py` (method inventory);
`.venv/.../qdrant_client/local/local_collection.py:535`.

**Return type.** `query_points` returns a `QueryResponse`:

```python
class QueryResponse(BaseModel):
    points: List["ScoredPoint"]
```
```python
class ScoredPoint(BaseModel):
    id: "ExtendedPointId"      # our point id — the UUID string we upserted
    version: int
    score: float               # similarity to the query vector
    payload: Optional["Payload"] = None
    vector: Optional["VectorStructOutput"] = None
    shard_key: Optional["ShardKey"] = None
    order_value: Optional["OrderValue"] = None
```
Source: `.venv/.../qdrant_client/http/models/models.py:2412-2413` (QueryResponse)
and `:2781-2792` (ScoredPoint). Note the result list is `response.points`, **not**
the response itself — iterate `query_points(...).points`.

The API-reference page mirrors this: *"Universally query points. This endpoint
covers all capabilities of search, recommend, discover, filters. But also enables
hybrid and multi-stage queries."* and documents `limit`, `offset`, `filter`,
`params`, `score_threshold`, `with_payload`, `with_vector`, returning scored
points with `id`, `score`, `payload`, `version`, `vector`.
Source: <https://api.qdrant.tech/api-reference/search/query-points>.

## 2. Local (embedded) mode parity

`build_vector_store` uses embedded mode for the CLI: `QdrantClient(path=str(settings.qdrant_path))`
(`packages/core/src/booksmart_core/vectors.py:104-111`). Local mode supports
`query_points` with the same parameters as server mode — the local client's
`query_points` takes `collection_name, query, using, prefetch, query_filter,
search_params, limit, offset, with_payload, with_vectors, score_threshold,
lookup_from` and forwards `query_filter`, `score_threshold`, `limit`, `offset`,
`with_payload`, `with_vectors` straight into the in-process collection.
Source: `.venv/.../qdrant_client/local/qdrant_local.py:391-427`.

Documented local-mode limitations relevant to search:

- **Brute-force, not HNSW; small-scale.** Local mode is recommended for
  prototyping / small collections (~20k points) and does an exact scan rather
  than approximate HNSW search. For our corpus sizes this is fine and actually
  gives exact results. Source: Qdrant local quickstart /
  `qdrant_client.local.qdrant_local` docs
  (<https://python-client.qdrant.tech/qdrant_client.local.qdrant_local>,
  <https://qdrant.tech/documentation/quickstart/>).
- **Single-process.** The on-disk `path=` directory is locked to one process at
  a time; concurrent opens raise. The CLI already accounts for this by calling
  `vector_store.client.close()` after a run
  (`packages/cli/src/booksmart_cli/runtime.py:116-119`) — a search command must
  likewise open, query, and close (or reuse one client per invocation).
- **Filtering and cosine scoring behave identically** to server mode (see §3, §4);
  `score_threshold` and `Filter` are honoured by the local engine.

Not separately verified against a primary source: whether local mode supports
quantization or named vectors identically. We use neither (single unnamed dense
vector, `Distance.COSINE`, no quantization — `vectors.py:_ensure_collection`), so
it does not affect this feature.

## 3. Filtering by `book_id` and `record_type`

Payload keys written per point are `record_type`, `record_id`, `book_id`, `text`
(`packages/core/src/booksmart_core/stages.py:403-408`), with
`record_type ∈ {"chapter", "section", "knowledge_object"}`
(`stages.py:57` `RecordType = Literal[...]`, and `:366`, `:369`, `:374`).

Build the filter with the same `qmodels` classes the write path already uses
(`vectors.py:79-98`). Single-value match on `book_id`:

```python
from qdrant_client import models as qmodels

qmodels.Filter(
    must=[qmodels.FieldCondition(key="book_id", match=qmodels.MatchValue(value=book_id))]
)
```
`FieldCondition`, `Filter`, `MatchValue`, and `MatchAny` all exist in 1.18:
`.venv/.../qdrant_client/http/models/models.py:991` (FieldCondition), `:1015`
(Filter), `:1773` (MatchValue), `:1731` (MatchAny).

- **Multiple conditions (AND):** put several `FieldCondition`s in `must=[...]` —
  e.g. `book_id == X` **and** `record_type == "section"`. `Filter` also takes
  `should` (OR) and `must_not` (the write path uses `must_not=[HasIdCondition(...)]`).
- **Several record types at once (OR over one field):** use `MatchAny`:
  ```python
  qmodels.FieldCondition(
      key="record_type",
      match=qmodels.MatchAny(any=["chapter", "section"]),
  )
  ```
  `MatchAny(any=[...])` matches a payload value against any element of the list —
  cleaner than several `should` conditions.
  Source: `.venv/.../qdrant_client/http/models/models.py:1731` (MatchAny).

Pass the assembled filter as `query_filter=` to `query_points`. `search <book>`
adds the `book_id` condition; `search all` omits it. An optional
`--type`/`record_types` argument adds a `MatchAny` condition on `record_type`.

## 4. Score semantics under COSINE

The collection is created with `distance=qmodels.Distance.COSINE`
(`vectors.py:48-49`). For cosine, **higher score = more similar**, and results
come back sorted by score descending:

- `distance_to_order(Distance.Cosine)` returns `DistanceOrder.BIGGER_IS_BETTER`.
  Source: `.venv/.../qdrant_client/local/distances.py:105-118`.
- The engine sorts descending for `BIGGER_IS_BETTER`
  (`np.argsort(scores)[::-1]`). Source:
  `.venv/.../qdrant_client/local/local_collection.py:655-670`.

**`score_threshold` keeps points at/above the threshold** under cosine. Local
engine logic: for `BIGGER_IS_BETTER` it stops as soon as `score < score_threshold`
(i.e. it retains scores `>= score_threshold`).
Source: `.venv/.../qdrant_client/local/local_collection.py:686-692`. The
`query_points` docstring says the same, version-aware: *"Score of the returned
result might be higher or smaller than the threshold depending on the Distance
function used. E.g. for cosine similarity only higher scores will be returned."*
Source: `qdrant_client.py:333-338`. So a caller-supplied `score_threshold` is a
minimum similarity; leave it `None` to get the top-`limit` unconditionally.

## 5. Embedding the query with the LOCKED model

The collection stores its locked model in **collection metadata** under
`EMBEDDING_MODEL_KEY = "embedding_model"`, written at create time via the
`metadata=` kwarg (`vectors.py:27`, `:45-51`). In 1.18 this is a first-class,
persisted field:

- `create_collection(..., metadata=...)` is accepted (local impl:
  `.venv/.../qdrant_client/local/qdrant_local.py:780-801`; the model is
  `CreateCollection.metadata`, `http/models/models.py:573`).
- **Read it back** via `get_collection(name).config.metadata` — `CollectionConfig`
  has a real `metadata: Optional[Payload]` field described as *"Arbitrary JSON
  metadata for the collection ... application-specific information such as ...
  inference model info"*. Source:
  `.venv/.../qdrant_client/http/models/models.py:258-278` (`CollectionConfig.metadata`).
  This is exactly what `_ensure_collection` already does on the write path:
  `metadata = self.client.get_collection(self.collection).config.metadata or {}`
  then `metadata.get(EMBEDDING_MODEL_KEY)` (`vectors.py:53-54`).

**The search path must reproduce ADR 0001's lock as a read check** before it
embeds anything: read the locked model, compare to the configured
`embedder.model`, and raise loudly on mismatch or on a pre-lock collection with no
recorded model (same two failure branches as `_ensure_collection`,
`vectors.py:55-71`, raising `ProviderConfigError`). Rationale: a query embedded by
a different model than the stored vectors is silently incomparable — the exact
degradation ADR 0001 forbids. Source: `docs/adr/0001-model-locked-vector-collection.md`.

The single-query embedding call is `embedder.embed([query])[0]` — `embed` takes a
list and returns a list of vectors (`EmbeddingProvider` Protocol,
`packages/core/src/booksmart_core/llm.py:317-321`). One query is well under any
`max_batch`, so no batching is needed for search.

**Suggested helper on `VectorStore`** (keeps the metadata read in one place,
next to `_ensure_collection`):

```python
def locked_model(self) -> str:
    """The embedding model this collection is locked to (ADR 0001)."""
    if not self.client.collection_exists(self.collection):
        raise ProviderConfigError(f"vector collection {self.collection!r} does not exist")
    metadata = self.client.get_collection(self.collection).config.metadata or {}
    locked = metadata.get(EMBEDDING_MODEL_KEY)
    if locked is None:
        raise ProviderConfigError(
            f"vector collection {self.collection!r} predates model locking (ADR 0001)"
        )
    return locked
```

## 6. Resolving hits back to relational rows

Each `ScoredPoint.payload` carries `record_type` and `record_id` (the relational
row id, a UUID **string** — written as `str(row.id)`, `stages.py:405`). The three
types map to three ORM classes (`packages/core/src/booksmart_core/models.py`):

| `record_type`      | ORM class        | model.py anchor |
|--------------------|------------------|-----------------|
| `chapter`          | `Chapter`        | `models.py:56`  |
| `section`          | `Section`        | `models.py:84`  |
| `knowledge_object` | `KnowledgeObject`| `models.py:120` |

Resolution preserving Qdrant's rank/score order:

```python
_MODELS = {"chapter": Chapter, "section": Section, "knowledge_object": KnowledgeObject}

hits: list[SearchHit] = []
for point in response.points:                    # already score-descending (§4)
    payload = point.payload or {}
    record_type = payload["record_type"]
    record_id = uuid.UUID(payload["record_id"])  # stored as a UUID string
    row = session.get(_MODELS[record_type], record_id)
    if row is None:
        continue   # vector outlived its row (see §8); skip rather than fail
    hits.append(SearchHit(score=point.score, record_type=record_type,
                          record_id=record_id, row=row))
```

`session.get(Model, id)` is a primary-key fetch. Iterating `response.points` in
order keeps Qdrant's ranking. Because payload already carries `text`, a light
projection (title + snippet) can be built without the row if a consumer wants to
avoid hydrating ORM objects — but resolving to rows needs the Session (see §7).

## 7. Recommended core API shape (ADR 0002-compatible)

ADR 0002: core is a library of plain functions taking serializable inputs; the
**consumer** owns the Session/engine and supplies providers
(`docs/adr/0002-stages-are-the-unit-of-durability.md`; mirrored by the CLI's
`Runtime` wiring `session_factory` + `build_vector_store` + `build_embedding_provider`,
`packages/cli/src/booksmart_cli/runtime.py`). Search is a **pure read** — no
writes, no `session.commit()`.

```python
# packages/core/src/booksmart_core/search.py  (new module)
import uuid
from dataclasses import dataclass
from typing import Literal, Sequence

from sqlalchemy.orm import Session

from booksmart_core.llm import EmbeddingProvider
from booksmart_core.models import Base
from booksmart_core.vectors import VectorStore

RecordType = Literal["chapter", "section", "knowledge_object"]


@dataclass(frozen=True)
class SearchHit:
    score: float                     # cosine similarity, higher = closer (§4)
    record_type: RecordType
    record_id: uuid.UUID
    book_id: uuid.UUID
    text: str                        # the embedded text, straight from payload
    row: Base                        # hydrated Chapter | Section | KnowledgeObject


def search(
    session: Session,
    vector_store: VectorStore,
    embedder: EmbeddingProvider,
    query: str,
    *,
    book_id: uuid.UUID | None = None,          # None -> search all books
    record_types: Sequence[RecordType] | None = None,   # None -> all types
    limit: int = 10,
    score_threshold: float | None = None,
    offset: int = 0,                            # pagination (§8)
) -> list[SearchHit]:
    ...
```

Where each responsibility lives:

- **Model-lock check + query embedding live inside `search` (or a thin
  `VectorStore.search` it calls).** Read the locked model (§5), assert it equals
  `embedder.model`, raise `ProviderConfigError` on mismatch — *before* the embed
  call, so a misconfigured embedder never even hits its API. Then
  `vector = embedder.embed([query])[0]`.
- **Filter construction** (§3) from `book_id` / `record_types`.
- **`query_points`** call with `query=vector, query_filter=..., limit=limit,
  offset=offset, score_threshold=score_threshold` (`with_payload` defaults to
  `True`, §8 — payload is required to resolve rows).
- **Resolution** (§6) needs the Session — that is why `session` is the first
  parameter. Yes, resolution genuinely requires it: the hydrated row (title,
  summary, content) lives only in Postgres/SQLite; Qdrant holds only the id +
  `text`.

This composes cleanly for all three consumers: the CLI opens a session and builds
`vector_store`/`embedder` from `Settings` (it already has the builders) and calls
`search`; a server consumer does the same inside a request handler and serialises
`list[SearchHit]` to JSON. Neither needs a Runner — search is not a Stage (no
durability, no Run record), so ADR 0002's stage machinery does not apply; it is
just a read function that happens to touch the vector store.

## 8. Empty / edge cases

- **Collection missing.** `query_points` on a non-existent collection raises. Gate
  with `client.collection_exists(name)` (as the write path does,
  `vectors.py:44`, `:95`) and return `[]` (or raise a typed "not embedded yet"
  error) rather than surfacing a raw Qdrant exception. The `locked_model()` helper
  (§5) already raises a clear `ProviderConfigError` for a missing collection.
- **Collection empty / book with no embeddings.** A valid `query_points` against
  an empty collection (or with a `book_id` filter matching nothing) simply returns
  `QueryResponse(points=[])` → `search` returns `[]`. No special-casing needed.
- **`with_payload` default.** In 1.18 `with_payload` **defaults to `True`**
  (`qdrant_client.py:288`), so payload comes back without asking. We rely on it to
  read `record_type`/`record_id`/`book_id`/`text`, so either leave the default or
  set `with_payload=True` explicitly for clarity. We do **not** need
  `with_vectors` (defaults `False`) — the stored vector is not useful to a caller.
- **Pagination.** Use `offset` (int) alongside `limit`; the docstring warns
  *"large offset values may cause performance issues"* (`qdrant_client.py:318-321`).
  For a first cut, `limit` alone is enough; expose `offset` only if the CLI/api
  wants paging.
- **Dangling vectors (row deleted).** Structure/knowledge stages replace rows
  wholesale, and `replace_book_points` prunes stale points, but a vector could
  briefly outlive its row between stages/runs. Resolution should **skip** a hit
  whose `session.get(...)` is `None` rather than fail the whole query (see §6).
- **Score ordering.** Do not re-sort — `query_points` already returns
  score-descending under cosine (§4); preserve that order into `SearchHit`s.

---

## Open decisions for the implementer

1. **Where the query-embed + lock check physically live** — a new
   `VectorStore.search(vector, *, filter, limit, ...)` that takes an *already
   embedded* vector (keeping `VectorStore` free of the embedder dependency, and
   the lock check in `search.py`), versus a `search()` free function that owns
   both. Leaning: keep `VectorStore` embedder-free; do the embed + lock in
   `search.py`, add only a small `VectorStore.locked_model()` read helper.
2. **Return shape granularity** — hydrated ORM `row` (rich, but couples core-search
   to session lifetime / detachment like `reads.py` does with `expunge`) versus a
   flat projection built from payload `text` + a few row columns. The CLI's
   `reads.py` pattern (detached rows / plain dicts) is the precedent to match for
   the CLI; the api may prefer a dict.
3. **CLI surface** — `booksmart search <book|all> "<query>"`: how to accept
   "all" vs a book id/selector, whether to expose `--type`, `--limit`,
   `--score-threshold`, and how to render mixed chapter/section/knowledge hits.
   The command should build `vector_store`/`embedder` from `Settings` and
   `close()` the embedded client afterwards (single-process lock, §2).
4. **Error taxonomy** — reuse `ProviderConfigError` (model-lock mismatch, missing
   collection) from `booksmart_core.errors`; decide whether "book has no
   embeddings" is an empty result or a typed CLI error (`reads.py` raises
   `BookNotFoundError` for a missing book — a parallel `NotEmbeddedError` may fit).
5. **Score exposure** — whether to surface the raw cosine score to users, apply a
   default `score_threshold`, or leave both to the caller. Recommend: no default
   threshold, expose the score, let the CLI decide presentation.
6. **Server-mode note** — a server consumer runs against a Qdrant *server*
   (`qdrant_url`), where search is HNSW/approximate; results may differ slightly
   from the CLI's exact local scan. Not a correctness issue, but worth a test that
   both paths return the same top hit for a fixed fixture.

### Not verified against a primary source

- Exact wording of the Qdrant *concepts/search* prose page (`qdrant.tech/documentation/concepts/search/`)
  could not be fetched during this research; the score/threshold/ordering claims
  are instead grounded in the installed 1.18 source and the API reference
  (<https://api.qdrant.tech/api-reference/search/query-points>), which are more
  authoritative for the pinned version's behaviour.
- Local-mode quantization / named-vector parity was not confirmed; this feature
  uses neither, so it is out of scope.
