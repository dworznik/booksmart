# HTTP surface (removed)

The full REST surface `booksmart-api` should recreate on top of `booksmart-core`.
Status codes and semantics below are the ones the FastAPI app served at
[`589857c`](https://github.com/dworznik/booksmart/tree/589857c/app).

## App wiring

[`app/main.py`](https://github.com/dworznik/booksmart/blob/589857c/app/main.py),
[`app/db.py`](https://github.com/dworznik/booksmart/blob/589857c/app/db.py)

`create_app(settings)` built one SQLAlchemy engine + `sessionmaker` + `BookStorage`
onto `app.state`, and `get_db` yielded a request-scoped `Session`. In core this
is now the consumer's job: core exposes `Settings`, the ORM models, `BookStorage`,
and `execute_run(session_factory, storage_root, book_id, scope)` — a Runner owns
session and engine lifecycle. Under Inngest each step opens its own session
(proven by `tests/test_stage_contract.py`), so there is no single app-scoped
session to model.

## Books

[`app/routers/books.py`](https://github.com/dworznik/booksmart/blob/589857c/app/routers/books.py)

| Method | Path | Success | Errors |
| --- | --- | --- | --- |
| POST | `/books` (multipart: file + form fields) | `201` `BookOut` | `415` unsupported/File-content mismatch; `409` `{detail: {message, existing_book_id}}` duplicate; `422` missing title/author |
| GET | `/books` | `200` `list[BookOut]` (by `uploaded_at, id`) | — |
| GET | `/books/{id}` | `200` `BookOut` | `404` |
| PATCH | `/books/{id}` | `200` `BookOut` | `404`; `422` on file-field or null-required (see [upload-validation.md](upload-validation.md)) |
| GET | `/books/{id}/structure` | `200` `list[ChapterOut]` | `404` |
| GET | `/books/{id}/profile` | `200` `BookProfileOut` (latest) | `404` book or no profile |

Registration also ran the byte-identical **dedup** (SHA-256 `file_hash`) and
stored the original with a rollback-on-DB-failure guard. Both are documented in
[upload-validation.md](upload-validation.md). The read shapes (`/structure`,
`/profile`) are reproduced for tests in `packages/core/tests/conftest.py`
(`book_structure`, `latest_profile`) — a useful reference for the query SQL.

## Runs

[`app/routers/jobs.py`](https://github.com/dworznik/booksmart/blob/589857c/app/routers/jobs.py)
(the router was renamed to speak of runs but kept `/jobs` URLs).

| Method | Path | Pre-#23 | Semantics |
| --- | --- | --- | --- |
| POST | `/books/{id}/ingest` | `202` queued job | trigger a full run |
| POST | `/books/{id}/reprocess` `{scope}` | `202` queued job | scoped re-run; `409` if an incremental scope has no prior succeeded run; `422` unknown scope; `404` unknown book |
| GET | `/books/{id}/jobs` | `200` history oldest-first | `404` unknown book |
| GET | `/jobs/{id}` | `200` `RunOut` | `404` |

**202 trigger flow (the shape for Inngest).** Originally the endpoint enqueued a
`queued` job and a polling worker claimed it — an async accept. ADR 0002 removed
the queued state, so post-#23 the endpoint ran the pipeline synchronously.
`booksmart-api` restores the async accept differently: the HTTP handler sends an
Inngest event and returns `202` with the run id immediately; the Inngest function
opens the run (`start_run`), wraps each stage in a `step.run`, and closes it
(`finalize_run`). Map `BooksmartError.retriable == False` to Inngest's
`NonRetriableError`.

**Reprocess guards live in the consumer now.** The `409` prior-success check was
`has_successful_run(session, book_id)` (still in `booksmart_core.runner`); the
`422` unknown-scope check was Pydantic's `Literal["profile","extraction","embeddings","full"]`
(`ReprocessScope`). Core no longer rejects an unknown scope up front — it records
a failed run (`StagePreconditionError: unknown scope`) — so the API should
validate the scope and prior-success before emitting the event to keep the
`409`/`422` contract.

**Run listing semantics.** History is every run for a book, `ORDER BY created_at, id`
(oldest first), failures included; created at execution start (no queued row).
Reproduced for tests as `runs_for_book` in `conftest.py`.

## Knowledge

[`app/routers/knowledge.py`](https://github.com/dworznik/booksmart/blob/589857c/app/routers/knowledge.py)

| Method | Path | Success | Errors |
| --- | --- | --- | --- |
| GET | `/books/{id}/knowledge-objects?type=` | `200` `list[KnowledgeObjectOut]` (by `created_at, id`) | `404` unknown book; `422` invalid `type` |
| GET | `/knowledge-objects/{id}` | `200` `KnowledgeObjectOut` | `404` |

`type` was validated against the `KnowledgeType` `Literal` (still exported from
`booksmart_core.extraction`). Read shape reproduced as `knowledge_objects` in
`conftest.py`.

## Schemas

[`app/schemas.py`](https://github.com/dworznik/booksmart/blob/589857c/app/schemas.py)

`BookOut`, `BookUpdate` (partial update with `extra="forbid"` and a null-required
validator), `ChapterOut`/`SectionOut`, `BookProfileOut`, `KnowledgeObjectOut`,
`RunOut`, `ReprocessRequest`/`ReprocessScope`. All derive cleanly from the core
ORM models (`ConfigDict(from_attributes=True)`), so a consumer regenerates them
against `booksmart_core.models` with no core change. `BookUpdate`'s partial-update
+ file-field-rejection rules are the only non-trivial bit — see
[upload-validation.md](upload-validation.md#metadata-updates).
