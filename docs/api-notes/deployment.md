# Server deployment & compose e2e (removed)

The `Dockerfile` and `docker-compose.yml` existed only to run the FastAPI server
(and, in CI, to smoke-test it end to end). With the server gone they were
removed. Originals:
[`Dockerfile`](https://github.com/dworznik/booksmart/blob/589857c/Dockerfile),
[`docker-compose.yml`](https://github.com/dworznik/booksmart/blob/589857c/docker-compose.yml),
[compose-e2e CI job](https://github.com/dworznik/booksmart/blob/589857c/.github/workflows/ci.yml).

## The deployment shape (for booksmart-api)

- Base `python:3.12-slim` + `tesseract-ocr` (the OCR fallback parser and
  pymupdf4llm's integrated OCR both need it).
- Startup ran migrations then the server:
  `sh -c "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"`.
  `booksmart-api` runs core's single alembic history (`packages/core/migrations`,
  a second alembic environment layers its api-only tables — see #22 decision 6)
  then boots its own ASGI app.
- Infra: Postgres 17 + Qdrant `v1.15.5` as sibling services; storage on a mounted
  volume; providers configured via `BOOKSMART_*` env.

## The compose e2e smoke test (for booksmart-api / CLI e2e)

Fake providers (`BOOKSMART_LLM_PROVIDER=fake`, `BOOKSMART_EMBEDDING_PROVIDER=fake`)
drove the full pipeline through real containers with no keys, network, or cost:
register a generated PDF → `POST /ingest` → poll `/jobs/{id}` until `succeeded`.
The `fake` providers still ship in `booksmart_core.fakes`, so both consumers can
reproduce this: the CLI's e2e (fake provider, sqlite, embedded Qdrant) replaces
it for the CLI; the API repeats the container smoke test against its own service.
