# Booksmart

Turn books into queryable knowledge, locally. Booksmart ingests PDFs and EPUBs
through a parsing → structure → profile → extraction → summaries → embeddings
pipeline, then lets you search the result in plain language.

## Packages

This is a [uv](https://docs.astral.sh/uv/) workspace holding two published
distributions:

| Package | Import | What it is |
| --- | --- | --- |
| [`booksmart-core`](packages/core) | `booksmart_core` | The ingestion pipeline as a typed library: Stage functions a consumer drives with its own Runner. |
| [`booksmart`](packages/cli) | `booksmart_cli` | The local CLI over core — SQLite and embedded Qdrant, no Docker, no server. |

A third consumer, `booksmart-api` (an Inngest-backed service on Postgres), lives
in its own repository and builds on `booksmart-core`.

## Install

Install the CLI — see [`packages/cli/README.md`](packages/cli/README.md) for the
current instructions, including the pre-release install from source.

## Development

```console
$ uv sync          # editable install of both packages
$ uv run pytest    # the whole suite, SQLite by default
$ uv run mypy      # strict, both packages
```

The core suite also runs against Postgres in CI; point
`BOOKSMART_TEST_DATABASE_URL` at a database to do the same locally.

## Design

`CONTEXT.md` holds the vocabulary (Stage, Scope, Run, Runner). The load-bearing
decisions are recorded as ADRs in [`docs/adr/`](docs/adr): stages are the unit of
durability, and the vector collection is locked to one embedding model.
