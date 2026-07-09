# Booksmart

Turn books into queryable knowledge, locally. Booksmart ingests PDFs and EPUBs
through a parsing → structure → profile → extraction → summaries → embeddings
pipeline, then lets you search the result in plain language.

## Packages

This is a [uv](https://docs.astral.sh/uv/) workspace holding two published
distributions:

| Package | Import | What it is |
| --- | --- | --- |
| [`booksmart-core`](https://pypi.org/project/booksmart-core/) ([source](packages/core)) | `booksmart_core` | The ingestion pipeline as a typed library: Stage functions a consumer drives with its own Runner. |
| [`booksmart`](https://pypi.org/project/booksmart/) ([source](packages/cli)) | `booksmart_cli` | The local CLI over core — SQLite and embedded Qdrant, no Docker, no server. |

A third consumer, `booksmart-api` (an Inngest-backed service on Postgres), lives
in its own repository and builds on `booksmart-core`.

## Install

Python 3.12+, with `uv tool install` or `pipx install`:

```console
$ uv tool install booksmart        # or: pipx install booksmart
$ booksmart config set anthropic_api_key   # hidden prompt; persists in ~/.booksmart
$ booksmart config set openai_api_key      # embeddings
$ booksmart add ./clean-code.pdf --title "Clean Code" --author "Robert C. Martin"
$ booksmart ingest <book-id>
$ booksmart search all "how do deep modules reduce complexity"
```

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

## License

MIT — see [`LICENSE`](LICENSE), or <https://dworznik.mit-license.org>.
