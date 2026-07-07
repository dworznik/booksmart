# booksmart

The Booksmart CLI — turn books into queryable knowledge, locally.

A single-user front end over [`booksmart-core`](https://pypi.org/project/booksmart-core/):
register PDFs/EPUBs, ingest them through the parsing → structure → profile →
extraction → summaries → embeddings pipeline, and browse the results. Everything
runs against an auto-migrated SQLite file and embedded Qdrant under
`~/.booksmart/` — no Docker, no Postgres, no server.

```console
$ uv tool install booksmart
$ booksmart add ./clean-code.pdf --title "Clean Code" --author "Robert C. Martin"
$ booksmart ingest <book-id>
$ booksmart structure <book-id>
$ booksmart knowledge list <book-id>
```

## Commands

`add`, `ingest`, `books list/show/update`, `runs list/show`, `structure`,
`profile`, `knowledge list/show`.

## Configuration

Providers and locations come from `BOOKSMART_*` environment variables (e.g.
`BOOKSMART_LLM_PROVIDER`, `BOOKSMART_HOME`). Set `BOOKSMART_QDRANT_URL` to use a
Qdrant server instead of the embedded on-disk store.
