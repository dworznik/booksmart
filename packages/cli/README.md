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
$ booksmart search all "how do deep modules reduce complexity"
```

## Commands

`add`, `ingest`, `books list/show/update`, `runs list/show`, `structure`,
`profile`, `knowledge list/show`, `search`.

### Search

`booksmart search <book-id|all> "<query>"` ranks the chapters, sections and
knowledge objects most similar to a natural-language query, over the embeddings
an ingest produced. Restrict it with `--type` (repeatable: `chapter`, `section`,
`knowledge_object`), cap it with `--limit`, and drop weak hits with
`--score-threshold` (a cosine similarity, `0`–`1`).

The query is embedded with the model the vector collection is locked to; if that
is not the currently configured embedding model, search refuses rather than
return plausible, silently wrong rankings (ADR 0001).

## Configuration

Providers and locations come from `BOOKSMART_*` environment variables (e.g.
`BOOKSMART_LLM_PROVIDER`, `BOOKSMART_HOME`). Set `BOOKSMART_QDRANT_URL` to use a
Qdrant server instead of the embedded on-disk store.
