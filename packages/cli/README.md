# booksmart

The Booksmart CLI — turn books into queryable knowledge, locally.

A single-user front end over
[`booksmart-core`](https://github.com/dworznik/booksmart/tree/main/packages/core):
register PDFs/EPUBs, ingest them through the parsing → structure → profile →
extraction → summaries → embeddings pipeline, and browse the results. Everything
runs against an auto-migrated SQLite file and embedded Qdrant under
`~/.booksmart/` — no Docker, no Postgres, no server.

## Install

**Not published to PyPI yet** — install from the repo, which is private, so this
needs an SSH key with access to it. `uv` clones the repo, resolves
`booksmart-core` from the same checkout, and puts a `booksmart` command on your
PATH:

```console
$ uv tool install "git+ssh://git@github.com/dworznik/booksmart.git#subdirectory=packages/cli"
```

Over HTTPS instead, with `gh auth login` (or a `repo`-scoped token) supplying git
credentials:

```console
$ uv tool install "git+https://github.com/dworznik/booksmart.git#subdirectory=packages/cli"
```

Re-run with `--force` to pick up new commits. Once the first `cli-v*` tag ships,
this becomes `uv tool install booksmart`, which pulls `booksmart-core` down as a
dependency.

## Quickstart

```console
$ booksmart add ./clean-code.pdf --title "Clean Code" --author "Robert C. Martin"
$ booksmart ingest <book-id>
$ booksmart structure <book-id>
$ booksmart knowledge list <book-id>
$ booksmart search all "how do deep modules reduce complexity"
```

`ingest` calls an LLM and an embedding provider, so it needs credentials —
`ANTHROPIC_API_KEY` and `OPENAI_API_KEY` by default. To drive the whole pipeline
with no keys, no network and no cost, select the deterministic fake providers:

```console
$ BOOKSMART_LLM_PROVIDER=fake BOOKSMART_EMBEDDING_PROVIDER=fake booksmart ingest <book-id>
```

## Commands

`add`, `ingest`, `books list/show/update`, `runs list/show`, `structure`,
`profile`, `knowledge list/show`, `search`.

### Search

`booksmart search <book-id|all> "<query>"` ranks the chapters, sections and
knowledge objects most similar to a natural-language query, over the embeddings
an ingest produced. Restrict it with `--type` (repeatable: `chapter`, `section`,
`knowledge_object`), cap it with `--limit`, and drop weak hits with
`--score-threshold` (a cosine similarity, `-1`–`1`).

The query is embedded with the model the vector collection is locked to; if that
is not the currently configured embedding model, search refuses rather than
return plausible, silently wrong rankings (ADR 0001).

## Configuration

Providers and locations come from `BOOKSMART_*` environment variables (e.g.
`BOOKSMART_LLM_PROVIDER`, `BOOKSMART_HOME`). Set `BOOKSMART_QDRANT_URL` to use a
Qdrant server instead of the embedded on-disk store.
