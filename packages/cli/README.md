# booksmart

The Booksmart CLI — turn books into queryable knowledge, locally.

A single-user front end over
[`booksmart-core`](https://pypi.org/project/booksmart-core/): register PDFs/EPUBs,
ingest them through the parsing → structure → profile → extraction → summaries →
embeddings pipeline, and browse the results. Everything runs against an
auto-migrated SQLite file and embedded Qdrant under `~/.booksmart/` — no Docker,
no Postgres, no server.

## Install

Needs Python 3.12 or newer. `booksmart` is a command-line tool, so install it
into its own environment rather than into a project's:

```console
$ uv tool install booksmart
$ pipx install booksmart
```

Either puts a `booksmart` command on your PATH. Plain `pip install booksmart`
also works if you would rather have it in the current environment.
`booksmart-core` comes along as a dependency in every case.

## Quickstart

`ingest` calls an LLM and an embedding provider, so it needs credentials —
an Anthropic key (LLM) and an OpenAI key (embeddings) by default. Set them
once; they persist in `~/.booksmart/config.toml`:

```console
$ booksmart config set anthropic_api_key   # hidden prompt, or pipe the key in
$ booksmart config set openai_api_key
$ booksmart add ./clean-code.pdf --title "Clean Code" --author "Robert C. Martin"
$ booksmart ingest <book-id>
$ booksmart structure <book-id>
$ booksmart knowledge list <book-id>
$ booksmart search all "how do deep modules reduce complexity"
```

To drive the whole pipeline with no keys, no network and no cost, select the
deterministic fake providers:

```console
$ BOOKSMART_LLM_PROVIDER=fake BOOKSMART_EMBEDDING_PROVIDER=fake booksmart ingest <book-id>
```

## Commands

`add`, `ingest`, `books list/show/update`, `runs list/show`, `structure`,
`profile`, `knowledge list/show`, `search`, `config set/get/unset/list`.

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

Any setting — provider, model, API keys, locations — can be persisted with
`booksmart config set <field> [value]` (omit the value to enter it via hidden
prompt or piped stdin, keeping keys out of shell history). Values live in
`~/.booksmart/config.toml`, created `0600` and safe to hand-edit.

Each setting resolves through one chain, highest first:

1. `BOOKSMART_*` environment variables (e.g. `BOOKSMART_LLM_PROVIDER`) —
   explicit targeting for scripts and one-offs.
2. `config.toml` — what `config set` writes.
3. The vendors' conventional variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
   `GEMINI_API_KEY`) — API keys only, so an already-exported key just works.
4. Defaults (SQLite, `storage/` and embedded Qdrant under `~/.booksmart`).

`booksmart config list` shows every field's effective value and which layer it
came from. Set `BOOKSMART_QDRANT_URL` (or `config set qdrant_url ...`) to use a
Qdrant server instead of the embedded on-disk store; `BOOKSMART_HOME` moves the
whole installation.
