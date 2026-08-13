# booksmart

Turn books into queryable knowledge, on your own machine.

Point it at a PDF or an EPUB. It reads the book, works out its chapters and
sections, extracts the ideas it teaches, summarises them, and makes the whole
thing searchable in plain language — so you can ask *"how do deep modules reduce
complexity?"* and get the passage that answers it, with the chapter it came from.

Everything stays local: an SQLite file and an embedded vector store under
`~/.booksmart/`. No Docker, no Postgres, no server, nothing to run alongside it.

## Install

Python 3.12 or newer. `booksmart` is a command-line tool, so install it into its
own environment rather than a project's:

```console
$ uv tool install booksmart
$ pipx install booksmart
```

Either puts a `booksmart` command on your PATH. Plain `pip install booksmart`
works too if you would rather have it in the current environment.

## Quickstart

Reading a book calls an LLM and an embedding provider, so it needs credentials —
by default an Anthropic key for the language model and an OpenAI key for
embeddings. Set them once; they persist:

```console
$ booksmart config set anthropic_api_key   # hidden prompt, or pipe the key in
$ booksmart config set openai_api_key
$ booksmart add ./a-book.pdf --title "A Book About Software" --author "A. N. Author"
$ booksmart ingest <book-id>
$ booksmart search all "how do deep modules reduce complexity"
```

Then look around: `booksmart structure <book-id>` for the chapter tree,
`booksmart knowledge list <book-id>` for the ideas it pulled out, and
`booksmart books list` for what you have.

Want to try the whole pipeline with no keys, no network and no cost? Use the
built-in fake providers:

```console
$ BOOKSMART_LLM_PROVIDER=fake BOOKSMART_EMBEDDING_PROVIDER=fake booksmart ingest <book-id>
```

## Searching

```console
$ booksmart search <book-id|all> "<query>"
```

Search matches two ways at once — by **meaning**, so a question finds a passage
phrased differently, and by **words**, so an exact term or a proper noun still
ranks even if the model never learned it. Results fuse both.

- `--dense-only` — match by meaning alone.
- `--type` — restrict to `chapter`, `section` or `knowledge_object`; repeatable.
- `--limit` — how many hits.
- `--score-threshold` — drop weak hits by similarity.

The `score` column means different things in the two modes, so the output says
which you are looking at. With `--dense-only` it is a similarity between -1 and
1. Fused, it measures how strongly both kinds of matching agreed, and only
compares within one set of results — a fused `0.583` can be the best hit there
is.

**Changing your embedding model means re-reading your books.** Search refuses to
run against a collection built with a different model rather than return
plausible, silently wrong rankings — and tells you so.

## Commands

`add`, `ingest`, `books list/show/update`, `runs list/show`, `structure`,
`profile`, `knowledge list/show`, `search`, `config set/get/unset/list`.

## Configuration

Any setting — provider, model, API keys, locations — persists with
`booksmart config set <field> [value]`. Omit the value to enter it at a hidden
prompt or pipe it in, which keeps keys out of your shell history. Values live in
`~/.booksmart/config.toml`, created `0600` and safe to hand-edit.

Settings resolve through one chain, highest wins:

1. `BOOKSMART_*` environment variables (e.g. `BOOKSMART_LLM_PROVIDER`) — for
   scripts and one-offs.
2. `config.toml` — what `config set` writes.
3. The vendors' own variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
   `GEMINI_API_KEY`) — API keys only, so an already-exported key just works.
4. Defaults.

`booksmart config list` shows every field's effective value and which layer it
came from. `BOOKSMART_QDRANT_URL` points at a Qdrant server instead of the
embedded store; `BOOKSMART_HOME` moves the whole installation.

Built on [`booksmart-core`](https://pypi.org/project/booksmart-core/), which is
the same pipeline as a library if you would rather call it than run it.
