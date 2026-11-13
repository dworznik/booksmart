# Booksmart

Booksmart turns books into queryable knowledge, locally. It ingests PDFs and
EPUBs through a pipeline of **Stages** — parsing, structure, profile,
extraction, summaries, embeddings — and then lets you search the result in plain
language.

## Two packages

| Package | Import | What it is |
| --- | --- | --- |
| [`booksmart`](https://pypi.org/project/booksmart/) | `booksmart_cli` | The local CLI: SQLite and embedded Qdrant, no Docker, no server. Start here if you want to search your own books. |
| [`booksmart-core`](https://pypi.org/project/booksmart-core/) | `booksmart_core` | The ingestion pipeline as a typed library: Stage functions a consumer drives with its own Runner. Start here if you are building on top of it. |

## Quick start

Booksmart needs Python 3.12 or newer. Install the CLI with `uv` or `pipx`:

```console
$ uv tool install booksmart        # or: pipx install booksmart
```

Give it the API keys it needs — Anthropic for extraction and summaries, OpenAI
for embeddings. Each is read from a hidden prompt and stored in `~/.booksmart`:

```console
$ booksmart config set anthropic_api_key
$ booksmart config set openai_api_key
```

Add a book, run the pipeline over it, then ask it something:

```console
$ booksmart add ./a-book.pdf --title "A Book About Software" --author "A. N. Author"
$ booksmart ingest <book-id>
$ booksmart search all "how do deep modules reduce complexity"
```

Search is hybrid by default: dense embeddings and BM25 together.

## How it fits together

- A **Stage** is one unit of the pipeline and the unit of durability: a
  completed Stage is permanent, and a failed one never undoes the Stages before
  it.
- A **Scope** is the set of Stages one execution covers — the full pipeline, or
  an incremental slice that reuses upstream output.
- A **Run** records one execution over a book: its Scope, outcome, models and
  token spend.
- A **Runner** executes Stages in order and owns the Run. The CLI brings its
  own; a service built on `booksmart-core` would bring another.

## Where to go next

- **Source and issues** — [github.com/dworznik/booksmart](https://github.com/dworznik/booksmart).
- **The CLI on PyPI** — [`booksmart`](https://pypi.org/project/booksmart/).
- **The library on PyPI** — [`booksmart-core`](https://pypi.org/project/booksmart-core/).
- **Vocabulary** — [`CONTEXT.md`](https://github.com/dworznik/booksmart/blob/main/CONTEXT.md)
  defines Stage, Scope, Run and Runner precisely.
- **Design decisions** — the architectural decision records live in
  [`docs/adr/`](https://github.com/dworznik/booksmart/tree/main/docs/adr).
