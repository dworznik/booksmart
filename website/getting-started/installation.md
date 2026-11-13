# Installation

The `booksmart` command line tool is published to PyPI as
[`booksmart`](https://pypi.org/project/booksmart/). It needs **Python 3.12 or
newer**.

## Install the CLI

`booksmart` is a tool, not a library you import, so install it into an
environment of its own. Either of these puts a `booksmart` command on your
`PATH`:

```console
$ uv tool install booksmart
```

```console
$ pipx install booksmart
```

`pip install booksmart` also works, if you would rather have the CLI inside an
environment you already manage.

Check that it is there:

```console
$ booksmart --help
```

You should see the command list: `add`, `ingest`, `structure`, `profile`,
`search`, and the `books`, `runs`, `knowledge` and `config` groups.

## What comes with it

Nothing else needs to run alongside the CLI. Everything it stores lives on your
machine, under `~/.booksmart/`:

| Path | Holds |
| --- | --- |
| `config.toml` | Your persisted configuration, written by `booksmart config set`. |
| `booksmart.db` | An SQLite database: books, Runs, structure, knowledge. |
| `storage/` | The book files you have added. |
| `qdrant/` | An embedded Qdrant vector store, for search. |

The database creates and migrates itself on first use. There is no Docker, no
Postgres and no server to start.

Two things are fetched over the network the first time they are needed:

- **API calls** to the model Providers you configure, during an ingest and a
  search. The [Quickstart](quickstart.md) sets these up.
- **The BM25 model** for keyword search, downloaded once on the first ingest
  and cached after that.

## Optional: OCR for scanned PDFs

A PDF with no text layer — a scan — is read by OCR, which uses the
[Tesseract](https://github.com/tesseract-ocr/tesseract) binary. It is not a
Python package, so install it from your system's package manager if you have
scanned books:

```console
$ sudo apt-get install tesseract-ocr    # Debian, Ubuntu
$ brew install tesseract                # macOS
```

EPUBs and PDFs that carry a text layer do not need it.

## Upgrade or remove

```console
$ uv tool upgrade booksmart      # or: pipx upgrade booksmart
$ uv tool uninstall booksmart    # or: pipx uninstall booksmart
```

Uninstalling leaves `~/.booksmart/` in place, so your books and configuration
survive a reinstall. Delete that directory to remove them too.

Next: the [Quickstart](quickstart.md) takes a book through the pipeline and
searches it.
