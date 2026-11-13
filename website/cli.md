# CLI reference

The `booksmart` command is a local, single-user front end over `booksmart-core`.
Everything runs on your machine against an SQLite file and an embedded Qdrant
store under `~/.booksmart` — no Docker and no server.

The commands follow the life of a book through the pipeline:

1. **Configure** — `booksmart config set` stores the API keys the pipeline's
   Stages need.
2. **Register** — `booksmart add` copies a PDF or EPUB into local storage and
   records its metadata. Nothing is parsed yet.
3. **Ingest** — `booksmart ingest` starts a **Run** over the book. The default
   `full` **Scope** covers every Stage, from parsing through embeddings; the
   narrower Scopes redo a slice and reuse the output of the Stages upstream of
   it.
4. **Inspect** — `books`, `runs`, `structure`, `profile` and `knowledge` read
   back what the Run produced: the book's record, its Run history, the chapter
   tree, the book profile and the extracted knowledge objects.
5. **Search** — `booksmart search` asks a question across everything embedded,
   hybrid (dense plus BM25) by default.

Every command takes `--help` and prints the same text as below.

!!! note "Generated, not written"
    Everything under *Commands* is generated from the CLI itself each time the
    site is built, so it always matches the CLI in the same commit. To change it,
    change the command's help text in `booksmart_cli`.

<!-- booksmart-cli-reference -->
