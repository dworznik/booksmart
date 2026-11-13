# Storage

`BookStorage` is the filesystem side of the pipeline. It holds three kinds of
file under one storage root:

- the **original** book, saved when a book is added and read by `run_parse`;
- the **parsed Markdown** that `run_parse` writes and every later Stage reads;
- the **run log** a Runner writes when a Run finishes.

Paths persisted in the database are stored *relative* to the storage root and
resolved at read time with `BookStorage.resolve`. That keeps the whole data
directory (database, storage and local Qdrant) a portable unit: move it to a
different absolute location and every artifact still resolves.

::: booksmart_core.storage
