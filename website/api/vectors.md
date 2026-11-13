# VectorStore

`VectorStore` is the Qdrant collection every embedded record lives in. The
`embeddings` Stage writes to it and [`search`](search.md) reads from it. The
relational database and the filesystem remain the source of truth: the
collection holds vectors, plus the payloads that link each point back to its
row and its book.

Every point carries two vectors, a dense one and a sparse (BM25) one, and they
are always written together. `VectorRecord` cannot be built with only one.

The collection is **locked** to the dense embedding model and the sparse recipe
it was created for, and writes from any other are rejected even when the
dimensions match. Mixing vectors from different models degrades search in a way
no operator can diagnose from the results, so switching models is an explicit
migration: drop the collection and re-embed every book. See
[ADR 0001](https://github.com/dworznik/booksmart/blob/main/docs/adr/0001-model-locked-vector-collection.md)
and [ADR 0004](https://github.com/dworznik/booksmart/blob/main/docs/adr/0004-the-sparse-lock-pins-a-recipe.md).

Build one from configuration with `build_vector_store`, and close it when done:
embedded on-disk Qdrant holds a single-process lock on its directory.

::: booksmart_core.vectors
