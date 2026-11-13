# API reference

This is the reference for `booksmart-core`, the ingestion pipeline as a typed
library. Everything below each page's introduction is generated from the
package's own docstrings and type hints, so it describes the code that ships
rather than a copy of it. The introductions are written by hand: what the seam
is for and where it sits in the pipeline.

## How the pieces fit

A book moves through the pipeline in two halves.

**The write side** turns a stored file into embedded knowledge. The work is
split into **Stages** (`parse`, `structure`, `profile`, `extraction`,
`summaries`, `embeddings`), each a plain function that commits its own output.
A **Runner** walks the Stages of a **Scope** in order and records the **Run**.
Stages share two pieces of infrastructure: `BookStorage` for the files on disk
and `VectorStore` for the embedded vectors.

**The read side** is `search`: embed a query, rank the stored vectors and
resolve each hit back to the row it came from. It writes nothing and needs no
Runner.

| Page | Module | Seam |
| --- | --- | --- |
| [Stages](stages.md) | `booksmart_core.stages` | The `run_<stage>` functions and the `StageReport` each returns |
| [Runner](runner.md) | `booksmart_core.runner` | The in-repo sequential Runner and the Scopes it executes |
| [Search](search.md) | `booksmart_core.search` | Hybrid and dense retrieval over the embedded records |
| [VectorStore](vectors.md) | `booksmart_core.vectors` | The model-locked Qdrant collection |
| [Storage](storage.md) | `booksmart_core.storage` | Originals, parsed Markdown and run logs on the filesystem |

The design decisions behind these seams are recorded as ADRs in the
repository. [Stages are the unit of durability](https://github.com/dworznik/booksmart/blob/main/docs/adr/0002-stages-are-the-unit-of-durability.md)
and [the model-locked vector collection](https://github.com/dworznik/booksmart/blob/main/docs/adr/0001-model-locked-vector-collection.md)
are the two that shape the API most.
