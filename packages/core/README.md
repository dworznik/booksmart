# booksmart-core

Booksmart's book-ingestion pipeline as a typed library: parsing, structure
detection, profile generation, knowledge extraction, summaries, and embeddings —
exposed as durable, synchronous Stage functions that a consumer drives with its
own Runner (see ADR 0002).

Core owns the domain: the ORM models, a single dialect-neutral Alembic history
(SQLite and Postgres), object storage, the provider abstractions (Anthropic /
OpenAI / Gemini, plus deterministic fakes), and the Qdrant vector store. It reads
no environment at all — a consumer constructs `Settings` explicitly (API keys
included; since 0.2.0 there is no env-var fallback) and owns engine and session
lifecycle.

Consumers:

- [`booksmart`](https://pypi.org/project/booksmart/) — the local CLI (SQLite,
  embedded Qdrant).

```python
from booksmart_core.database import create_engine, upgrade_to_head
from booksmart_core.runner import execute_run

upgrade_to_head(url)
run_id = execute_run(session_factory, storage_root, book_id, "full")
```

## Search

`booksmart_core.search.search` is the read side of the embedding pipeline, and
the single seam every consumer's search surface sits on — the CLI's `search`
command calls it against embedded Qdrant; a server consumer would call it with
the same arguments against a served instance.

```python
from booksmart_core.search import search

hits = search(session, vector_store, embedder, "how do deep modules help?", limit=5)
```

It embeds the query with the collection's locked model (refusing a mismatch, ADR
0001), ANN-searches Qdrant, and resolves each hit back to its detached ORM row.

