# booksmart-core

Booksmart's book-ingestion pipeline as a typed library: parsing, structure
detection, profile generation, knowledge extraction, summaries, and embeddings —
exposed as durable, synchronous Stage functions that a consumer drives with its
own Runner (see ADR 0002).

Core owns the domain: the ORM models, a single dialect-neutral Alembic history
(SQLite and Postgres), object storage, the provider abstractions (Anthropic /
OpenAI / Gemini, plus deterministic fakes), and the Qdrant vector store. It reads
no environment on import — a consumer constructs `Settings` and owns engine and
session lifecycle.

Consumers:

- [`booksmart`](https://pypi.org/project/booksmart/) — the local CLI (SQLite,
  embedded Qdrant).
- `booksmart-api` — an Inngest-backed service (Postgres) that wraps each Stage in
  a durable step.

```python
from booksmart_core.database import create_engine, upgrade_to_head
from booksmart_core.runner import execute_run

upgrade_to_head(url)
run_id = execute_run(session_factory, storage_root, book_id, "full")
```
