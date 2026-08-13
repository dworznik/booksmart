# booksmart-core

Turn PDFs and EPUBs into structured, searchable knowledge — a typed Python
library you drive from your own application.

Give it a book. It extracts the text, recovers the chapter and section structure,
pulls out the concepts the book teaches, writes summaries, and embeds all of it
so you can ask questions in natural language and get back the passage that
answers them.

If you want this as a command you run rather than a library you call, install
[`booksmart`](https://pypi.org/project/booksmart/) instead — it is this library
with a local front end, and needs no setup beyond an API key.

## Install

```console
$ pip install booksmart-core[sparse]
```

Python 3.12 or newer. The `sparse` extra adds keyword matching alongside meaning
matching, which is what makes search find an exact term the embedding model never
learned. Leave it off (`pip install booksmart-core`) if you only want semantic
search, or if you would rather plug in your own keyword provider.

You will also need an API key for whichever LLM and embedding providers you use —
Anthropic, OpenAI and Gemini are supported out of the box, and deterministic
fakes let you run the whole pipeline offline with no key and no cost.

## Ingesting a book

```python
from booksmart_core.database import upgrade_to_head
from booksmart_core.runner import execute_run

upgrade_to_head(url)
run_id = execute_run(session_factory, storage_root, book_id, "full")
```

Ingestion runs as a sequence of independent steps — parse, structure, profile,
extraction, summaries, embeddings — and each one records what it did, what it
cost, and which model produced it. A step that fails does not lose the work of
the ones before it, so a re-run resumes rather than restarts.

You keep control of the database session and the storage location. The library
reads no environment variables and holds no global state: everything it needs
arrives through an explicit `Settings` object, including API keys.

## Searching

```python
from booksmart_core.search import search
from booksmart_core.sparse import build_sparse_embedding_provider

results = search(
    session,
    vector_store,
    embedder,
    "how do deep modules help?",
    sparse_embedder=build_sparse_embedding_provider(settings),
    limit=5,
)
for hit in results.hits:
    print(hit.rank, hit.score, hit.title)
```

Search fuses two kinds of matching: meaning, via embeddings, and words, via BM25.
Meaning alone misses a proper noun or an exact term the embedding model never
saw; words alone miss a question phrased differently from the passage that
answers it. Fusing them ranks better than either.

Pass `mode="dense"` for meaning-only, and you can drop the `sparse_embedder`
argument and the extra along with it. Ask for the default without one and the
call raises rather than quietly serving you meaning-only results — the mismatch
between what you asked for and what you got is the thing worth refusing.

Each hit carries its rank, its score, and the database row it came from, so you
can render a result without a second query. `results.embedding_tokens` reports
what the query cost, when the provider says.

**A collection remembers the models it was built with**, and search refuses to
run against a mismatch rather than return plausible, silently wrong rankings.
Changing embedding model means re-embedding — which the library tells you,
instead of quietly degrading.

## What you get back

Books, chapters and sections with generated summaries; knowledge objects — the
principles, patterns and definitions a book teaches, each with the location it
came from; and vectors for all of it. Everything is an ordinary SQLAlchemy row
you can query directly, on SQLite or Postgres, with migrations included.
