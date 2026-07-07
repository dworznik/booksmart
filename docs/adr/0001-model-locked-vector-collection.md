# The vector collection is locked to one embedding model

The shared Qdrant collection is created for a specific embedding model and
records that model's name; every subsequent write is rejected unless the
configured embedder matches, even when the vector dimensions happen to
coincide. We chose this over per-model collections (more moving parts than a
single-corpus product needs) and over silent coexistence (vectors from
different models are incomparable — same-dimension mixing degrades search
quality in a way no operator can diagnose from symptoms). Switching embedding
models is therefore an explicit migration: drop the collection and reprocess
embeddings for every book.

## Considered Options

- Per-model collections with routing — reversible later if multi-model
  retrieval ever becomes a requirement.
- Dimension check only — rejected because the most dangerous case
  (different models, identical dimensions) passes a dimension check.
