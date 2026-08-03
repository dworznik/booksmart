# The sparse half of the collection lock pins a recipe, not a model name

ADR 0001 locks the shared Qdrant collection to one embedding model. Hybrid
retrieval adds a second vector per point, computed client-side by a sparse
model, so the lock now has a second half — and that half pins the model *plus
the parameters it ran with* (for BM25: k, b, avg_len, language), recorded in
collection metadata under `sparse_recipe`. A mismatch in any of them is refused
with the same drop-the-collection-and-reprocess guidance ADR 0001 gives.

This extends ADR 0001 rather than superseding it: the dense lock is unchanged,
and the reasoning is the same one applied to a different failure. Vectors from
different dense models are incomparable; term weights from differently
parameterised BM25 are equally incomparable, and both degrade retrieval with no
symptom an operator could trace back to the cause.

A name-only sparse lock was the obvious alternative and is the one this rejects.
The parameters are not incidental configuration — they *are* the weighting.
Two collections built by `Qdrant/bm25` with different `b` hold different numbers
for the same text, and a lock that compared only the name would call them
interchangeable. Because the degradation is a recall loss rather than an error,
nothing downstream would report it.

The recipe is read back off the constructed model rather than off our own
configuration, so a change in the *library's* defaults trips the lock too — the
drift most likely to happen, since it arrives with a routine dependency bump and
nobody's config changes. That choice has a consequence worth stating: a sparse
model that does not expose those parameters cannot be locked safely, and
constructing one is refused rather than silently locked to a bare name.

## Consequences

- Adopting hybrid storage is a migration for existing collections: they record
  no sparse recipe and their points carry no sparse vector, so they are refused
  on read and on write rather than adopted.
- Changing sparse model *or* any of its parameters costs a full reprocess, the
  same as changing the embedding model.
- The recipe check on the read path applies where the sparse vectors are
  actually read (hybrid search). A dense-only search compares no term weights,
  so a drifted recipe does not refuse it — and requiring a sparse provider there
  would cost a model download for a search that will never use it.

## Considered Options

- Lock the sparse model name only — rejected above: the dangerous case
  (same model, different parameters) passes a name check, exactly as the
  dangerous case in ADR 0001 (different models, identical dimensions) passes a
  dimension check.
- Do not lock the sparse side at all, treating BM25 as reproducible — false:
  it is reproducible only given the same parameters, which is the thing being
  pinned.
- Record the parameters from our own Settings — would miss library-default
  drift, the likeliest source.
