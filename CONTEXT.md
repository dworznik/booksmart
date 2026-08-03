# booksmart

A pipeline that turns uploaded books into queryable knowledge: parsed text,
detected structure, LLM-extracted knowledge objects, summaries, and embeddings.

## Language

### Pipeline

**Stage**:
One unit of the ingestion pipeline (parse, structure, profile, extraction,
summaries, embeddings). The unit of durability and retry: a completed Stage
is permanent, and a failed Stage never undoes earlier Stages. Every Stage
replaces its own output wholesale, so re-running one is safe.
_Avoid_: step, phase, task

**Scope**:
The set of Stages a run executes (full, profile, extraction, embeddings).
Incremental Scopes reuse upstream Stage output.

**Run**:
The record of one pipeline execution over a book: its Scope, outcome, and
provenance (models, prompt versions, token spend). Created the moment
execution starts — there is no queued state. Owned by the Runner; Stages
never see it.
_Avoid_: job

**Runner**:
Whatever executes Stages in order and owns the Run record. Each consumer
brings its own; Stages and Runs are shared, Runners are not.
_Avoid_: worker, orchestrator

### Model providers

**Provider**:
An adapter for one vendor's model API (LLM completion or embedding), selected
by configuration. The pipeline talks to providers, never to vendors directly.

**Limit**:
A provider-declared fact about its vendor's API — maximum batch size, maximum
output tokens, which reasoning efforts a model accepts, embedding dimensions.
Not configurable; violating one is a booksmart bug.
_Avoid_: capability, constraint, cap

**Preference**:
A user choice about how to use a provider — which model, reasoning effort,
provider selection itself. Set per deployment and validated against Limits
before any call is made.
_Avoid_: setting, option, knob

**Preference Snapshot**:
The Preferences resolved once when a run is triggered and carried with the
run, so every step of a durable run behaves consistently regardless of
config changes or deploys that happen mid-run. Limits are never snapshotted;
they are enforced live.

### Retrieval

**Dense vector**:
What an embedding Provider returns for a text: a fixed-size array of floats
compared by cosine similarity. Carries meaning, not words — it matches a
paraphrase and misses an unfamiliar proper noun.

**Sparse vector**:
The lexical counterpart, computed locally rather than by a vendor: term ids
with weights, one entry per word the text actually contains. Matches the
words, not the meaning. Stored beside the dense vector on the same point,
never instead of it.

**Recipe**:
A sparse model plus the parameters it ran with (for BM25: k, b, avg_len,
language). The unit the collection is locked to, because the parameters
change every term weight — a Recipe is what a sparse model *is*, and a bare
model name does not identify one.
_Avoid_: sparse model name (it is not enough to identify the weighting)

**Model lock**:
The collection metadata recording the embedding model and the sparse Recipe
it was created for, and the refusal to read or write it under any other
(ADR 0001). Changing either is an explicit migration: drop the collection
and reprocess every book.

**Hybrid search**:
The default: retrieve by Dense vector and by Sparse vector independently,
then merge the two rankings. Finds both the paraphrase and the exact term.
_Avoid_: semantic search (it names only the dense half)

**Fusion**:
How the two rankings are merged — Reciprocal Rank Fusion, which scores each
record by the sum of 1/(rank + k) over the branches it appeared in. Reads
positions, never scores, so the two branches need no common scale.

**Fused score**:
What Fusion produces. Describes how well the two rankings agreed, not how
close anything is; comparable within one result set and meaningless across
queries. Never call it a similarity — a Fused score of 1.0 and a cosine of
1.0 are unrelated facts.

**Branch**:
One side of a Hybrid search: the dense Branch or the sparse Branch. Filters
apply per Branch, never once at the root, and each Branch fetches at least
limit + offset candidates.
