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
