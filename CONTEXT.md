# booksmart

A pipeline that turns uploaded books into queryable knowledge: parsed text,
detected structure, LLM-extracted knowledge objects, summaries, and embeddings.

## Language

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
