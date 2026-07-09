# Stages are the unit of durability; orchestration lives in Runners

Core exposes the ingestion pipeline as plain synchronous Stage functions that
commit their own transaction, take serializable inputs (ids, not ORM objects
or document text), and return a typed report. Nothing in core claims work,
sequences stages, retries, or writes Run records — that is the Runner's job,
and every consumer brings its own Runner: the CLI walks the stages in a
foreground loop; a server consumer wraps each stage in a durable-execution step
and lets the runtime's re-invocation/memoization cycle provide durability and retry.

This trades away job atomicity: a failed stage leaves earlier stages
committed. We accept that because every stage replaces its own output
wholesale (structure, knowledge objects, and vectors are deleted and
rewritten, never appended), so re-running a committed stage is safe — and a
durable executor *requires* completed steps to stay completed. Errors carry
an explicit `retriable` flag so Runners can map them to their own retry
semantics (a runtime's non-retriable error, a CLI error message) without parsing
strings.

Core stays synchronous on purpose. A durable runtime's "async" is re-invocation
over HTTP, not an event loop; an async-first core would force async DB
drivers and SDK clients onto every consumer while buying neither durability
nor concurrency that step-level parallelism doesn't already provide.

## Considered Options

- One transaction per pipeline run (status quo) — rejected: a durable
  executor cannot roll back work committed by earlier steps, so job
  atomicity is unimplementable under re-invocation runtimes; keeping it CLI-only would
  fork the semantics between consumers.
- Async-first core — rejected: drags aiosqlite/asyncpg and async SDK
  clients into every consumer for no durability gain; sync functions wrap
  cleanly in durable steps and threadpools.
- Core-owned shared runner — rejected: a server's runner *is* its runtime;
  any orchestration shipped in core would either go unused there or leak
  scheduling assumptions into the library.
