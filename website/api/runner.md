# Runner

A **Runner** is whatever executes Stages in order and owns the **Run** record.
Stages and Runs are shared across consumers; Runners are not, and each consumer
brings the one that suits it. This module is the in-repo one: sequential,
synchronous, on a single session, which is the shape the CLI uses.

A Run is created when execution starts and finalised when it ends; there is no
queued state. `execute_run` is the entry point most consumers want. It resolves
a **Scope** to its Stages through `SCOPE_STAGES`, dispatches each one, and
records the outcome: an expected Stage failure is written onto the Run rather
than raised. `start_run` and `finalize_run` are exposed for a consumer building
its own Runner on the same Run record.

The `build_default_*` functions construct providers from configuration when the
caller passes none. They are also the seam tests use to substitute stubs
without every call site threading providers through.

::: booksmart_core.runner
