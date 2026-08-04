# booksmart-bench

The benchmark harness for booksmart: two families of measurement — **ingestion**
(structure fidelity, extraction coverage, summary faithfulness, cost) and
**recall** (location-level retrieval quality) — taken through the real pipeline
and the real search path.

Not published to PyPI. It is a workspace member so it can import `booksmart_core`
directly and measure the code sitting next to it.

## The three-artefact split

Measurement is TREC-shaped, and the three artefacts stay separable:

1. **Ground truth** — hand-authored, versioned in a separate assets repo.
2. **Run file** — emitted here; carries *resolved locations, never record ids*.
3. **Scorer** — a pure comparison of run file × truth; pipeline-agnostic.

One verb per artefact boundary:

```console
booksmart-bench ingest    # build a corpus (the only expensive verb)
booksmart-bench run       # execute queries, emit a run file
booksmart-bench score     # run file × truth -> scores (pure)
booksmart-bench report    # render two runs side by side (pure)
```

## Assets and corpora

The harness owns no data. Two locations, resolved separately:

- **Assets** — the ground-truth checkout, given by `--assets` or
  `BOOKSMART_BENCH_ASSETS`. There is no default: the assets live outside this
  repo and their path is machine-specific, so an unset value is an error with a
  remedy rather than a guess.
- **Corpus home** — `~/.booksmart-bench/<config-snapshot>/`, one auto-migrated
  SQLite file and embedded Qdrant per pipeline configuration. Ingest is
  expensive and recall is not, so corpora are keyed by everything that would
  invalidate them (models, prompt versions) and reused until one of those moves.
  Override the base with `BOOKSMART_BENCH_HOME`.
