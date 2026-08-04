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
$ booksmart-bench ingest <area|book> [--force]      # build a corpus
$ booksmart-bench run <family> <area>               # emit a run file
$ booksmart-bench score <run-file> [--out s.json]   # run file x truth -> scores
$ booksmart-bench report <base> <cand> [--out r.md] # two runs, side by side
```

`ingest` is the only verb that spends money, and it is idempotent: a book whose
bytes and configuration are unchanged is skipped. `score` and `report` are pure
— no pipeline, no corpus, no network — and `report` exits non-zero on a
regression so it can gate a change without anyone reading the table first.

## What is measured

| dimension | score |
| --- | --- |
| recall | MRR (primary) + hit@5, binary location-level judgements |
| structure fidelity | P/R/F1 over (chapter, section) nodes vs the authored ToC |
| extraction coverage | recall against the concept inventory |
| summary faithfulness | mean claim-support ratio |
| cost & throughput | tokens and wall time per Stage — **tracked, never scored** |

Per-slice figures are diagnostics: they say where a difference came from and
decide nothing. Only the pooled aggregate carries the regression verdict, and
there is no blended headline number — an MRR and an F1 do not average into
anything actionable.

Nothing is silently dropped. A gated slice, a query truth holds that a run never
asked, a book a run names that truth does not cover — each is reported under
"not measured", because a slice that disappears from a report reads as one that
passed.

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
