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
$ booksmart-bench run <family> <area|book>          # emit a run file
$ booksmart-bench score <run-file> [--out s.json]   # run file x truth -> scores
$ booksmart-bench report <base> <cand> [--out r.md] # two runs, side by side
```

...plus one verb that is not an artefact boundary:

```console
$ booksmart-bench sources [--pin]                   # check the book files, pin them
```

`sources` identifies each file in the assets checkout by the chapter and section
titles it contains — not by its filename, which is whatever the last person
typed. It exists because the only thing tying truth to a book is a sha256 in
`book.yaml`, and every way of getting that wrong is silent: a later edition keeps
most of its headings and so scores highly against truth authored for the earlier
one, and a scan with no text layer ingests happily while benchmarking the OCR
parser instead of the one every other book used. So a file has to beat the
runner-up by a margin to be named at all, an edition or copyright year the file
states about *itself* has to agree with `book.yaml`, and `--pin` writes nothing
for a book two files both claim. Reading every page of a long book is what makes
identification reliable, so allow a few seconds per book.

`ingest` is idempotent: a book whose bytes and configuration are unchanged is
skipped. `score` and `report` are pure — no pipeline, no corpus, no network —
and `report` exits non-zero on a regression so it can gate a change without
anyone reading the table first.

## Running a benchmark

`run` takes a family and a scope, and writes one run file into `<assets>/runs/`
(or wherever `--out` says):

```console
$ booksmart-bench run recall fp        # the query sets, through the real search path
$ booksmart-bench run ingestion sicp   # structure, coverage, faithfulness, cost
$ booksmart-bench run all fp           # both, into one file
```

A scope is a book slug or an area. An area also asks that area's cross-book
queries, which go to the whole corpus — which book answers is half of what they
measure — while a book's own set is filtered to that book, so whatever else the
corpus happens to hold cannot move the score.

Recall goes through `booksmart_core.search`, the same path a product consumer
uses. The run writer owns the **record -> location join**: a hit's chapter,
section or knowledge object is resolved to the ToC node id truth names, by
normalised title with position as the tiebreak, so the scorer never sees a
record id. A hit that joins to no node is dropped and counted — the surviving
ranks stay the ranks search produced — and every drop lands in the run file's
notes.

`run` spends money only when the judge is configured (below); `--no-judge`
turns that off.

## The faithfulness judge

Summary faithfulness has no ground truth to compare against — the summary is
written per run — so the check is a second model reading each summary beside the
exact source slice it should have come from. That makes the judge an
instrument, and it is pinned like one:

```console
$ export BOOKSMART_BENCH_JUDGE_PROVIDER=gemini   # never the summariser's family
$ export BOOKSMART_BENCH_JUDGE_MODEL=gemini-3.5-flash
```

Both are required. The judge must be **cross-family** — a model grading its own
family's prose measures its own preferences, so a judge provider equal to
`BOOKSMART_LLM_PROVIDER` is refused — and it must be **named**, because a
provider default is whatever the vendor decided this month, which is the ruler
moving between runs. Its identity, prompt version included, is stamped into
every run file that used it, and its spend lands in the cost dimension like any
other spend. Prompts live in `judge.py` and are versioned
like the pipeline's extraction prompts: a score produced under one prompt is not
comparable with a score produced under another.

With no judge configured, faithfulness simply goes unmeasured and the run says
so.

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

## The truth lint

Truth is hand-authored in another repository against books this one never sees,
so every verb lints it before spending anything. An **error** is truth that
cannot be scored — a `loc:` naming no node, a duplicate node id, a kind that does
not exist, a conceptual query that shares vocabulary with the node it expects. A
**warning** is truth that is incomplete but honest, like an unpinned source or a
book whose queries are not authored yet.

One guard is worth knowing about before editing a `toc.yaml`. Entries are YAML
flow mappings, and in a flow mapping a comma separates entries:

```yaml
- { id: "4.1", title: The call, apply, and bind methods }   # three keys, not two
```

That records a title of `"The call"`. Since structure fidelity matches nodes on
normalised title, the node then cannot match the section it describes, and it
scores as a detection failure rather than a quoting mistake.

So an entry saying anything the schema cannot read is an error, naming the keys
and the title as recorded. A chapter may carry `id`, `title` and `sections`;
front-matter, section and back-matter entries may carry only `id` and `title`,
because this schema has two levels and a third authored here would be read by
nobody. Quote titles containing `,` `?` `(` `)` `!` `#` `[` `]` `{` `}`.

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
