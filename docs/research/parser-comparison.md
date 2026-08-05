# Comparing the parsers: marker vs pymupdf

`ParserChain` prefers `marker` over `pymupdf` for PDFs
([`parsing.py`](../../packages/core/src/booksmart_core/parsing.py)), but
`marker-pdf` is not a declared dependency, is not in `uv.lock`, and is not
installed by CI or the devcontainer. **`MarkerParser` has therefore never run.**
Every PDF this project has parsed went through `pymupdf4llm`.

Before adopting marker — which brings `torch`, `torchvision` and `transformers`
— or removing the branch of the chain that prefers it, the difference has to be
measured on the kind of document booksmart actually ingests. That is
[booksmart#80](https://github.com/dworznik/booksmart/issues/80). This page is
how to run it.

## What the eval measures

`packages/core/tests/test_parser_eval.py` drives
`packages/core/tests/parser_eval.py`. Five countable proxies, none of which is
"quality" — read them beside a sample of the markdown, never instead of one:

| metric | why |
| --- | --- |
| headings | what structure detection works from; structure fidelity is scored against a hand-authored ToC |
| fenced code blocks | a flattened listing is wrong *and* misleads the summariser |
| letter-spaced runs | display type set with tracking extracts as one token per letter |
| bare page-number lines | page furniture reaching summaries and embeddings is pure noise |
| characters per page | the crude check that any text came out |

```console
$ uv run pytest packages/core/tests/test_parser_eval.py          # guards, always
$ BOOKSMART_PARSER_EVAL_PDF=~/Downloads/bash.pdf \
    uv run pytest packages/core/tests/test_parser_eval.py -k RealEval -s
```

## Which document

**Use two.** A single document cannot answer the question, and assuming
otherwise already produced one wrong conclusion here.

`pymupdf4llm`'s code detection is **font-dependent, not absent**. It fences code
set in a face it recognises as monospace and misses code set in one it does not:

| document | code fences per 20 pages |
| --- | --- |
| GNU Bash Reference Manual | 70 |
| *An Introduction to R* | 114 |
| a typeset programming book from the private corpus | **0** |

So run it over a well-behaved manual *and* an awkward book.

**Public fixture** — the GNU Bash Reference Manual, 214 pages, GFDL 1.3, code
listings throughout:

```console
$ curl -LO https://www.gnu.org/software/bash/manual/bash.pdf
```

Alternative: `https://cran.r-project.org/doc/manuals/R-intro.pdf` (103pp, GPL
documentation, denser in listings).

Neither is strictly public domain — both are free documentation under a copyleft
licence, which is why they are *downloaded* rather than committed. Nothing in
this repo needs to redistribute them.

**Awkward fixture** — point `BOOKSMART_PARSER_EVAL_PDF` at any book from the
private benchmark corpus. Those are the documents the pipeline actually ingests,
and they are the harder case.

## Baseline, so you can tell whether you reproduced it

`pymupdf` alone, on `bash.pdf` (214 pages), no marker, no tesseract:

```
parser                  pages chars/pg  headings  fences    spaced  pagenos     secs
marker (ParseFailure)     214        0         0       0         0        0      0.0
pymupdf                   214     2750       739     266         2      213    281.8
ocr (ParseFailure)        214        0         0       0         0        0      0.0
```

Two things worth noticing before marker enters the picture. `pymupdf4llm` does
well here — 739 headings and 266 fenced blocks. And it still leaves **213 bare
page-number lines in a 214-page document**, roughly one per page, in the text
that goes on to be summarised and embedded.

~1.3 s/page, so a 600-page book is around 13 minutes of parsing before a single
LLM call.

## Installing marker on macOS

Verified against marker's own README at the time of writing; check it again
before trusting any of this, since marker moves quickly.

### 1. Do not put it in the workspace lock

marker is deliberately undeclared (see #80). Installing it into the workspace
environment would add `torch` to `uv.lock` and change what every contributor
pulls. Keep it in a throwaway environment:

```console
$ uv venv .venv-marker --python 3.12
$ source .venv-marker/bin/activate
$ uv pip install marker-pdf booksmart-core
```

Or, for a single run without any venv at all:

```console
$ uv run --with marker-pdf pytest packages/core/tests/test_parser_eval.py -k RealEval -s
```

The `--with` form is the safer default: nothing persists, and `uv.lock` is
untouched.

### 2. What gets pulled in

`marker-pdf` itself is 0.2 MB. Its closure is not:

```
marker-pdf → surya-ocr → torch, torchvision, transformers
           → pdftext, scikit-learn, pillow, rapidfuzz
           → anthropic, openai, google-genai      ← see step 4
```

Budget **several GB** of disk for the wheels, plus model weights downloaded on
first run. Expect the first parse to be slow for that reason alone.

No Homebrew packages are needed for the PDF path. On Apple Silicon the PyPI
`torch` wheels are arm64-native and include MPS support; there is no CUDA on
macOS and no extra package index to configure, which makes mac a *easier* target
for this than Linux.

### 3. Choose the device

marker reads `TORCH_DEVICE`. On Apple Silicon:

```console
$ export TORCH_DEVICE=mps      # Metal; falls back to cpu if unavailable
```

On an Intel Mac, leave it unset or use `cpu`, and expect it to be slow enough
that you may prefer to run the comparison on fewer pages.

If you go on to use marker's VLM/LLM path (you should not — see below), its
README describes an inference server via `brew install llama.cpp` on Apple
Silicon. The plain PDF path used here does not need it.

### 4. Keep LLM mode off

marker 2.x has an opt-in LLM-assisted mode (`--use_llm`, defaulting to a Gemini
model), which is why `anthropic`, `openai` and `google-genai` are in its
dependency list. It is **off by default** and must stay off here:

- booksmart drives marker through `PdfConverter` directly
  (`parsing.py`, `MarkerParser.parse`), which does not enable it, and
- a parser that makes its own API calls spends money *outside* the per-Stage
  cost accounting, which would silently confound the benchmark's cost dimension.

If you are ever unsure whether it is active, watch for network traffic on the
first page, and check that no provider key is set in the environment you run in.

### 5. Confirm the chain actually picks it up

```console
$ python -c "import marker.converters.pdf; print('marker importable')"
$ uv run pytest packages/core/tests/test_parser_eval.py -k ChainSelection -q
```

`test_marker_is_preferred_when_it_is_installed` asserts `marker` wins when it is
importable and `pymupdf` wins when it is not, so it passes either way and tells
you which world you are in. The eval's table names the parser that produced each
row, so a run that silently fell back to `pymupdf` is visible rather than
mistaken for a marker result.

## What to record

The issue asks for a written finding, not a verdict in a shell history. Capture:

- **marker's exact version** — nothing pins it, so a result without a version is
  not reproducible;
- the table for both documents, easy and awkward;
- a **sample of each parser's markdown over the same two pages**, because the
  counts cannot show you reading order, table handling, or whether a listing was
  fenced *correctly* rather than merely fenced;
- wall-clock per page, and whether MPS was actually used;
- a recommendation among: adopt marker as an optional extra
  (`booksmart-core[marker]`) so it can be pinned; keep it out and delete
  `MarkerParser`, since a chain entry that never runs is misleading; or keep it
  out and document the opt-in.

## The measurement that actually decides it

Everything above is a proxy. The number that matters is **structure fidelity**
— P/R/F1 of detected `(chapter, section)` nodes against the authored ToC — which
needs a full ingest under each parser and a `booksmart-bench report` between the
two corpora.

That is currently blocked by a separate gap: the parser is not part of the
corpus key (`packages/bench/src/booksmart_bench/config.py`), so two ingests
under different parsers land in the *same* corpus directory, and `ingest` skips
books whose bytes are unchanged. Fix that first, or run the second ingest with
`--force` into an explicitly separate `BOOKSMART_BENCH_HOME`.
