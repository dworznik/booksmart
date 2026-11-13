# Quickstart

This page takes one book from a file on disk to a search that answers a
question about it. It assumes you have [installed](installation.md) the
`booksmart` CLI.

The example book is *Alice's Adventures in Wonderland*, a public-domain EPUB
from [Project Gutenberg](https://www.gutenberg.org/ebooks/11). Any PDF or EPUB
of your own works the same way.

## 1. Choose your Providers

Reading a book calls a language model (for the book's profile, its summaries
and the knowledge extracted from it) and an embedding model (for search). Each
comes from a **Provider**, and each Provider needs its vendor's API key. Pick
one of these and run its commands.

### Anthropic + OpenAI (the default)

Anthropic for the language model and OpenAI for embeddings — Anthropic has no
embeddings API, so the default needs both keys:

```console
$ booksmart config set anthropic_api_key
$ booksmart config set openai_api_key
```

### Gemini

Google's Gemini serves both halves, so one key covers everything:

```console
$ booksmart config set llm_provider gemini
$ booksmart config set embedding_provider gemini
$ booksmart config set gemini_api_key
```

### No keys (fake Providers)

Deterministic canned responses, with no API key and no cost. Every
command below runs and succeeds, but the profile, knowledge and search results
are placeholders rather than a reading of the book — use this to see the
mechanics, not to learn anything about Alice.

```console
$ booksmart config set llm_provider fake
$ booksmart config set embedding_provider fake
```

Leaving the value off `config set` makes it ask for the key at a hidden
prompt, so the key never lands in your shell history. If you already export
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `GEMINI_API_KEY`, booksmart picks it up
and you can skip that line. [Configuration](configuration.md) has the details.

## 2. Get a book

```console
$ curl -L -o alice.epub https://www.gutenberg.org/ebooks/11.epub3.images
```

## 3. Add it

`add` validates the file, stores a copy, and registers it as a book. Title and
author are required:

```console
$ booksmart add ./alice.epub --title "Alice's Adventures in Wonderland" --author "Lewis Carroll"
Registered Alice's Adventures in Wonderland as c146c43a-444d-49ae-949a-1eb865f0bc66
```

That id is how every later command names the book. Yours will be different —
copy it from your own output wherever this page writes `<book-id>`. You can
always find it again with `booksmart books list`.

Adding does not read the book yet. Adding the same file twice is refused, so
re-running this step is harmless:

```console
$ booksmart add ./alice.epub --title "Alice's Adventures in Wonderland" --author "Lewis Carroll"
error: This file is already registered as book c146c43a-444d-49ae-949a-1eb865f0bc66 (byte-identical content).
```

## 4. Ingest it

`ingest` runs the pipeline over the book in the foreground. The pipeline is a
fixed sequence of **Stages**, and the output names each one as it starts:

```console
$ booksmart ingest <book-id>
Ingesting c146c43a-444d-49ae-949a-1eb865f0bc66 (scope: full)
  • parse…
  • structure…
  • profile…
  • extraction…
  • summaries…
  • embeddings…
✓ run 74fdadc8-caea-4141-b337-73897dfc7169 succeeded
```

What each Stage does:

| Stage | Does |
| --- | --- |
| `parse` | Reads the EPUB or PDF into text. |
| `structure` | Finds the chapters and sections. |
| `profile` | Writes a short profile of the whole book. |
| `extraction` | Pulls out the knowledge the book teaches — principles, techniques and the like. |
| `summaries` | Summarises each chapter and section. |
| `embeddings` | Turns everything above into vectors, so it can be searched. |

The **Scope** of this execution is `full`: every Stage. With a real Provider a
short book like this one takes a couple of minutes; the fake Providers finish
in seconds.

Each execution is recorded as a **Run** — its Scope, its outcome, the models
used and the tokens spent. A completed Stage is permanent: if a later Stage
fails, the ones before it are kept, `ingest` exits non-zero, and running it
again is safe. A missing key fails before anything is spent:

```console
$ booksmart ingest <book-id>
Ingesting c146c43a-444d-49ae-949a-1eb865f0bc66 (scope: full)
error: Anthropic API key is not set (anthropic_api_key). Set it with: booksmart config set anthropic_api_key
```

## 5. Search it

```console
$ booksmart search <book-id> "a tea party that never ends" --limit 3
┏━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ rank ┃ score ┃ type    ┃ title                            ┃ match                                ┃
┡━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1    │ 1.000 │ chapter │ CHAPTER VII. A Mad Tea-Party     │ Alice joins the March Hare, the      │
│      │       │         │                                  │ Hatter, and a sleepy Dormou…         │
│ 2    │ 0.583 │ chapter │ CHAPTER XII. Alice’s Evidence    │ Called to testify at the trial, a    │
│      │       │         │                                  │ giant Alice accidentally…            │
│ 3    │ 0.333 │ chapter │ CHAPTER XI. Who Stole the Tarts? │ Alice attends the trial of the Knave │
│      │       │         │                                  │ of Hearts, who is accu…              │
└──────┴───────┴─────────┴──────────────────────────────────┴──────────────────────────────────────┘
meaning + keyword · score is a merged-rank score, comparable only within these results — read the
rank
```

That output came from the Gemini Providers. The exact rows depend on the models
you chose, but the shape is the same: the chapters, sections and knowledge
that best answer the question, best first.

Search matches two ways at once — by **meaning**, so a question finds a
passage phrased nothing like it, and by **keyword**, so an exact term or a name
still ranks — and merges the two rankings. Read the `rank` column. The `score`
says how strongly the two kinds of matching agreed and only compares within one
set of results, so a `0.583` can be a good hit. Pass `--dense-only` to match by
meaning alone; the score then becomes a similarity between -1 and 1.

Use `all` instead of a book id to search every book you have ingested:

```console
$ booksmart search all "why does Alice keep changing size"
```

Other options: `--type chapter`, `--type section` or `--type knowledge_object`
to restrict what comes back (repeatable), and `--limit` for how many hits.

## 6. Look around

The ingest left more behind than search vectors:

```console
$ booksmart structure <book-id>
1. The Project Gutenberg eBook of Alice's Adventures in Wonderland
2. by Lewis Carroll
    1. THE MILLENNIUM FULCRUM EDITION 3.0
3. Contents (front_matter)
4. CHAPTER I. Down the Rabbit-Hole
5. CHAPTER II. The Pool of Tears
6. CHAPTER III. A Caucus-Race and a Long Tale
7. CHAPTER IV. The Rabbit Sends in a Little Bill
…
```

- `booksmart profile <book-id>` — the book's generated profile.
- `booksmart knowledge list <book-id>` — the knowledge extracted from it. This
  is aimed at books that teach something; a novel like this one may yield none,
  and says so.
- `booksmart runs list <book-id>`, then `booksmart runs show <run-id>` — the
  Run history, with the models used and the tokens each Run spent.
- `booksmart books list` — every book you have added.

## Re-running part of the pipeline

`ingest --scope` runs a subset of the Stages instead of all of them, reusing
what the earlier Stages produced — after changing the model, say:

| Scope | Re-runs |
| --- | --- |
| `full` | Every Stage (the default). |
| `profile` | The book's profile only. |
| `extraction` | Extraction, then embeddings, so search sees the new knowledge. |
| `embeddings` | Embeddings only. |

```console
$ booksmart ingest <book-id> --scope embeddings
```

Any Scope but `full` needs an earlier successful Run of the book to build on.
