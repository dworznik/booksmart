# Configuration

Every setting — which Providers to use, which models, API keys, where data
lives — is read and written with `booksmart config`, and persisted in one file:
`~/.booksmart/config.toml`.

## The `config` commands

| Command | Does |
| --- | --- |
| `booksmart config set <field> [value]` | Validate a value and write it to `config.toml`. |
| `booksmart config get <field>` | Print the value in effect, unredacted. |
| `booksmart config unset <field>` | Remove a value from `config.toml`. |
| `booksmart config list` | Every field, its value in effect, and where that value came from. |

### `set`

```console
$ booksmart config set llm_provider gemini
Set llm_provider = gemini
```

A value is validated before it is written, so a typo fails here rather than at
your next ingest:

```console
$ booksmart config set llm_provider nope
error: llm_provider must be one of anthropic, fake, gemini, openai; got 'nope'
```

**Leave the value off to keep it out of your shell history.** `set` then asks
for it at a prompt — hidden, for API keys — or reads it from standard input
when you pipe it in:

```console
$ booksmart config set anthropic_api_key
Value for anthropic_api_key:
Set anthropic_api_key = sk-a…0000
```

API keys are shown redacted whenever `set` or `list` prints them.

### `get`

`get` prints the value that is actually in effect, after every layer described
[below](#where-a-value-comes-from) has been considered. It is unredacted, so it
is the one to use in a script:

```console
$ booksmart config get llm_provider
gemini
```

A field with no value prints `<field> is not set` and exits non-zero.

### `unset`

`unset` removes a field from `config.toml`. The value does not necessarily
disappear: whatever the next layer down says takes over — an environment
variable, or the default.

```console
$ booksmart config unset llm_provider
Unset llm_provider
```

### `list`

`list` answers *why is this the value?* for every field at once. The `source`
column names the layer that won:

```console
$ booksmart config list
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ field                ┃ value                                 ┃ source           ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ database_url         │ sqlite:////home/you/.booksmart/books… │ default          │
│ storage_root         │ /home/you/.booksmart/storage          │ default          │
│ qdrant_url           │ http://localhost:6333                 │ default          │
│ qdrant_path          │ /home/you/.booksmart/qdrant           │ default          │
│ llm_provider         │ gemini                                │ config           │
│ llm_model            │ (unset)                               │                  │
│ llm_reasoning_effort │ (unset)                               │                  │
│ embedding_provider   │ gemini                                │ config           │
│ embedding_model      │ (unset)                               │                  │
│ sparse_provider      │ fastembed                             │ default          │
│ sparse_model         │ (unset)                               │                  │
│ embed_passages       │ False                                 │ default          │
│ anthropic_api_key    │ (unset)                               │                  │
│ openai_api_key       │ (unset)                               │                  │
│ gemini_api_key       │ AIza…xxxx                             │ conventional-env │
└──────────────────────┴───────────────────────────────────────┴──────────────────┘
```

## Where a value comes from

Each field resolves through the same four layers. The highest layer that has a
value wins:

1. **`BOOKSMART_<FIELD>` environment variables** (`source: env`) — for example
   `BOOKSMART_LLM_PROVIDER=fake`. These win over everything, so use them for
   a script or a one-off command.
2. **`config.toml`** (`source: config`) — what `booksmart config set` writes.
3. **The vendors' own variables** (`source: conventional-env`) —
   `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`. API keys only. A
   key you already export for other tools just works, but a value you
   deliberately `config set` beats it.
4. **Defaults** (`source: default`).

```console
$ BOOKSMART_LLM_PROVIDER=fake booksmart config get llm_provider
fake
```

## The config file

`config.toml` is plain TOML, one `field = "value"` line per setting:

```toml
llm_provider = "gemini"
embedding_provider = "gemini"
```

It is created with mode `0600`, because it can hold API keys; the CLI warns if
it ever finds the file readable by other users. You can edit it by hand —
`config set` preserves your comments and formatting when it rewrites it. An
unknown field name in the file is an error, so a misspelt setting cannot be
silently ignored.

## Fields

| Field | Default | What it chooses |
| --- | --- | --- |
| `llm_provider` | `anthropic` | The Provider for language-model calls: `anthropic`, `openai`, `gemini`, or `fake`. |
| `llm_model` | the Provider's default | Which model that Provider uses. |
| `llm_reasoning_effort` | unset | How much the model reasons before answering (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`), for the `openai` and `gemini` Providers. Which values a model accepts is fixed by the model. |
| `embedding_provider` | `openai` | The Provider for embeddings: `openai`, `gemini`, or `fake`. Anthropic has no embeddings API. |
| `embedding_model` | the Provider's default | Which embedding model that Provider uses. |
| `sparse_provider` | `fastembed` | Keyword (BM25) vectors, computed locally — no key needed. `fake` for tests. |
| `sparse_model` | the Provider's default | Which sparse model to use. |
| `embed_passages` | `false` | Whether an ingest also embeds the book's own text, cut into short passages, so search can return the words themselves. Off by default because it embeds roughly ten times as much text per book. Takes effect the next time a book's `embeddings` Stage runs; search passages with `--type passage`. |
| `anthropic_api_key` | unset | Needed when `llm_provider` is `anthropic`. |
| `openai_api_key` | unset | Needed when either Provider is `openai`. |
| `gemini_api_key` | unset | Needed when either Provider is `gemini`. |
| `database_url` | `~/.booksmart/booksmart.db` (SQLite) | Where books, Runs and knowledge are stored. |
| `storage_root` | `~/.booksmart/storage` | Where added book files are kept. |
| `qdrant_path` | `~/.booksmart/qdrant` | The embedded vector store's directory. |
| `qdrant_url` | `http://localhost:6333` | A Qdrant server. Ignored until you set it: while neither `qdrant_url` nor `qdrant_path` is set, the embedded store is used. Setting `qdrant_path` makes the embedded store win again. |

The `fake` Providers return deterministic canned output with no key, no network
and no cost. They are for trying the pipeline's mechanics and for tests — their
knowledge and search results are placeholders, not readings of your book.

!!! warning "Changing the embedding model means re-ingesting"

    The vector collection is locked to the embedding model, and the keyword
    model, it was built with. Search refuses to run against a collection built
    with a different one, rather than return rankings that look plausible and
    are silently wrong. Switching `embedding_provider` or `embedding_model`
    after books are ingested means starting the collection again and
    re-ingesting them.

## Moving the whole installation

`BOOKSMART_HOME` is the one setting that is not a field: it moves the directory
everything above lives in, `config.toml` included. Point it somewhere else to
keep a separate set of books — a throwaway one for experiments, say:

```console
$ BOOKSMART_HOME=/tmp/booksmart-scratch booksmart books list
No books registered yet. Add one with `booksmart add <file>`.
```

To use a Qdrant server instead of the embedded store, set `qdrant_url`:

```console
$ booksmart config set qdrant_url http://localhost:6333
```
