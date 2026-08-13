---
name: verify
description: Drive the booksmart CLI end-to-end against a throwaway home dir to verify changes at the real surface.
---

# Verifying booksmart changes

The surface is the `booksmart` CLI (typer entry point `booksmart_cli.main:app`);
`uv run booksmart …` runs the workspace's editable install — no build step.

## Recipe

Isolate with a throwaway home and a scrubbed environment (the resolution chain
reads `BOOKSMART_*` and the conventional key vars, so strip both):

```bash
export BOOKSMART_HOME=$(mktemp -d)/vhome
env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY -u GEMINI_API_KEY \
    BOOKSMART_HOME=$BOOKSMART_HOME uv run booksmart <command>
```

Full pipeline with no keys/network: `booksmart config set llm_provider fake`
and `config set embedding_provider fake` (or the `BOOKSMART_*_PROVIDER=fake`
env vars), then a sample PDF:

```bash
uv run python -c "
import pymupdf
doc = pymupdf.open(); page = doc.new_page()
page.insert_text((72,72), '# Chapter One\n\nBody text.\n\n# Chapter Two\n\nMore body.')
doc.save('$BOOKSMART_HOME/book.pdf')"
```

Then `add` → `ingest <id>` → `search all "query"` exercises every stage,
SQLite auto-migration, embedded Qdrant, and the settings chain.

## Gotchas

- `booksmart config list/get` shows the effective value and source — use it to
  check precedence claims (env > config.toml > conventional key env > default).
- The embedded Qdrant dir holds a lock; run commands sequentially, and reuse
  the same `BOOKSMART_HOME` only after the prior command exited.
