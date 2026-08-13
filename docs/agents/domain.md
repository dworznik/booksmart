# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root, or
- **`CONTEXT-MAP.md`** at the repo root if it exists — it points at one `CONTEXT.md` per context. Read each one relevant to the topic.
- **`docs/adr/`** — read ADRs that touch the area you're about to work in. In multi-context repos, also check `src/<context>/docs/adr/` for context-scoped decisions.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## Living docs vs frozen docs

Not every document under `docs/` tracks the code, and reading a frozen one as
current is how stale API shapes get copied into new work.

**Living** — must stay true, and a change that contradicts one is a bug in the
change or a reason to amend the doc:

- `CONTEXT.md` — the glossary.
- `docs/adr/` — decisions. Superseding an ADR means writing the next ADR, never
  editing the old one's decision.

**Frozen** — a record of what was known at one moment, correct as of the commit
it pins and not maintained afterwards. Every file in these trees carries a pin
banner naming that commit:

- `docs/research/` — pre-implementation research: primary-source vendor facts and
  a recommended shape, gathered before an issue was implemented.
- `docs/api-notes/` — what the removed HTTP surface did, for a consumer
  reimplementing it.
- `docs/prds/` — the product requirements a version was built against.

Do **not** refresh a frozen doc's code snippets when a seam moves — that erases
the record of what was actually known then, and buys nothing the code doesn't
already say. Read the code for the current shape; read these for the reasoning.
If a frozen doc's conclusion is now wrong (not merely dated), say so in the
issue or ADR that supersedes it.

## File structure

Single-context repo (most repos):

```
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-event-sourced-orders.md
│   └── 0002-postgres-for-write-model.md
└── src/
```

Multi-context repo (presence of `CONTEXT-MAP.md` at the root):

```
/
├── CONTEXT-MAP.md
├── docs/adr/                          ← system-wide decisions
└── src/
    ├── ordering/
    │   ├── CONTEXT.md
    │   └── docs/adr/                  ← context-specific decisions
    └── billing/
        ├── CONTEXT.md
        └── docs/adr/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders) — but worth reopening because…_
