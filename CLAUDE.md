# booksmart

Project instructions for coding agents. Domain language and architectural decisions live in
[`CONTEXT.md`](./CONTEXT.md) and [`docs/adr/`](./docs/adr/) — read those before changing behaviour.

## Commits

**Every commit message must follow [Conventional Commits](https://www.conventionalcommits.org).**

```
<type>(<optional scope>): <description>

<optional body>
```

`<type>` is one of: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `build`, `ci`, `chore`,
`revert`. Use `!` after the type/scope (`feat(store)!: …`) for a breaking change.

The subject line is imperative, lower-case, and has no trailing period. Scope is optional but
preferred — use the module it touches (`core`, `cli`, `search`, `vectors`, `stages`, `config`,
`agents`).

Commits predating this rule do not follow it; do not rewrite them. Everything from here does.

Explain **why** in the body, not what — the diff already says what. If a change is a judgement
call, a workaround, or contradicts an ADR, the body is where that belongs.

A `commit-msg` hook enforces this. Enable it once per clone:

```bash
git config core.hooksPath .githooks
```

Auto-generated merge, revert, and fixup subjects are exempt — git writes those, not us.

**Do not pass `--no-verify`.** It exists for genuine emergencies, and skipping the hook to avoid
writing a conventional subject is not one.

## Branches

`<type>/<short-kebab-description>`, reusing the commit type vocabulary above:
`feat/agent-skills-config`, `fix/dense-vector-validation`, `docs/mkdocs-site`.

Never commit directly to `main` — branch first, then open a PR.

## Agent skills

### Issue tracker

GitHub Issues in `dworznik/booksmart`, via the `gh` CLI. External PRs are **not** a triage
surface. See [`docs/agents/issue-tracker.md`](./docs/agents/issue-tracker.md).

### Triage labels

The five canonical roles verbatim: `needs-triage`, `needs-info`, `ready-for-agent`,
`ready-for-human`, `wontfix`. See [`docs/agents/triage-labels.md`](./docs/agents/triage-labels.md).

### Domain docs

Single-context: one `CONTEXT.md` and one `docs/adr/`, both at the repo root. See
[`docs/agents/domain.md`](./docs/agents/domain.md).
