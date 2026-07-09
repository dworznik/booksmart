# API notes

When `booksmart-core` became a library (issue #24), the FastAPI server was
removed: the app factory, the three routers, the Pydantic schemas, and the
request-scoped DB dependency. None of that is data-processing — it is HTTP
plumbing a *consumer* owns. A future server consumer can re-implement the
surface it wants on top of core; the CLI implements its
own thin front end.

This directory is the value assessment the split agreed to leave behind: for
every removed unit, what it did, which shapes are worth reusing, and a permalink
to the original so the reimplementation starts from working code rather than
memory.

**Pre-refactor revision:** everything below lived under `app/` at
[`589857c`](https://github.com/dworznik/booksmart/tree/589857c) (the commit this
PR branched from). Permalinks pin that SHA.

| Removed unit | Notes |
| --- | --- |
| `app/main.py` — app factory, engine/session/storage wiring | [http-surface.md](http-surface.md#app-wiring) |
| `app/routers/books.py` — register/list/get/patch, upload validation, dedup | [http-surface.md](http-surface.md#books), [upload-validation.md](upload-validation.md) |
| `app/routers/jobs.py` — trigger (202), reprocess, run listing | [http-surface.md](http-surface.md#runs) |
| `app/routers/knowledge.py` — knowledge-object listing/fetch | [http-surface.md](http-surface.md#knowledge) |
| `app/schemas.py` — response/request models | [http-surface.md](http-surface.md#schemas) |
| `app/db.py` — request-scoped `Session` dependency | [http-surface.md](http-surface.md#app-wiring) |
| `docker-compose.yml` / `Dockerfile` — server deployment + compose e2e | [deployment.md](deployment.md) |
| Settings construction across durable steps | [preference-snapshot.md](preference-snapshot.md) |
