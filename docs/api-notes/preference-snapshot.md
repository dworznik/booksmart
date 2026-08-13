# Preference Snapshot: rebuild Settings from a FULL model_dump

Not a removed unit — a footnote for a future server consumer, recorded here so it isn't
rediscovered painfully. See CONTEXT.md's *Preference Snapshot* term.

A durable run must behave consistently across steps even if config or a deploy
changes mid-run. So the Runner resolves `Settings` **once** when the run is
triggered and carries that snapshot to every step, rather than reconstructing
`Settings()` inside each durable step.

The trap: `pydantic-settings` fills any field you omit from the **live
environment**. So a snapshot must serialize the *entire* settings object and
rehydrate from that full mapping — never a partial dict:

```python
# At trigger time, on the machine that owns the decision:
snapshot: dict = settings.model_dump()          # FULL dump — every field

# Inside each durable step, reconstruct from the snapshot ONLY:
settings = Settings.model_validate(snapshot)     # no env reads fill gaps,
                                                 # because nothing is omitted
```

If you snapshot a partial dict (say, only `llm_model`) and rebuild with
`Settings(**partial)`, every unlisted field — provider, keys, qdrant url — is
re-read from whatever env the consumer's process happens to have, silently
diverging from the run's intent. A full `model_dump()` closes that hole.

Limits are **never** snapshotted (they are provider/API facts enforced live);
only Preferences are. `booksmart_core.config.Settings` reads no env on import, so
the consumer constructs it explicitly and controls exactly when env is read.
