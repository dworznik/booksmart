# Gemini LLM limits (issue #53) — findings

> Research notes for [#53](https://github.com/dworznik/booksmart/issues/53):
> `_GEMINI_LLM_LIMITS` (`packages/core/src/booksmart_core/llm.py`) held only
> `gemini-2.5-pro` and `gemini-2.5-flash`, so the tiers callers now select fell
> through `resolve_limits()` to a guessed vendor default, and the two rows that
> did exist named deprecated models. This establishes both `LLMLimits` fields for
> the current tiers and re-checks the existing rows.
>
> **Frozen at [`868c341`](https://github.com/dworznik/booksmart/tree/868c341)**
> (2026-07-21), the state of the code this research was gathered against and
> describes throughout; the entries it recommends landed in the PR that closes
> #53. Every number below is a live-API observation on that date; Gemini's
> roster churns faster than any other vendor's here, so treat them as
> refreshable and read the tables in `llm.py` for what the code enforces (see
> [`docs/agents/domain.md`](../agents/domain.md#living-docs-vs-frozen-docs)).

## Summary

- **All four live models share `max_output_tokens=65536`**, straight from the
  Models API's own `outputTokenLimit` field (§1) — half again what the
  `_GEMINI_LLM_DEFAULT` guess (32000) reports.
- **The flash tiers accept the endpoint's full effort set**,
  `("none", "minimal", "low", "medium", "high")` — confirmed one call per
  (model, effort) against the live compat endpoint (§2). This is what
  `gemini-2.5-flash` already had, so the existing row is re-verified unchanged.
- **`gemini-3.1-pro-preview` accepts only `("low", "medium", "high")`.** It is
  stricter than the `gemini-2.5-pro` row it replaces, which listed `minimal`:
  3.1 Pro rejects `none` *and* `minimal` with two distinct errors (§2.2).
- **The issue understates gap 2: the deprecated ids do not merely have a
  shutdown date, they already 404.** `gemini-2.5-pro`, `gemini-2.5-flash-lite`
  and `gemini-3-pro-preview` all answer HTTP 404 *"no longer available to new
  users"* for this key (§3). The table's only pro entry named a model that
  cannot be called.
- **Consequence the issue does not mention: `DEFAULT_MODELS["gemini"]` was
  `gemini-2.5-pro`** (`llm.py:51`), so selecting the Gemini provider without
  naming a model 404'd at the first call (§4).
- **`models.list` is not a liveness signal.** It still advertises all three dead
  ids, with full metadata. Only a completion call distinguishes them (§3.2) —
  which is why every row in the shipped table was probed with a real call.

Every claim below is a live observation against
`https://generativelanguage.googleapis.com/v1beta/` on 2026-07-21, using the
same OpenAI-compat path `GeminiProvider` itself uses (`openai==2.44.0`).

---

## Proposed table entries

```python
_GEMINI_LLM_LIMITS = {
    "gemini-3.1-flash-lite": LLMLimits(
        max_output_tokens=65536,
        valid_reasoning_efforts=("none", "minimal", "low", "medium", "high"),
    ),
    "gemini-3.5-flash": LLMLimits(
        max_output_tokens=65536,
        valid_reasoning_efforts=("none", "minimal", "low", "medium", "high"),
    ),
    "gemini-2.5-flash": LLMLimits(
        max_output_tokens=65536,
        valid_reasoning_efforts=("none", "minimal", "low", "medium", "high"),
    ),
    "gemini-3.1-pro-preview": LLMLimits(
        max_output_tokens=65536,
        valid_reasoning_efforts=("low", "medium", "high"),
    ),
}
_GEMINI_LLM_DEFAULT = LLMLimits(max_output_tokens=32000)   # unchanged
DEFAULT_MODELS["gemini"] = "gemini-3.5-flash"              # was gemini-2.5-pro
```

(Comments omitted here; the shipped table carries them.)

---

## 1. `max_output_tokens = 65536`, from the Models API

`GET /v1beta/models?key=…` returns an `outputTokenLimit` per model — the vendor
declaring its own Limit, which beats any doc page:

| Model | `inputTokenLimit` | `outputTokenLimit` |
|---|---|---|
| `gemini-3.1-flash-lite` | 1,048,576 | **65,536** |
| `gemini-3.1-pro-preview` | 1,048,576 | **65,536** |
| `gemini-3.5-flash` | 1,048,576 | **65,536** |
| `gemini-2.5-flash` | 1,048,576 | **65,536** |

So the `32000` vendor default under-reports every current model by half. Nothing
in the Gemini path lowers this further: `OpenAIProvider.complete` passes
`max_completion_tokens=self.max_output_tokens`, and unlike the Anthropic path
there is no client-side non-streaming Limit (that bound comes from the
`anthropic` SDK, which is not involved here). So the declared
`outputTokenLimit` is the Limit the code can actually use.

The full listing carried 54 models; the table above is the subset this repo
selects from. Multimodal and live variants (`-image`, `-tts`, `-native-audio`,
`-live`) report different, smaller limits and are out of scope — this provider
only does text completion.

## 2. `valid_reasoning_efforts`, from one call per (model, effort)

Method: one Chat Completions call per pair through the compat endpoint, with the
`reasoning_effort` in `extra_body` exactly where `OpenAIProvider.complete` puts
it, `max_completion_tokens=64`. All six values in the OpenAI SDK's
`ReasoningEffort` literal were tried, so the probe covers `xhigh` too.

| Model | Accepted | Rejected |
|---|---|---|
| `gemini-3.1-flash-lite` | `none`, `minimal`, `low`, `medium`, `high` | `xhigh` |
| `gemini-3.5-flash` | `none`, `minimal`, `low`, `medium`, `high` | `xhigh` |
| `gemini-2.5-flash` | `none`, `minimal`, `low`, `medium`, `high` | `xhigh` |
| `gemini-3.1-pro-preview` | `low`, `medium`, `high` | `none`, `minimal`, `xhigh` |

### 2.1 The endpoint enumerates its own universe

`xhigh` fails on *every* model with the same message, and it names the whole set:

```
400 INVALID_ARGUMENT: Invalid reasoning_effort: xhigh.
Valid values are: high, low, medium, minimal, none
```

That is a parameter-level whitelist, checked before any model-specific rule — so
Gemini's effort universe is the five values, and `xhigh` is OpenAI's alone. It
also means no Gemini row can ever contain `xhigh`, and a future model can only
ever be a *subset* of the five.

### 2.2 The pro tier's two distinct refusals

`gemini-3.1-pro-preview` rejects the bottom two efforts for different stated
reasons, both HTTP 400:

- `none` → *"Budget 0 is invalid. This model only works in thinking mode."*
- `minimal` → *"Thinking level MINIMAL is not supported for this model. Please
  retry with other thinking level."*

The first matches the old `gemini-2.5-pro` behaviour the previous code comment
described ("thinking cannot be turned off"). The second is new: 2.5 Pro accepted
`minimal` as a 1024-token thinking budget, and 3.1 Pro has no level below `low`.
Carrying the old tuple forward to the new model would therefore have been wrong
in exactly one value — an argument for probing rather than porting.

### 2.3 "Accepted" means accepted, not answered

Two accepted pairs (`gemini-3.1-flash-lite` at `low`, `gemini-3.1-pro-preview`
at `medium`) returned HTTP 200 with `finish_reason="length"` and empty content:
the 64-token budget went to reasoning. That is the trap #47's research recorded
(§1.3 there) — a token-starved success is not a rejection. Classifying on HTTP
status rather than on whether text came back is what keeps the two apart; had
this probe scored empty content as failure, both rows would be wrong.

## 3. The deprecated ids are already gone

| Model | Completion call |
|---|---|
| `gemini-2.5-pro` | 404 *"no longer available to new users"* |
| `gemini-2.5-flash-lite` | 404 *"no longer available to new users"* |
| `gemini-3-pro-preview` | 404 *"no longer available"* |
| `gemini-2.5-flash` | 200 — still live |

So of the two rows the table shipped, one names a model that cannot be called.
The issue expected a future shutdown ("shutdown ≥ 2026-10-16"); the id is
unusable now, at least for a key without prior access.

**The 404 wording is key-scoped.** *"No longer available to new users"* implies a
key with established usage may still reach it. The table cannot express
"depends on the key", so the entry is removed: a caller with legacy access gets
the vendor default and one warning, which is the documented behaviour for a model
the table does not know, and is the safe direction of error (it under-reports
`max_output_tokens` and skips effort validation rather than inventing either).

### 3.1 The vendor's own schedule, and what it recommends instead

The deprecations page (<https://ai.google.dev/gemini-api/docs/deprecations>)
gives every 2.5 model the same shutdown date and names a replacement:

| Model | Shutdown date | Vendor's recommended replacement |
|---|---|---|
| `gemini-2.5-pro` | 2026-10-16 | `gemini-3.1-pro-preview` |
| `gemini-2.5-flash` | 2026-10-16 | `gemini-3.5-flash` |
| `gemini-2.5-flash-lite` | 2026-10-16 | `gemini-3.1-flash-lite` |

Two things follow. First, the replacements Google names are exactly the four
models this table now holds — arrived at independently, from the probe, which is
a useful cross-check on the selection. Second, **the schedule under-describes
reality**: the page's own note says the listed dates "indicate the *earliest
possible dates* on which a model might be retired", yet two of these three
already 404 (§3), months ahead of the date. So the date is not a liveness
guarantee, and `gemini-2.5-flash` — still answering — is the next row to fall.

### 3.2 `models.list` lists the dead

All three 404ing ids appear in `models.list` with complete metadata —
`outputTokenLimit`, `supportedGenerationMethods: [generateContent, …]`, the lot.
The listing describes what the API knows about, not what it will answer. Any
future refresh of this table must probe with a **completion call**; reading
`models.list` alone would have kept `gemini-2.5-pro` and added
`gemini-2.5-flash-lite`, both dead.

## 4. The default model was one of the dead ids

`DEFAULT_MODELS["gemini"] = "gemini-2.5-pro"` (`llm.py:51`) is what
`build_llm_provider` uses when `Settings.llm_model` is None. Since that id 404s,
any deployment selecting `llm_provider = "gemini"` without naming a model failed
at its first call — a live break, not a warning, and invisible to the test suite
because no test called the API.

The recommendation is `gemini-3.5-flash`: probed live, and a *stable* id rather
than a preview one, which matters more for a fallback than raw model strength
does — it is what a deployment that expressed no opinion gets, so it should be
the entry least likely to move underneath it. `gemini-3.1-pro-preview` is the
stronger model but carries preview churn; a deployment that wants it can name it.
It is also the vendor's own nominated replacement for the flash tier (§3.1).

## 5. Open questions

- **Preview churn.** `gemini-3.1-pro-preview` is a preview id: its limits and its
  effort set may move without notice, and the id itself may retire the way
  `gemini-3-pro-preview` just did. Its row is the first to re-probe.
- **65,536 is declared, not observed.** As with #47's OpenAI figures, confirming
  the declared Limit would mean paying for a 65k-token completion. It is taken
  from `outputTokenLimit`, which is the vendor's own declaration rather than a
  doc page, and under-reporting is the safe direction of error.
- **The 404s are key-scoped** (§3), so a key with legacy access will see
  different behaviour from the one probed here. Nothing in the table can capture
  that, and the fallback path handles it.
- **A keyed drift test would settle this class of question permanently.** #47's
  research floated the same idea for Anthropic's Models API. The Gemini case is
  stronger: `models.list` is one unauthenticated-ish call that yields
  `outputTokenLimit` for every id in the table, so a keyed, optional CI check
  could assert the table has not drifted — without asserting liveness, which
  needs a real completion (§3.2). Deliberately not added here: it would put a
  network call in a suite whose keylessness is a documented property.
- **Models not tabulated.** `gemini-3.1-flash-lite-preview` (a preview alias of a
  tabulated model) and the multimodal/live variants were listed but not probed;
  they are outside what this provider does. `gemini-3-flash-preview` was listed
  and not probed.
