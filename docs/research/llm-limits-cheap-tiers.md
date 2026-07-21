# LLM limits for the cheap tiers (issue #47) — findings

> Research notes for [#47](https://github.com/dworznik/booksmart/issues/47): the
> per-vendor `LLMLimits` tables in `packages/core/src/booksmart_core/llm.py` cover
> only each vendor's frontier tier, so `gpt-5-mini`, `gpt-5-nano`, and
> `claude-haiku-4-5` fall through `resolve_limits()` to a guessed vendor default.
> This establishes the two `LLMLimits` fields for those three models from vendor
> primary sources, and re-checks the existing entries against the same sources.
>
> **Frozen at [`fe99e43`](https://github.com/dworznik/booksmart/tree/fe99e43)**
> (2026-07-21), the commit that merged this research with its implementation.
> Vendor facts are true as of that date and are not re-verified as models ship —
> read the tables in `llm.py` for what the code enforces (see
> [`docs/agents/domain.md`](../agents/domain.md#living-docs-vs-frozen-docs)).

## Summary

- **`gpt-5-mini` and `gpt-5-nano`: `max_output_tokens=128000`,
  `valid_reasoning_efforts=("minimal", "low", "medium", "high")`.** Both model
  pages document a 400k context window and **128,000 max output tokens** — the
  same output ceiling as `gpt-5.5`, not the 32000 the vendor default guesses. The
  effort set differs from `gpt-5.5`'s in **both** directions: mini/nano accept
  `"minimal"` (which `gpt-5.5` rejects) and reject `"none"` and `"xhigh"` (which
  `gpt-5.5` accepts). The OpenAI spec text is explicit that only `gpt-5.1`-and-later
  models support `none`, and that `xhigh` arrived after `gpt-5.1-codex-max`; **all
  three tuples are additionally confirmed against the live API** (§1.3), with `gpt-5.5`
  reproducing its existing table entry exactly as a control.
- **`claude-haiku-4-5`: `max_output_tokens=20000`, `valid_reasoning_efforts=None`.**
  The *model's* documented ceiling is **64k** output tokens (not 128k like Opus 4.8 /
  Sonnet 5), but that number is unreachable from this codebase: `AnthropicProvider`
  calls `messages.create()` **non-streaming**, and the SDK refuses any non-streaming
  request whose `max_tokens` exceeds **21,333**. So the same 20000 the frontier
  entries use applies here — for a different reason than the current comment gives
  (it says "both models cap output at 128k"; Haiku caps at 64k, and the number that
  actually binds is the SDK's, not the model's).
- **The ~21.3k SDK ceiling is model-independent**, derived from a hardcoded
  `128_000` denominator in `_calculate_nonstreaming_timeout` — see §2.1 for the
  arithmetic. There is no per-model non-streaming entry for any Claude 4.5+ model.
- **`valid_reasoning_efforts=None` for Anthropic is still correct.**
  `AnthropicProvider.__init__` takes no `reasoning_effort` argument, `complete()`
  passes no effort/thinking parameter, and `build_llm_provider` does not forward
  `settings.llm_reasoning_effort` to it. Nothing to validate. (Anthropic *does* have
  an `effort` parameter — we simply never send it.)
- **All five existing entries check out** against the same sources. Two stale
  *comments* (not values) are flagged in §5, plus one observation about `gpt-5.5`
  not appearing in the installed SDK's `ChatModel` literal.

Every claim below is cited to a primary source: the installed SDK source in this
repo's venv (`anthropic==0.116.0`, `openai==2.44.0` — the most authoritative
description of the SDKs' own behaviour), official vendor documentation, or a
`path:line` into this repo. Claims that could not be confirmed from a primary
source are labelled as such in §6.

---

## Proposed table entries

```python
_ANTHROPIC_LLM_LIMITS = {
    # Model output caps differ (Opus 4.8 and Sonnet 5 cap at 128k, Haiku 4.5 at
    # 64k), but none of them is the binding limit: this provider calls the API
    # non-streaming and the SDK refuses non-streaming requests above 21,333
    # tokens ("streaming is required for operations that may take longer than
    # 10 minutes" — anthropic/_base_client.py:769-778, where the check is
    # 3600 * max_tokens / 128_000 > 600, independent of the model). So the
    # usable Limit is the non-streaming ceiling, the same for every model.
    # valid_reasoning_efforts stays None: the Anthropic provider does not take
    # the reasoning-effort Preference, so there is nothing to validate.
    "claude-opus-4-8": LLMLimits(max_output_tokens=20000),
    "claude-sonnet-5": LLMLimits(max_output_tokens=20000),
    "claude-haiku-4-5": LLMLimits(max_output_tokens=20000),
}
_ANTHROPIC_LLM_DEFAULT = LLMLimits(max_output_tokens=20000)

_OPENAI_LLM_LIMITS = {
    # gpt-5.5 does not accept "minimal" (unlike earlier gpt-5 models).
    "gpt-5.5": LLMLimits(
        max_output_tokens=128000,
        valid_reasoning_efforts=("none", "low", "medium", "high", "xhigh"),
    ),
    # The gpt-5 generation is the mirror image of gpt-5.5: it accepts "minimal"
    # but not "none" (only gpt-5.1 and later support it) and not "xhigh" (added
    # after gpt-5.1-codex-max). Same 128k output ceiling as gpt-5.5.
    "gpt-5-mini": LLMLimits(
        max_output_tokens=128000,
        valid_reasoning_efforts=("minimal", "low", "medium", "high"),
    ),
    "gpt-5-nano": LLMLimits(
        max_output_tokens=128000,
        valid_reasoning_efforts=("minimal", "low", "medium", "high"),
    ),
}
_OPENAI_LLM_DEFAULT = LLMLimits(max_output_tokens=32000)
```

`_GEMINI_LLM_LIMITS` needs no change (see §4).

---

## 1. OpenAI — `gpt-5-mini`, `gpt-5-nano`

### 1.1 `max_output_tokens = 128000`

| Model | Context window | Max output tokens | Source |
|---|---|---|---|
| `gpt-5-mini` | 400,000 | **128,000** | <https://developers.openai.com/api/docs/models/gpt-5-mini> |
| `gpt-5-nano` | 400,000 | **128,000** | <https://developers.openai.com/api/docs/models/gpt-5-nano> |
| `gpt-5` | 400,000 | 128,000 | <https://developers.openai.com/api/docs/models/gpt-5> |

(The `platform.openai.com/docs/models/...` URLs 301-redirect to
`developers.openai.com/api/docs/models/...`; the developers.openai.com pages are the
same first-party model pages.)

So the vendor default's `32000` under-reports both models by 4×. Nothing in the
OpenAI path caps this further: `OpenAIProvider.complete` passes
`max_completion_tokens=self.max_output_tokens`
(`packages/core/src/booksmart_core/llm.py:279`), and the SDK documents
`max_completion_tokens` as *"An upper bound for the number of tokens that can be
generated for a completion, including visible output tokens and reasoning tokens"*
(`.venv/lib/python3.13/site-packages/openai/types/chat/completion_create_params.py:113-118`)
— an upper bound, with no client-side validation and no streaming requirement.

### 1.2 `valid_reasoning_efforts = ("minimal", "low", "medium", "high")`

The full enum the SDK's type accepts is six values:

```python
ReasoningEffort: TypeAlias = Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh"]]
```
Source: `.venv/lib/python3.13/site-packages/openai/types/shared/reasoning_effort.py:8`
(identical in `shared_params/reasoning_effort.py:10`).

Which subset each model accepts is documented in the OpenAPI-generated docstring on
the `reasoning_effort` field, verbatim
(`.venv/lib/python3.13/site-packages/openai/types/chat/completion_create_params.py:208-223`;
the same text appears on the Responses-API `Reasoning.effort` field,
`openai/types/shared/reasoning.py:28-41`):

> Currently supported values are `none`, `minimal`, `low`, `medium`, `high`, and `xhigh`.
> […]
> - `gpt-5.1` defaults to `none`, which does not perform reasoning. The supported reasoning values for `gpt-5.1` are `none`, `low`, `medium`, and `high`. […]
> - **All models before `gpt-5.1` default to `medium` reasoning effort, and do not support `none`.**
> - The `gpt-5-pro` model defaults to (and only supports) `high` reasoning effort.
> - **`xhigh` is supported for all models after `gpt-5.1-codex-max`.**

`gpt-5-mini` and `gpt-5-nano` are "models before `gpt-5.1`" (they shipped with `gpt-5`
— see the SDK's `ChatModel` literal,
`.venv/lib/python3.13/site-packages/openai/types/shared/chat_model.py:24-29`, which
groups `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-2025-08-07`,
`gpt-5-mini-2025-08-07`, `gpt-5-nano-2025-08-07`). Therefore, from the two bullets
above:

- **`none` is out** — "All models before `gpt-5.1` … do not support `none`."
- **`xhigh` is out** — it postdates `gpt-5.1-codex-max`, which itself postdates `gpt-5`.

That leaves `minimal`, `low`, `medium`, `high`. The positive confirmation for
`minimal` on this generation is the `gpt-5` model page, which lists the supported
reasoning effort values as **"minimal, low, medium, and high"**
(<https://developers.openai.com/api/docs/models/gpt-5>). The reasoning guide adds
only that *"support for these values is model-dependent… Some models support only a
subset of these values, so check the relevant model page before choosing a setting"*
(<https://developers.openai.com/api/docs/guides/reasoning>).

The `gpt-5-mini` and `gpt-5-nano` model pages themselves do *not* enumerate the effort
values — they only state "Reasoning token support" — so the tuple above was initially
an inference from (a) the two spec bullets, which exclude `none` and `xhigh` for the
whole pre-5.1 generation, and (b) the `gpt-5` page's enumeration for the same
generation. **It has since been confirmed directly against the live API** (§1.3).

This is exactly the mirror image of the existing `gpt-5.5` entry, and the issue's
guess ("the mini/nano tuple likely differs from `gpt-5.5`'s") is right.

### 1.3 Confirmed against the live API

The docs leave the `minimal` question open for mini/nano, so it was settled empirically:
one Chat Completions call per (model, effort) across all six values in the SDK's
`ReasoningEffort` literal, `max_completion_tokens=2000`.

| Model | Accepted | Rejected |
|---|---|---|
| `gpt-5-mini` | `minimal`, `low`, `medium`, `high` | `none`, `xhigh` |
| `gpt-5-nano` | `minimal`, `low`, `medium`, `high` | `none`, `xhigh` |
| `gpt-5.5` | `none`, `low`, `medium`, `high`, `xhigh` | `minimal` |

Rejections come back as HTTP 400 `Unsupported value: 'reasoning_effort' does not
support '<x>' with this model`. `gpt-5.5` was probed as a control: it reproduces the
existing table entry exactly, which is what licenses trusting the same method on the
two new rows.

**One trap worth recording.** A first pass used `max_completion_tokens=16` and read
*any* 400 as a rejection. That misreads the result: an unsupported effort and a
too-small token budget both 400, but the second one says `Could not finish the message
because max_tokens or model output limit was reached` — which means the effort *was*
accepted and the budget merely failed to fit the reasoning tokens. Under the small
budget, `minimal` on mini/nano and `xhigh` on `gpt-5.5` both hit that second error and
would have been scored as rejections, inverting two of the three answers. The
discriminating signal is the *kind* of 400, not the presence of one. (That `xhigh` on
`gpt-5.5` is genuinely valid — it is in the shipped table — is the tell that exposed
the bug.)

---

## 2. Anthropic — `claude-haiku-4-5`

### 2.1 `max_output_tokens = 20000` — the SDK ceiling, not the model ceiling

**The model's ceiling is 64k.** The vendor comparison table gives Claude Haiku 4.5 a
200k-token context window and a **64k-token max output** — distinct from Opus 4.8
and Sonnet 5, which are 1M / **128k**:

| | Claude Opus 4.8 | Claude Sonnet 5 | Claude Haiku 4.5 |
|---|---|---|---|
| Claude API alias | `claude-opus-4-8` | `claude-sonnet-5` | `claude-haiku-4-5` |
| Claude API ID | `claude-opus-4-8` | `claude-sonnet-5` | `claude-haiku-4-5-20251001` |
| Context window | 1M tokens | 1M tokens | 200k tokens |
| **Max output** | **128k tokens** | **128k tokens** | **64k tokens** |

Source: <https://platform.claude.com/docs/en/about-claude/models/overview> ("Latest
models comparison" table; `docs.claude.com/en/docs/about-claude/models/overview`
302-redirects here). Cross-checked against the bundled `claude-api` skill's model
catalog (`claude-api/shared/models.md`, "Current Models" table), which gives the same
figures: Opus 4.8 and Sonnet 5 at 1M / 128K, Haiku 4.5 at 200K / 64K.

The docs also note the 128k figures apply to the *synchronous* Messages API; the
Batches API can go to 300k with a beta header. We use the synchronous API, so the
batch number is irrelevant.

**The number that actually binds is 21,333 — the SDK's non-streaming ceiling.**
`AnthropicProvider.complete` calls `self._client.messages.create(...)` with no
`stream=True` (`packages/core/src/booksmart_core/llm.py:223-228`), i.e. the
non-streaming path. On that path, `messages.create` computes a timeout via

```python
timeout = self._client._calculate_nonstreaming_timeout(
    max_tokens, MODEL_NONSTREAMING_TOKENS.get(model, None)
)
```
Source: `.venv/lib/python3.13/site-packages/anthropic/resources/messages/messages.py:1031-1032`
(same call at `:1229`, and in `resources/beta/messages/messages.py:1204`, `:1317`).

And that function is:

```python
def _calculate_nonstreaming_timeout(self, max_tokens: int, max_nonstreaming_tokens: int | None) -> Timeout:
    maximum_time = 60 * 60
    default_time = 60 * 10

    expected_time = maximum_time * max_tokens / 128_000
    if expected_time > default_time or (max_nonstreaming_tokens and max_tokens > max_nonstreaming_tokens):
        raise ValueError(
            "Streaming is required for operations that may take longer than 10 minutes. "
            + "See https://github.com/anthropics/anthropic-sdk-python#long-requests for more details",
        )
    return Timeout(default_time, connect=5.0)
```
Source: `.venv/lib/python3.13/site-packages/anthropic/_base_client.py:769-782`.

**Deriving the ~21.3k figure.** The guard raises when
`expected_time > default_time`, i.e. when

```
3600 * max_tokens / 128_000 > 600
⇔ max_tokens > 600 * 128_000 / 3600
⇔ max_tokens > 21_333.33…
```

So the largest `max_tokens` a non-streaming `messages.create` will accept is
**21,333**; 21,334 raises `ValueError`. Note the `128_000` is a **hardcoded constant
in the timeout heuristic** (a nominal tokens-per-hour throughput), *not* a lookup of
the model's output cap — so this bound is **identical for every Claude model**,
including 64k-output Haiku 4.5. The current `20000` sits comfortably under it, and
is the right value for the new entry too.

**No tighter per-model cap applies.** The second half of the guard consults
`MODEL_NONSTREAMING_TOKENS`, whose *entire* contents are:

```python
MODEL_NONSTREAMING_TOKENS = {
    "claude-opus-4-20250514": 8_192,
    "claude-opus-4-0": 8_192,
    "claude-4-opus-20250514": 8_192,
    "anthropic.claude-opus-4-20250514-v1:0": 8_192,
    "claude-opus-4@20250514": 8_192,
    "claude-opus-4-1-20250805": 8192,
    "anthropic.claude-opus-4-1-20250805-v1:0": 8192,
    "claude-opus-4-1@20250805": 8192,
}
```
Source: `.venv/lib/python3.13/site-packages/anthropic/_constants.py:20-29`
(`anthropic==0.116.0`).

Only Opus 4 and Opus 4.1 have a tighter (8,192) non-streaming cap. `claude-haiku-4-5`,
`claude-haiku-4-5-20251001`, `claude-opus-4-8`, and `claude-sonnet-5` are all absent,
so `MODEL_NONSTREAMING_TOKENS.get(model)` returns `None` for them and only the 21,333
bound applies.

`claude-haiku-4-5` is a valid model ID in the installed SDK
(`.venv/lib/python3.13/site-packages/anthropic/types/model.py`, `Model` literal —
lists both `claude-haiku-4-5` and `claude-haiku-4-5-20251001`, alongside
`claude-opus-4-8` and `claude-sonnet-5`).

### 2.2 `valid_reasoning_efforts = None` — verified, the provider never sends one

The current `None` is not a "we don't know" — it is "there is nothing to validate",
and that is still true in the code:

- `AnthropicProvider.__init__` has no `reasoning_effort` parameter at all
  (`packages/core/src/booksmart_core/llm.py:208-220`) — unlike `OpenAIProvider`,
  which does (`:247-254`) and calls `_validate_reasoning_effort` (`:259-261`).
- `AnthropicProvider.complete` sends only `model`, `max_tokens`, `system`, `messages`
  (`llm.py:223-228`). No `effort`, no `thinking`.
- `build_llm_provider` forwards `settings.llm_reasoning_effort` to `GeminiProvider`
  (`llm.py:419`) and `OpenAIProvider` (`llm.py:424`) but **not** to `AnthropicProvider`
  (`llm.py:414`).

So no user Preference can reach the Anthropic API as an effort value, and
`_validate_reasoning_effort` is never called on the Anthropic path. `None` is
correct for all three Anthropic entries.

For the record, Anthropic *does* expose an `effort` parameter — the models overview
notes *"On Claude Opus 4.8, the `effort` parameter defaults to `high` on all
surfaces… On Claude Sonnet 5, it defaults to `high` on the Claude API and Claude
Code"* (<https://platform.claude.com/docs/en/about-claude/models/overview>). Wiring it
up would be a feature, not part of #47; if it ever is wired up, these `None`s become
real tuples and Haiku 4.5's row will need its own set (Haiku 4.5 supports *extended*
thinking, not adaptive thinking, per the same comparison table — a different
mechanism from Opus/Sonnet's `effort`).

---

## 3. Sanity-check: existing OpenAI entry (`gpt-5.5`)

**Both fields are correct.**

- `max_output_tokens=128000` — the model page gives a **1,050,000** context window and
  **128,000** max output tokens (<https://developers.openai.com/api/docs/models/gpt-5.5>).
- `valid_reasoning_efforts=("none", "low", "medium", "high", "xhigh")` — the same page
  states the `reasoning.effort` parameter supports **"none, low, medium (default), high
  and xhigh"**. `minimal` is absent, which is exactly what the existing code comment
  says. Consistent with the SDK spec bullets in §1.2: `gpt-5.5` is after `gpt-5.1` (so
  `none` is supported) and after `gpt-5.1-codex-max` (so `xhigh` is supported).

---

## 4. Sanity-check: existing Gemini entries

**Both entries are correct; no change needed.**

- `max_output_tokens=65536` for both — the model pages give an **Output token limit of
  65,536** (and an input limit of 1,048,576) for both Gemini 2.5 Pro and Gemini 2.5
  Flash. Sources: <https://ai.google.dev/gemini-api/docs/models/gemini-2.5-pro>,
  <https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash>.
- The effort sets and the code comment ("Thinking cannot be turned off on 2.5 Pro: it
  rejects `none` but accepts `minimal` (a 1024-token thinking budget); Flash accepts
  both") are confirmed by the OpenAI-compatibility page
  (<https://ai.google.dev/gemini-api/docs/openai>), which maps `reasoning_effort` to
  Gemini thinking budgets:

  | OpenAI parameter | Gemini 2.5 |
  |---|---|
  | `minimal` | 1,024 |
  | `low` | 1,024 |
  | `medium` | 8,192 |
  | `high` | 24,576 |

  and states that `reasoning_effort` may be set to `"none"` to disable thinking on
  Gemini 2.5 models, but that **"Reasoning cannot be turned off for Gemini 2.5 Pro or
  3 models."** Hence Pro's tuple omits `none` and Flash's includes it — as coded.

The `extra_body` workaround in `OpenAIProvider.complete` (`llm.py:274-281`) is what
lets `"none"` reach Gemini's compat layer. It still works; only its stated *reason* is
now stale — see §5.

---

## 5. Discrepancies found in the existing code

1. **`llm.py:89-92` — the Anthropic table comment is now half-wrong.**
   "Both models cap output at 128k" is true of Opus 4.8 and Sonnet 5 but **not** of
   Haiku 4.5, which caps at 64k (§2.1). Adding Haiku means the comment must be
   reworded: the model caps *differ*, and the reason 20000 applies to all of them is
   that the SDK's model-independent 21,333 non-streaming bound binds first. (The
   comment's substantive claim — that the usable limit is the non-streaming ceiling,
   not the model's — is correct and is the thing worth preserving.) The proposed
   comment above does this.

2. **`llm.py:274-276` — the `extra_body` comment's rationale is stale.**
   It says: *"Via extra_body because Gemini's compat layer accepts `"none"`, which the
   OpenAI SDK's `reasoning_effort` type does not."* In the pinned `openai>=2.44.0`
   (`packages/core/pyproject.toml:13`; installed 2.44.0), `ReasoningEffort` **does**
   include `"none"`:
   `Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh"]]`
   (`openai/types/shared/reasoning_effort.py:8`). The workaround is harmless and still
   functions, but the stated reason no longer holds — presumably the SDK added `none`
   when `gpt-5.1` shipped. Worth a comment fix, not a behaviour change. (Out of scope
   for #47 but adjacent; flagging as asked.)

   **The workaround is still load-bearing, for a different reason** — worth recording so
   nobody deletes it on the strength of the paragraph above. `reasoning_effort` is typed
   `Literal["none", "minimal", "low", "medium", "high", "xhigh"] | Omit`, while a
   Preference reaches this seam as a plain `str` (from `Settings.llm_reasoning_effort`).
   Passing it directly fails `mypy --strict`:
   *`Argument "reasoning_effort" … has incompatible type "str | None"; expected
   "Literal[…] | None | Omit | None"`*. And widening the seam to that Literal would be
   wrong regardless: it enumerates the efforts **OpenAI** accepts, but the same
   `complete()` also serves Gemini's compat layer, whose valid set is its own. The
   effort is validated against this module's tables, not the SDK's type — `extra_body`
   is what keeps the SDK type from imposing one vendor's enum on every vendor.

3. **Observation, not a bug: `gpt-5.5` is not in the installed SDK's `ChatModel`
   literal.** `openai/types/shared/chat_model.py` (2.44.0) tops out at `gpt-5.4`. This
   is fine — the request field is typed `model: Required[Union[str, ChatModel]]`
   (`openai/types/chat/completion_create_params.py:49`), so a bare string is accepted
   and the API resolves it. Nothing to change; noting it so nobody "fixes" the default
   model on a type-checker complaint. (`gpt-5-mini` and `gpt-5-nano` *are* in the
   literal, so the two new entries are on firmer ground than the existing default.)

4. **Nothing wrong with the vendor defaults.** `_OPENAI_LLM_DEFAULT` (32000) and
   `_GEMINI_LLM_DEFAULT` (32000) remain sensibly conservative for genuinely unknown
   models; `_ANTHROPIC_LLM_DEFAULT` (20000) is correct by construction, since 21,333 is
   the model-independent hard bound for every non-streaming Anthropic call (§2.1).

---

## 6. Unverified / open questions

- **~~`minimal` on `gpt-5-mini` / `gpt-5-nano` is inferred, not directly quoted.~~
  RESOLVED — confirmed against the live API; see §1.3.** The docs still do not
  enumerate effort values on the mini/nano model pages, so §1.2's reasoning is what a
  future reader should re-run if these models change; the probe in §1.3 is what settles
  it today.
- **The OpenAI `max_output_tokens=128000` figures are documentation-only.** The §1.3
  probe exercised `reasoning_effort`, not the output ceiling — confirming 128k would
  mean paying for a 128k-token completion, which is not worth it. The number is taken
  from the vendor model pages (§1.1) and is a documented API fact, but unlike the effort
  tuples it has not been observed. Under-reporting it (as today's `32000` default does)
  is the safe direction of error, so the risk of trusting the docs here is low.
- **The 21,333 bound is the SDK's, not the API's.** The Anthropic *API* may well accept
  a non-streaming request with a larger `max_tokens`; we cannot know, because the
  client raises `ValueError` before any HTTP request is made
  (`anthropic/_base_client.py:775`). This does not matter — the table records what the
  provider code can actually use, and the provider goes through the SDK — but it means
  20000 is a *client-side* limit, and would change if `AnthropicProvider` ever switched
  to streaming (in which case Haiku's real limit becomes 64k and Opus/Sonnet's 128k).
- **Anthropic `effort` value sets are not recorded here.** Since `AnthropicProvider`
  never sends the parameter, its accepted values were not researched exhaustively.
  (The skill catalog notes Sonnet 5's `effort` supports `low`/`medium`/`high`/`xhigh`/`max`;
  that is a cross-check, not a vendor citation, and it is not needed for #47.)
- **The Models API would settle all of the Anthropic numbers programmatically.**
  `client.models.retrieve("claude-haiku-4-5").max_tokens` returns the model's max output
  tokens, and the docs recommend it over cached tables
  (<https://platform.claude.com/docs/en/about-claude/models/overview>, "You can query
  model capabilities and token limits programmatically with the Models API"). Not used
  here — it needs a key and a live call. Worth considering as a *test* (a keyed,
  optional CI check that the table has not drifted) rather than as runtime behaviour,
  which would contradict the module docstring's "Limits change only via the tables in
  this module".
