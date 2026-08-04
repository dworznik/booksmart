"""The summary-faithfulness judge: a pinned, cross-family ruler.

Faithfulness is the one dimension no ground truth can hold. A summary is written
per run, so there is nothing to author it against — the only available check is
another model reading the summary beside the text it was written from. That
makes the judge a measuring instrument, and the design (SPEC §5) is entirely
about keeping the instrument still while the thing it measures moves:

- **Cross-family.** The judge always comes from a different provider family than
  the summariser under test. A model grading its own family's prose is a mirror,
  not a ruler — self-preference bias is the best-documented failure of
  LLM-as-judge setups, and it flatters exactly the pipeline change you are
  trying to evaluate.
- **Pinned, and outside the Preference Snapshot.** Judge provider, model and
  prompt version are bench configuration, changed deliberately, and stamped into
  every run file. The pipeline's own Preferences never reach it: a run that
  switched summariser *and* judge together would produce two numbers that cannot
  be compared to anything.
- **RAGAS-shaped, per claim.** Extract the summary's atomic claims, then verify
  each one on its own against the exact source slice it should have come from.
  One claim per call: a claim checked beside its neighbours is checked in their
  company, which is not the question.

**Temperature.** SPEC §5 asks for temperature 0. Core's provider seam does not
carry a temperature, and adding one would not buy it here: every model a
cross-family judge can currently be (the gpt-5 and Gemini 3 generations) is a
reasoning model whose API rejects an explicit temperature outright. Determinism
therefore rests where the design already put it — a pinned judge plus the
−0.05 faithfulness threshold absorb the sampling noise, and majority voting
stays the documented fallback if a single pass proves unstable in practice.

A judge that malfunctions must never be read as a summary that failed, so an
answer this module cannot parse (after one retry) excludes that summary from the
dimension and says so by name. The alternative — counting an unreadable verdict
as "unsupported" — manufactures a faithfulness regression out of a bad sample.
"""

import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from booksmart_core.config import Settings
from booksmart_core.llm import DEFAULT_MODELS, LLMProvider, build_llm_provider, strip_fences

from booksmart_bench.errors import BenchError

# Bump whenever either prompt below changes wording, exactly as the pipeline's
# extraction and summary prompts are versioned: a score produced under one
# prompt is not comparable with a score produced under another, and the run file
# stamps this so nobody has to remember which was which.
JUDGE_PROMPT_VERSION = "1"

JUDGE_PROVIDER_ENV = "BOOKSMART_BENCH_JUDGE_PROVIDER"
JUDGE_MODEL_ENV = "BOOKSMART_BENCH_JUDGE_MODEL"

CLAIM_SYSTEM_PROMPT = (
    "You break a summary of a technical book into atomic factual claims. "
    "Respond with a JSON array of strings only - no prose, no markdown fences. "
    "Each element is one self-contained claim, stated plainly and without "
    "pronouns or references to 'the chapter', 'the section' or 'the summary'. "
    "Split compound sentences into one claim each. Do not add anything the "
    "summary does not say. Return [] when the summary asserts nothing "
    "checkable."
)

VERIFICATION_SYSTEM_PROMPT = (
    "You verify one claim against an exact slice of a technical book. Respond "
    "with a JSON object only - no prose, no markdown fences - with fields: "
    '"supported" (boolean) and "why" (one sentence). Set "supported" to true '
    "only when the source text states the claim or directly entails it. Judge "
    "against the source text alone: whether the claim is true in general, "
    "plausible, or something you know from elsewhere is irrelevant. A claim the "
    "source neither states nor entails is not supported."
)


class JudgeError(BenchError):
    """The judge's answer could not be read as a verdict."""


@dataclass(frozen=True)
class JudgeConfig:
    """The pinned identity of the ruler, as stamped into a run file."""

    provider: str
    model: str
    prompt_version: str = JUDGE_PROMPT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
        }


@dataclass(frozen=True)
class SummaryUnderTest:
    """One summary and the exact text it should have been written from."""

    book: str
    loc: str  # a ToC node id; the join has already happened
    summary: str
    source: str


@dataclass(frozen=True)
class SummaryVerdict:
    book: str
    loc: str
    supported: int
    claims: int
    # Why this summary carries no ratio, when it carries none. Travels into the
    # run file so the scorer can report the reason instead of the reader having
    # to guess between "nothing to check" and "the judge broke".
    note: str = ""


@dataclass(frozen=True)
class JudgeReport:
    verdicts: tuple[SummaryVerdict, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    failures: tuple[str, ...] = ()


# --- configuration ----------------------------------------------------------


def resolve_judge(settings: Settings) -> JudgeConfig | None:
    """The configured judge, or None when there is none.

    An absent judge is a normal state, not an error: faithfulness is one
    dimension of five, and a run that measures the other four is still a run.
    The scorer reports the gap rather than scoring around it.
    """
    provider = os.environ.get(JUDGE_PROVIDER_ENV, "").strip()
    if not provider:
        return None
    if provider not in DEFAULT_MODELS:
        raise BenchError(
            f"Unknown judge provider {provider!r} in {JUDGE_PROVIDER_ENV}; "
            f"expected one of {', '.join(sorted(DEFAULT_MODELS))}."
        )
    if provider == settings.llm_provider:
        raise BenchError(
            f"The judge and the summariser under test are both {provider!r}. The judge must "
            "be cross-family (SPEC §5) — a model grading its own family's summaries measures "
            f"its own preferences. Set {JUDGE_PROVIDER_ENV} to a different provider."
        )

    model = os.environ.get(JUDGE_MODEL_ENV, "").strip()
    if not model:
        # Falling back to the provider's default would leave the ruler as
        # whatever the vendor decided this month — the one thing a pinned judge
        # is defined not to be. Naming it costs one environment variable.
        raise BenchError(
            f"{JUDGE_PROVIDER_ENV} is set but {JUDGE_MODEL_ENV} is not. The judge is "
            "pinned deliberately (SPEC §5): a run scored against a moving ruler is "
            f"comparable with nothing. Name a model, e.g. {DEFAULT_MODELS[provider]!r}."
        )
    return JudgeConfig(provider=provider, model=model)


def build_judge(config: JudgeConfig, settings: Settings) -> LLMProvider:
    """The judge's provider, built from the pipeline's keys but none of its
    Preferences — including reasoning effort, which belongs to the summariser's
    model and may not even be valid for the judge's."""
    return build_llm_provider(
        settings.model_copy(
            update={
                "llm_provider": config.provider,
                "llm_model": config.model,
                "llm_reasoning_effort": None,
            }
        )
    )


# --- prompts ----------------------------------------------------------------


def build_claim_prompt(summary: str) -> str:
    return "\n".join(
        ["Summary:", summary, "", "List the atomic claims this summary makes, as JSON."]
    )


def build_verification_prompt(claim: str, source: str) -> str:
    return "\n".join(
        [
            "Source text:",
            source,
            "",
            f"Claim: {claim}",
            "",
            "Is the claim supported by the source text? Answer as JSON.",
        ]
    )


# --- parsing ----------------------------------------------------------------


def parse_claims(text: str) -> tuple[str, ...]:
    """The claims in the judge's answer.

    Both shapes a model drifts between over a long sweep are accepted — bare
    strings and ``{"claim": ...}`` objects — because the alternative is losing a
    whole book's faithfulness to a formatting preference.
    """
    payload = _load(text, "claim list")
    if not isinstance(payload, list):
        raise JudgeError("judge returned no claim array")
    claims: list[str] = []
    for element in payload:
        if isinstance(element, str) and element.strip():
            claims.append(element.strip())
        elif isinstance(element, dict):
            claim = element.get("claim")
            if isinstance(claim, str) and claim.strip():
                claims.append(claim.strip())
    return tuple(claims)


def parse_verdict(text: str) -> bool:
    """Whether the judge said the claim was supported.

    A missing or non-boolean verdict raises rather than defaulting: reading an
    unusable answer as "unsupported" would invent a faithfulness failure out of
    a judge malfunction.
    """
    payload = _load(text, "verdict")
    if not isinstance(payload, dict):
        raise JudgeError("judge returned no verdict object")
    supported = payload.get("supported")
    if not isinstance(supported, bool):
        raise JudgeError("judge verdict has no boolean 'supported' field")
    return supported


def _load(text: str, what: str) -> object:
    try:
        return json.loads(strip_fences(text.strip()))
    except json.JSONDecodeError as exc:
        raise JudgeError(f"judge {what} is not valid JSON: {exc}") from exc


# --- the sweep --------------------------------------------------------------


def judge_summaries(
    llm: LLMProvider,
    summaries: Sequence[SummaryUnderTest],
    *,
    on_progress: Callable[[SummaryUnderTest], None] | None = None,
) -> JudgeReport:
    """Verify every summary, claim by claim, and report what that cost."""
    spend = _Spend()
    verdicts: list[SummaryVerdict] = []
    failures: list[str] = []
    started = time.monotonic()

    for summary in summaries:
        if on_progress is not None:
            on_progress(summary)
        try:
            verdicts.append(_judge_one(llm, summary, spend))
        except Exception as exc:
            # Deliberately broad. A sweep is the expensive half of a benchmark
            # run, and every failure mode here — an unreadable answer, a rate
            # limit, a slice too long for the judge's context — costs one
            # summary if it is caught and the whole sweep if it is not. The
            # summary is excluded from the dimension and named, never scored as
            # a failing one.
            failures.append(f"{summary.book}/{summary.loc}: {type(exc).__name__}: {exc}")
            verdicts.append(
                SummaryVerdict(
                    book=summary.book,
                    loc=summary.loc,
                    supported=0,
                    claims=0,
                    note=f"the judge could not verify this summary ({exc})",
                )
            )

    return JudgeReport(
        verdicts=tuple(verdicts),
        input_tokens=spend.input_tokens,
        output_tokens=spend.output_tokens,
        seconds=time.monotonic() - started,
        failures=tuple(failures),
    )


def _judge_one(llm: LLMProvider, summary: SummaryUnderTest, spend: "_Spend") -> SummaryVerdict:
    claims = _ask(llm, build_claim_prompt(summary.summary), CLAIM_SYSTEM_PROMPT, parse_claims, spend)
    if not claims:
        return SummaryVerdict(
            book=summary.book,
            loc=summary.loc,
            supported=0,
            claims=0,
            note="the judge found no checkable claims in this summary",
        )

    supported = sum(
        _ask(
            llm,
            build_verification_prompt(claim, summary.source),
            VERIFICATION_SYSTEM_PROMPT,
            parse_verdict,
            spend,
        )
        for claim in claims
    )
    return SummaryVerdict(
        book=summary.book, loc=summary.loc, supported=supported, claims=len(claims)
    )


class _Spend:
    """What the sweep has cost so far, accumulated across every call."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, input_tokens: int | None, output_tokens: int | None) -> None:
        self.input_tokens += input_tokens or 0
        self.output_tokens += output_tokens or 0


ParsedT = TypeVar("ParsedT")


def _ask(
    llm: LLMProvider,
    prompt: str,
    system: str,
    parse: Callable[[str], ParsedT],
    spend: _Spend,
) -> ParsedT:
    """One judge call plus its parse, retried once on an unreadable answer —
    the same allowance core's stages give a model's occasional bad sample."""
    for attempt in (1, 2):
        response = llm.complete(prompt, system=system)
        spend.add(response.input_tokens, response.output_tokens)
        try:
            return parse(response.text)
        except JudgeError:
            if attempt == 2:
                raise
    raise RuntimeError("unreachable")
