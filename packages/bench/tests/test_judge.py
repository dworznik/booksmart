"""The pinned cross-family faithfulness judge.

Driven here through a stub provider rather than core's fake: the fake answers
the *pipeline's* prompts, and the judge is deliberately not part of the
pipeline. Every test here is about the ruler — what it is allowed to be, what it
does with a source slice, and what it does when it cannot read its own answer.
"""

import pytest

from booksmart_bench.errors import BenchError
from booksmart_bench.judge import (
    JUDGE_MODEL_ENV,
    JUDGE_PROMPT_VERSION,
    JUDGE_PROVIDER_ENV,
    JudgeConfig,
    SummaryUnderTest,
    build_judge,
    judge_summaries,
    parse_claims,
    parse_verdict,
    resolve_judge,
)
from booksmart_core.config import Settings
from booksmart_core.llm import LLMResponse


class StubJudge:
    """A judge whose every answer is scripted, in call order."""

    def __init__(self, *answers: str, model: str = "stub-judge-1") -> None:
        self.model = model
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        self.prompts.append(prompt)
        self.systems.append(system)
        text = self.answers.pop(0) if self.answers else "{}"
        return LLMResponse(text=text, model=self.model, input_tokens=3, output_tokens=5)


def summary(text: str = "Grommets seal the joint.", source: str = "A grommet seals a joint.") -> SummaryUnderTest:
    return SummaryUnderTest(book="a-book", loc="1.2", summary=text, source=source)


SUPPORTED = '{"supported": true, "why": "the slice says so"}'
UNSUPPORTED = '{"supported": false, "why": "the slice does not say that"}'


class TestClaimParsing:
    def test_a_json_array_of_strings(self) -> None:
        assert parse_claims('["one", "two"]') == ("one", "two")

    def test_objects_with_a_claim_field_are_accepted(self) -> None:
        """Models drift between the two shapes across a long sweep; a run that
        died on the drift would cost the whole sweep."""
        assert parse_claims('[{"claim": "one"}, {"claim": "two"}]') == ("one", "two")

    def test_fenced_json_survives(self) -> None:
        assert parse_claims('```json\n["one"]\n```') == ("one",)

    def test_a_summary_with_no_claims_is_an_empty_tuple_not_an_error(self) -> None:
        """Not every summary asserts something checkable, and that is not a
        judge failure."""
        assert parse_claims("[]") == ()

    def test_prose_is_a_judge_error(self) -> None:
        with pytest.raises(BenchError):
            parse_claims("Sure! Here are the claims:")

    def test_an_element_in_neither_shape_is_a_judge_error(self) -> None:
        """Skipping it would shrink the denominator: faithfulness is
        supported/total, so a summary would be scored against a claim set
        nobody chose."""
        with pytest.raises(BenchError, match="claim 1"):
            parse_claims('["one", {"text": "two"}]')

    def test_a_blank_claim_is_a_judge_error(self) -> None:
        with pytest.raises(BenchError):
            parse_claims('["one", "   "]')


class TestVerdictParsing:
    def test_a_boolean_verdict(self) -> None:
        assert parse_verdict(SUPPORTED) is True
        assert parse_verdict(UNSUPPORTED) is False

    def test_a_missing_verdict_is_a_judge_error(self) -> None:
        """Reading an unusable answer as "unsupported" would invent a
        faithfulness failure out of a judge malfunction."""
        with pytest.raises(BenchError):
            parse_verdict('{"why": "hmm"}')


class TestJudging:
    def test_the_ratio_is_supported_over_total(self) -> None:
        judge = StubJudge('["a", "b", "c"]', SUPPORTED, UNSUPPORTED, SUPPORTED)

        report = judge_summaries(judge, [summary()])

        assert (report.verdicts[0].supported, report.verdicts[0].claims) == (2, 3)

    def test_each_claim_is_verified_against_the_source_slice(self) -> None:
        judge = StubJudge('["a"]', SUPPORTED)

        judge_summaries(judge, [summary(source="the exact slice")])

        assert "the exact slice" in judge.prompts[1]

    def test_claims_are_verified_one_at_a_time(self) -> None:
        """RAGAS-shaped: a claim verified beside its neighbours is verified in
        their company, which is not the question being asked."""
        judge = StubJudge('["a", "b"]', SUPPORTED, SUPPORTED)

        judge_summaries(judge, [summary()])

        assert len(judge.prompts) == 3  # one extraction, two verifications

    def test_judge_spend_is_recorded(self) -> None:
        """The judge is spend like any other spend, and lands in the cost
        dimension rather than vanishing into the score."""
        judge = StubJudge('["a"]', SUPPORTED)

        report = judge_summaries(judge, [summary()])

        assert report.input_tokens == 6 and report.output_tokens == 10
        assert report.seconds >= 0

    def test_a_summary_with_no_claims_is_reported_not_scored(self) -> None:
        judge = StubJudge("[]")

        report = judge_summaries(judge, [summary()])

        assert report.verdicts[0].claims == 0
        assert "no" in report.verdicts[0].note.lower()


class TestJudgeFailure:
    def test_an_unparseable_answer_is_retried_once(self) -> None:
        judge = StubJudge("not json", '["a"]', SUPPORTED)

        report = judge_summaries(judge, [summary()])

        assert report.verdicts[0].claims == 1

    def test_a_summary_the_judge_could_not_read_is_excluded_and_named(self) -> None:
        """A malfunctioning judge must never be scored as a failing summary."""
        judge = StubJudge("not json", "still not json")

        report = judge_summaries(judge, [summary()])

        assert report.verdicts[0].claims == 0
        assert report.verdicts[0].note
        assert any("a-book" in failure for failure in report.failures)

    def test_one_bad_summary_does_not_cost_the_sweep(self) -> None:
        judge = StubJudge("not json", "still not json", '["a"]', SUPPORTED)

        report = judge_summaries(
            judge, [summary(), SummaryUnderTest("a-book", "1.1", "Widgets mesh.", "Widgets mesh.")]
        )

        assert [j.claims for j in report.verdicts] == [0, 1]


class TestPinning:
    def test_a_judge_from_the_summarisers_own_family_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Self-preference bias: a model grading its own family's summaries is
        not a ruler, it is a mirror."""
        monkeypatch.setenv(JUDGE_PROVIDER_ENV, "openai")

        with pytest.raises(BenchError, match="cross-family"):
            resolve_judge(Settings(llm_provider="openai"))

    def test_a_hand_built_config_is_checked_too(self) -> None:
        """`JudgeConfig` is a plain value a caller can build; a rule that only
        guarded the environment would be a rule with a way around it."""
        with pytest.raises(BenchError, match="cross-family"):
            build_judge(
                JudgeConfig(provider="openai", model="a-model"),
                Settings(llm_provider="openai"),
            )

    def test_no_judge_configured_is_not_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Faithfulness is one dimension of five; an unconfigured judge means
        that dimension goes unmeasured and says so, not that a run dies."""
        monkeypatch.delenv(JUDGE_PROVIDER_ENV, raising=False)

        assert resolve_judge(Settings()) is None

    def test_the_judge_identity_is_pinned_and_versioned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(JUDGE_PROVIDER_ENV, "gemini")
        monkeypatch.setenv(JUDGE_MODEL_ENV, "a-judge-model")

        config = resolve_judge(Settings(llm_provider="anthropic"))

        assert config is not None
        assert (config.provider, config.model) == ("gemini", "a-judge-model")
        assert config.prompt_version == JUDGE_PROMPT_VERSION

    def test_a_judge_with_no_model_named_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A provider default is whatever the vendor decided this month, which
        is the ruler moving under you between runs."""
        monkeypatch.setenv(JUDGE_PROVIDER_ENV, "gemini")
        monkeypatch.delenv(JUDGE_MODEL_ENV, raising=False)

        with pytest.raises(BenchError, match=JUDGE_MODEL_ENV):
            resolve_judge(Settings(llm_provider="anthropic"))

    def test_an_unknown_judge_provider_names_the_providers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(JUDGE_PROVIDER_ENV, "a-vendor-that-does-not-exist")

        with pytest.raises(BenchError, match="gemini"):
            resolve_judge(Settings())
