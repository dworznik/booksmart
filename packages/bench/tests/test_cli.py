"""The four verbs: one per artefact boundary, all stubbed for now.

Each verb resolves its inputs for real before refusing to do the work it does
not implement yet — so a misconfigured assets path is reported now, not in the
wave that fills the verb in.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from booksmart_bench.main import VERBS, app

VERB_NAMES = sorted(VERBS)


class TestSurface:
    def test_help_lists_every_verb(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        for verb in VERB_NAMES:
            assert verb in result.stdout

    def test_the_four_verbs_are_the_artefact_boundaries(self) -> None:
        assert VERB_NAMES == ["ingest", "report", "run", "score"]

    @pytest.mark.parametrize("verb", VERB_NAMES)
    def test_every_verb_takes_assets(
        self, runner: CliRunner, make_assets: Callable[..., Path], verb: str
    ) -> None:
        result = runner.invoke(app, [verb, "--assets", str(make_assets())])

        assert "--assets" not in result.stderr  # i.e. not an unknown-option error


class TestUnresolvedAssets:
    @pytest.mark.parametrize("verb", VERB_NAMES)
    def test_missing_assets_fails_with_the_remedy(self, runner: CliRunner, verb: str) -> None:
        result = runner.invoke(app, [verb])

        assert result.exit_code == 1
        assert "BOOKSMART_BENCH_ASSETS" in result.stderr

    @pytest.mark.parametrize("verb", VERB_NAMES)
    def test_a_wrong_path_fails_before_any_work(
        self, runner: CliRunner, verb: str, tmp_path: Path
    ) -> None:
        result = runner.invoke(app, [verb, "--assets", str(tmp_path / "nowhere")])

        assert result.exit_code == 1
        assert "does not exist" in result.stderr


class TestTruthGate:
    """Truth is checked before anything is spent, in every verb — including the
    expensive one."""

    @pytest.mark.parametrize("verb", VERB_NAMES)
    def test_unscoreable_truth_stops_the_verb(
        self, runner: CliRunner, write_truth: Callable[..., Path], verb: str
    ) -> None:
        assets = write_truth(
            "a-book",
            queries=[
                {
                    "q": "a term",
                    "kind": "exact-term",
                    "expects": [{"loc": "9.9"}],
                    "why": "Names a location that does not exist.",
                }
            ],
        )

        result = runner.invoke(app, [verb, "--assets", str(assets)])

        assert result.exit_code == 1
        assert "9.9" in result.stdout
        assert "not implemented" not in result.stderr

    def test_bracketed_truth_survives_into_the_report(
        self, runner: CliRunner, write_truth: Callable[..., Path]
    ) -> None:
        """Findings quote hand-authored truth, and the console reads brackets as
        markup — so a bracketed term would be dropped from the very report that
        exists to name it."""
        assets = write_truth(
            "a-book",
            index_pairs=[{"term": "list [xs]", "loc": "nowhere"}],
        )

        result = runner.invoke(app, ["score", "--assets", str(assets)])

        assert "[xs]" in result.stdout

    def test_a_stray_closing_tag_in_truth_does_not_crash_the_verb(
        self, runner: CliRunner, write_truth: Callable[..., Path]
    ) -> None:
        """Unescaped, this raises MarkupError out of the reporter itself."""
        assets = write_truth(
            "a-book",
            index_pairs=[{"term": "see [/ref]", "loc": "nowhere"}],
        )

        result = runner.invoke(app, ["score", "--assets", str(assets)])

        assert result.exit_code == 1
        assert "[/ref]" in result.stdout

    def test_warnings_are_shown_but_survived(
        self, runner: CliRunner, write_truth: Callable[..., Path]
    ) -> None:
        """Half-authored truth — a source not yet pinned — has to stay workable."""
        assets = write_truth(
            "a-book",
            book={
                "title": "Placeholder Book",
                "area": "placeholder-area",
                "source": {"file": "sources/x.pdf", "sha256": "TBD-when-file-lands"},
            },
        )

        result = runner.invoke(app, ["score", "--assets", str(assets)])

        assert "sha256" in result.stdout
        assert "not implemented" in result.stderr  # got past the gate


class TestStubs:
    @pytest.mark.parametrize("verb", VERB_NAMES)
    def test_a_stub_exits_non_zero_and_says_what_it_will_do(
        self, runner: CliRunner, make_assets: Callable[..., Path], verb: str
    ) -> None:
        """A stub that exited 0 would read as a passing benchmark in a script."""
        result = runner.invoke(app, [verb, "--assets", str(make_assets())])

        assert result.exit_code == 1
        assert "not implemented" in result.stderr
        assert VERBS[verb] in result.stderr

    @pytest.mark.parametrize("verb", VERB_NAMES)
    def test_a_stub_reports_where_it_resolved_its_inputs(
        self, runner: CliRunner, make_assets: Callable[..., Path], verb: str
    ) -> None:
        assets = make_assets()

        result = runner.invoke(app, [verb, "--assets", str(assets)])

        assert str(assets) in result.stdout
