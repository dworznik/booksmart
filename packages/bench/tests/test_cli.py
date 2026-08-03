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


class TestStubs:
    @pytest.mark.parametrize("verb", VERB_NAMES)
    def test_a_stub_exits_non_zero_and_names_its_issue(
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
