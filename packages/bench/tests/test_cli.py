"""The four verbs: one per artefact boundary.

Every verb resolves its assets checkout and lints truth before doing anything —
so a misconfigured path or a location nothing can satisfy is reported before an
afternoon of ingesting, not after it.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from booksmart_bench.main import VERBS, app

VERB_NAMES = sorted(VERBS)


def args_for(verb: str, assets: Path, run_file: Path) -> list[str]:
    """The positional arguments a verb needs, so a test about assets handling
    is not really a test about missing arguments."""
    positional = {
        # The scope is never reached in these tests — the assets and truth gates
        # fire first — but typer still demands the argument.
        "ingest": ["a-scope"],
        "run": ["recall", "a-scope"],
        "score": [str(run_file)],
        "report": [str(run_file), str(run_file)],
    }.get(verb, [])
    return [verb, *positional, "--assets", str(assets)]


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
        self,
        runner: CliRunner,
        make_assets: Callable[..., Path],
        write_run: Callable[..., Path],
        verb: str,
    ) -> None:
        result = runner.invoke(app, args_for(verb, make_assets(), write_run()))

        assert "--assets" not in result.stderr  # i.e. not an unknown-option error


    def test_a_long_path_is_reported_whole(
        self,
        runner: CliRunner,
        write_truth: Callable[..., Path],
        write_run: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        """A path is one long token, and Rich breaks it mid-token to fit the
        console width — so the path a verb just reported is not literally in its
        own output, and cannot be copied out of it.

        The depth here is the point. Under a non-tty Rich assumes 80 columns; a
        Linux temp path is around 68 characters and fits, a macOS one is around
        120 and does not. Left to the ambient temp directory this passes on CI
        and fails on a developer's machine, which is exactly what happened.
        """
        deep = tmp_path.joinpath(*["nested-enough-to-wrap"] * 4)
        assets = write_truth("a-book", root=deep)
        assert len(str(assets)) > 80, "fixture must exceed the assumed console width"

        result = runner.invoke(app, ["score", str(write_run()), "--assets", str(assets)])

        assert str(assets) in result.stdout


class TestUnresolvedAssets:
    @pytest.mark.parametrize("verb", VERB_NAMES)
    def test_missing_assets_fails_with_the_remedy(
        self, runner: CliRunner, write_run: Callable[..., Path], verb: str
    ) -> None:
        without_assets = args_for(verb, Path("unused"), write_run())[:-2]
        result = runner.invoke(app, without_assets)

        assert result.exit_code == 1
        assert "BOOKSMART_BENCH_ASSETS" in result.stderr

    @pytest.mark.parametrize("verb", VERB_NAMES)
    def test_a_wrong_path_fails_before_any_work(
        self, runner: CliRunner, write_run: Callable[..., Path], verb: str, tmp_path: Path
    ) -> None:
        result = runner.invoke(app, args_for(verb, tmp_path / "nowhere", write_run()))

        assert result.exit_code == 1
        assert "does not exist" in result.stderr


class TestTruthGate:
    """Truth is checked before anything is spent, in every verb — including the
    expensive one."""

    @pytest.mark.parametrize("verb", VERB_NAMES)
    def test_unscoreable_truth_stops_the_verb(
        self,
        runner: CliRunner,
        write_truth: Callable[..., Path],
        write_run: Callable[..., Path],
        verb: str,
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

        result = runner.invoke(app, args_for(verb, assets, write_run()))

        assert result.exit_code == 1
        assert "9.9" in result.stdout
        assert "not implemented" not in result.stderr

    def test_bracketed_truth_survives_into_the_report(
        self,
        runner: CliRunner,
        write_truth: Callable[..., Path],
        write_run: Callable[..., Path],
    ) -> None:
        """Findings quote hand-authored truth, and the console reads brackets as
        markup — so a bracketed term would be dropped from the very report that
        exists to name it."""
        assets = write_truth(
            "a-book",
            index_pairs=[{"term": "list [xs]", "loc": "nowhere"}],
        )

        result = runner.invoke(app, ["score", str(write_run()), "--assets", str(assets)])

        assert "[xs]" in result.stdout

    def test_a_stray_closing_tag_in_truth_does_not_crash_the_verb(
        self,
        runner: CliRunner,
        write_truth: Callable[..., Path],
        write_run: Callable[..., Path],
    ) -> None:
        """Unescaped, this raises MarkupError out of the reporter itself."""
        assets = write_truth(
            "a-book",
            index_pairs=[{"term": "see [/ref]", "loc": "nowhere"}],
        )

        result = runner.invoke(app, ["score", str(write_run()), "--assets", str(assets)])

        assert result.exit_code == 1
        assert "[/ref]" in result.stdout

    def test_warnings_are_shown_but_survived(
        self,
        runner: CliRunner,
        write_truth: Callable[..., Path],
        write_run: Callable[..., Path],
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

        result = runner.invoke(app, ["score", str(write_run()), "--assets", str(assets)])

        assert "sha256" in result.stdout
        assert result.exit_code == 0  # got past the gate and scored


class TestRun:
    """The wiring only — what `run` measures is `test_execute.py`'s subject."""

    def test_an_empty_corpus_names_the_verb_that_fills_it(
        self, runner: CliRunner, make_assets: Callable[..., Path]
    ) -> None:
        """A run over a corpus that holds nothing would emit a run file of
        zeroes, which scores exactly like a pipeline that lost every record."""
        assets = make_assets()

        result = runner.invoke(
            app, ["run", "recall", "placeholder-book", "--assets", str(assets)]
        )

        assert result.exit_code == 1
        assert "ingest" in result.stderr

    def test_an_unknown_family_names_the_families(
        self, runner: CliRunner, make_assets: Callable[..., Path]
    ) -> None:
        result = runner.invoke(
            app, ["run", "everything", "placeholder-book", "--assets", str(make_assets())]
        )

        assert result.exit_code == 1
        assert "recall" in result.stderr
