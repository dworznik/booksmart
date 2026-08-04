"""``booksmart-bench`` — one verb per artefact boundary.

Measurement is TREC-shaped and the three artefacts stay separable: truth is
hand-authored and versioned elsewhere, a *run file* records resolved locations
(never record ids), and scoring is a pure comparison of the two. The verbs cut
along exactly those seams:

    ingest   build a corpus for one pipeline configuration — the only expensive
             verb, and the reason corpora are keyed and reused
    run      execute the benchmarks against a corpus, emit a run file
    score    run file x truth -> scores (pure)
    report   render two runs side by side (pure)

Every verb resolves its assets checkout up front, so a wrong ``--assets`` is a
one-line error before anything is spent.
"""

import functools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Optional, TypeVar

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from booksmart_core.errors import BooksmartError

from booksmart_bench.config import corpus_home, load_settings, resolve_assets
from booksmart_bench.errors import BenchError
from booksmart_bench.report import render
from booksmart_bench.scoring import Scores, compare, load_run
from booksmart_bench.scoring import score as score_run
from booksmart_bench.truth import Finding, Truth, errors_in, lint, load_truth

F = TypeVar("F", bound=Callable[..., object])


def render_error(message: str) -> None:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)


def handle_errors(fn: F) -> F:
    """Render expected failures as one clean line and exit non-zero, instead of
    dumping a traceback. The harness is run by hand, often after a long ingest,
    so a message naming the remedy is worth more here than anywhere else.

    Lives here rather than beside the error types because catching the
    pipeline's own ``BooksmartError`` means importing the pipeline, and the
    scorer must be able to reach the error types without doing that.
    """

    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> object:
        try:
            return fn(*args, **kwargs)
        except (BenchError, BooksmartError) as exc:
            render_error(str(exc))
            raise typer.Exit(1) from exc

    return wrapper  # type: ignore[return-value]

# The verb -> what it does, in one line. Used in help and in the message a
# not-yet-built verb refuses with.
VERBS = {
    "ingest": "build a corpus for this configuration",
    "run": "execute the benchmarks and emit a run file",
    "score": "compare a run file against truth",
    "report": "render two runs side by side",
}

# Specced but not built. Each still resolves and lints its inputs for real
# before refusing, and exits non-zero — a stub exiting 0 would read as a passing
# benchmark in a script. This set shrinks as the waves land.
UNIMPLEMENTED = frozenset({"ingest", "run"})

app = typer.Typer(
    help="Benchmark booksmart's ingestion and recall against hand-authored truth.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

AssetsOption = Annotated[
    Optional[Path],
    typer.Option(
        "--assets",
        help="The benchmark assets checkout (defaults to $BOOKSMART_BENCH_ASSETS).",
    ),
]


def _prepare(assets: Path | None) -> tuple[Path, Truth]:
    """Resolve and check what every verb needs, before anything is spent.

    Truth is linted here rather than inside the verbs because every verb depends
    on it and the failures it catches are silent ones: a location nothing can
    satisfy scores zero and reads as a retrieval regression, not as a typo.
    """
    resolved = resolve_assets(assets)
    console.print(f"assets: [cyan]{escape(str(resolved))}[/cyan]")

    truth = load_truth(resolved)
    console.print(
        f"truth: [cyan]{len(truth.books)}[/cyan] book(s), "
        f"[cyan]{len(truth.areas)}[/cyan] area(s)"
    )
    _report_findings(lint(truth))
    return resolved, truth


def _resolve_and_report(verb: str, assets: Path | None) -> None:
    """A verb that is specced but not built yet: check the inputs for real, then
    refuse. Exiting 0 here would read as a passing benchmark in a script."""
    _prepare(assets)
    console.print(f"corpus: [cyan]{escape(str(corpus_home(load_settings())))}[/cyan]")
    raise BenchError(f"`{verb}` is not implemented yet — it will {VERBS[verb]}.")


def _report_findings(findings: tuple[Finding, ...]) -> None:
    """Show every finding, then stop on the ones that make truth unscoreable.
    Warnings are printed and survived: truth for an area is authored before every
    book in it exists, and that half-finished state has to stay workable.

    Findings quote hand-authored truth — query text, terms, ToC titles — and Rich
    reads ``[...]`` as markup, so anything bracketed would be dropped from the
    report or, for a stray closing tag, raise instead of printing. A reporter
    that silently deletes part of its report is worse than no reporter, hence
    the escaping.
    """
    for finding in findings:
        colour = "red" if finding.severity == "error" else "yellow"
        console.print(
            f"[{colour}]{finding.severity}[/{colour}] "
            f"{escape(finding.where)}: {escape(finding.message)}"
        )

    failures = errors_in(findings)
    if failures:
        raise BenchError(
            f"{len(failures)} error(s) in truth; fix them before running a benchmark."
        )


@app.command()
@handle_errors
def ingest(assets: AssetsOption = None) -> None:
    """Build (or extend) this configuration's corpus by ingesting the books
    truth is authored against. The only verb that spends money."""
    _resolve_and_report("ingest", assets)


@app.command()
@handle_errors
def run(assets: AssetsOption = None) -> None:
    """Execute the benchmarks against an existing corpus, emitting a run file."""
    _resolve_and_report("run", assets)


@app.command()
@handle_errors
def score(
    run_file: Annotated[Path, typer.Argument(help="A run file emitted by `run`.")],
    assets: AssetsOption = None,
    out: Annotated[
        Optional[Path], typer.Option(help="Write the scores as JSON to this path.")
    ] = None,
) -> None:
    """Score a run file against truth. Pure: no pipeline, no network."""
    _, truth = _prepare(assets)
    scores = score_run(load_run(run_file), truth)
    _print_scores(scores)
    if out is not None:
        out.write_text(json.dumps(scores.as_dict(), indent=2) + "\n")
        console.print(f"wrote [cyan]{escape(str(out))}[/cyan]")


@app.command()
@handle_errors
def report(
    baseline: Annotated[Path, typer.Argument(help="The run to compare against.")],
    candidate: Annotated[Path, typer.Argument(help="The run under test.")],
    assets: AssetsOption = None,
    out: Annotated[
        Optional[Path], typer.Option(help="Write the Markdown to this path.")
    ] = None,
) -> None:
    """Render two runs side by side as Markdown, with regression thresholds
    applied. Pure.

    Exits non-zero when the candidate regressed, so this can gate a change
    without anyone having to read the table first.
    """
    _, truth = _prepare(assets)
    before = score_run(load_run(baseline), truth)
    after = score_run(load_run(candidate), truth)
    verdict = compare(before, after)

    markdown = render(before, after, verdict)
    if out is not None:
        out.write_text(markdown)
        console.print(f"wrote [cyan]{escape(str(out))}[/cyan]")
    else:
        console.print(escape(markdown), markup=False, highlight=False)

    if verdict.is_regression:
        raise BenchError(
            f"{len(verdict.regressions)} dimension(s) regressed beyond threshold."
        )


def _print_scores(scores: Scores) -> None:
    pooled = scores.pooled
    if not pooled:
        console.print("[yellow]nothing scored[/yellow] — the run measured no dimension")
    table = Table("dimension", "score")
    for dimension, value in sorted(pooled.items()):
        table.add_row(dimension, f"{value:.3f}")
    if pooled:
        console.print(table)
    for note in (*scores.skipped, *scores.unasked):
        console.print(f"[yellow]not measured[/yellow] {escape(note)}")


if __name__ == "__main__":
    app()
