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
from booksmart_bench.execute import DEFAULT_LIMIT, FAMILIES, RunOutcome, execute_run
from booksmart_bench.ingest import IngestOutcome, ingest_scope
from booksmart_bench.judge import resolve_judge
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


def print_path(label: str, path: Path) -> None:
    """Report a path, whole.

    Rich word-wraps at the console width, and a path is one long token with no
    break points it is willing to respect — so a long one is split across lines
    mid-token. That makes it uncopyable, and it makes the break look like part
    of the path.

    Not a cosmetic complaint. Under a non-tty Rich assumes 80 columns, and a
    macOS temp path runs to about 120 characters, so a path a verb had just
    reported was not literally present in its own output. Linux temp paths are
    around 68 and fit, which is why the whole suite passed in CI and failed on
    the first developer machine to try it.
    """
    console.print(f"{label}: [cyan]{escape(str(path))}[/cyan]", soft_wrap=True)


def _prepare(assets: Path | None) -> tuple[Path, Truth]:
    """Resolve and check what every verb needs, before anything is spent.

    Truth is linted here rather than inside the verbs because every verb depends
    on it and the failures it catches are silent ones: a location nothing can
    satisfy scores zero and reads as a retrieval regression, not as a typo.
    """
    resolved = resolve_assets(assets)
    print_path("assets", resolved)

    truth = load_truth(resolved)
    console.print(
        f"truth: [cyan]{len(truth.books)}[/cyan] book(s), "
        f"[cyan]{len(truth.areas)}[/cyan] area(s)"
    )
    _report_findings(lint(truth))
    return resolved, truth


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
def ingest(
    scope: Annotated[
        str, typer.Argument(help="A book slug, or an area name to ingest every book in it.")
    ],
    assets: AssetsOption = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-ingest books already in this corpus.")
    ] = False,
) -> None:
    """Build (or extend) this configuration's corpus by ingesting the books
    truth is authored against. The only verb that spends money.

    Idempotent: a book whose bytes and configuration are unchanged is skipped,
    because ingest is expensive and recall is not.
    """
    resolved, truth = _prepare(assets)
    settings = load_settings()
    print_path("corpus", corpus_home(settings))

    outcome = ingest_scope(
        resolved,
        truth,
        scope,
        settings,
        force=force,
        on_stage=lambda slug, stage: console.print(f"  {escape(slug)} • {stage}…"),
    )
    _print_ingest(outcome)
    if any(book.status == "failed" for book in outcome.books):
        raise BenchError("one or more books failed to ingest; see the table above")


def _print_ingest(outcome: IngestOutcome) -> None:
    table = Table("book", "status", "wall", "in", "out", "embed")
    for book in outcome.books:
        table.add_row(
            escape(book.slug),
            _status_markup(book.status),
            f"{book.wall_seconds:.1f}s" if book.wall_seconds else "—",
            *(
                f"{sum(getattr(stage, field) for stage in book.stages):,}"
                for field in ("input_tokens", "output_tokens", "embedding_tokens")
            ),
        )
    console.print(table)
    for book in outcome.books:
        if book.error:
            console.print(f"[red]{escape(book.slug)}[/red]: {escape(book.error)}")
    for note in outcome.notes:
        console.print(f"[yellow]note[/yellow] {escape(note)}")


def _status_markup(status: str) -> str:
    colour = {"succeeded": "green", "failed": "red", "skipped": "yellow"}.get(status, "white")
    return f"[{colour}]{status}[/{colour}]"


@app.command()
@handle_errors
def run(
    family: Annotated[
        str, typer.Argument(help=f"Which family to execute: {', '.join(sorted(FAMILIES))}.")
    ],
    scope: Annotated[
        str,
        typer.Argument(
            help="A book slug, or an area name to also ask that area's cross-book queries."
        ),
    ],
    assets: AssetsOption = None,
    limit: Annotated[
        int, typer.Option(help="How many hits each query asks the search path for.")
    ] = DEFAULT_LIMIT,
    out: Annotated[
        Optional[Path], typer.Option(help="Write the run file here instead of <assets>/runs/.")
    ] = None,
    judge: Annotated[
        bool,
        typer.Option(
            "--judge/--no-judge",
            help="Verify summary faithfulness with the pinned cross-family judge.",
        ),
    ] = True,
) -> None:
    """Execute the benchmarks against an existing corpus, emitting a run file.

    The run file carries resolved locations, never record ids: the join from a
    hit's record to the ToC node truth names happens here, which is what lets
    `score` stay a pure comparison.
    """
    resolved, truth = _prepare(assets)
    settings = load_settings()
    console.print(f"corpus: [cyan]{escape(str(corpus_home(settings)))}[/cyan]")

    judge_config = resolve_judge(settings) if judge else None
    if judge_config is not None:
        console.print(
            f"judge: [cyan]{escape(judge_config.provider)}[/cyan] "
            f"{escape(judge_config.model)} (prompt v{judge_config.prompt_version})"
        )

    outcome = execute_run(
        resolved,
        truth,
        family,
        scope,
        settings,
        limit=limit,
        out=out,
        judge=judge_config,
        on_progress=lambda line: console.print(f"  {escape(line)}"),
    )
    _print_run(outcome)


def _print_run(outcome: RunOutcome) -> None:
    table = Table("run", "books", "queries asked", "summaries judged")
    table.add_row(
        escape(outcome.run_id),
        escape(", ".join(outcome.books)),
        str(outcome.asked),
        str(outcome.judged),
    )
    console.print(table)
    for note in outcome.notes:
        console.print(f"[yellow]note[/yellow] {escape(note)}")
    console.print(f"wrote [cyan]{escape(str(outcome.path))}[/cyan]")


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
        print_path("wrote", out)


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
        print_path("wrote", out)
    else:
        # markup=False already stops Rich reading brackets as tags; escaping as
        # well would print the backslashes and disagree with what --out writes.
        console.print(markdown, markup=False, highlight=False)

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
