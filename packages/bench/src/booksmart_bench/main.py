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

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.markup import escape

from booksmart_bench.config import corpus_home, load_settings, resolve_assets
from booksmart_bench.errors import BenchError, handle_errors
from booksmart_bench.truth import Finding, errors_in, lint, load_truth

# The verb -> what it will do once implemented. Until then the stub says so and
# exits non-zero, rather than reporting a benchmark that never ran.
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


def _resolve_and_report(verb: str, assets: Path | None) -> None:
    """Resolve and check what every verb needs, then refuse the unimplemented work.

    Truth is linted here rather than inside the verbs because every verb depends
    on it and the failures it catches are silent ones: a location nothing can
    satisfy scores zero and reads as a retrieval regression, not as a typo.
    """
    resolved = resolve_assets(assets)
    console.print(f"assets: [cyan]{escape(str(resolved))}[/cyan]")
    console.print(f"corpus: [cyan]{escape(str(corpus_home(load_settings())))}[/cyan]")

    truth = load_truth(resolved)
    console.print(
        f"truth: [cyan]{len(truth.books)}[/cyan] book(s), "
        f"[cyan]{len(truth.areas)}[/cyan] area(s)"
    )
    _report_findings(lint(truth))

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
def score(assets: AssetsOption = None) -> None:
    """Score a run file against truth. Pure: no pipeline, no network."""
    _resolve_and_report("score", assets)


@app.command()
@handle_errors
def report(assets: AssetsOption = None) -> None:
    """Render two runs side by side as Markdown, with regression thresholds
    applied. Pure."""
    _resolve_and_report("report", assets)


if __name__ == "__main__":
    app()
