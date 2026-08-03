"""``booksmart-bench`` — one verb per artefact boundary.

Measurement is TREC-shaped and the three artefacts stay separable: truth is
hand-authored and versioned elsewhere, a *run file* records resolved locations
(never record ids), and scoring is a pure comparison of the two. The verbs cut
along exactly those seams:

    ingest   build a corpus for one pipeline configuration — the only expensive
             verb, and the reason corpora are keyed and reused
    run      execute the benchmarks against a corpus, emit a run file
    score    run file × truth -> scores (pure)
    report   render two runs side by side (pure)

Every verb resolves its assets checkout up front, so a wrong ``--assets`` is a
one-line error before anything is spent.
"""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from booksmart_bench.config import corpus_home, load_settings, resolve_assets
from booksmart_bench.errors import BenchError, handle_errors

# The verb -> the issue that fills it in. Until then the stub says so and exits
# non-zero, rather than reporting a benchmark that never ran.
VERBS = {
    "ingest": "booksmart-bench#14",
    "run": "booksmart-bench#15",
    "score": "booksmart-bench#16",
    "report": "booksmart-bench#16",
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
    """Resolve what every verb needs, show it, then refuse the unimplemented work."""
    resolved = resolve_assets(assets)
    console.print(f"assets: [cyan]{resolved}[/cyan]")
    console.print(f"corpus: [cyan]{corpus_home(load_settings())}[/cyan]")
    raise BenchError(f"`{verb}` is not implemented yet ({VERBS[verb]}).")


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
