"""Booksmart CLI entrypoint.

Skeleton only: this issue lands the package and its console-script wiring so the
workspace builds and booksmart-core is consumable. The real parity commands
(add, ingest, books, runs, structure, profile, knowledge) — with embedded
Qdrant and auto-migration — arrive in the CLI issue.
"""

import typer

app = typer.Typer(
    help="Booksmart — turn books into queryable knowledge, locally.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Booksmart CLI. Commands arrive in a later slice."""


if __name__ == "__main__":
    app()
