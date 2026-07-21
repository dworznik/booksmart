"""Booksmart CLI — a local, single-user front end over booksmart-core.

Register and ingest books, browse their runs, structure, profile and extracted
knowledge, and search across everything embedded. Everything runs locally against
an auto-migrated SQLite file and embedded Qdrant — no Docker, no Postgres, no
server. Every command but ``search`` mirrors the removed HTTP surface exactly
(docs/api-notes/); ``search`` is the first post-split feature (issue #30)."""

import uuid
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from booksmart_core.models import Book, KnowledgeObject, Run
from booksmart_core.runner import SCOPE_STAGES, has_successful_run

from booksmart_cli import reads, registration
from booksmart_cli.config import config_app
from booksmart_cli.errors import CliError, handle_errors, render_error
from booksmart_cli.runtime import Runtime

app = typer.Typer(
    help="Booksmart — turn books into queryable knowledge, locally.",
    no_args_is_help=True,
    add_completion=False,
)
books_app = typer.Typer(help="Register, list, inspect, and edit books.", no_args_is_help=True)
runs_app = typer.Typer(help="Inspect ingestion run history.", no_args_is_help=True)
knowledge_app = typer.Typer(help="Browse extracted knowledge objects.", no_args_is_help=True)
app.add_typer(books_app, name="books")
app.add_typer(runs_app, name="runs")
app.add_typer(knowledge_app, name="knowledge")
app.add_typer(config_app, name="config")

console = Console()


def _parse_uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError:
        raise CliError(f"Not a valid id: {value}") from None


# --- add / ingest -----------------------------------------------------------


@app.command()
@handle_errors
def add(
    file: Annotated[Path, typer.Argument(help="Path to a .pdf or .epub file.")],
    title: Annotated[str, typer.Option(help="Book title (required).")],
    author: Annotated[str, typer.Option(help="Book author (required).")],
    edition: Annotated[Optional[str], typer.Option()] = None,
    publication_year: Annotated[Optional[int], typer.Option()] = None,
    isbn: Annotated[Optional[str], typer.Option()] = None,
    primary_topic: Annotated[Optional[str], typer.Option()] = None,
    language: Annotated[Optional[str], typer.Option()] = None,
    framework: Annotated[Optional[str], typer.Option()] = None,
    methodology: Annotated[Optional[str], typer.Option()] = None,
    notes: Annotated[Optional[str], typer.Option()] = None,
    trust_level: Annotated[Optional[str], typer.Option()] = None,
    intended_use: Annotated[Optional[str], typer.Option()] = None,
) -> None:
    """Register a book from a local file (validate, dedup, store). Does not
    ingest — run `booksmart ingest <id>` next."""
    runtime = Runtime.load()
    metadata: dict[str, object] = {
        "edition": edition,
        "publication_year": publication_year,
        "isbn": isbn,
        "primary_topic": primary_topic,
        "language": language,
        "framework": framework,
        "methodology": methodology,
        "notes": notes,
        "trust_level": trust_level,
        "intended_use": intended_use,
    }
    book = registration.register_book(
        runtime, file, title=title, author=author, metadata=metadata
    )
    console.print(f"Registered [bold]{book.title}[/bold] as [cyan]{book.id}[/cyan]")


@app.command()
@handle_errors
def ingest(
    book: Annotated[str, typer.Argument(help="Book id.")],
    scope: Annotated[
        str, typer.Option(help="full | profile | extraction | embeddings.")
    ] = "full",
) -> None:
    """Run the pipeline over a book, foreground, streaming stage progress. Exits
    non-zero if the run fails."""
    if scope not in SCOPE_STAGES:
        raise CliError(
            f"Unknown scope {scope!r}; expected one of {', '.join(sorted(SCOPE_STAGES))}"
        )
    book_id = _parse_uuid(book)
    runtime = Runtime.load()

    # Incremental scopes build on a prior successful run (the old reprocess 409).
    if scope != "full" and not _has_prior_success(runtime, book_id):
        raise CliError(
            f"Scope {scope!r} needs a prior successful run; run `ingest {book}` first"
        )

    console.print(f"Ingesting [cyan]{book_id}[/cyan] (scope: {scope})")
    run_id = runtime.ingest(
        book_id,
        scope,
        on_stage=lambda stage: console.print(f"  • {stage}…"),
    )
    run = reads.get_run(runtime, run_id)
    if run.status == "succeeded":
        console.print(f"[green]✓[/green] run [cyan]{run.id}[/cyan] succeeded")
    else:
        render_error(f"run {run.id} failed: {run.error}")
        raise typer.Exit(1)


def _has_prior_success(runtime: Runtime, book_id: uuid.UUID) -> bool:
    with runtime.session_factory() as session:
        return has_successful_run(session, book_id)


# --- books -------------------------------------------------------------------


@books_app.command("list")
@handle_errors
def books_list() -> None:
    """List every registered book, oldest first."""
    runtime = Runtime.load()
    books = reads.list_books(runtime)
    if not books:
        console.print("No books registered yet. Add one with `booksmart add <file>`.")
        return
    table = Table("id", "title", "author", "format")
    for book in books:
        table.add_row(str(book.id), book.title, book.author, book.file_format)
    console.print(table)


@books_app.command("show")
@handle_errors
def books_show(book: Annotated[str, typer.Argument(help="Book id.")]) -> None:
    """Show one book's full metadata."""
    runtime = Runtime.load()
    _print_book_detail(reads.get_book(runtime, _parse_uuid(book)))


@books_app.command("update")
@handle_errors
def books_update(
    book: Annotated[str, typer.Argument(help="Book id.")],
    title: Annotated[Optional[str], typer.Option()] = None,
    author: Annotated[Optional[str], typer.Option()] = None,
    edition: Annotated[Optional[str], typer.Option()] = None,
    # A string (not int) so an empty value can clear it, matching the other
    # optional fields; a non-empty value must still parse as an integer.
    publication_year: Annotated[Optional[str], typer.Option()] = None,
    isbn: Annotated[Optional[str], typer.Option()] = None,
    primary_topic: Annotated[Optional[str], typer.Option()] = None,
    language: Annotated[Optional[str], typer.Option()] = None,
    framework: Annotated[Optional[str], typer.Option()] = None,
    methodology: Annotated[Optional[str], typer.Option()] = None,
    notes: Annotated[Optional[str], typer.Option()] = None,
    trust_level: Annotated[Optional[str], typer.Option()] = None,
    intended_use: Annotated[Optional[str], typer.Option()] = None,
) -> None:
    """Edit a book's metadata. Only the flags you pass change; title/author can
    be changed but not cleared; any other field clears with an empty value."""
    if title is not None and not title.strip():
        raise CliError("title may be changed but not cleared")
    if author is not None and not author.strip():
        raise CliError("author may be changed but not cleared")

    # Only the flags the user actually passed become changes; title/author only
    # ever set a new value, other fields also clear on an empty value (-> None).
    provided: dict[str, Optional[str]] = {
        "title": title,
        "author": author,
        "edition": edition,
        "publication_year": publication_year,
        "isbn": isbn,
        "primary_topic": primary_topic,
        "language": language,
        "framework": framework,
        "methodology": methodology,
        "notes": notes,
        "trust_level": trust_level,
        "intended_use": intended_use,
    }
    changes: dict[str, object] = {}
    for field, value in provided.items():
        if value is None:
            continue
        if field in ("title", "author"):
            changes[field] = value
        elif value == "":
            changes[field] = None  # explicit clear of an optional field
        elif field == "publication_year":
            changes[field] = _parse_year(value)
        else:
            changes[field] = value
    if not changes:
        raise CliError("Nothing to update; pass at least one field to change.")

    runtime = Runtime.load()
    _print_book_detail(registration.update_book(runtime, _parse_uuid(book), changes))


def _parse_year(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise CliError(f"publication-year must be an integer, got {value!r}") from None


def _print_book_detail(book: Book) -> None:
    table = Table("field", "value", show_header=False)
    table.add_row("id", str(book.id))
    table.add_row("title", book.title)
    table.add_row("author", book.author)
    for field in (*registration.METADATA_FIELDS, "file_format", "parser_used"):
        value = getattr(book, field)
        if value is not None:
            table.add_row(field, str(value))
    console.print(table)


# --- runs --------------------------------------------------------------------


@runs_app.command("list")
@handle_errors
def runs_list(book: Annotated[str, typer.Argument(help="Book id.")]) -> None:
    """A book's run history, oldest first (failures included)."""
    runtime = Runtime.load()
    runs = reads.list_runs(runtime, _parse_uuid(book))
    if not runs:
        console.print("No runs yet for this book.")
        return
    table = Table("id", "scope", "status", "created")
    for run in runs:
        table.add_row(str(run.id), run.scope, _status_markup(run.status), _when(run))
    console.print(table)


@runs_app.command("show")
@handle_errors
def runs_show(run: Annotated[str, typer.Argument(help="Run id.")]) -> None:
    """Show one run's outcome, versions, and token spend."""
    runtime = Runtime.load()
    run_row = reads.get_run(runtime, _parse_uuid(run))
    table = Table("field", "value", show_header=False)
    table.add_row("id", str(run_row.id))
    table.add_row("book_id", str(run_row.book_id))
    table.add_row("scope", run_row.scope)
    table.add_row("status", _status_markup(run_row.status))
    for field in (
        "error",
        "parser_used",
        "extraction_version",
        "model_version",
        "prompt_version",
        "input_tokens",
        "output_tokens",
    ):
        value = getattr(run_row, field)
        if value is not None:
            table.add_row(field, str(value))
    console.print(table)


def _status_markup(status: str) -> str:
    colour = {"succeeded": "green", "failed": "red", "running": "yellow"}.get(status, "white")
    return f"[{colour}]{status}[/{colour}]"


def _when(run: Run) -> str:
    return run.created_at.isoformat() if run.created_at else ""


# --- structure / profile -----------------------------------------------------


@app.command()
@handle_errors
def structure(book: Annotated[str, typer.Argument(help="Book id.")]) -> None:
    """Show a book's detected chapter/section tree."""
    runtime = Runtime.load()
    chapters = reads.book_structure(runtime, _parse_uuid(book))
    if not chapters:
        console.print("No structure yet; run an ingest first.")
        return
    for chapter in chapters:
        suffix = "" if chapter.kind == "chapter" else f" [dim]({chapter.kind})[/dim]"
        console.print(f"[bold]{chapter.position + 1}. {chapter.title}[/bold]{suffix}")
        for section in chapter.sections:
            console.print(f"    {section.position + 1}. {section.title}")


@app.command()
@handle_errors
def profile(book: Annotated[str, typer.Argument(help="Book id.")]) -> None:
    """Show a book's latest generated profile."""
    runtime = Runtime.load()
    book_profile = reads.latest_profile(runtime, _parse_uuid(book))
    console.print(book_profile.content)
    console.print(
        f"[dim]— {book_profile.model} (prompt v{book_profile.prompt_version})[/dim]"
    )


# --- knowledge ---------------------------------------------------------------


@knowledge_app.command("list")
@handle_errors
def knowledge_list(
    book: Annotated[str, typer.Argument(help="Book id.")],
    type: Annotated[Optional[str], typer.Option(help="Filter by knowledge type.")] = None,
) -> None:
    """List a book's extracted knowledge objects."""
    runtime = Runtime.load()
    objects = reads.list_knowledge(runtime, _parse_uuid(book), type)
    if not objects:
        console.print("No knowledge objects for this book.")
        return
    table = Table("id", "type", "title", "confidence")
    for obj in objects:
        table.add_row(str(obj.id), obj.type, obj.title, f"{obj.confidence:.2f}")
    console.print(table)


@knowledge_app.command("show")
@handle_errors
def knowledge_show(
    object_id: Annotated[str, typer.Argument(help="Knowledge object id.")],
) -> None:
    """Show one knowledge object in full."""
    runtime = Runtime.load()
    _print_knowledge_detail(reads.get_knowledge(runtime, _parse_uuid(object_id)))


# --- search ------------------------------------------------------------------

SEARCH_SCOPE_ALL = "all"


@app.command()
@handle_errors
def search(
    scope: Annotated[
        str, typer.Argument(help="Book id to search within, or `all` for every book.")
    ],
    query: Annotated[str, typer.Argument(help="What to look for, in plain language.")],
    type: Annotated[
        Optional[list[str]],
        typer.Option(help="Restrict to a record type; repeatable."),
    ] = None,
    limit: Annotated[int, typer.Option(help="Maximum number of hits.")] = 10,
    score_threshold: Annotated[
        Optional[float],
        typer.Option(help="Drop hits below this cosine similarity (-1 to 1)."),
    ] = None,
) -> None:
    """Find the chapters, sections and knowledge objects most similar to a query.

    Searches the embeddings an ingest produced, so a book has to be ingested
    before it can be found.
    """
    book_id = None if scope == SEARCH_SCOPE_ALL else _parse_uuid(scope)
    runtime = Runtime.load()
    hits = reads.semantic_search(
        runtime,
        query,
        book_id=book_id,
        record_types=type,
        limit=limit,
        score_threshold=score_threshold,
    ).hits
    if not hits:
        console.print("No matches. Ingest a book first, or try a broader query.")
        return

    table = Table("score", "type", "title", "match")
    if book_id is None:
        table.add_column("book")
    for hit in hits:
        row = [f"{hit.score:.3f}", hit.record_type, hit.title, _snippet(hit.text)]
        if book_id is None:
            row.append(str(hit.book_id))
        table.add_row(*row)
    console.print(table)


def _snippet(text: str, width: int = 60) -> str:
    """The embedded text as one line, short enough to sit in a table cell."""
    flattened = " ".join(text.split())
    if len(flattened) <= width:
        return flattened
    return flattened[: width - 1].rstrip() + "…"


def _print_knowledge_detail(obj: KnowledgeObject) -> None:
    console.print(f"[bold]{obj.type}: {obj.title}[/bold]")
    console.print(obj.summary)
    console.print()
    console.print(obj.content)
    console.print(
        f"[dim]source: {obj.source_location} · confidence {obj.confidence:.2f} · "
        f"{obj.extraction_model}[/dim]"
    )


if __name__ == "__main__":
    app()
