"""CLI-facing errors and their one-line rendering.

The pipeline's own failures are recorded on the Run (never raised out of
``execute_run``); these cover the front-end concerns the removed HTTP layer used
to answer with 4xx — an unsupported file, a duplicate, an unknown book. Every
command is wrapped in ``handle_errors``, which prints the message as one clean
line, exit code 1."""

import functools
from collections.abc import Callable
from typing import TypeVar

import typer

from booksmart_core.errors import BooksmartError, MissingAPIKeyError

F = TypeVar("F", bound=Callable[..., object])


def render_error(message: str) -> None:
    """One red error line to stderr, via click's stream (captured in tests)."""
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)


def handle_errors(fn: F) -> F:
    """Render expected failures (bad input, pipeline errors) as one clean line
    and exit non-zero, instead of dumping a traceback. The pipeline's own
    BooksmartError carries retriability for API consumers; the CLI stays human
    and shows only the message — plus the exact remedy for a missing key."""

    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> object:
        try:
            return fn(*args, **kwargs)
        except MissingAPIKeyError as exc:
            render_error(f"{exc}. Set it with: booksmart config set {exc.field}")
            raise typer.Exit(1) from exc
        except (CliError, BooksmartError) as exc:
            render_error(str(exc))
            raise typer.Exit(1) from exc

    return wrapper  # type: ignore[return-value]


class CliError(Exception):
    """Base for expected, user-facing CLI failures (rendered as one line)."""


class UnsupportedFileError(CliError):
    """The file is not a supported format, by extension or by content (the old
    ``415``)."""


class DuplicateBookError(CliError):
    """A byte-identical book is already registered (the old ``409``)."""

    def __init__(self, existing_book_id: str) -> None:
        super().__init__(
            f"This file is already registered as book {existing_book_id} "
            f"(byte-identical content)."
        )
        self.existing_book_id = existing_book_id


class BookNotFoundError(CliError):
    """No book with the given id (the old ``404``)."""


class RunNotFoundError(CliError):
    """No run with the given id (the old ``404`` on /jobs/{id})."""


class KnowledgeNotFoundError(CliError):
    """No knowledge object with the given id (the old ``404``)."""


class NoProfileError(CliError):
    """The book has no generated profile yet (the old ``404`` on /profile)."""
