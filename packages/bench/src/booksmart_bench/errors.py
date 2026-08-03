"""Bench-facing errors and their one-line rendering.

Same shape as the CLI's: an expected failure is one clean line and exit code 1,
never a traceback. The harness is run by hand, often after a long ingest, so a
message that names the remedy is worth more here than anywhere else.
"""

import functools
from collections.abc import Callable
from typing import TypeVar

import typer

from booksmart_core.errors import BooksmartError

F = TypeVar("F", bound=Callable[..., object])


def render_error(message: str) -> None:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)


def handle_errors(fn: F) -> F:
    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> object:
        try:
            return fn(*args, **kwargs)
        except (BenchError, BooksmartError) as exc:
            render_error(str(exc))
            raise typer.Exit(1) from exc

    return wrapper  # type: ignore[return-value]


class BenchError(Exception):
    """Base for expected, user-facing bench failures (rendered as one line)."""


class AssetsNotFoundError(BenchError):
    """No assets checkout was given, or the given path is not one."""
