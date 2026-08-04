"""Bench-facing error types.

Deliberately dependency-free — no typer, and nothing from the pipeline. The
scorer reaches these (a malformed run file is a ``BenchError``) and #16 requires
it to import nothing from the pipeline, transitively included. Rendering these
as one clean line is a front-end concern and lives with the front end, in
``main``, where typer already is.
"""


class BenchError(Exception):
    """Base for expected, user-facing bench failures (rendered as one line)."""


class AssetsNotFoundError(BenchError):
    """No assets checkout was given, or the given path is not one."""


class SourceMissingError(BenchError):
    """A book's source file is absent from ``sources/``, or does not match the
    sha256 its ``book.yaml`` pins."""
