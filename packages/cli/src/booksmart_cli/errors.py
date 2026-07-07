"""CLI-facing errors.

The pipeline's own failures are recorded on the Run (never raised out of
``execute_run``); these cover the front-end concerns the removed HTTP layer used
to answer with 4xx — an unsupported file, a duplicate, an unknown book. ``main``
catches ``CliError`` and prints its message as one clean line, exit code 1."""


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
