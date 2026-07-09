"""booksmart-core: the book-ingestion pipeline as a library.

The alembic history and its config ship inside the package (not just the sdist)
so any consumer installing core as a wheel — the CLI, a server — can locate
and run the single migration history from the installed location.
"""

from pathlib import Path

MIGRATIONS_PATH = Path(__file__).resolve().parent / "migrations"
ALEMBIC_INI_PATH = Path(__file__).resolve().parent / "alembic.ini"

__all__ = ["ALEMBIC_INI_PATH", "MIGRATIONS_PATH"]
