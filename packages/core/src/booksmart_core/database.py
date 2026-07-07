"""Engine construction and migration helpers a consumer uses to stand up a
database before driving the pipeline.

A Runner owns engine/session lifecycle (core exposes only Stage functions over a
``Session``). These helpers give a consumer — the CLI, or any embedded user — a
correct SQLite engine (foreign keys enforced, so the stages' ON DELETE CASCADE
works) and a one-call migration to head against the single packaged history.
booksmart-api runs Postgres and manages its own engine, so it needs neither.
"""

from sqlalchemy import Engine, event
from sqlalchemy import create_engine as _sa_create_engine

from booksmart_core import MIGRATIONS_PATH


def create_engine(url: str) -> Engine:
    """A SQLAlchemy engine that enforces SQLite foreign keys.

    SQLite honours foreign keys (and thus ON DELETE CASCADE) only when asked,
    per connection; without this the structure/extraction stages' bulk deletes
    leave orphaned rows. A no-op for other dialects."""
    engine = _sa_create_engine(url)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_conn: object, _: object) -> None:
            cursor = dbapi_conn.cursor()  # type: ignore[attr-defined]
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def upgrade_to_head(url: str) -> None:
    """Migrate the database at ``url`` to head using core's single packaged
    alembic history. Idempotent — a database already at head is a no-op — so a
    consumer can call it unconditionally on startup."""
    # Imported lazily: alembic pulls in a fair bit, and only migration paths need it.
    from alembic import command
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(MIGRATIONS_PATH))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")
