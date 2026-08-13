"""The squashed baseline migration: empty databases migrate cleanly, and the one
existing Postgres deployment can adopt the new single history.

These run on SQLite (the default suite dialect) but exercise dialect-neutral
Alembic mechanics: the same single history migrates an empty database to head,
and an already-provisioned database adopts it without re-running any DDL.
"""

from pathlib import Path

from alembic import command
from alembic.config import Config as AlembicConfig
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text

from booksmart_core import MIGRATIONS_PATH
from booksmart_core.models import Base

EXPECTED_TABLES = {
    "books",
    "chapters",
    "sections",
    "book_profiles",
    "knowledge_objects",
    "runs",
}


def _alembic_config(url: str) -> AlembicConfig:
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(MIGRATIONS_PATH))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def _head_revision() -> str:
    return ScriptDirectory.from_config(_alembic_config("sqlite://")).get_current_head()


def _current_revision(url: str) -> str | None:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return MigrationContext.configure(conn).get_current_revision()
    finally:
        engine.dispose()


def test_history_squashed_to_a_single_baseline() -> None:
    script = ScriptDirectory.from_config(_alembic_config("sqlite://"))
    revisions = list(script.walk_revisions())
    assert len(revisions) == 1
    assert revisions[0].down_revision is None


def test_empty_database_migrates_to_head(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'fresh.db'}"
    command.upgrade(_alembic_config(url), "head")

    assert _current_revision(url) == _head_revision()
    engine = create_engine(url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert EXPECTED_TABLES <= tables


def test_existing_deployment_adopts_baseline_via_stamp_purge(tmp_path: Path) -> None:
    """The one existing Postgres deployment already holds the full schema and an
    ``alembic_version`` pointing at the pre-squash ``0012``. The baseline
    docstring's ``alembic stamp --purge head`` must move it onto this history
    without re-running DDL."""
    url = f"sqlite:///{tmp_path / 'legacy.db'}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)  # schema already provisioned
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('0012')"))
    engine.dispose()

    # --purge clears the stale (now-unlocatable) 0012 row before stamping.
    command.stamp(_alembic_config(url), "head", purge=True)
    assert _current_revision(url) == _head_revision()

    # A subsequent upgrade is a no-op — it must not try to re-create tables.
    command.upgrade(_alembic_config(url), "head")
    assert _current_revision(url) == _head_revision()
