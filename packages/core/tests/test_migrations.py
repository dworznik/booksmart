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

EXPECTED_TABLES = {
    "books",
    "chapters",
    "sections",
    "book_profiles",
    "knowledge_objects",
    "runs",
    "run_stages",
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


def test_the_history_is_one_line_rooted_in_the_squashed_baseline() -> None:
    """What the squash bought, stated as the invariant rather than as a count.

    One root and no branching is the part that has to hold: two roots would mean
    the old per-change chain had come back, and a branch is a history that
    cannot be walked to a single head. The number of revisions above the root is
    just how much has happened since — a migration is the ordinary way to change
    the schema, and forbidding a second one would forbid that.
    """
    script = ScriptDirectory.from_config(_alembic_config("sqlite://"))
    revisions = list(script.walk_revisions())

    roots = [revision for revision in revisions if revision.down_revision is None]
    assert [root.revision for root in roots] == ["0001"]
    assert len(script.get_heads()) == 1


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


def _legacy_database(tmp_path: Path) -> str:
    """A database holding the pre-squash schema and pointing at the old ``0012``.

    Built by migrating an empty database to ``0001`` rather than from
    ``Base.metadata``, and that is the whole point of this helper. The metadata
    is *today's* — it grows with every model added — so a legacy database built
    from it already has the tables the migrations under test are supposed to
    deliver, and the adoption procedure passes by construction however wrong it
    is. Migrating to the baseline pins the fixture to the schema that deployment
    actually holds.
    """
    url = f"sqlite:///{tmp_path / 'legacy.db'}"
    command.upgrade(_alembic_config(url), "0001")
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM alembic_version"))
        conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('0012')"))
    engine.dispose()
    return url


def test_existing_deployment_adopts_the_baseline_then_upgrades(tmp_path: Path) -> None:
    """The one existing Postgres deployment already holds the baseline schema and
    an ``alembic_version`` pointing at the pre-squash ``0012``. The procedure in
    the baseline's docstring must move it onto this history without re-running
    the DDL it already has — and must then leave it at head, with every revision
    since the baseline actually applied."""
    url = _legacy_database(tmp_path)

    # --purge clears the stale (now-unlocatable) 0012 row before stamping. It
    # stamps *the baseline*, which is the revision whose schema is already there.
    command.stamp(_alembic_config(url), "0001", purge=True)
    assert _current_revision(url) == "0001"

    command.upgrade(_alembic_config(url), "head")

    assert _current_revision(url) == _head_revision()
    engine = create_engine(url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert EXPECTED_TABLES <= tables


def test_stamping_head_would_skip_every_revision_after_the_baseline(
    tmp_path: Path,
) -> None:
    """Why the procedure names the baseline rather than ``head``.

    ``stamp --purge head`` was correct while the baseline was the only revision,
    and stopped being correct the moment a second one landed — silently, because
    it still succeeds. The deployment is then marked as holding every migration
    since while having applied none, and the upgrade that would have fixed it is
    a no-op forever.
    """
    url = _legacy_database(tmp_path)

    command.stamp(_alembic_config(url), "head", purge=True)
    command.upgrade(_alembic_config(url), "head")

    engine = create_engine(url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert not EXPECTED_TABLES <= tables
