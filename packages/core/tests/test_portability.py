"""Portability proof: a data dir is a relocatable unit.

Every persisted path (``Book.storage_path`` / ``Book.parsed_path`` /
``Run.output_path``) is stored relative to the storage root, so the whole data
dir — the SQLite file plus ``storage/`` — can be moved under a different
absolute root (tarball to a beta tester, ``mv`` to another disk) and every
artifact still resolves. This test ingests a book under root A, physically moves
the data dir to root B, deletes A, then opens the relocated copy and reads every
artifact back.
"""

import uuid
from pathlib import Path

from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import sessionmaker

from booksmart_core import MIGRATIONS_PATH
from booksmart_core.models import Book, Run
from booksmart_core.runner import execute_run
from booksmart_core.stages import run_structure
from booksmart_core.storage import BookStorage

from .conftest import enable_sqlite_foreign_keys, store_book
from .test_ingestion_api import EXTRACT_TEXT, make_pdf_bytes


def _sqlite_engine(data_dir: Path) -> Engine:
    """A SQLite engine over ``<data_dir>/booksmart.db`` with foreign keys on."""
    engine = create_engine(f"sqlite:///{data_dir / 'booksmart.db'}")
    enable_sqlite_foreign_keys(engine)
    return engine


def test_data_dir_relocates_to_a_new_absolute_root(tmp_path: Path) -> None:
    root_a = tmp_path / "A"
    root_a.mkdir()

    # --- ingest under root A --------------------------------------------------
    engine_a = _sqlite_engine(root_a)
    alembic_cfg = AlembicConfig()
    alembic_cfg.set_main_option("script_location", str(MIGRATIONS_PATH))
    alembic_cfg.set_main_option("sqlalchemy.url", str(engine_a.url))
    command.upgrade(alembic_cfg, "head")

    session_factory_a = sessionmaker(bind=engine_a)
    storage_a = BookStorage(root_a / "storage")
    original_bytes = make_pdf_bytes()  # pymupdf output is not byte-stable across calls
    book_id = store_book(
        session_factory_a,
        storage_a,
        title="Clean Code",
        author="Robert C. Martin",
        filename="clean-code.pdf",
        content=original_bytes,
    )
    # The autouse stub-provider fixture stands in for the real LLM/embedder.
    run_id = execute_run(session_factory_a, root_a / "storage", uuid.UUID(book_id), "full")
    with session_factory_a() as session:
        run = session.get(Run, run_id)
        assert run is not None and run.status == "succeeded"
    engine_a.dispose()  # release the SQLite file handle before moving it

    # Nothing persisted may be an absolute path, or relocation would break.
    engine_check = _sqlite_engine(root_a)
    with sessionmaker(bind=engine_check)() as session:
        for book in session.scalars(select(Book)):
            assert not Path(book.storage_path).is_absolute()
            assert book.parsed_path is not None
            assert not Path(book.parsed_path).is_absolute()
        for stored_run in session.scalars(select(Run)):
            assert stored_run.output_path is not None
            assert not Path(stored_run.output_path).is_absolute()
    engine_check.dispose()

    # --- relocate the whole data dir A -> B, then delete A --------------------
    root_b = tmp_path / "B" / "nested" / "elsewhere"
    root_b.parent.mkdir(parents=True)
    root_a.rename(root_b)
    assert not root_a.exists()

    # --- reopen under root B and read every artifact back ---------------------
    engine_b = _sqlite_engine(root_b)
    session_factory_b = sessionmaker(bind=engine_b)
    storage_b = BookStorage(root_b / "storage")
    try:
        with session_factory_b() as session:
            book = session.get(Book, uuid.UUID(book_id))
            assert book is not None
            # The original upload resolves under the new root.
            original = storage_b.resolve(book.storage_path)
            assert original.exists()
            assert original.read_bytes() == original_bytes
            # The parsed artifact resolves and still holds the extracted text.
            parsed = storage_b.resolve(book.parsed_path)
            assert EXTRACT_TEXT in parsed.read_text(encoding="utf-8")
            # The run's output pointer resolves to the same artifact.
            run = session.get(Run, run_id)
            assert run is not None
            assert storage_b.resolve(run.output_path).read_text(encoding="utf-8") == (
                parsed.read_text(encoding="utf-8")
            )

        # Read-time resolution works for a stage re-run in the new root too:
        # run_structure re-reads the parsed markdown via storage.resolve and
        # would raise StagePreconditionError if the artifact did not resolve.
        with session_factory_b() as session:
            report = run_structure(session, uuid.UUID(book_id), storage=storage_b)
        assert report.stage == "structure"
        assert "chapters" in report.counts
    finally:
        engine_b.dispose()
