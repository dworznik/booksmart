"""End-to-end CLI flow — the no-Docker, no-Postgres replacement for the removed
compose e2e: register and ingest a real PDF with the fake providers against a
SQLite file and embedded Qdrant, then read every artifact back through the
commands. Also proves the whole data dir relocates (issue #25) at the CLI level.
"""

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from booksmart_cli.main import app


def test_add_then_ingest_produces_every_artifact(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
) -> None:
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))

    ingest = runner.invoke(app, ["ingest", book_id])
    assert ingest.exit_code == 0, ingest.stdout
    assert "succeeded" in ingest.stdout
    # Progress streamed each stage as it ran.
    for stage in ("parse", "structure", "profile", "extraction", "summaries", "embeddings"):
        assert stage in ingest.stdout

    assert book_id in runner.invoke(app, ["books", "list"]).stdout

    structure = runner.invoke(app, ["structure", book_id])
    assert structure.exit_code == 0
    assert "Chapter" in structure.stdout

    profile = runner.invoke(app, ["profile", book_id])
    assert profile.exit_code == 0
    assert "fake-llm-1" in profile.stdout

    knowledge = runner.invoke(app, ["knowledge", "list", book_id])
    assert knowledge.exit_code == 0
    assert "Principle" in knowledge.stdout

    runs = runner.invoke(app, ["runs", "list", book_id])
    assert runs.exit_code == 0
    assert "succeeded" in runs.stdout


def test_data_dir_relocates_to_a_new_home(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
    ingest_book: Callable[[str], None],
) -> None:
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    ingest_book(book_id)

    # Relocate the entire data dir (SQLite file + storage/ + embedded Qdrant) to
    # a different absolute path, as a tarball to a fresh machine would.
    relocated = tmp_path / "elsewhere" / "booksmart-home"
    relocated.parent.mkdir(parents=True)
    shutil.move(str(home), str(relocated))
    monkeypatch.setenv("BOOKSMART_HOME", str(relocated))

    listing = runner.invoke(app, ["books", "list"])
    assert listing.exit_code == 0
    assert book_id in listing.stdout

    structure = runner.invoke(app, ["structure", book_id])
    assert structure.exit_code == 0
    assert "Chapter" in structure.stdout
