"""Command surface parity with the removed HTTP API: metadata updates, run/scope
guards, knowledge filtering, and the not-found / bad-input cases that used to be
4xx (docs/api-notes/http-surface.md).
"""

import uuid
from collections.abc import Callable
from pathlib import Path

from typer.testing import CliRunner

from booksmart_cli.main import app


def _ingested(
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
    ingest_book: Callable[[str], None],
) -> str:
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    ingest_book(book_id)
    return book_id


# --- books update ------------------------------------------------------------


def test_update_changes_only_passed_fields(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
) -> None:
    book_id = add_book(make_pdf(tmp_path / "book.pdf"), "--edition", "1st")
    result = runner.invoke(app, ["books", "update", book_id, "--title", "Refactoring"])
    assert result.exit_code == 0
    show = runner.invoke(app, ["books", "show", book_id]).stdout
    assert "Refactoring" in show  # changed
    assert "1st" in show  # untouched


def test_update_can_clear_an_optional_field(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
) -> None:
    book_id = add_book(make_pdf(tmp_path / "book.pdf"), "--notes", "temporary")
    assert "temporary" in runner.invoke(app, ["books", "show", book_id]).stdout
    assert runner.invoke(app, ["books", "update", book_id, "--notes", ""]).exit_code == 0
    assert "temporary" not in runner.invoke(app, ["books", "show", book_id]).stdout


def test_update_cannot_clear_title(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
) -> None:
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    result = runner.invoke(app, ["books", "update", book_id, "--title", ""])
    assert result.exit_code == 1
    assert "not cleared" in result.stderr


def test_update_publication_year_can_be_set_and_cleared(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
) -> None:
    # The one optional integer column must clear like the text ones (HTTP null).
    book_id = add_book(make_pdf(tmp_path / "book.pdf"), "--publication-year", "1999")
    assert "1999" in runner.invoke(app, ["books", "show", book_id]).stdout
    assert runner.invoke(app, ["books", "update", book_id, "--publication-year", ""]).exit_code == 0
    assert "1999" not in runner.invoke(app, ["books", "show", book_id]).stdout


def test_update_publication_year_must_be_an_integer(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
) -> None:
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    result = runner.invoke(app, ["books", "update", book_id, "--publication-year", "soon"])
    assert result.exit_code == 1
    assert "integer" in result.stderr


def test_update_unknown_book_is_an_error(runner: CliRunner, home: Path) -> None:
    result = runner.invoke(app, ["books", "update", str(uuid.uuid4()), "--title", "X"])
    assert result.exit_code == 1
    assert "No book" in result.stderr


# --- ingest scope guards -----------------------------------------------------


def test_unknown_scope_is_rejected(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
) -> None:
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    result = runner.invoke(app, ["ingest", book_id, "--scope", "bogus"])
    assert result.exit_code == 1
    assert "Unknown scope" in result.stderr


def test_incremental_scope_without_prior_success_is_rejected(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
) -> None:
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    result = runner.invoke(app, ["ingest", book_id, "--scope", "profile"])
    assert result.exit_code == 1
    assert "prior successful run" in result.stderr


def test_incremental_scope_after_full_succeeds(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
    ingest_book: Callable[[str], None],
) -> None:
    book_id = _ingested(tmp_path, make_pdf, add_book, ingest_book)
    result = runner.invoke(app, ["ingest", book_id, "--scope", "profile"])
    assert result.exit_code == 0
    assert "succeeded" in result.stdout


# --- knowledge / runs lookups ------------------------------------------------


def test_knowledge_list_filters_by_type(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
    ingest_book: Callable[[str], None],
) -> None:
    book_id = _ingested(tmp_path, make_pdf, add_book, ingest_book)
    matching = runner.invoke(app, ["knowledge", "list", book_id, "--type", "Principle"])
    assert matching.exit_code == 0
    assert "Principle" in matching.stdout
    empty = runner.invoke(app, ["knowledge", "list", book_id, "--type", "Checklist"])
    assert empty.exit_code == 0
    assert "No knowledge objects" in empty.stdout


def test_knowledge_list_invalid_type_is_rejected(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
    ingest_book: Callable[[str], None],
) -> None:
    book_id = _ingested(tmp_path, make_pdf, add_book, ingest_book)
    result = runner.invoke(app, ["knowledge", "list", book_id, "--type", "Nonsense"])
    assert result.exit_code == 1
    assert "Unknown knowledge type" in result.stderr


def test_knowledge_show_unknown_is_an_error(runner: CliRunner, home: Path) -> None:
    result = runner.invoke(app, ["knowledge", "show", str(uuid.uuid4())])
    assert result.exit_code == 1


def test_run_show_unknown_is_an_error(runner: CliRunner, home: Path) -> None:
    result = runner.invoke(app, ["runs", "show", str(uuid.uuid4())])
    assert result.exit_code == 1


# --- not-found reads on an unknown book --------------------------------------


def test_reads_on_unknown_book_error_cleanly(runner: CliRunner, home: Path) -> None:
    missing = str(uuid.uuid4())
    for args in (["structure", missing], ["runs", "list", missing], ["knowledge", "list", missing]):
        result = runner.invoke(app, args)
        assert result.exit_code == 1
        assert "No book" in result.stderr


def test_profile_before_ingest_is_an_error(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
) -> None:
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    result = runner.invoke(app, ["profile", book_id])
    assert result.exit_code == 1
    assert "no profile" in result.stderr.lower()
