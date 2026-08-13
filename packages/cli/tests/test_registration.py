"""`add`: format validation, byte-identical dedup, metadata — the rules the
removed upload endpoint enforced (docs/api-notes/upload-validation.md).
"""

from collections.abc import Callable
from pathlib import Path

from typer.testing import CliRunner

from booksmart_cli.main import app


def test_unsupported_extension_is_rejected(runner: CliRunner, home: Path, tmp_path: Path) -> None:
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    result = runner.invoke(app, ["add", str(bad), "--title", "T", "--author", "A"])
    assert result.exit_code == 1
    assert "Unsupported file type" in result.stderr


def test_content_not_matching_extension_is_rejected(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    # A .pdf whose bytes are not a PDF (magic-byte check, not just the suffix).
    fake = tmp_path / "book.pdf"
    fake.write_bytes(b"<html>not a pdf</html>")
    result = runner.invoke(app, ["add", str(fake), "--title", "T", "--author", "A"])
    assert result.exit_code == 1
    assert "does not look like a PDF" in result.stderr


def test_byte_identical_duplicate_is_rejected_with_existing_id(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
) -> None:
    pdf = make_pdf(tmp_path / "book.pdf")
    book_id = add_book(pdf)

    # A different file path, byte-identical content, different metadata: still a dup.
    copy = tmp_path / "renamed.pdf"
    copy.write_bytes(pdf.read_bytes())
    result = runner.invoke(app, ["add", str(copy), "--title", "Other", "--author", "Someone"])
    assert result.exit_code == 1
    assert "already registered" in result.stderr
    assert book_id in result.stderr


def test_missing_required_metadata_is_a_usage_error(
    runner: CliRunner, home: Path, tmp_path: Path, make_pdf: Callable[..., Path]
) -> None:
    result = runner.invoke(app, ["add", str(make_pdf(tmp_path / "book.pdf"))])
    # typer/click rejects missing required options before the command runs.
    assert result.exit_code == 2


def test_metadata_and_hints_are_stored(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
) -> None:
    book_id = add_book(
        make_pdf(tmp_path / "book.pdf"),
        "--edition",
        "2nd",
        "--framework",
        "DDD",
        "--publication-year",
        "2003",
    )
    show = runner.invoke(app, ["books", "show", book_id])
    assert show.exit_code == 0
    assert "2nd" in show.stdout
    assert "DDD" in show.stdout
    assert "2003" in show.stdout


def test_failed_registration_leaves_no_stored_file(
    runner: CliRunner, home: Path, tmp_path: Path
) -> None:
    # Rejected uploads must not litter storage/ with an orphaned original.
    bad = tmp_path / "book.pdf"
    bad.write_bytes(b"<html>not a pdf</html>")
    runner.invoke(app, ["add", str(bad), "--title", "T", "--author", "A"])
    books_dir = home / "storage" / "books"
    assert not books_dir.exists() or not any(books_dir.iterdir())


def test_add_does_not_ingest(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
) -> None:
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    runs = runner.invoke(app, ["runs", "list", book_id])
    assert runs.exit_code == 0
    assert "No runs yet" in runs.stdout
