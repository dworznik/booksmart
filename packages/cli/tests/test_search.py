"""The `booksmart search` command, end to end against embedded Qdrant.

Driven with the deterministic fake providers, so the fake embedder's vectors —
not a real model's — decide the ranking. These tests assert what the command
promises (which records it finds, how it filters, how it fails), never a
particular order of similar hits; the ranking itself is core's contract, proven
against exact geometry in the core suite.
"""

import uuid
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from booksmart_cli import reads
from booksmart_cli.main import app
from booksmart_cli.runtime import Runtime


@pytest.fixture(autouse=True)
def wide_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rich sizes its table columns to the terminal and elides what will not fit;
    at CliRunner's default 80 columns a book id is squeezed out entirely. Give
    the tests room so they assert on the rendering, not on the elision."""
    monkeypatch.setenv("COLUMNS", "200")


def search(runner: CliRunner, *args: str) -> tuple[int, str, str]:
    result = runner.invoke(app, ["search", *args])
    return result.exit_code, result.stdout, result.stderr


def test_finds_records_embedded_by_an_ingest(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
    ingest_book: Callable[[str], None],
) -> None:
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    ingest_book(book_id)

    code, out, _ = search(runner, book_id, "deep modules")

    assert code == 0
    # The fake pipeline embeds chapter summaries and one knowledge object per book.
    assert "chapter" in out
    assert "knowledge_object" in out


def test_all_scope_searches_every_book_and_names_the_book(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
    ingest_book: Callable[[str], None],
) -> None:
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    ingest_book(book_id)

    code, out, _ = search(runner, "all", "deep modules")

    assert code == 0
    # The book column identifies which book each hit came from.
    assert book_id in out.replace("\n", "")


def test_type_restricts_the_records_searched(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
    ingest_book: Callable[[str], None],
) -> None:
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    ingest_book(book_id)

    code, out, _ = search(runner, book_id, "deep modules", "--type", "knowledge_object")

    assert code == 0
    assert "knowledge_object" in out
    assert "chapter" not in out


def test_limit_caps_the_number_of_hits(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
    ingest_book: Callable[[str], None],
) -> None:
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    ingest_book(book_id)

    unlimited = search(runner, book_id, "deep modules")[1]
    limited = search(runner, book_id, "deep modules", "--limit", "1")[1]

    assert _hit_rows(limited) == 1
    assert _hit_rows(unlimited) > 1


def test_score_threshold_can_exclude_everything(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
    ingest_book: Callable[[str], None],
) -> None:
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    ingest_book(book_id)

    # Cosine similarity never exceeds 1.0, so nothing can clear this bar.
    code, out, _ = search(runner, book_id, "deep modules", "--score-threshold", "1.1")

    assert code == 0
    assert "No matches" in out


def test_a_registered_but_un_ingested_book_has_no_matches(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
) -> None:
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))

    code, out, _ = search(runner, book_id, "deep modules")

    # Nothing embedded yet is a normal state, not an error.
    assert code == 0
    assert "No matches" in out


def test_unknown_book_fails_cleanly(runner: CliRunner, home: Path) -> None:
    code, _, err = search(runner, str(uuid.uuid4()), "deep modules")

    assert code == 1
    assert "No book with id" in err


def test_malformed_book_id_fails_cleanly(runner: CliRunner, home: Path) -> None:
    code, _, err = search(runner, "not-a-uuid", "deep modules")

    assert code == 1
    assert "Not a valid id" in err


def test_empty_query_fails_cleanly(runner: CliRunner, home: Path) -> None:
    code, _, err = search(runner, "all", "   ")

    assert code == 1
    assert "must not be empty" in err


def test_unknown_type_fails_cleanly(runner: CliRunner, home: Path) -> None:
    code, _, err = search(runner, "all", "deep modules", "--type", "paragraph")

    assert code == 1
    assert "paragraph" in err
    assert "knowledge_object" in err


def test_non_positive_limit_fails_cleanly(runner: CliRunner, home: Path) -> None:
    code, _, err = search(runner, "all", "deep modules", "--limit", "0")

    assert code == 1
    assert "at least 1" in err


def test_search_leaves_the_embedded_qdrant_lock_free_for_the_next_command(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
    ingest_book: Callable[[str], None],
) -> None:
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    ingest_book(book_id)

    assert search(runner, book_id, "deep modules")[0] == 0
    # Embedded Qdrant takes a single-process lock on its directory; a search that
    # forgot to close its client would make every later command fail to open it.
    second_code, _, second_err = search(runner, book_id, "deep modules")
    assert second_code == 0, second_err
    reingest = runner.invoke(app, ["ingest", book_id, "--scope", "embeddings"])
    assert reingest.exit_code == 0, reingest.stdout + reingest.stderr


def test_the_read_seam_carries_the_query_embedding_usage(
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
    ingest_book: Callable[[str], None],
) -> None:
    """The command renders hits only, but the seam under it reports what the
    query cost — so a non-CLI consumer of reads.py can cost its search traffic
    (issue #57). Nothing else in this file looks past the terminal surface."""
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    ingest_book(book_id)

    results = reads.semantic_search(Runtime.load(), "deep modules")

    assert results.hits
    # The fake embedding provider reports a truthful zero: no call was billed.
    assert results.embedding_tokens == 0


def _hit_rows(output: str) -> int:
    """Count result rows: every hit renders its score as `0.xxx` in the table."""
    return sum(1 for line in output.splitlines() if "│ 0." in line or "| 0." in line)
