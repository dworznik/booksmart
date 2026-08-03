"""The `booksmart search` command, end to end against embedded Qdrant.

Driven with the deterministic fake providers, so the fake embedder's vectors —
not a real model's — decide the ranking. These tests assert what the command
promises (which records it finds, how it filters, how it fails), never a
particular order of similar hits; the ranking itself is core's contract, proven
against exact geometry in the core suite.
"""

import re
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
    code, out, _ = search(
        runner, book_id, "deep modules", "--dense-only", "--score-threshold", "1.1"
    )

    assert code == 0
    assert "No matches" in out


def test_an_impossible_threshold_still_lets_keyword_matches_through(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
    ingest_book: Callable[[str], None],
) -> None:
    """The documented hybrid behaviour, and the reason --score-threshold is not
    simply refused in hybrid mode: the floor bounds the meaning half, so a record
    that matches the words survives a bar no similarity could clear."""
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    ingest_book(book_id)

    # "determinism" is literally in the fake pipeline's knowledge object.
    code, out, _ = search(runner, book_id, "fake determinism", "--score-threshold", "1.1")

    assert code == 0
    assert "knowledge_object" in out


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
    """Count result rows by their rank cell.

    Not by the score: a fused score can reach 1.0, so a `0.xxx` match would
    silently skip the top hit — which is exactly the row a limit test cares
    about."""
    return len(_ranks(output))


_RANK_CELL = re.compile(r"^[│|]\s*(\d+)\s*[│|]")

# rank │ score │ type — the type cell of a result row.
_HIT_ROW = re.compile(r"^[│|]\s*\d+\s*[│|]\s*[\d.]+\s*[│|]\s*(\w+)\s*[│|]")


def _ranks(output: str) -> list[int]:
    return [int(m.group(1)) for line in output.splitlines() if (m := _RANK_CELL.match(line))]


def _first_hit_type(output: str) -> str:
    for line in output.splitlines():
        match = _HIT_ROW.match(line)
        if match:
            return match.group(1)
    raise AssertionError(f"no result rows in output:\n{output}")


# --- hybrid retrieval (issue #39) -------------------------------------------


def test_search_is_hybrid_by_default_and_says_so(
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
    # The legend is not decoration: the same column holds two kinds of number,
    # and without it 0.583 reads as a weak match rather than the best hit here.
    assert "meaning + keyword" in out
    assert "read the rank" in out


def test_dense_only_says_its_scores_are_similarities(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
    ingest_book: Callable[[str], None],
) -> None:
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    ingest_book(book_id)

    code, out, _ = search(runner, book_id, "deep modules", "--dense-only")

    assert code == 0
    assert "cosine similarity" in out
    assert "meaning + keyword" not in out


def test_hits_are_ranked_from_one(
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
    ranks = _ranks(out)
    assert ranks == list(range(1, len(ranks) + 1))
    assert len(ranks) > 1


def test_an_exact_term_dense_only_ranks_low_climbs_under_hybrid(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
    ingest_book: Callable[[str], None],
) -> None:
    """The demo issue #39 asks for, at the real command surface.

    The fake embedder derives its vectors from text length, so it has no notion
    of meaning at all — which makes it a fair stand-in for a dense model that has
    never seen the term. The knowledge object is the only record containing the
    word, and hybrid is what surfaces it.
    """
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    ingest_book(book_id)

    hybrid = search(runner, book_id, "determinism")[1]
    dense = search(runner, book_id, "determinism", "--dense-only")[1]

    # The chapter summaries say "deterministic"; only the knowledge object says
    # "determinism". Dense-only cannot tell those apart and ranks the chapters
    # first; the keyword half puts the exact term on top.
    assert _first_hit_type(hybrid) == "knowledge_object"
    assert _first_hit_type(dense) == "chapter"


def test_dense_only_needs_no_sparse_model(
    runner: CliRunner,
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
    ingest_book: Callable[[str], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A --dense-only search must not construct a sparse provider: the real one
    downloads a model, and this search will never use it."""
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    ingest_book(book_id)

    def explode(_: object) -> object:
        raise AssertionError("dense-only search built a sparse provider")

    monkeypatch.setattr("booksmart_cli.reads.build_sparse_embedding_provider", explode)

    code, out, err = search(runner, book_id, "deep modules", "--dense-only")

    assert code == 0, err
    assert "chapter" in out


def test_the_read_seam_reports_which_mode_produced_the_hits(
    home: Path,
    tmp_path: Path,
    make_pdf: Callable[..., Path],
    add_book: Callable[..., str],
    ingest_book: Callable[[str], None],
) -> None:
    """Scores mean different things per mode, so a non-CLI consumer of reads.py
    needs the mode to know what it is holding."""
    book_id = add_book(make_pdf(tmp_path / "book.pdf"))
    ingest_book(book_id)

    hybrid = reads.semantic_search(Runtime.load(), "deep modules")
    dense = reads.semantic_search(Runtime.load(), "deep modules", mode="dense")

    assert hybrid.mode == "hybrid"
    assert dense.mode == "dense"
    assert dense.hits[0].score <= 1.0  # a cosine similarity
