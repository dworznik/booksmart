"""CLI test fixtures.

Every test drives the real console script through typer's CliRunner against an
isolated home directory (fresh SQLite file, storage/, embedded Qdrant) with the
deterministic fake providers selected — the CLI's own no-Docker, no-Postgres
end-to-end shape, replacing the removed compose e2e.

Shared helpers are exposed as fixtures (not module imports) so this test suite
stays a plain directory — no ``__init__.py`` — and never collides with the
core suite's ``tests`` package.
"""

import re
from collections.abc import Callable, Iterator
from pathlib import Path

import pymupdf
import pytest
from typer.testing import CliRunner

from booksmart_cli.main import app

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    home_dir = tmp_path / "home"
    monkeypatch.setenv("BOOKSMART_HOME", str(home_dir))
    monkeypatch.setenv("BOOKSMART_LLM_PROVIDER", "fake")
    monkeypatch.setenv("BOOKSMART_EMBEDDING_PROVIDER", "fake")
    # Fake here too, for a reason the other two do not have: the real sparse
    # provider is local, but constructing it downloads the BM25 model, so the
    # suite would otherwise need a network and a warm cache to run at all.
    monkeypatch.setenv("BOOKSMART_SPARSE_PROVIDER", "fake")
    # Make sure no stray server/db config or ambient API key (BOOKSMART_-prefixed
    # or the vendors' conventional variables) leaks in from the developer's env.
    for leaked in (
        "BOOKSMART_DATABASE_URL",
        "BOOKSMART_STORAGE_ROOT",
        "BOOKSMART_QDRANT_URL",
        "BOOKSMART_SPARSE_MODEL",
        "BOOKSMART_ANTHROPIC_API_KEY",
        "BOOKSMART_OPENAI_API_KEY",
        "BOOKSMART_GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(leaked, raising=False)
    yield home_dir


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def make_pdf() -> Callable[..., Path]:
    """Factory: a real PDF with two chapter headings.

    Structure is **typographic**, not literal `#` characters written into the
    page. This fixture used to write Markdown as PDF text and rely on the parser
    passing it through; the pdf route reads type — a heading is a short line set
    larger than the body — so a page in one face at one size correctly declines
    and has no structure at all. A page also carries a page of text, because the
    router probes characters per page and routes a PDF under the threshold to OCR.
    """

    CHAPTER_POINTS = 24
    BODY_POINTS = 11
    FILLER = (
        "Ordinary body prose, enough of it that a page reads as a page of a book "
        "rather than as a scan with no text layer. " * 3
    )

    def _make(path: Path, body: str = "Ubiquitous Language rules the domain.") -> Path:
        doc = pymupdf.open()
        for chapter, text in (
            ("Chapter One", body),
            ("Chapter Two", "Deep modules hide complexity."),
        ):
            page = doc.new_page()
            page.insert_text((72, 80), chapter, fontsize=CHAPTER_POINTS)
            page.insert_textbox(
                pymupdf.Rect(72, 120, 520, 700), f"{text} {FILLER}", fontsize=BODY_POINTS
            )
        path.write_bytes(doc.tobytes())
        doc.close()
        return path

    return _make


@pytest.fixture()
def add_book(runner: CliRunner) -> Callable[..., str]:
    """Factory: register a book via the CLI and return its id."""

    def _add(pdf: Path, *extra: str) -> str:
        result = runner.invoke(
            app, ["add", str(pdf), "--title", "DDD", "--author", "Evans", *extra]
        )
        assert result.exit_code == 0, result.stdout + result.stderr
        match = UUID_RE.search(result.stdout)
        assert match is not None
        return match.group()

    return _add


@pytest.fixture()
def ingest_book(runner: CliRunner) -> Callable[[str], None]:
    """Factory: run a full ingest and assert it succeeded."""

    def _ingest(book_id: str) -> None:
        result = runner.invoke(app, ["ingest", book_id])
        assert result.exit_code == 0, result.stdout + result.stderr

    return _ingest
