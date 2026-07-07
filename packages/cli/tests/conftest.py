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
    # Make sure no stray server/db config leaks in from the developer's env.
    for leaked in ("BOOKSMART_DATABASE_URL", "BOOKSMART_STORAGE_ROOT", "BOOKSMART_QDRANT_URL"):
        monkeypatch.delenv(leaked, raising=False)
    yield home_dir


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def make_pdf() -> Callable[..., Path]:
    """Factory: write a minimal but real PDF with two chapter headings."""

    def _make(path: Path, body: str = "Ubiquitous Language rules the domain.") -> Path:
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            f"# Chapter One\n\n{body}\n\n# Chapter Two\n\nDeep modules hide complexity.",
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
