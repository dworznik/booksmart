"""Bench test fixtures.

Everything here is synthetic. The real assets checkout is private and this repo
is public, so tests build their own throwaway truth trees with placeholder slugs
— never a real title, never a real corpus shape.

Shared helpers are fixtures rather than module imports so this suite stays a
plain directory (no ``__init__.py``) alongside the core and CLI suites.
"""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """No ambient bench or booksmart configuration reaches a test."""
    for leaked in (
        "BOOKSMART_BENCH_ASSETS",
        "BOOKSMART_BENCH_HOME",
        "BOOKSMART_LLM_PROVIDER",
        "BOOKSMART_LLM_MODEL",
        "BOOKSMART_EMBEDDING_PROVIDER",
        "BOOKSMART_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(leaked, raising=False)
    # A home under tmp_path by default, so nothing ever touches the real one.
    monkeypatch.setenv("BOOKSMART_BENCH_HOME", str(tmp_path / "bench-home"))
    yield


@pytest.fixture()
def make_assets(tmp_path: Path) -> Callable[..., Path]:
    """Factory: a directory shaped like an assets checkout."""

    def _make(name: str = "assets", *, truth: bool = True) -> Path:
        root = tmp_path / name
        if truth:
            (root / "truth" / "placeholder-book").mkdir(parents=True)
        else:
            root.mkdir(parents=True)
        return root

    return _make


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()
