"""Bench test fixtures.

Everything here is synthetic. The real assets checkout is private and this repo
is public, so tests build their own throwaway truth trees with placeholder slugs
— never a real title, never a real corpus shape.

Shared helpers are fixtures rather than module imports so this suite stays a
plain directory (no ``__init__.py``) alongside the core and CLI suites.
"""

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """No ambient bench or booksmart configuration reaches a test.

    By prefix rather than by name: ``load_settings`` reads ``BOOKSMART_<FIELD>``
    for every field ``Settings`` has, so a list of names would have to be
    extended in lockstep with a model in another package — and the failure mode
    is a developer's own shell quietly changing what a test measures.
    """
    for leaked in [name for name in os.environ if name.startswith("BOOKSMART_")]:
        monkeypatch.delenv(leaked, raising=False)
    # A home under tmp_path by default, so nothing ever touches the real one.
    monkeypatch.setenv("BOOKSMART_BENCH_HOME", str(tmp_path / "bench-home"))
    yield


@pytest.fixture()
def make_assets(tmp_path: Path, write_truth: Callable[..., Path]) -> Callable[..., Path]:
    """Factory: an assets checkout holding one valid, minimal book.

    Valid on purpose — the verbs lint truth before doing anything, so a fixture
    that was merely *shaped* like a checkout would fail every test for a reason
    the test was not about.
    """

    def _make(name: str = "assets", *, truth: bool = True) -> Path:
        root = tmp_path / name
        if not truth:
            root.mkdir(parents=True)
            return root
        return write_truth(root=root)

    return _make


@pytest.fixture()
def write_truth(tmp_path: Path) -> Callable[..., Path]:
    """Factory: a truth tree on disk, from plain Python structures.

    Takes dicts rather than YAML strings so a test reads as the shape it is
    exercising, not as an indented blob.
    """

    def _write(
        slug: str = "placeholder-book",
        *,
        root: Path | None = None,
        book: dict[str, object] | None = None,
        toc: dict[str, object] | None = None,
        queries: list[dict[str, object]] | None = None,
        concepts: list[dict[str, object]] | None = None,
        index_pairs: list[dict[str, object]] | None = None,
        area: str | None = None,
        area_queries: list[dict[str, object]] | None = None,
    ) -> Path:
        assets = root or (tmp_path / "assets")
        book_dir = assets / "truth" / slug
        book_dir.mkdir(parents=True, exist_ok=True)

        _dump(book_dir / "book.yaml", book if book is not None else _DEFAULT_BOOK | {"slug": slug})
        _dump(book_dir / "toc.yaml", toc if toc is not None else _DEFAULT_TOC)
        _dump(book_dir / "queries.yaml", queries if queries is not None else [])
        _dump(book_dir / "concepts.yaml", concepts if concepts is not None else [])
        _dump(book_dir / "index-pairs.yaml", index_pairs if index_pairs is not None else [])

        if area is not None:
            area_dir = assets / "truth" / "areas" / area
            area_dir.mkdir(parents=True, exist_ok=True)
            _dump(area_dir / "queries.yaml", area_queries or [])
        return assets

    return _write


_DEFAULT_BOOK: dict[str, object] = {
    "title": "Placeholder Book",
    "edition": "1st",
    "authors": ["A. Author"],
    "area": "placeholder-area",
    "source": {"file": "sources/placeholder.pdf", "sha256": "0" * 64},
}

_DEFAULT_TOC: dict[str, object] = {
    "chapters": [
        {
            "id": "1",
            "title": "First Chapter",
            "sections": [
                {"id": "1.1", "title": "Widgets And Sprockets"},
                {"id": "1.2", "title": "Grommets"},
            ],
        }
    ]
}


def _dump(path: Path, payload: object) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


@pytest.fixture()
def write_run(tmp_path: Path) -> Callable[..., Path]:
    """Factory: a run file on disk, defaulting to the smallest valid one."""

    def _write(
        name: str = "run.json",
        *,
        run_id: str = "2026-01-01T00-00-00Z-recall",
        snapshot: str = "fake-fake-00000000",
        books: list[str] | None = None,
        results: list[dict[str, object]] | None = None,
        **sections: object,
    ) -> Path:
        payload: dict[str, object] = {
            "run_id": run_id,
            "corpus": {"snapshot": snapshot, "books": books or ["placeholder-book"]},
            "search": {"mode": "hybrid", "limit": 10},
            "results": results if results is not None else [],
        }
        payload.update(sections)
        path = tmp_path / name
        path.write_text(json.dumps(payload, indent=2))
        return path

    return _write


def hits(*locs: str, book: str = "placeholder-book") -> list[dict[str, object]]:
    """Ranked hits from a bare list of locations, rank 1 first."""
    return [
        {"rank": rank, "book": book, "loc": loc, "record_type": "section"}
        for rank, loc in enumerate(locs, start=1)
    ]


@pytest.fixture()
def book_defaults() -> dict[str, object]:
    """A valid book identity, for tests that need to vary exactly one field."""
    return dict(_DEFAULT_BOOK)


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()
