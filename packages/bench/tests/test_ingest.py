"""Building a corpus: the only expensive verb, driven here against fake providers.

The pipeline is real — the same public ``run_<stage>`` functions the product
runs — but the providers are core's deterministic stand-ins and the book is a
two-page PDF generated in the test. That keeps the ingest path genuinely
exercised without a network, an API key, or any book text in this repo.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
from sqlalchemy import func, select

from booksmart_bench.corpus import Corpus
from booksmart_bench.errors import BenchError, SourceMissingError
from booksmart_bench.ingest import ingest_scope, read_provenance
from booksmart_bench.truth import load_truth
from booksmart_core.config import Settings
from booksmart_core.models import Chapter

pytestmark = pytest.mark.usefixtures("fake_providers")


@pytest.fixture()
def assets(write_truth: Callable[..., Path], place_source: Callable[..., str]) -> Path:
    """An assets checkout with one book whose source is present and pinned."""
    root = write_truth("a-book", book={"title": "A Book", "area": "an-area"})
    place_source(root, "a-book")
    return root


def ingest(assets: Path, scope: str = "a-book", **kwargs: object) -> object:
    from booksmart_bench.config import load_settings

    return ingest_scope(assets, load_truth(assets), scope, load_settings(), **kwargs)  # type: ignore[arg-type]


class TestCorpusHome:
    def test_opens_under_the_snapshot_directory(self) -> None:
        from booksmart_bench.config import config_snapshot, load_settings

        settings = load_settings()
        with Corpus.open(settings) as corpus:
            assert corpus.home.name == config_snapshot(settings)
            assert corpus.home.is_dir()

    def test_keeps_database_storage_and_vectors_together(self) -> None:
        """The corpus is a single portable directory, like the product CLI's
        home — so a snapshot can be deleted by deleting one folder."""
        from booksmart_bench.config import load_settings

        with Corpus.open(load_settings()) as corpus:
            assert corpus.settings.storage_root.is_relative_to(corpus.home)
            assert corpus.settings.qdrant_path is not None
            assert corpus.settings.qdrant_path.is_relative_to(corpus.home)
            assert str(corpus.home) in corpus.settings.database_url

    def test_relocating_the_corpus_does_not_change_the_snapshot(self) -> None:
        """Store locations are excluded from the snapshot on purpose: a corpus
        that moved is still the same corpus."""
        from booksmart_bench.config import config_snapshot

        one = config_snapshot(Settings(database_url="sqlite:///a.db"))
        other = config_snapshot(Settings(database_url="sqlite:///b.db"))

        assert one == other


class TestIngest:
    def test_runs_the_pipeline_and_records_what_it_did(self, assets: Path) -> None:
        outcome = ingest(assets)

        assert [book.slug for book in outcome.books] == ["a-book"]
        assert outcome.books[0].status == "succeeded"

    def test_the_book_lands_in_the_corpus(self, assets: Path) -> None:
        from booksmart_bench.config import load_settings

        ingest(assets)

        with Corpus.open(load_settings()) as corpus, corpus.session_factory() as session:
            chapters = session.scalar(select(func.count()).select_from(Chapter))
        assert chapters and chapters > 0

    def test_every_stage_reports_its_own_time_and_spend(self, assets: Path) -> None:
        """The interim for booksmart#65: the Run row carries a total, so the
        harness captures the per-Stage breakdown in-process instead."""
        outcome = ingest(assets)

        stages = outcome.books[0].stages
        assert [stage.stage for stage in stages] == [
            "parse",
            "structure",
            "profile",
            "extraction",
            "summaries",
            "embeddings",
        ]
        assert all(stage.seconds >= 0 for stage in stages)
        assert outcome.books[0].wall_seconds >= 0

    def test_provenance_survives_the_process(self, assets: Path) -> None:
        from booksmart_bench.config import load_settings

        ingest(assets)

        with Corpus.open(load_settings()) as corpus:
            recorded = read_provenance(corpus.home)
        assert "a-book" in recorded.books


class TestSources:
    def test_a_missing_source_is_named_before_anything_is_spent(
        self, write_truth: Callable[..., Path]
    ) -> None:
        root = write_truth("a-book", book={"title": "A Book", "area": "an-area"})

        with pytest.raises(SourceMissingError, match="a-book"):
            ingest(root)

    def test_a_sha256_mismatch_refuses_to_ingest(
        self, assets: Path, write_truth: Callable[..., Path]
    ) -> None:
        """Truth authored against one artifact must never silently score
        another."""
        (assets / "sources" / "a-book.pdf").write_bytes(b"%PDF-1.4 different bytes")

        with pytest.raises(SourceMissingError, match="sha256"):
            ingest(assets)

    def test_an_unpinned_source_is_ingested_but_its_hash_is_reported(
        self, write_truth: Callable[..., Path], place_source: Callable[..., str]
    ) -> None:
        """Every book is unpinned until the handover lands, so refusing would
        block the whole harness — but the hash to paste in has to be offered."""
        root = write_truth("a-book", book={"title": "A Book", "area": "an-area"})
        digest = place_source(root, "a-book", pin=False)

        outcome = ingest(root)

        assert outcome.books[0].status == "succeeded"
        assert any(digest in note for note in outcome.notes)


class TestIdempotency:
    def test_re_ingesting_an_unchanged_config_is_a_no_op(self, assets: Path) -> None:
        """Ingest once, recall often — re-running must not spend the afternoon
        again."""
        ingest(assets)

        second = ingest(assets)

        assert second.books[0].status == "skipped"

    def test_force_re_ingests(self, assets: Path) -> None:
        ingest(assets)

        second = ingest(assets, force=True)

        assert second.books[0].status == "succeeded"

    def test_a_changed_source_is_re_ingested_without_being_forced(
        self, assets: Path, place_source: Callable[..., str]
    ) -> None:
        """The corpus is keyed by config, but a book is keyed by its bytes; a
        new edition is not the book that was ingested."""
        ingest(assets)
        place_source(assets, "a-book")  # a fresh PDF, new hash, re-pinned

        second = ingest(assets)

        assert second.books[0].status == "succeeded"


class TestStageFailure:
    def test_a_failing_stage_is_recorded_not_raised(
        self, assets: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same contract as core's Runner: an expected Stage failure lands on
        the Run, it does not come out of the Runner."""

        def boom(*args: object, **kwargs: object) -> object:
            raise RuntimeError("stage exploded")

        monkeypatch.setattr("booksmart_bench.ingest.run_summaries", boom)

        outcome = ingest(assets)

        assert outcome.books[0].status == "failed"
        assert "stage exploded" in (outcome.books[0].error or "")

    def test_a_failed_stage_does_not_commit_its_partial_writes(
        self, assets: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_extraction deletes the book's knowledge objects before writing
        the replacements. Without a rollback, a failure between the two commits
        the deletion and leaves the corpus permanently emptied for that book."""
        from booksmart_bench.config import load_settings
        from booksmart_core.models import KnowledgeObject

        ingest(assets)
        with Corpus.open(load_settings()) as corpus, corpus.session_factory() as session:
            before = session.scalar(select(func.count()).select_from(KnowledgeObject))
        assert before

        def delete_then_fail(session: object, book_id: object, **kwargs: object) -> object:
            # Exactly run_extraction's shape: clear the book's objects, then die
            # before the commit that would write the replacements.
            from sqlalchemy import delete

            session.execute(  # type: ignore[attr-defined]
                delete(KnowledgeObject).where(KnowledgeObject.book_id == book_id)
            )
            raise RuntimeError("failed after clearing knowledge")

        monkeypatch.setattr("booksmart_bench.ingest.run_extraction", delete_then_fail)
        ingest(assets, force=True)

        with Corpus.open(load_settings()) as corpus, corpus.session_factory() as session:
            after = session.scalar(select(func.count()).select_from(KnowledgeObject))
        assert after == before

    def test_a_failure_keeps_the_provenance_of_books_already_done(
        self,
        write_truth: Callable[..., Path],
        place_source: Callable[..., str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Losing the record of an hour already spent is the one thing ingest
        must never do."""
        from booksmart_bench.config import load_settings

        root = write_truth("book-one", book={"title": "One", "area": "an-area"})
        write_truth("book-two", root=root, book={"title": "Two", "area": "an-area"})
        for slug in ("book-one", "book-two"):
            place_source(root, slug)

        # An error the Runner does not catch — registration, or anything else
        # outside the Stage loop — escapes ingest_scope entirely.
        calls = {"n": 0}
        real = __import__("booksmart_bench.ingest", fromlist=["_register"])._register

        def fail_on_second_book(*args: object, **kwargs: object) -> object:
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("registration exploded")
            return real(*args, **kwargs)

        monkeypatch.setattr("booksmart_bench.ingest._register", fail_on_second_book)
        with pytest.raises(RuntimeError, match="registration exploded"):
            ingest(root, "an-area")

        with Corpus.open(load_settings()) as corpus:
            recorded = read_provenance(corpus.home)
        assert "book-one" in recorded.books
        assert "book-two" not in recorded.books


class TestScope:
    def test_an_area_ingests_every_book_in_it(
        self, write_truth: Callable[..., Path], place_source: Callable[..., str]
    ) -> None:
        root = write_truth("book-one", book={"title": "One", "area": "an-area"})
        write_truth("book-two", root=root, book={"title": "Two", "area": "an-area"})
        write_truth("book-three", root=root, book={"title": "Three", "area": "other"})
        for slug in ("book-one", "book-two", "book-three"):
            place_source(root, slug)

        outcome = ingest(root, "an-area")

        assert sorted(book.slug for book in outcome.books) == ["book-one", "book-two"]

    def test_an_unknown_scope_names_what_is_available(
        self, assets: Path
    ) -> None:
        with pytest.raises(BenchError, match="an-area"):
            ingest(assets, "no-such-thing")
