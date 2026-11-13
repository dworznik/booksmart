"""Integration tests: with the passage Preference on, the embeddings Stage cuts
the book's own text into passages and makes them searchable.

Driven through the Runner and read back through `search`, the way a consumer
would — the cutting itself is specified in `test_passages.py`.
"""

import uuid

import pymupdf
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from booksmart_core.config import Settings
from booksmart_core.models import Book, Chapter, Passage, Run, Section
from booksmart_core.runner import execute_run
from booksmart_core.search import SearchHit, SearchMode, search
from booksmart_core.stages import run_embeddings, run_structure
from booksmart_core.storage import BookStorage
from booksmart_core.vectors import VectorStore

from .conftest import StubEmbeddingProvider, StubLLMProvider, StubSparseProvider
from .test_embeddings_api import count_book_points, prime_summaries
from .test_ingestion_api import BODY_POINTS, PAGE_FILLER, register_book
from .test_knowledge_api import prime_extraction
from .test_profile_api import register_book_with_hints

BODY = "Body text explaining the idea in plain words."


def make_pdf_with_chapter_opening() -> bytes:
    """A book whose first chapter's text starts before its first section, so it
    has a passage that belongs to a chapter and to no section. Sized like
    `make_structured_pdf_bytes`: 24pt chapter, 16pt section, body at the body
    size so nothing else reads as a heading. Two chapters, because a lone
    top-level heading is read as the book's title rather than as a chapter."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "Chapter One: Tides", fontsize=24)
    page.insert_text(
        (72, 130), "The tide opens the chapter before any section.", fontsize=BODY_POINTS
    )
    page.insert_text((72, 170), "Spring Tides", fontsize=16)
    page.insert_text((72, 200), "The tide is highest at the full moon.", fontsize=BODY_POINTS)
    page.insert_textbox(pymupdf.Rect(72, 560, 520, 760), PAGE_FILLER, fontsize=BODY_POINTS)
    page = doc.new_page()
    page.insert_text((72, 80), "Chapter Two: Currents", fontsize=24)
    page.insert_textbox(pymupdf.Rect(72, 560, 520, 760), PAGE_FILLER, fontsize=BODY_POINTS)
    data: bytes = doc.tobytes()
    doc.close()
    return data


def ingest(
    session_factory: sessionmaker[Session],
    settings: Settings,
    book_id: uuid.UUID,
    stub_llm: StubLLMProvider,
    *,
    scope: str = "full",
    embed_passages: bool,
) -> Run:
    """Run a Scope with the passage Preference set, and return its Run."""
    if scope == "full":
        prime_extraction(stub_llm)
        prime_summaries(stub_llm)
    run_id = execute_run(
        session_factory, settings.storage_root, book_id, scope, embed_passages=embed_passages
    )
    with session_factory() as session:
        run = session.get(Run, run_id)
        assert run is not None
        assert run.status == "succeeded", run.error
        session.expunge(run)
        return run


def passage_hits(
    session_factory: sessionmaker[Session],
    vector_store: VectorStore,
    stub_embedder: StubEmbeddingProvider,
    stub_sparse: StubSparseProvider,
    query: str,
) -> list[SearchHit]:
    with session_factory() as session:
        return search(
            session,
            vector_store,
            stub_embedder,
            query,
            sparse_embedder=stub_sparse,
            record_types=["passage"],
            limit=50,
        ).hits


class TestPassageEmbedding:
    def test_the_books_own_words_become_searchable_under_their_sections(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
        stub_embedder: StubEmbeddingProvider,
        stub_sparse: StubSparseProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id = uuid.UUID(register_book_with_hints(session_factory, storage))

        ingest(session_factory, settings, book_id, stub_llm, embed_passages=True)

        hits = passage_hits(session_factory, vector_store, stub_embedder, stub_sparse, "plain words")
        # One passage per section of the fixture: each section's heading and the
        # prose under it. The two chapter headings sit directly above a section
        # heading, and a heading alone is not a passage.
        assert sorted(hit.text.splitlines()[0] for hit in hits) == [
            "## Deep Modules",
            "## Shallow Modules",
            "## Symptoms",
        ]
        assert all(hit.record_type == "passage" and BODY in hit.text for hit in hits)
        assert all(hit.book_id == book_id for hit in hits)
        with session_factory() as session:
            sections = {
                section.id: section.title
                for section in session.scalars(
                    select(Section).join(Chapter).where(Chapter.book_id == book_id)
                )
            }
        by_heading = {hit.text.splitlines()[0]: hit for hit in hits}
        deep = by_heading["## Deep Modules"]
        assert isinstance(deep.row, Passage)
        assert deep.row.section_id is not None
        assert sections[deep.row.section_id] == "Deep Modules"
        assert deep.title == "chapter 1: Chapter One: Modules > Deep Modules"

    def test_a_passage_row_holds_a_span_of_the_parsed_markdown_not_the_text(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
        stub_embedder: StubEmbeddingProvider,
        stub_sparse: StubSparseProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id = uuid.UUID(register_book_with_hints(session_factory, storage))

        ingest(session_factory, settings, book_id, stub_llm, embed_passages=True)

        with session_factory() as session:
            book = session.get(Book, book_id)
            assert book is not None and book.parsed_path is not None
            markdown = storage.resolve(book.parsed_path).read_text(encoding="utf-8")
        hits = passage_hits(session_factory, vector_store, stub_embedder, stub_sparse, "plain words")
        assert len(hits) == 3
        for hit in hits:
            assert isinstance(hit.row, Passage)
            # The point's payload is a copy; the file is where the words live.
            assert markdown[hit.row.start_char : hit.row.end_char] == hit.text
            assert hit.row.embedding_model == "stub-embed-1"

    def test_passages_are_off_unless_the_preference_turns_them_on(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
        stub_embedder: StubEmbeddingProvider,
        stub_sparse: StubSparseProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id = uuid.UUID(register_book_with_hints(session_factory, storage))
        prime_extraction(stub_llm)
        prime_summaries(stub_llm)

        # No embed_passages argument at all: the caller who never heard of it.
        execute_run(session_factory, settings.storage_root, book_id)

        assert passage_hits(session_factory, vector_store, stub_embedder, stub_sparse, "plain") == []
        # 2 chapters + 3 sections + 3 knowledge objects, as before passages existed.
        assert count_book_points(vector_store, str(book_id)) == 8
        embedded = [text for batch in stub_embedder.batches for text in batch]
        assert not any(BODY in text for text in embedded)

    def test_the_run_reports_how_many_passages_it_cut(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
    ) -> None:
        book_id = uuid.UUID(register_book_with_hints(session_factory, storage))

        run = ingest(session_factory, settings, book_id, stub_llm, embed_passages=True)

        [embeddings] = [stage for stage in run.stages if stage.stage == "embeddings"]
        assert embeddings.counts == {"passages": 3, "vectors": 11}
        # 25 tokens per embedded text from the stub: passages are paid for like
        # every other record, and the Run says so.
        assert embeddings.embedding_tokens == 11 * StubEmbeddingProvider.INPUT_TOKENS_PER_TEXT

    def test_no_request_exceeds_the_providers_per_request_token_limit(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
        stub_embedder: StubEmbeddingProvider,
    ) -> None:
        """Passages are what makes this Limit reachable: a book's summaries
        never add up to a vendor's per-request token cap, and its text does.
        A token is at least a byte, so a batch under the cap in bytes is under
        it in tokens whatever the vendor's tokenizer."""
        book_id = uuid.UUID(register_book_with_hints(session_factory, storage))
        stub_embedder.max_batch_tokens = 600

        ingest(session_factory, settings, book_id, stub_llm, embed_passages=True)

        sizes = [sum(len(text.encode()) for text in batch) for batch in stub_embedder.batches]
        assert len(sizes) > 1
        assert max(sizes) <= 600
        # Split, not dropped: every record still reached the provider.
        assert sum(len(batch) for batch in stub_embedder.batches) == 11

    def test_reembedding_replaces_a_books_passages_rather_than_adding_to_them(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
        stub_embedder: StubEmbeddingProvider,
        stub_sparse: StubSparseProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id = uuid.UUID(register_book_with_hints(session_factory, storage))
        ingest(session_factory, settings, book_id, stub_llm, embed_passages=True)
        first = passage_hits(session_factory, vector_store, stub_embedder, stub_sparse, "plain words")

        ingest(
            session_factory, settings, book_id, stub_llm, scope="embeddings", embed_passages=True
        )

        second = passage_hits(session_factory, vector_store, stub_embedder, stub_sparse, "plain words")
        assert sorted(hit.text for hit in second) == sorted(hit.text for hit in first)
        assert {hit.record_id for hit in second}.isdisjoint(hit.record_id for hit in first)
        assert count_book_points(vector_store, str(book_id)) == 11
        with session_factory() as session:
            rows = list(session.scalars(select(Passage).where(Passage.book_id == book_id)))
        assert len(rows) == 3

    def test_turning_the_preference_off_and_reembedding_removes_the_passages(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
        stub_embedder: StubEmbeddingProvider,
        stub_sparse: StubSparseProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id = uuid.UUID(register_book_with_hints(session_factory, storage))
        ingest(session_factory, settings, book_id, stub_llm, embed_passages=True)

        ingest(
            session_factory, settings, book_id, stub_llm, scope="embeddings", embed_passages=False
        )

        assert passage_hits(session_factory, vector_store, stub_embedder, stub_sparse, "plain") == []
        assert count_book_points(vector_store, str(book_id)) == 8
        with session_factory() as session:
            assert list(session.scalars(select(Passage).where(Passage.book_id == book_id))) == []

    def test_a_passage_does_not_outlive_the_structure_it_was_cut_from(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
        stub_embedder: StubEmbeddingProvider,
        stub_sparse: StubSparseProvider,
        vector_store: VectorStore,
    ) -> None:
        """A span is an offset into one particular parse. Once the structure
        Stage has replaced the book's chapters — which it does after every
        re-parse — the old spans describe a file that may no longer exist, so
        the rows go with the chapters rather than waiting for the embeddings
        Stage to catch up; their points, rowless, stop being hits."""
        book_id = uuid.UUID(
            register_book(session_factory, storage, content=make_pdf_with_chapter_opening())
        )
        ingest(session_factory, settings, book_id, stub_llm, embed_passages=True)
        before = passage_hits(session_factory, vector_store, stub_embedder, stub_sparse, "tide")
        # One passage under the section, and two that belong to a chapter alone
        # — so both foreign keys have a row to carry away.
        assert sorted(
            (hit.row.section_id is None, hit.text.splitlines()[0])
            for hit in before
            if isinstance(hit.row, Passage)
        ) == [
            (False, "## Spring Tides"),
            (True, "# Chapter One: Tides"),
            (True, "# Chapter Two: Currents"),
        ]

        with session_factory() as session:
            run_structure(session, book_id, storage=storage)

        with session_factory() as session:
            assert list(session.scalars(select(Passage).where(Passage.book_id == book_id))) == []
        assert passage_hits(session_factory, vector_store, stub_embedder, stub_sparse, "tide") == []


class TestSearchingPassages:
    @pytest.mark.parametrize("mode", ["hybrid", "dense"])
    def test_a_search_that_names_no_record_types_never_returns_a_passage(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
        stub_embedder: StubEmbeddingProvider,
        stub_sparse: StubSparseProvider,
        vector_store: VectorStore,
        mode: SearchMode,
    ) -> None:
        book_id = uuid.UUID(register_book_with_hints(session_factory, storage))
        ingest(session_factory, settings, book_id, stub_llm, embed_passages=True)

        # "plain words" is in every passage and in no summary, so the passages
        # are the best lexical matches in the collection — and still not hits.
        with session_factory() as session:
            hits = search(
                session,
                vector_store,
                stub_embedder,
                "plain words",
                sparse_embedder=stub_sparse,
                mode=mode,
                limit=50,
            ).hits

        assert sorted({hit.record_type for hit in hits}) == [
            "chapter",
            "knowledge_object",
            "section",
        ]
        assert len(hits) == 8

    def test_passages_are_searched_beside_other_records_when_both_are_named(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        storage: BookStorage,
        stub_llm: StubLLMProvider,
        stub_embedder: StubEmbeddingProvider,
        stub_sparse: StubSparseProvider,
        vector_store: VectorStore,
    ) -> None:
        book_id = uuid.UUID(register_book_with_hints(session_factory, storage))
        ingest(session_factory, settings, book_id, stub_llm, embed_passages=True)

        with session_factory() as session:
            hits = search(
                session,
                vector_store,
                stub_embedder,
                "plain words",
                sparse_embedder=stub_sparse,
                record_types=["passage", "chapter"],
                limit=50,
            ).hits

        assert sorted(hit.record_type for hit in hits) == ["chapter"] * 2 + ["passage"] * 3


class TestStageContract:
    def test_embedding_passages_without_storage_is_a_caller_bug(
        self,
        session_factory: sessionmaker[Session],
        stub_embedder: StubEmbeddingProvider,
        stub_sparse: StubSparseProvider,
        vector_store: VectorStore,
    ) -> None:
        with session_factory() as session, pytest.raises(ValueError, match="storage"):
            run_embeddings(
                session,
                uuid.uuid4(),
                embedder=stub_embedder,
                sparse_embedder=stub_sparse,
                vector_store=vector_store,
                embed_passages=True,
            )
