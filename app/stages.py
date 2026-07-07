"""The ingestion pipeline as public, synchronous Stage functions (ADR 0002).

Each ``run_<stage>(session, book_id, <providers>) -> StageReport`` is a unit of
durability: it takes serializable inputs (ids, never ORM objects or document
text), resolves its own data from the database and the book's parsed artifact,
replaces its output wholesale, and commits its own transaction. Stages are
Run-blind — they never read or write a Run row; a Runner owns that (app.runner)
and every consumer brings its own.

Stages do not wrap their failures. A missing book or parsed artifact raises
StagePreconditionError (non-retriable); a bad model response surfaces as a
ProviderResponseError (retriable, via LLMError / ExtractionError /
SummaryError); vendor SDK exceptions pass straight through. The Runner maps
those to its own retry semantics.
"""

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeVar

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.errors import StagePreconditionError
from app.extraction import (
    EXTRACTION_PROMPT_VERSION,
    EXTRACTION_SYSTEM_PROMPT,
    ExtractionError,
    build_extraction_prompt,
    iter_chapter_bodies,
    parse_extraction_response,
    resolve_source,
)
from app.llm import EmbeddingProvider, LLMProvider, LLMResponse
from app.models import Book, BookProfile, Chapter, KnowledgeObject, Section
from app.parsing import ParserChain
from app.profile import PROFILE_PROMPT_VERSION, PROFILE_SYSTEM_PROMPT, build_profile_prompt
from app.storage import BookStorage
from app.structure import detect_structure
from app.summaries import (
    SUMMARY_PROMPT_VERSION,
    SUMMARY_SYSTEM_PROMPT,
    SummaryError,
    build_summary_prompt,
    parse_summary_response,
)
from app.vectors import VectorRecord, VectorStore

Stage = Literal["parse", "structure", "profile", "extraction", "summaries", "embeddings"]

# Stages that call an LLM provider (used by the Runner to decide what to build
# and which version stamps to record).
LLM_STAGES: frozenset[Stage] = frozenset({"profile", "extraction", "summaries"})

RecordType = Literal["chapter", "section", "knowledge_object"]


@dataclass(frozen=True)
class StageReport:
    """What one Stage did, for the Runner to aggregate into a Run.

    Frozen and self-contained: it replaces the mutable ``log`` callback and
    ``TokenUsage`` the poll-worker threaded through every stage. Token counts
    are honest lower bounds — a provider that reports no usage contributes
    zero and logs '?'."""

    stage: Stage
    log_lines: tuple[str, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    counts: Mapping[str, int] = field(default_factory=dict)


def _show_count(count: int | None) -> str:
    return str(count) if count is not None else "?"


class _ReportBuilder:
    """Mutable accumulator a stage fills, then freezes into a StageReport."""

    def __init__(self, stage: Stage) -> None:
        self.stage = stage
        self._log_lines: list[str] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.counts: dict[str, int] = {}

    def log(self, line: str) -> None:
        self._log_lines.append(f"{datetime.now(UTC).isoformat()} {line}")

    def add_usage(self, response: LLMResponse) -> str:
        """Accumulate one call's usage; returns the log fragment describing it."""
        self.input_tokens += response.input_tokens or 0
        self.output_tokens += response.output_tokens or 0
        return (
            f"tokens in={_show_count(response.input_tokens)} "
            f"out={_show_count(response.output_tokens)}"
        )

    def finish(self) -> StageReport:
        return StageReport(
            stage=self.stage,
            log_lines=tuple(self._log_lines),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            counts=dict(self.counts),
        )


def _load_book(session: Session, book_id: uuid.UUID) -> Book:
    book = session.get(Book, book_id)
    if book is None:
        raise StagePreconditionError(f"book {book_id} does not exist")
    return book


def _read_parsed_markdown(book: Book) -> str:
    """The book's current parsed markdown, resolved from Book.parsed_path.

    A stage that consumes markdown but was run before the book was ever parsed
    (or whose artifact has vanished from storage) cannot proceed — that is a
    precondition failure, not a transient one."""
    if book.parsed_path is None:
        raise StagePreconditionError(
            "no parsed markdown for this book; run a full ingest first"
        )
    artifact = Path(book.parsed_path)
    if not artifact.exists():
        raise StagePreconditionError(
            f"parsed markdown artifact missing from storage: {artifact}"
        )
    return artifact.read_text(encoding="utf-8")


def _book_chapters(session: Session, book: Book) -> list[Chapter]:
    return list(
        session.scalars(
            select(Chapter).where(Chapter.book_id == book.id).order_by(Chapter.position)
        )
    )


ParsedT = TypeVar("ParsedT")


def _complete_and_parse(
    llm: LLMProvider,
    prompt: str,
    system: str,
    parse: Callable[[str], ParsedT],
    label: str,
    builder: _ReportBuilder,
) -> tuple[ParsedT, LLMResponse]:
    """One LLM call plus its parse, retried once on an unparseable response — a
    model's occasional bad sample shouldn't roll back a whole multi-chapter
    run. Both attempts are logged and counted; the second failure propagates as
    a ProviderResponseError."""
    for attempt in (1, 2):
        response = llm.complete(prompt, system=system)
        builder.log(f"{label} {builder.add_usage(response)}")
        try:
            return parse(response.text), response
        except (ExtractionError, SummaryError) as exc:
            if attempt == 2:
                raise
            builder.log(f"{label} response unparseable ({exc}); retrying once")
    raise RuntimeError("unreachable")


def run_parse(
    session: Session, book_id: uuid.UUID, *, chain: ParserChain, storage: BookStorage
) -> StageReport:
    """Parse the stored original into markdown through the parser chain and
    point Book.parsed_path / parser_used at the result."""
    builder = _ReportBuilder("parse")
    book = _load_book(session, book_id)
    result = chain.extract(Path(book.storage_path), book.file_format, builder.log)
    parsed_path = storage.save_parsed(book.id, result.markdown)
    book.parsed_path = str(parsed_path)
    book.parser_used = result.parser
    builder.counts["characters"] = len(result.markdown)
    session.commit()
    return builder.finish()


def run_structure(session: Session, book_id: uuid.UUID) -> StageReport:
    """Replace the book's chapter/section tree from its parsed markdown."""
    builder = _ReportBuilder("structure")
    book = _load_book(session, book_id)
    markdown = _read_parsed_markdown(book)
    builder.log("structure: detecting chapters and sections")
    detected = detect_structure(markdown)
    session.execute(delete(Chapter).where(Chapter.book_id == book.id))
    for chapter_position, chapter in enumerate(detected):
        session.add(
            Chapter(
                book_id=book.id,
                position=chapter_position,
                title=chapter.title,
                kind=chapter.kind,
                source_line=chapter.line,
                sections=[
                    Section(
                        position=section_position,
                        title=section.title,
                        source_line=section.line,
                    )
                    for section_position, section in enumerate(chapter.sections)
                ],
            )
        )
    section_count = sum(len(chapter.sections) for chapter in detected)
    builder.counts["chapters"] = len(detected)
    builder.counts["sections"] = section_count
    builder.log(f"structure: {len(detected)} chapters, {section_count} sections")
    session.commit()
    return builder.finish()


def run_profile(session: Session, book_id: uuid.UUID, *, llm: LLMProvider) -> StageReport:
    """Append a new versioned profile from metadata, hints, and structure."""
    builder = _ReportBuilder("profile")
    book = _load_book(session, book_id)
    builder.log("profile: generating book profile")
    prompt = build_profile_prompt(book, _book_chapters(session, book))
    response = llm.complete(prompt, system=PROFILE_SYSTEM_PROMPT)
    builder.log(f"profile: {builder.add_usage(response)}")
    session.add(
        BookProfile(
            book_id=book.id,
            content=response.text,
            model=response.model,
            prompt_version=PROFILE_PROMPT_VERSION,
        )
    )
    builder.counts["profiles"] = 1
    builder.log(f"profile: generated by {response.model} (prompt v{PROFILE_PROMPT_VERSION})")
    session.commit()
    return builder.finish()


def run_extraction(session: Session, book_id: uuid.UUID, *, llm: LLMProvider) -> StageReport:
    """Replace the book's knowledge objects, one LLM call per chapter."""
    builder = _ReportBuilder("extraction")
    book = _load_book(session, book_id)
    markdown = _read_parsed_markdown(book)
    builder.log("extraction: extracting knowledge objects")
    chapters = _book_chapters(session, book)
    session.execute(delete(KnowledgeObject).where(KnowledgeObject.book_id == book.id))
    total = 0
    if not chapters:
        builder.log("extraction: no chapters detected; nothing to extract")
    for chapter, body in iter_chapter_bodies(chapters, markdown):
        (extracted_objects, dropped), response = _complete_and_parse(
            llm,
            build_extraction_prompt(book, chapter, body),
            EXTRACTION_SYSTEM_PROMPT,
            parse_extraction_response,
            f"extraction: chapter {chapter.position + 1}",
            builder,
        )
        for reason in dropped:
            builder.log(f"extraction: chapter {chapter.position + 1} dropped element — {reason}")
        for extracted in extracted_objects:
            section, source_location = resolve_source(chapter, extracted.section_index)
            if extracted.section_index is not None and section is None:
                builder.log(
                    f"extraction: section_index {extracted.section_index} out of range "
                    f"for chapter {chapter.position + 1}; keeping chapter link only"
                )
            session.add(
                KnowledgeObject(
                    book_id=book.id,
                    chapter_id=chapter.id,
                    section_id=section.id if section is not None else None,
                    type=extracted.type,
                    title=extracted.title,
                    content=extracted.content,
                    summary=extracted.summary,
                    source_location=source_location,
                    confidence=extracted.confidence,
                    edition=book.edition,
                    page=extracted.page,
                    paragraph=extracted.paragraph,
                    extraction_model=response.model,
                    extraction_prompt_version=EXTRACTION_PROMPT_VERSION,
                )
            )
            total += 1
    builder.counts["knowledge_objects"] = total
    builder.log(
        f"extraction: {total} knowledge objects from {len(chapters)} chapters "
        f"(prompt v{EXTRACTION_PROMPT_VERSION})"
    )
    session.commit()
    return builder.finish()


def run_summaries(session: Session, book_id: uuid.UUID, *, llm: LLMProvider) -> StageReport:
    """One LLM call per chapter fills chapter/section summaries."""
    builder = _ReportBuilder("summaries")
    book = _load_book(session, book_id)
    markdown = _read_parsed_markdown(book)
    builder.log("summaries: summarizing chapters and sections")
    chapters = _book_chapters(session, book)
    if not chapters:
        builder.log("summaries: no chapters detected; nothing to summarize")
    for chapter, body in iter_chapter_bodies(chapters, markdown):
        section_count = len(chapter.sections)

        def parse_for_chapter(
            text: str, count: int = section_count
        ) -> tuple[str, list[str | None]]:
            return parse_summary_response(text, count)

        (chapter_summary, section_summaries), response = _complete_and_parse(
            llm,
            build_summary_prompt(book, chapter, body),
            SUMMARY_SYSTEM_PROMPT,
            parse_for_chapter,
            f"summaries: chapter {chapter.position + 1}",
            builder,
        )
        chapter.summary = chapter_summary
        chapter.summary_model = response.model
        chapter.summary_prompt_version = SUMMARY_PROMPT_VERSION
        for section, summary in zip(chapter.sections, section_summaries, strict=True):
            section.summary = summary
            if summary is not None:
                section.summary_model = response.model
                section.summary_prompt_version = SUMMARY_PROMPT_VERSION
    builder.counts["chapters"] = len(chapters)
    builder.log(f"summaries: {len(chapters)} chapters summarized (prompt v{SUMMARY_PROMPT_VERSION})")
    session.commit()
    return builder.finish()


def run_embeddings(
    session: Session,
    book_id: uuid.UUID,
    *,
    embedder: EmbeddingProvider,
    vector_store: VectorStore,
) -> StageReport:
    """Embed chapter/section summaries and knowledge objects, store the vectors
    in Qdrant, and link each row via embedding_id."""
    builder = _ReportBuilder("embeddings")
    book = _load_book(session, book_id)
    builder.log("embeddings: generating embeddings")
    chapters = _book_chapters(session, book)
    objects = list(
        session.scalars(select(KnowledgeObject).where(KnowledgeObject.book_id == book.id))
    )

    items: list[tuple[Chapter | Section | KnowledgeObject, RecordType, str]] = []
    for chapter in chapters:
        if chapter.summary:
            items.append((chapter, "chapter", chapter.summary))
        for section in chapter.sections:
            if section.summary:
                items.append((section, "section", section.summary))
    for knowledge_object in objects:
        items.append(
            (
                knowledge_object,
                "knowledge_object",
                f"{knowledge_object.type}: {knowledge_object.title}\n"
                f"{knowledge_object.summary}\n{knowledge_object.content}",
            )
        )

    if not items:
        vector_store.replace_book_points(str(book.id), [], embedder.model)
        builder.counts["vectors"] = 0
        builder.log("embeddings: nothing to embed")
        session.commit()
        return builder.finish()

    texts = [text for _, _, text in items]
    embedded_vectors: list[list[float]] = []
    # Batch size is the provider's Limit for its model, never a caller guess.
    for start in range(0, len(texts), embedder.max_batch):
        embedded_vectors.extend(embedder.embed(texts[start : start + embedder.max_batch]))
    embedded_at = datetime.now(UTC)
    records: list[VectorRecord] = []
    for (row, record_type, text), vector in zip(items, embedded_vectors, strict=True):
        point_id = uuid.uuid4()
        row.embedding_id = point_id
        row.embedding_model = embedder.model
        row.embedded_at = embedded_at
        records.append(
            VectorRecord(
                id=str(point_id),
                vector=vector,
                payload={
                    "record_type": record_type,
                    "record_id": str(row.id),
                    "book_id": str(book.id),
                    "text": text,
                },
            )
        )
    vector_store.replace_book_points(str(book.id), records, embedder.model)
    builder.counts["vectors"] = len(records)
    builder.log(f"embeddings: {len(records)} vectors stored ({embedder.model})")
    session.commit()
    return builder.finish()
