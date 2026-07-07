"""knowledge-worker (v1): poll-the-table ingestion worker, no message broker.

Claims the oldest queued ingestion job with SELECT ... FOR UPDATE SKIP LOCKED,
extracts structured Markdown from the stored original through the parser
preference chain (Marker -> PyMuPDF -> OCR), and writes it to
storage/parsed/<book_id>/<job_id>.md with a per-job parse log under
storage/logs/. Run with `python -m app.worker`.
"""

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeVar

from qdrant_client import QdrantClient
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.extraction import (
    EXTRACTION_PROMPT_VERSION,
    EXTRACTION_SYSTEM_PROMPT,
    ExtractionError,
    build_extraction_prompt,
    iter_chapter_bodies,
    parse_extraction_response,
    resolve_source,
)
from app.llm import (
    EmbeddingProvider,
    LLMProvider,
    LLMResponse,
    build_embedding_provider,
    build_llm_provider,
)
from app.models import Book, BookProfile, Chapter, IngestionJob, KnowledgeObject, Section
from app.parsing import EXTRACTION_VERSION, ParserChain, build_default_chain
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

POLL_INTERVAL_SECONDS = 1.0

# Module-level so Marker's model load (when installed) happens once per process.
DEFAULT_CHAIN = build_default_chain()


def build_default_llm() -> LLMProvider:
    """Provider used when the caller injects none. A separate function so tests
    can substitute a stub without touching every process_one_job call site."""
    return build_llm_provider(Settings())


def build_default_embedder() -> EmbeddingProvider:
    """Test seam, like build_default_llm."""
    return build_embedding_provider(Settings())


def build_default_vector_store() -> VectorStore:
    """Test seam, like build_default_llm."""
    return VectorStore(QdrantClient(url=Settings().qdrant_url))


# The tightest provider limit wins: Gemini rejects batches over 100 inputs,
# OpenAI caps at 2048. Staying at 100 also keeps token limits far away.
EMBED_BATCH_SIZE = 100

RecordType = Literal["chapter", "section", "knowledge_object"]

Stage = Literal["parse", "structure", "profile", "extraction", "summaries", "embeddings"]

# Which stages each job scope runs. Incremental scopes reuse upstream
# artifacts; extraction pulls embeddings along because the replaced knowledge
# objects would otherwise be left with stale or missing vectors.
SCOPE_STAGES: dict[str, tuple[Stage, ...]] = {
    "full": ("parse", "structure", "profile", "extraction", "summaries", "embeddings"),
    "profile": ("profile",),
    "extraction": ("extraction", "embeddings"),
    "embeddings": ("embeddings",),
}

LLM_STAGES: frozenset[Stage] = frozenset({"profile", "extraction", "summaries"})
MARKDOWN_STAGES: frozenset[Stage] = frozenset({"structure", "extraction", "summaries"})


def _show_count(count: int | None) -> str:
    return str(count) if count is not None else "?"


@dataclass
class TokenUsage:
    """Provider-reported LLM usage summed across a job's calls. Unreported
    counts (None) log as '?' and add zero, so totals stay honest lower bounds."""

    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, response: LLMResponse) -> str:
        """Accumulate one call's usage; returns the log fragment describing it."""
        self.input_tokens += response.input_tokens or 0
        self.output_tokens += response.output_tokens or 0
        return (
            f"tokens in={_show_count(response.input_tokens)} "
            f"out={_show_count(response.output_tokens)}"
        )


ParsedT = TypeVar("ParsedT")


def _complete_and_parse(
    llm: LLMProvider,
    prompt: str,
    system: str,
    parse: Callable[[str], ParsedT],
    label: str,
    usage: TokenUsage,
    log: Callable[[str], None],
) -> tuple[ParsedT, LLMResponse]:
    """One LLM call plus its parse, retried once on an unparseable response —
    a model's occasional bad sample shouldn't roll back a whole multi-chapter
    run. Both attempts are logged and counted; the second failure propagates."""
    for attempt in (1, 2):
        response = llm.complete(prompt, system=system)
        log(f"{label} {usage.add(response)}")
        try:
            return parse(response.text), response
        except (ExtractionError, SummaryError) as exc:
            if attempt == 2:
                raise
            log(f"{label} response unparseable ({exc}); retrying once")
    raise RuntimeError("unreachable")


def _version_stamps(
    stages: tuple[Stage, ...],
    llm: LLMProvider | None,
    embedder: EmbeddingProvider | None,
) -> tuple[str, str | None, str | None]:
    """(extraction_version, model_version, prompt_version) describing exactly
    what this run used, limited to the stages it actually ran."""
    models = []
    if llm is not None and any(stage in LLM_STAGES for stage in stages):
        models.append(f"llm={llm.model}")
    if embedder is not None and "embeddings" in stages:
        models.append(f"embedding={embedder.model}")
    prompts = []
    if "profile" in stages:
        prompts.append(f"profile={PROFILE_PROMPT_VERSION}")
    if "extraction" in stages:
        prompts.append(f"extraction={EXTRACTION_PROMPT_VERSION}")
    if "summaries" in stages:
        prompts.append(f"summary={SUMMARY_PROMPT_VERSION}")
    return EXTRACTION_VERSION, ",".join(models) or None, ",".join(prompts) or None


def _latest_parsed_artifact(session: Session, book: Book) -> Path:
    """The newest parsed-markdown artifact from the book's successful runs."""
    path = session.scalars(
        select(IngestionJob.output_path)
        .where(
            IngestionJob.book_id == book.id,
            IngestionJob.status == "succeeded",
            IngestionJob.output_path.is_not(None),
        )
        .order_by(IngestionJob.finished_at.desc())
        .limit(1)
    ).first()
    if path is None:
        raise RuntimeError("no parsed markdown artifact to reuse; run a full ingest first")
    artifact = Path(path)
    if not artifact.exists():
        raise RuntimeError(f"parsed markdown artifact missing from storage: {artifact}")
    return artifact


def _book_chapters(session: Session, book: Book) -> list[Chapter]:
    return list(
        session.scalars(
            select(Chapter).where(Chapter.book_id == book.id).order_by(Chapter.position)
        )
    )


def _claim_next_job(session: Session) -> IngestionJob | None:
    return session.scalars(
        select(IngestionJob)
        .where(IngestionJob.status == "queued")
        .order_by(IngestionJob.created_at, IngestionJob.id)
        .limit(1)
        .with_for_update(skip_locked=True)
    ).first()


def _detect_and_replace_structure(
    session: Session, book: Book, markdown: str, log: Callable[[str], None]
) -> None:
    """Structure detection stage: replace the book's chapter/section tree."""
    log("structure: detecting chapters and sections")
    try:
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
        log(f"structure: {len(detected)} chapters, {section_count} sections")
    except Exception as exc:
        log(f"structure: failed — {type(exc).__name__}: {exc}")
        raise RuntimeError(f"structure detection stage failed: {exc}") from exc


def _generate_profile(
    session: Session,
    book: Book,
    llm: LLMProvider,
    usage: TokenUsage,
    log: Callable[[str], None],
) -> None:
    """Profile stage: append a new versioned profile from metadata, hints, and structure."""
    log("profile: generating book profile")
    try:
        prompt = build_profile_prompt(book, _book_chapters(session, book))
        response = llm.complete(prompt, system=PROFILE_SYSTEM_PROMPT)
        log(f"profile: {usage.add(response)}")
        session.add(
            BookProfile(
                book_id=book.id,
                content=response.text,
                model=response.model,
                prompt_version=PROFILE_PROMPT_VERSION,
            )
        )
        log(f"profile: generated by {response.model} (prompt v{PROFILE_PROMPT_VERSION})")
    except Exception as exc:
        log(f"profile: failed — {type(exc).__name__}: {exc}")
        raise RuntimeError(f"profile generation stage failed: {exc}") from exc


def _extract_knowledge(
    session: Session,
    book: Book,
    markdown: str,
    llm: LLMProvider,
    usage: TokenUsage,
    log: Callable[[str], None],
) -> None:
    """Extraction stage: replace the book's knowledge objects, one LLM call per chapter."""
    log("extraction: extracting knowledge objects")
    try:
        chapters = _book_chapters(session, book)
        session.execute(delete(KnowledgeObject).where(KnowledgeObject.book_id == book.id))
        if not chapters:
            log("extraction: no chapters detected; nothing to extract")
            return
        total = 0
        for chapter, body in iter_chapter_bodies(chapters, markdown):
            (extracted_objects, dropped), response = _complete_and_parse(
                llm,
                build_extraction_prompt(book, chapter, body),
                EXTRACTION_SYSTEM_PROMPT,
                parse_extraction_response,
                f"extraction: chapter {chapter.position + 1}",
                usage,
                log,
            )
            for reason in dropped:
                log(f"extraction: chapter {chapter.position + 1} dropped element — {reason}")
            for extracted in extracted_objects:
                section, source_location = resolve_source(chapter, extracted.section_index)
                if extracted.section_index is not None and section is None:
                    log(
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
        log(
            f"extraction: {total} knowledge objects from {len(chapters)} chapters "
            f"(prompt v{EXTRACTION_PROMPT_VERSION})"
        )
    except Exception as exc:
        log(f"extraction: failed — {type(exc).__name__}: {exc}")
        raise RuntimeError(f"knowledge extraction stage failed: {exc}") from exc


def _summarize_structure(
    session: Session,
    book: Book,
    markdown: str,
    llm: LLMProvider,
    usage: TokenUsage,
    log: Callable[[str], None],
) -> None:
    """Summary stage: one LLM call per chapter fills chapter/section summaries."""
    log("summaries: summarizing chapters and sections")
    try:
        chapters = _book_chapters(session, book)
        if not chapters:
            log("summaries: no chapters detected; nothing to summarize")
            return
        for chapter, body in iter_chapter_bodies(chapters, markdown):
            section_count = len(chapter.sections)

            def parse_for_chapter(text: str, count: int = section_count) -> tuple[str, list[str | None]]:
                return parse_summary_response(text, count)

            (chapter_summary, section_summaries), response = _complete_and_parse(
                llm,
                build_summary_prompt(book, chapter, body),
                SUMMARY_SYSTEM_PROMPT,
                parse_for_chapter,
                f"summaries: chapter {chapter.position + 1}",
                usage,
                log,
            )
            chapter.summary = chapter_summary
            chapter.summary_model = response.model
            chapter.summary_prompt_version = SUMMARY_PROMPT_VERSION
            for section, summary in zip(chapter.sections, section_summaries, strict=True):
                section.summary = summary
                if summary is not None:
                    section.summary_model = response.model
                    section.summary_prompt_version = SUMMARY_PROMPT_VERSION
        log(f"summaries: {len(chapters)} chapters summarized (prompt v{SUMMARY_PROMPT_VERSION})")
    except Exception as exc:
        log(f"summaries: failed — {type(exc).__name__}: {exc}")
        raise RuntimeError(f"summary stage failed: {exc}") from exc


def _generate_embeddings(
    session: Session,
    book: Book,
    embedder: EmbeddingProvider,
    vector_store: VectorStore,
    log: Callable[[str], None],
) -> None:
    """Embedding stage: embed chapter/section summaries and knowledge objects,
    store the vectors in Qdrant, and link each row via embedding_id."""
    log("embeddings: generating embeddings")
    try:
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
            vector_store.replace_book_points(str(book.id), [])
            log("embeddings: nothing to embed")
            return

        texts = [text for _, _, text in items]
        embedded_vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            embedded_vectors.extend(embedder.embed(texts[start : start + EMBED_BATCH_SIZE]))
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
        vector_store.replace_book_points(str(book.id), records)
        log(f"embeddings: {len(records)} vectors stored ({embedder.model})")
    except Exception as exc:
        log(f"embeddings: failed — {type(exc).__name__}: {exc}")
        raise RuntimeError(f"embedding stage failed: {exc}") from exc


def process_one_job(
    session_factory: sessionmaker[Session],
    storage_root: Path,
    chain: ParserChain | None = None,
    llm: LLMProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
) -> bool:
    """Claim and run the oldest queued job. Returns whether a job was processed."""
    storage = BookStorage(storage_root)
    chain = chain or DEFAULT_CHAIN
    with session_factory() as session:
        job = _claim_next_job(session)
        if job is None:
            return False
        job.status = "running"
        job.started_at = datetime.now(UTC)
        session.commit()  # releases the claim lock; the job is now visibly running

        log_lines: list[str] = []

        def log(line: str) -> None:
            log_lines.append(f"{datetime.now(UTC).isoformat()} {line}")

        stamps: tuple[str, str | None, str | None] = (EXTRACTION_VERSION, None, None)
        usage: TokenUsage | None = None
        try:
            book = session.get(Book, job.book_id)
            if book is None:
                raise RuntimeError(f"Book {job.book_id} no longer exists")
            stages = SCOPE_STAGES.get(job.scope)
            if stages is None:
                raise RuntimeError(f"Unknown job scope {job.scope!r}")
            log(f"scope: {job.scope} -> stages {', '.join(stages)}")

            if any(stage in LLM_STAGES for stage in stages):
                llm = llm or build_default_llm()
                usage = TokenUsage()
            if "embeddings" in stages:
                embedder = embedder or build_default_embedder()
                vector_store = vector_store or build_default_vector_store()
            stamps = _version_stamps(stages, llm, embedder)

            markdown: str | None = None
            if "parse" in stages:
                result = chain.extract(Path(book.storage_path), book.file_format, log)
                markdown = result.markdown
                job.output_path = str(storage.save_parsed(book.id, job.id, result.markdown))
                job.parser_used = result.parser
            elif any(stage in MARKDOWN_STAGES for stage in stages):
                artifact = _latest_parsed_artifact(session, book)
                markdown = artifact.read_text(encoding="utf-8")
                job.output_path = str(artifact)
                log(f"reuse: parsed markdown from {artifact}")

            if "structure" in stages:
                assert markdown is not None
                _detect_and_replace_structure(session, book, markdown, log)
            if "profile" in stages:
                assert llm is not None and usage is not None
                _generate_profile(session, book, llm, usage, log)
            if "extraction" in stages:
                assert markdown is not None and llm is not None and usage is not None
                _extract_knowledge(session, book, markdown, llm, usage, log)
            if "summaries" in stages:
                assert markdown is not None and llm is not None and usage is not None
                _summarize_structure(session, book, markdown, llm, usage, log)
            if "embeddings" in stages:
                assert embedder is not None and vector_store is not None
                _generate_embeddings(session, book, embedder, vector_store, log)
            job.status = "succeeded"
        except Exception as exc:
            # Discard partial stage writes (e.g. a structure delete whose
            # replacement inserts never landed) before recording the failure.
            session.rollback()
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        # Stamped outside the try so even failed runs record what they used.
        job.extraction_version, job.model_version, job.prompt_version = stamps
        if usage is not None:
            job.input_tokens = usage.input_tokens
            job.output_tokens = usage.output_tokens
        job.finished_at = datetime.now(UTC)
        try:
            storage.save_log(job.id, "".join(f"{line}\n" for line in log_lines))
        except OSError:
            pass  # the parse log is diagnostics; never let it sink the run record
        session.commit()
        return True


def run_forever() -> None:
    settings = Settings()
    engine = create_engine(settings.database_url)
    session_factory = sessionmaker(bind=engine)
    # Fail fast on misconfiguration; build everything once per process.
    llm = build_llm_provider(settings)
    embedder = build_embedding_provider(settings)
    vector_store = VectorStore(QdrantClient(url=settings.qdrant_url))
    while True:
        if not process_one_job(
            session_factory,
            settings.storage_root,
            llm=llm,
            embedder=embedder,
            vector_store=vector_store,
        ):
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_forever()
