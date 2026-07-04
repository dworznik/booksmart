"""knowledge-worker (v1): poll-the-table ingestion worker, no message broker.

Claims the oldest queued ingestion job with SELECT ... FOR UPDATE SKIP LOCKED,
extracts structured Markdown from the stored original through the parser
preference chain (Marker -> PyMuPDF -> OCR), and writes it to
storage/parsed/<book_id>/<job_id>.md with a per-job parse log under
storage/logs/. Run with `python -m app.worker`.
"""

import time
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models import Book, IngestionJob
from app.parsing import ParserChain, build_default_chain
from app.storage import BookStorage

POLL_INTERVAL_SECONDS = 1.0

# Module-level so Marker's model load (when installed) happens once per process.
DEFAULT_CHAIN = build_default_chain()


def _claim_next_job(session: Session) -> IngestionJob | None:
    return session.scalars(
        select(IngestionJob)
        .where(IngestionJob.status == "queued")
        .order_by(IngestionJob.created_at, IngestionJob.id)
        .limit(1)
        .with_for_update(skip_locked=True)
    ).first()


def process_one_job(
    session_factory: sessionmaker[Session],
    storage_root: Path,
    chain: ParserChain | None = None,
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

        try:
            book = session.get(Book, job.book_id)
            if book is None:
                raise RuntimeError(f"Book {job.book_id} no longer exists")
            result = chain.extract(Path(book.storage_path), book.file_format, log)
            job.output_path = str(storage.save_parsed(book.id, job.id, result.markdown))
            job.parser_used = result.parser
            job.status = "succeeded"
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
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
    while True:
        if not process_one_job(session_factory, settings.storage_root):
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_forever()
