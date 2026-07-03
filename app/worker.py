"""knowledge-worker (v1): poll-the-table ingestion worker, no message broker.

Claims the oldest queued ingestion job with SELECT ... FOR UPDATE SKIP LOCKED,
extracts structured Markdown from the stored original via PyMuPDF, and writes
it to storage/parsed/<book_id>/<job_id>.md. Run with `python -m app.worker`.
"""

import time
from datetime import UTC, datetime
from pathlib import Path

import pymupdf4llm
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models import Book, IngestionJob
from app.storage import BookStorage

POLL_INTERVAL_SECONDS = 1.0


def _claim_next_job(session: Session) -> IngestionJob | None:
    return session.scalars(
        select(IngestionJob)
        .where(IngestionJob.status == "queued")
        .order_by(IngestionJob.created_at, IngestionJob.id)
        .limit(1)
        .with_for_update(skip_locked=True)
    ).first()


def process_one_job(session_factory: sessionmaker[Session], storage_root: Path) -> bool:
    """Claim and run the oldest queued job. Returns whether a job was processed."""
    storage = BookStorage(storage_root)
    with session_factory() as session:
        job = _claim_next_job(session)
        if job is None:
            return False
        job.status = "running"
        job.started_at = datetime.now(UTC)
        session.commit()  # releases the claim lock; the job is now visibly running

        try:
            book = session.get(Book, job.book_id)
            if book is None:
                raise RuntimeError(f"Book {job.book_id} no longer exists")
            markdown = pymupdf4llm.to_markdown(book.storage_path)
            job.output_path = str(storage.save_parsed(book.id, job.id, markdown))
            job.status = "succeeded"
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        job.finished_at = datetime.now(UTC)
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
