import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Book, IngestionJob
from app.schemas import IngestionJobOut, ReprocessRequest

router = APIRouter(tags=["jobs"])


@router.post("/books/{book_id}/ingest", status_code=202, response_model=IngestionJobOut)
def trigger_ingestion(book_id: uuid.UUID, db: Session = Depends(get_db)) -> IngestionJob:
    if db.get(Book, book_id) is None:
        raise HTTPException(status_code=404, detail="Book not found")
    job = IngestionJob(book_id=book_id, status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@router.post("/books/{book_id}/reprocess", status_code=202, response_model=IngestionJobOut)
def reprocess_book(
    book_id: uuid.UUID, payload: ReprocessRequest, db: Session = Depends(get_db)
) -> IngestionJob:
    """Queue a scoped re-run. Incremental scopes reuse earlier artifacts, so
    they need at least one successful run to build on; full rebuilds always
    start from the preserved original."""
    if db.get(Book, book_id) is None:
        raise HTTPException(status_code=404, detail="Book not found")
    if payload.scope != "full":
        prior_success = db.scalars(
            select(IngestionJob.id)
            .where(IngestionJob.book_id == book_id, IngestionJob.status == "succeeded")
            .limit(1)
        ).first()
        if prior_success is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Cannot reprocess scope {payload.scope!r}: the book has no "
                    "successful ingestion to build on; run a full ingest first"
                ),
            )
    job = IngestionJob(book_id=book_id, status="queued", scope=payload.scope)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@router.get("/books/{book_id}/jobs", response_model=list[IngestionJobOut])
def list_book_jobs(
    book_id: uuid.UUID, db: Session = Depends(get_db)
) -> list[IngestionJob]:
    """The book's complete ingestion history, oldest first."""
    if db.get(Book, book_id) is None:
        raise HTTPException(status_code=404, detail="Book not found")
    return list(
        db.scalars(
            select(IngestionJob)
            .where(IngestionJob.book_id == book_id)
            .order_by(IngestionJob.created_at, IngestionJob.id)
        )
    )


@router.get("/jobs/{job_id}", response_model=IngestionJobOut)
def get_job(job_id: uuid.UUID, db: Session = Depends(get_db)) -> IngestionJob:
    job = db.get(IngestionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return job
