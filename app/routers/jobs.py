import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Book, IngestionJob
from app.schemas import IngestionJobOut

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


@router.get("/jobs/{job_id}", response_model=IngestionJobOut)
def get_job(job_id: uuid.UUID, db: Session = Depends(get_db)) -> IngestionJob:
    job = db.get(IngestionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return job
