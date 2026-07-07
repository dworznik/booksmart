import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Book, Run
from app.runner import execute_run, has_successful_run
from app.schemas import ReprocessRequest, RunOut

router = APIRouter(tags=["runs"])


def _execute(request: Request, book_id: uuid.UUID, scope: str) -> uuid.UUID:
    """Run a scope to completion on the app's own session factory. With the
    polling worker gone (ADR 0002), triggering *is* execution — the request
    blocks until the Run finishes and returns its final record."""
    return execute_run(
        request.app.state.session_factory,
        request.app.state.settings.storage_root,
        book_id,
        scope,
    )


@router.post("/books/{book_id}/ingest", response_model=RunOut)
def trigger_ingestion(
    book_id: uuid.UUID, request: Request, db: Session = Depends(get_db)
) -> Run:
    if db.get(Book, book_id) is None:
        raise HTTPException(status_code=404, detail="Book not found")
    run_id = _execute(request, book_id, "full")
    return db.get(Run, run_id)  # type: ignore[return-value]


@router.post("/books/{book_id}/reprocess", response_model=RunOut)
def reprocess_book(
    book_id: uuid.UUID,
    payload: ReprocessRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> Run:
    """Run a scoped re-run to completion. Incremental scopes reuse earlier
    artifacts, so they need at least one successful run to build on; full
    rebuilds always start from the preserved original."""
    if db.get(Book, book_id) is None:
        raise HTTPException(status_code=404, detail="Book not found")
    if payload.scope != "full" and not has_successful_run(db, book_id):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot reprocess scope {payload.scope!r}: the book has no "
                "successful ingestion to build on; run a full ingest first"
            ),
        )
    run_id = _execute(request, book_id, payload.scope)
    return db.get(Run, run_id)  # type: ignore[return-value]


@router.get("/books/{book_id}/jobs", response_model=list[RunOut])
def list_book_runs(book_id: uuid.UUID, db: Session = Depends(get_db)) -> list[Run]:
    """The book's complete run history, oldest first."""
    if db.get(Book, book_id) is None:
        raise HTTPException(status_code=404, detail="Book not found")
    return list(
        db.scalars(
            select(Run).where(Run.book_id == book_id).order_by(Run.created_at, Run.id)
        )
    )


@router.get("/jobs/{run_id}", response_model=RunOut)
def get_run(run_id: uuid.UUID, db: Session = Depends(get_db)) -> Run:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run
