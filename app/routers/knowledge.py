import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.extraction import KnowledgeType
from app.models import Book, KnowledgeObject
from app.schemas import KnowledgeObjectOut

router = APIRouter(tags=["knowledge"])


@router.get("/books/{book_id}/knowledge-objects", response_model=list[KnowledgeObjectOut])
def list_knowledge_objects(
    book_id: uuid.UUID,
    type_filter: Annotated[KnowledgeType | None, Query(alias="type")] = None,
    db: Session = Depends(get_db),
) -> list[KnowledgeObject]:
    if db.get(Book, book_id) is None:
        raise HTTPException(status_code=404, detail="Book not found")
    query = (
        select(KnowledgeObject)
        .where(KnowledgeObject.book_id == book_id)
        .order_by(KnowledgeObject.created_at, KnowledgeObject.id)
    )
    if type_filter is not None:
        query = query.where(KnowledgeObject.type == type_filter)
    return list(db.scalars(query))


@router.get("/knowledge-objects/{object_id}", response_model=KnowledgeObjectOut)
def get_knowledge_object(object_id: uuid.UUID, db: Session = Depends(get_db)) -> KnowledgeObject:
    knowledge_object = db.get(KnowledgeObject, object_id)
    if knowledge_object is None:
        raise HTTPException(status_code=404, detail="Knowledge object not found")
    return knowledge_object
