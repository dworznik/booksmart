from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.routers import books, jobs, knowledge
from app.storage import BookStorage


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        app.state.engine.dispose()

    app = FastAPI(title="booksmart knowledge-api", lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = create_engine(settings.database_url)
    app.state.session_factory = sessionmaker(bind=app.state.engine)
    app.state.storage = BookStorage(settings.storage_root)
    app.include_router(books.router)
    app.include_router(jobs.router)
    app.include_router(knowledge.router)
    return app


app = create_app()
