"""The CLI's local runtime: home-dir settings, an auto-migrated SQLite database,
embedded Qdrant, and a synchronous foreground Runner.

Everything lives under one home directory (``~/.booksmart`` by default) so the
whole installation is a single portable data dir — the SQLite file, ``storage/``,
and the embedded Qdrant directory move together (issue #25's portability, proven
end to end by the CLI e2e). No Docker, no Postgres, no server. Settings resolve
through ``booksmart_cli.config``'s precedence chain (env > config.toml >
conventional key env vars > home-dir defaults).
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from booksmart_core.database import create_engine, upgrade_to_head
from booksmart_core.llm import (
    EmbeddingProvider,
    LLMProvider,
    build_embedding_provider,
    build_llm_provider,
)
from booksmart_core.runner import SCOPE_STAGES, execute_run
from booksmart_core.stages import LLM_STAGES, Stage
from booksmart_core.storage import BookStorage
from booksmart_core.vectors import VectorStore, build_vector_store

from booksmart_core.config import Settings

from booksmart_cli.config import default_home, load_settings

__all__ = ["Runtime", "default_home", "load_settings"]


def _build_providers(
    settings: Settings, scope: str
) -> tuple[LLMProvider | None, EmbeddingProvider | None, VectorStore | None]:
    """Build only the providers this scope's stages need — so a profile-only run
    never constructs an embedder, and an embeddings-only run never an LLM."""
    stages = SCOPE_STAGES.get(scope, ())
    llm = build_llm_provider(settings) if any(s in LLM_STAGES for s in stages) else None
    embedder = build_embedding_provider(settings) if "embeddings" in stages else None
    vector_store = build_vector_store(settings) if "embeddings" in stages else None
    return llm, embedder, vector_store


@dataclass
class Runtime:
    """A ready-to-use local environment: settings, a migrated session factory,
    and object storage. Build it once per command with ``Runtime.load()``."""

    settings: Settings
    session_factory: sessionmaker[Session]
    storage: BookStorage

    @classmethod
    def load(cls, home: Path | None = None) -> "Runtime":
        """Resolve settings, auto-migrate the SQLite file to head (invisible to
        the user), and wire up storage."""
        settings = load_settings(home)
        upgrade_to_head(settings.database_url)
        engine = create_engine(settings.database_url)
        return cls(
            settings=settings,
            session_factory=sessionmaker(bind=engine),
            storage=BookStorage(settings.storage_root),
        )

    def ingest(
        self,
        book_id: uuid.UUID,
        scope: str = "full",
        *,
        on_stage: Callable[[Stage], None] | None = None,
    ) -> uuid.UUID:
        """Run a scope over a book to completion, foreground and synchronous,
        streaming stage progress through ``on_stage``. Returns the Run id; the
        outcome is recorded on the Run (this never raises for a Stage failure)."""
        llm, embedder, vector_store = _build_providers(self.settings, scope)
        try:
            return execute_run(
                self.session_factory,
                self.settings.storage_root,
                book_id,
                scope,
                llm=llm,
                embedder=embedder,
                vector_store=vector_store,
                on_stage=on_stage,
            )
        finally:
            # Release the embedded Qdrant on-disk lock so the next command (or a
            # relocated copy of the data dir) can open it.
            if vector_store is not None:
                vector_store.close()
