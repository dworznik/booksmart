"""A corpus: one auto-migrated database, one vector store, one storage root.

Everything for a pipeline configuration lives in a single directory under the
corpus home, the same portable shape the product CLI uses — so deleting a
snapshot is deleting a folder, and a corpus can be moved without becoming a
different corpus.

That last part is the reason the store locations are rewritten here rather than
read from the environment. The snapshot deliberately excludes database URL,
storage root and vector path, so `Corpus` is free to point all three inside the
snapshot directory without changing which snapshot it is. The alternative —
honouring an ambient ``BOOKSMART_DATABASE_URL`` — would let two configurations
write into one database and score against each other's records.
"""

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

from sqlalchemy.orm import Session, sessionmaker

from booksmart_core.config import Settings
from booksmart_core.database import create_engine, upgrade_to_head
from booksmart_core.storage import BookStorage
from booksmart_core.vectors import VectorStore, build_vector_store

from booksmart_bench.config import corpus_home


@dataclass
class Corpus:
    """An open corpus. Use as a context manager — the embedded vector store
    holds an on-disk lock that the next command needs released."""

    home: Path
    settings: Settings
    session_factory: sessionmaker[Session]
    storage: BookStorage
    _vector_store: VectorStore | None = None

    @classmethod
    def open(cls, settings: Settings, home: Path | None = None) -> Self:
        """Create or reopen this configuration's corpus, migrated to head."""
        root = home or corpus_home(settings)
        root.mkdir(parents=True, exist_ok=True)

        located = settings.model_copy(
            update={
                "database_url": f"sqlite:///{root / 'booksmart.db'}",
                "storage_root": root / "storage",
                "qdrant_path": root / "qdrant",
                # qdrant_path wins over qdrant_url in core, but leaving a stray
                # server URL in the settings a run stamps would misdescribe it.
                "qdrant_url": "",
            }
        )
        upgrade_to_head(located.database_url)
        return cls(
            home=root,
            settings=located,
            session_factory=sessionmaker(bind=create_engine(located.database_url)),
            storage=BookStorage(located.storage_root),
        )

    @property
    def vector_store(self) -> VectorStore:
        """Opened lazily and once: embedded Qdrant takes an exclusive on-disk
        lock, so a second client in the same process would fail."""
        if self._vector_store is None:
            self._vector_store = build_vector_store(self.settings)
        return self._vector_store

    def close(self) -> None:
        if self._vector_store is not None:
            self._vector_store.close()
            self._vector_store = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
