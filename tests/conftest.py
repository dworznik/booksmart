"""Integration test fixtures.

Tests run against a real PostgreSQL instance (docker compose up -d postgres).
A dedicated test database is created per session and migrated with Alembic.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from app.config import Settings
from app.main import create_app

TEST_DATABASE_URL = os.environ.get(
    "BOOKSMART_TEST_DATABASE_URL",
    "postgresql+psycopg://booksmart:booksmart@localhost:5432/booksmart_test",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def database_url() -> str:
    admin_url = TEST_DATABASE_URL.rsplit("/", 1)[0] + "/postgres"
    test_db_name = TEST_DATABASE_URL.rsplit("/", 1)[1]
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"),
            {"name": test_db_name},
        ).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{test_db_name}"'))
    admin_engine.dispose()

    alembic_cfg = AlembicConfig(str(PROJECT_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    command.upgrade(alembic_cfg, "head")
    return TEST_DATABASE_URL


@pytest.fixture()
def settings(database_url: str, tmp_path: Path) -> Settings:
    return Settings(database_url=database_url, storage_root=tmp_path / "storage")


@pytest.fixture()
def client(settings: Settings, database_url: str) -> Iterator[TestClient]:
    engine = create_engine(database_url)
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE books"))
        conn.commit()
    engine.dispose()

    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
