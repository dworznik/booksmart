from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BOOKSMART_")

    database_url: str = "postgresql+psycopg://booksmart:booksmart@localhost:5432/booksmart"
    storage_root: Path = Path("storage")
