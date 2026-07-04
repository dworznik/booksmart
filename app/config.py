from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BOOKSMART_")

    database_url: str = "postgresql+psycopg://booksmart:booksmart@localhost:5432/booksmart"
    storage_root: Path = Path("storage")

    qdrant_url: str = "http://localhost:6333"

    llm_provider: str = "anthropic"
    llm_model: str | None = None  # None -> the selected provider's default model
    embedding_provider: str = "openai"  # Anthropic has no embeddings API
    embedding_model: str | None = None
    # API keys; when unset the providers fall back to the conventional
    # ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY environment variables.
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
