"""Core configuration.

``Settings`` is a plain model, deliberately: core never reads the process
environment (or any other ambient source) — the caller constructs the whole
configuration explicitly. The CLI resolves its env vars and config file into
one of these; any other consumer does its own resolution the same way.
"""

from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    database_url: str = "postgresql+psycopg://booksmart:booksmart@localhost:5432/booksmart"
    storage_root: Path = Path("storage")

    # Vector store location. A server URL by default (a service's shape); set
    # qdrant_path to run Qdrant embedded on-disk instead (the CLI's default, no
    # service). When qdrant_path is set it wins and qdrant_url is ignored.
    qdrant_url: str = "http://localhost:6333"
    qdrant_path: Path | None = None

    llm_provider: str = "anthropic"
    llm_model: str | None = None  # None -> the selected provider's default model
    # Reasoning/thinking control for OpenAI-compatible providers ("none", "low",
    # "medium", "high"). "none" disables Gemini 2.5 Flash thinking so structured
    # stages don't spend the completion budget deliberating; gemini-2.5-pro
    # rejects "none". Ignored by the anthropic and fake providers.
    llm_reasoning_effort: str | None = None
    embedding_provider: str = "openai"  # Anthropic has no embeddings API
    embedding_model: str | None = None
    # API keys. Required (non-None) by the matching provider at construction;
    # there is no environment fallback — resolving keys is the caller's job.
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
