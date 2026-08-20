"""
app/core/config.py
──────────────────
Central settings for picasso-rag-chat. All values are read from environment
variables (or a .env file via pydantic-settings).

Reusability note: The ``active_profile`` setting determines which project
profile is loaded at startup. Switching profiles requires only an env-var
change — no code modification.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings resolved from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        enable_decoding=False,
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model identifier",
    )
    chat_model: str = Field(
        default="gpt-4.1",
        description="OpenAI chat completion model identifier",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        ...,
        description="Async SQLAlchemy DB URL (postgresql+asyncpg://...)",
    )

    # ── RAG ───────────────────────────────────────────────────────────────────
    retrieval_top_k: Annotated[int, Field(ge=1, le=20)] = Field(
        default=5,
        description="Number of chunks returned per similarity search",
    )
    history_window: Annotated[int, Field(ge=1, le=100)] = Field(
        default=20,
        description="Max conversation turns included in LLM prompt",
    )
    extraction_confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.7,
        description="Min confidence for an extracted answer to count as captured",
    )
    vector_similarity_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.7,
        description="Min cosine similarity for vector search chunks",
    )

    # ── Profile ───────────────────────────────────────────────────────────────
    active_profile: str = Field(
        default="picasso_fusion",
        description=(
            "Active project profile folder name inside app/project_profiles/. "
            "Changing this switches the entire domain configuration."
        ),
    )

    # ── Server ────────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        description="CORS allowed origins (Vite dev server etc.)",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors(cls, v: str | list[str]) -> list[str]:
        """Accept comma-separated string or list (or JSON list string)."""
        if isinstance(v, str):
            if v.strip().startswith("["):
                import json
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            return [o.strip() for o in v.split(",") if o.strip()]
        return v


# Module-level singleton — import this everywhere.
settings = Settings()
