"""Application settings loaded from environment / .env (pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Providers
    openai_api_key: str | None = None
    openai_tts_model: str = "gpt-4o-mini-tts"

    # Engine defaults
    default_provider: str = "openai"
    default_voice: str = "alloy"

    # Output (used by later phases)
    output_dir: str = "./output"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
