"""Application settings loaded from environment / .env (pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Providers
    # Kokoro: local, CPU-only, no API key. Registered when installed + enabled.
    kokoro_enabled: bool = True
    kokoro_voice: str = "af_heart"
    # How many Kokoro syntheses may run at once (each one uses all CPU cores anyway).
    kokoro_max_concurrency: int = Field(1, ge=1, le=8)
    openai_api_key: str | None = None
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_voice: str = "alloy"
    # Per-attempt timeout in seconds, and retries on network/429/5xx errors (by the SDK).
    openai_timeout: float = Field(30.0, gt=0)
    openai_max_retries: int = Field(2, ge=0, le=10)

    # Engine defaults (local Kokoro needs no key)
    default_provider: str = "kokoro"
    default_voice: str = "af_heart"

    # Podcast output + job store
    output_dir: str = "./output"
    jobs_db: str = "./output/jobs.db"
    podcast_pause_ms: int = 600
    # Podcast renders that may run at once; the rest wait as "queued".
    max_concurrent_jobs: int = Field(1, ge=1, le=8)
    # Finished jobs (and their audio files) older than this are deleted; 0 keeps all.
    jobs_retention_days: float = Field(7, ge=0)

    # Logging (app + openai loggers): DEBUG, INFO, WARNING, ERROR
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
