"""Configuration settings for paperless-rocm-reindexer."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PAPERLESS_BASE_URL: str = "http://paperless-web:8000"
    PAPERLESS_API_TOKEN: str = ""
    PAPERLESS_MEDIA_ROOT: Path = Path("/media")
    PAPERLESS_DB_URL: str = ""
    ROCM_OCR_BASE_URL: str = "http://rocm-ocr:8000"
    ROCM_OCR_API_KEY: str = ""
    OCR_CONCURRENCY: int = 4
    BATCH_LIMIT: int = 0
    TAG_SUCCESS: str = "ocr:rocm-v4"
    TAG_FAILURE: str = "ocr:failed"
    DAEMON_INTERVAL_SEC: int = 0
    DRY_RUN: bool = False
    STATE_DB_PATH: Path = Path("/data/reindexer_state.db")


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of Settings."""
    return Settings()
