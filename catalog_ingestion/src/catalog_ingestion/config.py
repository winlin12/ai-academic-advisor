from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://catalog:catalog@localhost:5433/catalog_ingestion"
    catalog_base_url: str = "https://catalog.purdue.edu"
    catalog_user_agent: str = (
        "PurdueAcademicPlannerBot/1.0 (student hobby project; not affiliated with Purdue)"
    )
    crawl_delay_seconds: float = 120.0
    max_crawl_pages: int = 500
    page_cache_dir: Path = Path(".page_cache")
    fetcher_backend: str = "playwright"
    playwright_browser: str = ""
    purdueapi_odata_url: str = "https://api.purdue.io/odata"
    log_level: str = "INFO"

    @field_validator("fetcher_backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        allowed = {"playwright", "httpx"}
        if v not in allowed:
            raise ValueError(f"fetcher_backend must be one of {allowed}, got {v!r}")
        return v


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
