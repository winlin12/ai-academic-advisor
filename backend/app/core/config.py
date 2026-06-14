from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_local_only: bool = True
    academic_db_path: str | None = None
    academic_database_url: str = "postgresql://purdueio:purdueio@localhost:5432/purdueio"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
