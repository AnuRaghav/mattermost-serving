from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    model_path: Path = Path("models/toxicity_pipeline.joblib")
    model_version: str = "placeholder-1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
