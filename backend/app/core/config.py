from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    APP_NAME: str = "APIL"
    APP_ENV: str = "development"
    APP_VERSION: str = "0.1.0"
    CORS_ORIGINS: str = ""

    DATABASE_URL: str

    REDIS_URL: str | None = None

    OPENAI_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None

    OLLAMA_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen3:4b"
    OLLAMA_TIMEOUT_SECONDS: float = Field(
    default=600.0,
    validation_alias=AliasChoices(
        "OLLAMA_TIMEOUT_SECONDS",
        "OLLAMA_TIMEOUT",
    ),
)
    OLLAMA_THINK: bool = False
    OLLAMA_KEEP_ALIVE: str = "10m"
    OLLAMA_NUM_PREDICT: int = 4096

    APIL_API_KEY: str | None = None
    APIL_MAX_IMPROVEMENT_ATTEMPTS: int = 1
    APIL_ENABLE_DEBUG_METADATA: bool = True

    RATE_LIMIT_REQUESTS: int = 30
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
