"""Application configuration using Pydantic Settings."""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for ClaimAssist AI."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_NAME: str = "ClaimAssist AI"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    APP_HOST: str = "127.0.0.1"
    APP_PORT: int = 8000

    # LLM Settings
    LLM_PROVIDER: str = "openrouter"  # "mock", "openai", or "openrouter"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_MODEL: str = "anthropic/claude-3.5-sonnet"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_SITE_URL: str = "https://github.com/ayush300302/claim_assist_ai"
    OPENROUTER_APP_NAME: str = "ClaimAssist AI"

    # Retrieval Settings
    RETRIEVAL_THRESHOLD: float = 0.25
    TOP_K_CHUNKS: int = 3
    POLICY_PATH: str = "data/policy.md"
    CLAIMS_PATH: str = "data/claims.json"

    # Tool execution safety
    CLAIM_TOOL_TIMEOUT_SECONDS: float = 5.0
    MAX_TOOL_CALL_ROUNDS: int = 3

    # Logging
    LOG_FILE_PATH: str = "logs/conversations.jsonl"

    @property
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def resolved_policy_path(self) -> Path:
        return self.base_dir / self.POLICY_PATH

    @property
    def resolved_claims_path(self) -> Path:
        return self.base_dir / self.CLAIMS_PATH

    @property
    def resolved_log_file_path(self) -> Path:
        return self.base_dir / self.LOG_FILE_PATH


settings = Settings()
