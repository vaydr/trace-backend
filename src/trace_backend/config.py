from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRACE_", env_file=".env", extra="ignore")
    graph_db: Path = Path("data/graph.sqlite")
    checkpoint: Path = Path("artifacts/webqsp/best.pt")
    embedding_cache: Path = Path("artifacts/embedding-cache.sqlite")
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"
    llm_api_key: SecretStr | None = None
    api_key: SecretStr | None = None
    device: str = "auto"
    cors_origins: list[str] = ["http://localhost:3000"]
    search_max_states: int = Field(default=200000, ge=1)
    candidate_max_triples: int = Field(default=50000, ge=1)
    llm_timeout: float = Field(default=90, gt=0)
    llm_attempts: int = Field(default=3, ge=1, le=8)
    max_prompt_chars: int = Field(default=250000, ge=1000)
    max_concurrent_queries: int = Field(default=2, ge=1)
