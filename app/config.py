from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///data/reviewermatch.db"

    @field_validator("database_url", mode="before")
    @classmethod
    def coerce_db_url(cls, v: str) -> str:
        if v.startswith("postgres://"):
            return "postgresql+asyncpg://" + v[len("postgres://"):]
        if v.startswith("postgresql://"):
            return "postgresql+asyncpg://" + v[len("postgresql://"):]
        return v

    embedding_model: str = "all-MiniLM-L6-v2"
    faiss_index_path: str = "data/authors.faiss"
    api_key: str = "changeme"
    max_results: int = 50

    openalex_email: str = ""
    openalex_base_url: str = "https://api.openalex.org"
    openalex_api_key: str = ""
    anthropic_api_key: str = ""
    free_tier_daily_limit: int = 3

    model_config = {"env_file": ".env", "protected_namespaces": ()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
