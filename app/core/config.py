from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Gemini
    gemini_api_key: str

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "documents"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # PostgreSQL
    database_url: str

    # App
    app_env: str = "development"

    #  Reduced embedding size requested from Gemini gemini-embedding-001
    embedding_dim: int = 768

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
