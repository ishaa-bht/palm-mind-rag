from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Gemini
    gemini_api_key: str

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "documents_bge_base_en_v1_5"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # PostgreSQL
    database_url: str

    # App
    app_env: str = "development"

    # Local sentence-transformers embeddings
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    embedding_dim: int = 768

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
