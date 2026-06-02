import asyncio
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the local embedding model once when it is first needed."""
    model = SentenceTransformer(
        settings.embedding_model_name,
        device=settings.embedding_device,
    )
    actual_dim = model.get_embedding_dimension()
    if actual_dim != settings.embedding_dim:
        raise ValueError(
            f"Embedding model '{settings.embedding_model_name}' produces "
            f"{actual_dim} dimensions, but EMBEDDING_DIM is "
            f"{settings.embedding_dim}."
        )
    return model


def _encode(texts: list[str]) -> list[list[float]]:
    """Encode text locally without blocking the FastAPI event loop."""
    vectors = _get_model().encode(
        texts,
        batch_size=settings.embedding_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.tolist()


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate local BGE embeddings for document text."""
    if not texts:
        return []

    return await asyncio.to_thread(_encode, texts)


async def embed_query(text: str) -> list[float]:
    """Generate a local BGE retrieval embedding for a user query."""
    vectors = await asyncio.to_thread(_encode, [f"{QUERY_PREFIX}{text}"])
    return vectors[0]
