import google.generativeai as genai
from app.core.config import settings

# Configure Gemini client once at module level
genai.configure(api_key=settings.gemini_api_key)

EMBEDDING_MODEL = "models/gemini-embedding-001"


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of text chunks using Gemini.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (each vector is list[float] of dim 768)
    """
    embeddings: list[list[float]] = []

    for text in texts:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document",
            output_dimensionality=settings.embedding_dim,
        )
        embeddings.append(result["embedding"])

    return embeddings


async def embed_query(text: str) -> list[float]:
    """
    Generate embedding for a single query string.
    Uses retrieval_query task type for better search results.

    Args:
        text: Query string from user

    Returns:
        Single embedding vector of dim 768
    """
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query",
        output_dimensionality=settings.embedding_dim,
    )
    return result["embedding"]
