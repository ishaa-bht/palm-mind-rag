import uuid
import re
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    ScoredPoint,
)
from app.core.config import settings
from app.schemas.schemas import ChunkingStrategy


# Single shared async client
client = AsyncQdrantClient(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
)


def normalize_retrieved_text(text: str) -> str:
    """Remove PDF line wrapping before retrieved text reaches the LLM prompt."""
    return re.sub(r"\s+", " ", text).strip()


async def init_qdrant_collection() -> None:
    """
    Create Qdrant collection on startup if it doesn't exist.
    Vector size 768 matches the configured Gemini embedding output.
    """
    existing = await client.get_collections()
    names = [col.name for col in existing.collections]

    if settings.qdrant_collection_name not in names:
        await client.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dim,   # 768 for Gemini
                distance=Distance.COSINE,
            ),
        )
        print(f"Qdrant collection '{settings.qdrant_collection_name}' created.")
    else:
        print(f"Qdrant collection '{settings.qdrant_collection_name}' already exists.")


async def upsert_chunks(
    doc_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    filename: str,
    strategy: ChunkingStrategy,
) -> None:
    """
    Store chunks and their embeddings in Qdrant.

    Each point contains:
      - vector: embedding of the chunk
      - payload: source and chunk metadata for retrieval inspection

    Args:
        doc_id: UUID of the parent document
        chunks: List of text chunks
        embeddings: Corresponding list of embedding vectors
        filename: Original filename for metadata
        strategy: Chunking strategy selected for the document
    """
    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have exactly one embedding.")

    points: list[PointStruct] = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "doc_id": doc_id,
                "text": chunk,
                "filename": filename,
                "chunk_index": index,
                "chunk_count": len(chunks),
                "strategy": strategy.value,
            },
        )
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    await client.upsert(
        collection_name=settings.qdrant_collection_name,
        points=points,
    )


async def search_similar(
    query_vector: list[float],
    top_k: int = 5,
) -> list[str]:
    """
    Search Qdrant for the most similar chunks to the query vector.

    Args:
        query_vector: Embedded query from user
        top_k: Number of top results to return

    Returns:
        List of chunk texts ranked by similarity
    """
    results: list[ScoredPoint] = await client.search(
        collection_name=settings.qdrant_collection_name,
        query_vector=query_vector,
        limit=top_k,
    )

    return [
        normalize_retrieved_text(hit.payload["text"])
        for hit in results
        if hit.payload and isinstance(hit.payload.get("text"), str)
    ]
