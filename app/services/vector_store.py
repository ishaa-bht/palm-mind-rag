import uuid
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    ScoredPoint,
)
from app.core.config import settings


# Single shared async client
client = AsyncQdrantClient(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
)


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
) -> None:
    """
    Store chunks and their embeddings in Qdrant.

    Each point contains:
      - vector: embedding of the chunk
      - payload: doc_id, chunk text, filename for retrieval

    Args:
        doc_id: UUID of the parent document
        chunks: List of text chunks
        embeddings: Corresponding list of embedding vectors
        filename: Original filename for metadata
    """
    points: list[PointStruct] = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "doc_id": doc_id,
                "text": chunk,
                "filename": filename,
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
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

    return [hit.payload["text"] for hit in results if hit.payload]
