import nltk
from app.schemas.schemas import ChunkingStrategy

# Download required nltk data if not already present
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def fixed_size_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[str]:
    """
    Strategy 1: Split text into fixed-size character chunks with overlap.
    Overlap ensures context is not lost at chunk boundaries.
    """
    chunks: list[str] = []
    start: int = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def sentence_chunk(
    text: str,
    max_sentences: int = 5
) -> list[str]:
    """
    Strategy 2: Split text on sentence boundaries using nltk.
    Groups max_sentences sentences per chunk to preserve meaning.
    """
    sentences: list[str] = nltk.sent_tokenize(text)
    chunks: list[str] = []

    for i in range(0, len(sentences), max_sentences):
        group = sentences[i: i + max_sentences]
        chunk = " ".join(group).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def chunk_text(
    text: str,
    strategy: ChunkingStrategy
) -> list[str]:
    """
    Dispatcher: routes to the correct chunking strategy.

    Args:
        text: Raw extracted text from document
        strategy: ChunkingStrategy enum (fixed or sentence)

    Returns:
        List of text chunks
    """
    if strategy == ChunkingStrategy.fixed:
        return fixed_size_chunk(text)
    return sentence_chunk(text)