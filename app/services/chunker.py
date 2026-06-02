import math
import re

from app.schemas.schemas import ChunkingStrategy
from app.services.embedder import embed_texts

MAX_CHUNK_WORDS = 300
CHUNK_OVERLAP_WORDS = 45
SEMANTIC_SIMILARITY_THRESHOLD = 0.65


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", text)
    ]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences without requiring runtime model downloads."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def _deduplicate(chunks: list[str]) -> list[str]:
    return list(dict.fromkeys(chunk for chunk in chunks if chunk.strip()))


def _word_windows(
    text: str,
    max_words: int = MAX_CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """Split oversized text into overlapping word windows safely."""
    _validate_window(max_words, overlap_words)

    words = text.split()
    if not words:
        return []

    step = max_words - overlap_words

    return [
        " ".join(words[start: start + max_words])
        for start in range(0, len(words), step)
        if words[start: start + max_words]
    ]


def recursive_chunk(
    text: str,
    max_words: int = MAX_CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """
    Split text using natural boundaries before falling back to word windows.
    """
    _validate_window(max_words, overlap_words)

    normalized = normalize_text(text)
    if not normalized:
        return []

    segments: list[str] = []
    for paragraph in normalized.split("\n\n"):
        if len(paragraph.split()) <= max_words:
            segments.append(paragraph)
            continue

        for sentence in split_sentences(paragraph):
            segments.append(sentence)

    chunks: list[str] = []
    current_words: list[str] = []

    for segment in segments:
        segment_words = segment.split()
        if len(segment_words) > max_words:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
            chunks.extend(_word_windows(segment, max_words, overlap_words))
            continue

        if current_words and len(current_words) + len(segment_words) > max_words:
            chunks.append(" ".join(current_words))
            current_words = current_words[-overlap_words:] if overlap_words else []
            if len(current_words) + len(segment_words) > max_words:
                current_words = []

        current_words.extend(segment_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return _deduplicate(chunks)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimensions.")

    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _validate_window(max_words: int, overlap_words: int) -> None:
    if max_words <= 0:
        raise ValueError("max_words must be greater than zero.")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must satisfy 0 <= overlap_words < max_words.")


async def semantic_chunk(
    text: str,
    max_words: int = MAX_CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
    similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> list[str]:
    """
    Group adjacent sentences dynamically using cosine similarity.
    """
    _validate_window(max_words, overlap_words)
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must satisfy -1.0 <= threshold <= 1.0.")

    normalized = normalize_text(text)
    sentences = split_sentences(normalized)
    if not sentences:
        return []
    if len(sentences) == 1:
        return recursive_chunk(sentences[0], max_words=max_words, overlap_words=overlap_words)

    embeddings = await embed_texts(sentences)
    chunks: list[str] = []

    current_sentences = [sentences[0]]
    current_word_count = len(sentences[0].split())

    for index in range(1, len(sentences)):
        sentence = sentences[index]
        sentence_words = len(sentence.split())

        similarity = _cosine_similarity(embeddings[index - 1], embeddings[index])

        topic_changed = similarity < similarity_threshold
        exceeds_budget = current_word_count + sentence_words > max_words

        if topic_changed or exceeds_budget:
            if current_sentences:
                chunks.extend(
                    recursive_chunk(
                        " ".join(current_sentences),
                        max_words=max_words,
                        overlap_words=overlap_words,
                    )
                )

            current_sentences = []
            current_word_count = 0

        current_sentences.append(sentence)
        current_word_count += sentence_words

    if current_sentences:
        chunks.extend(
            recursive_chunk(
                " ".join(current_sentences),
                max_words=max_words,
                overlap_words=overlap_words,
            )
        )

    return _deduplicate(chunks)


async def chunk_text(text: str, strategy: ChunkingStrategy) -> list[str]:
    """Route text to the selected production-oriented chunking strategy."""
    if strategy == ChunkingStrategy.recursive:
        return recursive_chunk(text)

    return await semantic_chunk(text)
