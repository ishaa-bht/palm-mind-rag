import uuid
import fitz  # PyMuPDF
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.db_models import DocumentMetadata
from app.schemas.schemas import ChunkingStrategy, IngestResponse
from app.services.chunker import chunk_text
from app.services.embedder import embed_texts
from app.services.vector_store import upsert_chunks

router = APIRouter()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract plain text from PDF bytes using PyMuPDF.

    Args:
        file_bytes: Raw PDF file content

    Returns:
        Extracted text string
    """
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()


def extract_text_from_txt(file_bytes: bytes) -> str:
    """
    Decode plain text file bytes to string.

    Args:
        file_bytes: Raw TXT file content

    Returns:
        Decoded text string
    """
    return file_bytes.decode("utf-8", errors="ignore").strip()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Upload and ingest a document",
    description="Upload a .pdf or .txt file, chunk it using the selected strategy, embed and store in Qdrant, and save metadata to PostgreSQL.",
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF or TXT file to ingest"),
    strategy: ChunkingStrategy = Form(..., description="Chunking strategy: 'recursive' or 'semantic'"),
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    """
    Document ingestion endpoint.

    Flow:
        1. Validate file type (.pdf or .txt)
        2. Extract text from file
        3. Chunk text using selected strategy
        4. Generate embeddings via Gemini
        5. Store vectors in Qdrant
        6. Save document metadata in PostgreSQL
        7. Return IngestResponse
    """
    # ── Step 1: Validate file type ──────────────────────────────────
    filename = file.filename or "unknown"
    extension = filename.rsplit(".", 1)[-1].lower()

    if extension not in ("pdf", "txt"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{extension}'. Only .pdf and .txt are allowed.",
        )

    # ── Step 2: Read file bytes ─────────────────────────────────────
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Step 3: Extract text ────────────────────────────────────────
    if extension == "pdf":
        text = extract_text_from_pdf(file_bytes)
    else:
        text = extract_text_from_txt(file_bytes)

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Could not extract any text from the uploaded file.",
        )

    # ── Step 4: Chunk text ──────────────────────────────────────────
    chunks: list[str] = await chunk_text(text, strategy)

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="Text extraction succeeded but chunking produced no chunks.",
        )

    # ── Step 5: Generate embeddings ─────────────────────────────────
    embeddings: list[list[float]] = await embed_texts(chunks)

    # ── Step 6: Store in Qdrant ─────────────────────────────────────
    doc_id = str(uuid.uuid4())

    await upsert_chunks(
        doc_id=doc_id,
        chunks=chunks,
        embeddings=embeddings,
        filename=filename,
        strategy=strategy,
    )

    # ── Step 7: Save metadata to PostgreSQL ─────────────────────────
    metadata = DocumentMetadata(
        id=doc_id,
        filename=filename,
        file_type=extension,
        chunking_strategy=strategy.value,
        chunk_count=len(chunks),
    )
    db.add(metadata)
    await db.flush()

    return IngestResponse(
        doc_id=doc_id,
        filename=filename,
        file_type=extension,
        chunking_strategy=strategy,
        chunk_count=len(chunks),
        uploaded_at=metadata.uploaded_at,
    )
