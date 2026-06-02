from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.db_models import Booking
from app.schemas.schemas import ChatRequest, ChatResponse
from app.services.embedder import embed_query
from app.services.vector_store import search_similar
from app.services.memory import get_history, add_message
from app.services.llm import generate_answer
from app.services.booking import extract_booking

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with ingested documents",
    description="Send a query and get an answer grounded in uploaded documents. Supports multi-turn conversation via session_id. Detects interview booking intent automatically.",
)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    Conversational RAG endpoint.

    Flow:
        1. Load chat history from Redis
        2. Check for booking intent via Gemini
           ├── If booking found → save to PostgreSQL → return confirmation
           └── If no booking → continue to RAG
        3. Embed the user query
        4. Search Qdrant for similar chunks
        5. Generate answer via Gemini with context + history
        6. Save user message and answer to Redis
        7. Return ChatResponse
    """
    session_id = request.session_id
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # ── Step 1: Load history from Redis ─────────────────────────────
    history: list[dict] = await get_history(session_id)

    # ── Step 2: Check for booking intent ────────────────────────────
    booking_info = await extract_booking(query, history)

    if booking_info:
        # Save booking to PostgreSQL
        booking = Booking(
            name=booking_info.name,
            email=booking_info.email,
            date=booking_info.date,
            time=booking_info.time,
            session_id=session_id,
        )
        db.add(booking)
        await db.flush()

        # Save to Redis history
        await add_message(session_id, "user", query)

        confirmation = (
            f"Your interview has been booked!\n"
            f"Name: {booking_info.name}\n"
            f"Email: {booking_info.email}\n"
            f"Date: {booking_info.date}\n"
            f"Time: {booking_info.time}\n"
            f"We'll send a confirmation to {booking_info.email} shortly."
        )

        await add_message(session_id, "assistant", confirmation)

        return ChatResponse(
            session_id=session_id,
            answer=confirmation,
            booking_detected=True,
        )

    # ── Step 3: Embed the query ──────────────────────────────────────
    query_vector: list[float] = await embed_query(query)

    # ── Step 4: Search Qdrant for similar chunks ─────────────────────
    context_chunks: list[str] = await search_similar(query_vector, top_k=5)

    if not context_chunks:
        answer = (
            "I couldn't find any relevant information in the "
            "uploaded documents. Please upload a document first."
        )
        await add_message(session_id, "user", query)
        await add_message(session_id, "assistant", answer)

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            booking_detected=False,
        )

    # ── Step 5: Generate answer via Gemini ───────────────────────────
    answer: str = await generate_answer(
        query=query,
        context_chunks=context_chunks,
        history=history,
    )

    # ── Step 6: Save to Redis history ───────────────────────────────
    await add_message(session_id, "user", query)
    await add_message(session_id, "assistant", answer)

    # ── Step 7: Return response ──────────────────────────────────────
    return ChatResponse(
        session_id=session_id,
        answer=answer,
        booking_detected=False,
    )