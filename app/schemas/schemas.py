from pydantic import BaseModel, EmailStr
from enum import Enum
from datetime import datetime
from typing import Optional


# ── Chunking Strategy Enum ──────────────────────────────────────────
class ChunkingStrategy(str, Enum):
    fixed = "fixed"
    sentence = "sentence"


# ── Ingest Schemas ──────────────────────────────────────────────────
class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    chunking_strategy: ChunkingStrategy
    chunk_count: int
    uploaded_at: datetime

    class Config:
        from_attributes = True


# ── Chat Schemas ────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    query: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    booking_detected: bool = False


# ── Booking Schemas ─────────────────────────────────────────────────
class BookingInfo(BaseModel):
    name: str
    email: str
    date: str
    time: str


class BookingResponse(BaseModel):
    id: str
    name: str
    email: str
    date: str
    time: str
    session_id: str
    created_at: datetime

    class Config:
        from_attributes = True