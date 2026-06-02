from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import ingest, chat
from app.services.vector_store import init_qdrant_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    Initializes Qdrant collection on startup.
    """
    print("Starting Palm Mind RAG API...")
    await init_qdrant_collection()
    print("All services initialized.")
    yield
    print("Shutting down Palm Mind RAG API...")


app = FastAPI(
    title="Palm Mind AI — RAG Backend",
    description=(
        "A production-grade RAG backend with document ingestion, "
        "conversational Q&A, and interview booking support."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware ──────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────
app.include_router(
    ingest.router,
    prefix="/api/v1",
    tags=["Document Ingestion"],
)
app.include_router(
    chat.router,
    prefix="/api/v1",
    tags=["Conversational RAG"],
)


@app.get("/", tags=["Health"])
async def root() -> dict:
    return {"status": "ok", "message": "Palm Mind RAG API is running."}


@app.get("/health", tags=["Health"])
async def health() -> dict:
    return {"status": "healthy"}