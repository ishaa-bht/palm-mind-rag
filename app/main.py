import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.api_core.exceptions import (
    GoogleAPIError,
    PermissionDenied,
    ResourceExhausted,
)

from app.api import ingest, chat
from app.services.vector_store import init_qdrant_collection

logger = logging.getLogger(__name__)


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


@app.exception_handler(GoogleAPIError)
async def handle_google_api_error(
    request: Request,
    exc: GoogleAPIError,
) -> JSONResponse:
    """Return actionable JSON when Gemini rejects an upstream request."""
    logger.exception("Gemini API request failed for %s", request.url.path)

    if isinstance(exc, PermissionDenied):
        detail = (
            "Gemini API access was denied. Check GEMINI_API_KEY and the "
            "Google AI project status."
        )
    elif isinstance(exc, ResourceExhausted):
        detail = (
            "Gemini API quota was exhausted. Wait for the quota window to "
            "reset or use a project with available quota."
        )
    else:
        detail = "Gemini API request failed. Please try again later."

    return JSONResponse(status_code=503, content={"detail": detail})


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
