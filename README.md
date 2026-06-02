# Palm Mind RAG — AI Backend

A production-grade Retrieval-Augmented Generation (RAG) backend built with FastAPI. Upload documents, ask questions grounded in their content, and book interviews — all through a conversational API.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| LLM & Embeddings | Google Gemini 2.5 Flash / gemini-embedding-001 |
| Vector Store | Qdrant |
| Session Memory | Redis |
| Database | PostgreSQL + SQLAlchemy (async) + Alembic |
| File Parsing | PyMuPDF (PDF), built-in (TXT) |

---

## Project Structure

```
app/
├── api/
│   ├── chat.py          # Conversational RAG endpoint
│   └── ingest.py        # Document upload and ingestion endpoint
├── core/
│   ├── config.py        # Environment-based settings
│   └── database.py      # Async SQLAlchemy engine and session
├── models/
│   └── db_models.py     # ORM models: DocumentMetadata, Booking
├── schemas/
│   └── schemas.py       # Pydantic request/response schemas
├── services/
│   ├── booking.py       # Gemini-powered booking intent extraction
│   ├── chunker.py       # Recursive and semantic chunking strategies
│   ├── embedder.py      # Gemini embedding generation
│   ├── llm.py           # RAG answer generation
│   ├── memory.py        # Redis session history management
│   └── vector_store.py  # Qdrant collection and similarity search
└── main.py              # FastAPI app, lifespan, routers, CORS
alembic/                 # Database migrations
docker-compose.yml       # PostgreSQL, Qdrant, Redis services
```

---

## Setup

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### 1. Clone and install dependencies

```bash
git clone https://github.com/ishaa-bht/palm-mind-rag.git
cd palm-mind-rag

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql+asyncpg://palmuser:palmpass@localhost:5433/palmdb
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents
REDIS_URL=redis://localhost:6379
APP_ENV=development
EMBEDDING_DIM=768
```

See [Environment Variables](#environment-variables) below for a full description of each.

### 3. Start infrastructure services

```bash
docker compose up -d
```

This starts PostgreSQL, Qdrant, and Redis.

### 4. Run database migrations

```bash
alembic upgrade head
```

### 5. Start the API server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | — | Google Gemini API key for LLM and embeddings |
| `DATABASE_URL` | Yes | — | Async PostgreSQL connection string (`postgresql+asyncpg://...`) |
| `QDRANT_HOST` | No | `localhost` | Hostname of the Qdrant vector store |
| `QDRANT_PORT` | No | `6333` | Port of the Qdrant vector store |
| `QDRANT_COLLECTION_NAME` | No | `documents` | Name of the Qdrant collection used to store embeddings |
| `REDIS_URL` | No | `redis://localhost:6379` | Redis connection URL for session history storage |
| `APP_ENV` | No | `development` | Application environment; enables SQL logging in `development` |
| `EMBEDDING_DIM` | No | `768` | Embedding output dimensionality requested from Gemini |

---

## API Endpoints

### `POST /api/v1/ingest` — Upload a Document

Accepts a `.pdf` or `.txt` file, extracts text, chunks it, generates embeddings, stores vectors in Qdrant, and saves metadata to PostgreSQL.

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | File | `.pdf` or `.txt` file to ingest |
| `strategy` | string | Chunking strategy: `recursive` or `semantic` |

**Example**

```bash
curl -s -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@/path/to/document.pdf" \
  -F "strategy=recursive" \
  | python -m json.tool
```

**Response**

```json
{
  "doc_id": "a3f1c2d4-...",
  "filename": "document.pdf",
  "file_type": "pdf",
  "chunking_strategy": "recursive",
  "chunk_count": 42,
  "uploaded_at": "2026-06-02T09:00:00Z"
}
```

---

### `POST /api/v1/chat` — Conversational Q&A

Embeds the user query, retrieves the most relevant chunks from Qdrant, and generates a grounded answer using Gemini. Maintains multi-turn conversation history in Redis. Automatically detects and processes interview booking intent.

**Request** — `application/json`

| Field | Type | Description |
|---|---|---|
| `session_id` | string | Unique identifier for the conversation session |
| `query` | string | User's message or question |

**Example**

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-123", "query": "What is the refund policy?"}' \
  | python -c 'import json, sys; print(json.load(sys.stdin)["answer"])'
```

**Response**

```json
{
  "session_id": "user-123",
  "answer": "According to the documents, the refund policy states...",
  "booking_detected": false
}
```

**Booking detected response**

```json
{
  "session_id": "user-123",
  "answer": "Your interview has been booked!\nName: Jane Doe\nEmail: jane@example.com\nDate: 2026-06-10\nTime: 14:00\n...",
  "booking_detected": true
}
```

---

## Valid Mock Payloads

The following commands are copy-pasteable from the project root while the API
is running at `http://localhost:8000`.

### 1. Ingest `sample.pdf`

Use recursive chunking for a fast general-purpose ingestion:

```bash
curl -s -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@sample.pdf" \
  -F "strategy=recursive" \
  | python -m json.tool
```

To test topic-boundary detection instead, run:

```bash
curl -s -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@sample.pdf" \
  -F "strategy=semantic" \
  | python -m json.tool
```

### 2. Ask research-paper questions

Reuse the same `session_id` to preserve Redis-backed conversation history
across the five requests. Each command prints the decoded `answer` text
directly, including readable Unicode punctuation.

**Question 1: What is the title of the research paper?**

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"sample-pdf-session","query":"What is the title of the research paper?"}' \
  | python -c 'import json, sys; print(json.load(sys.stdin)["answer"])'
```

**Question 2: What percentage of students started using mobile devices for English learning at university?**

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"sample-pdf-session","query":"What percentage of students started using mobile devices for English learning at university?"}' \
  | python -c 'import json, sys; print(json.load(sys.stdin)["answer"])'
```

**Question 3: How did Holec define learner autonomy?**

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"sample-pdf-session","query":"How did Holec define learner autonomy?"}' \
  | python -c 'import json, sys; print(json.load(sys.stdin)["answer"])'
```

**Question 4: Why did students prefer using mobile devices for learning English?**

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"sample-pdf-session","query":"Why did students prefer using mobile devices for learning English?"}' \
  | python -c 'import json, sys; print(json.load(sys.stdin)["answer"])'
```

**Question 5: What evidence suggests that mobile devices promoted autonomous learning?**

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"sample-pdf-session","query":"What evidence suggests that mobile devices promoted autonomous learning?"}' \
  | python -c 'import json, sys; print(json.load(sys.stdin)["answer"])'
```

### 3. Book an interview

Use a separate session ID for the booking flow. This payload includes all four
required fields: name, email, date, and time.

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"booking-demo-session","query":"I would like to book an interview. My name is Jane Doe, my email is jane@example.com, and I am available on 2026-06-10 at 14:00."}' \
  | python -c 'import json, sys; print(json.load(sys.stdin)["answer"])'
```

Verify that PostgreSQL stored the booking:

```bash
docker compose exec postgres \
  psql -U palmuser -d palmdb \
  -c "SELECT name, email, date, time, session_id, created_at FROM bookings ORDER BY created_at DESC LIMIT 5;"
```

---

## Chunking Strategies

Text extracted from uploaded documents is split into chunks before embedding. Two strategies are available:

### `recursive`

Splits text using natural boundaries before falling back to overlapping word
windows. It prefers paragraphs, then sentences, then words.

- **Maximum size:** 300 words
- **Overlap:** 45 words
- **Best for:** General-purpose ingestion with predictable performance

### `semantic`

Embeds adjacent sentences with Gemini and starts a new chunk when cosine
similarity drops below the configured topic-boundary threshold. Oversized
chunks fall back to the recursive splitter.

- **Maximum size:** 300 words
- **Similarity threshold:** 0.65
- **Best for:** Reports and documents containing multiple topics
- **Trade-off:** Additional Gemini embedding calls make ingestion slower

---

## How Booking Works

The chat endpoint runs a parallel booking detection step on every message using Gemini. No special command or keyword is required from the user.

**Detection flow:**

1. The full conversation history and the latest user message are sent to Gemini with a structured extraction prompt.
2. Gemini checks whether the user intends to book an interview **and** whether all four required fields are present in the conversation: `name`, `email`, `date`, and `time`.
3. If all fields are present, Gemini returns a structured JSON object. The booking is saved to PostgreSQL and a confirmation message is returned.
4. If any field is missing, `null` is returned and the conversation continues normally through the RAG pipeline.

**Required fields:**

| Field | Example |
|---|---|
| `name` | Jane Doe |
| `email` | jane@example.com |
| `date` | 2026-06-10 |
| `time` | 14:00 |

The user can provide these fields across multiple turns — the system accumulates context from the full conversation history before deciding whether to trigger a booking.

---

## Health Checks

```bash
curl -s http://localhost:8000/ | python -m json.tool
curl -s http://localhost:8000/health | python -m json.tool
```
