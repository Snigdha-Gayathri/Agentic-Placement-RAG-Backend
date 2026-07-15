# Agentic Placement RAG — Backend Service

A production-ready, highly secure, and observable **FastAPI + ChromaDB + Gemini** backend powering the **Agentic Placement RAG Assistant**.

This backend features state-of-the-art hybrid search (`BM25` + `Dense Cosine Embeddings`), cross-encoder reranking, multi-hop query rewriting, autonomous agentic planning cycles, multi-layered security guardrails, and real-time Server-Sent Events (SSE) telemetry.

---

## Key Features

- **Automatic ChromaDB Initialization on Startup**:
  The Chroma database is never committed to Git. Instead, when the service starts up on a new environment (such as Render or a fresh local setup) and detects that no vector index exists (`total_chunks == 0`), the backend automatically scans all raw PDF knowledge base files in `data/`, computes embeddings, builds the persistent ChromaDB index, and initializes the BM25 corpus automatically. Zero manual database setup required!
- **Multi-Layered Security Guardrails (`security/`)**:
  - **Input Validator**: Normalizes queries, limits token length, and filters forbidden patterns.
  - **Prompt Injection Detector**: Blocks jailbreak vectors and heuristic injection attempts.
  - **Sliding Window Rate Limiter**: Protects endpoints against DDoS and excessive token consumption.
  - **Retrieval Guard**: Filters out non-relevant candidate chunks (`similarity_threshold`).
  - **Context Sanitizer**: Removes potential prompt injection payloads embedded in retrieved documents before passing them to the LLM.
  - **Grounding & Hallucination Guard**: Verifies LLM output against retrieved chunks to ensure fidelity.
- **Hybrid Search & Reranking**:
  Combines exact keyword matching via `rank-bm25` with dense semantic search via Gemini embeddings (`text-embedding-004`), followed by cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for peak precision.
- **Observability & SSE Streaming**:
  Emits live pipeline execution stages (`/api/pipeline-status/{request_id}`) and detailed developer dashboard metrics (`/api/dashboard/{request_id}`).

---

## Architecture & Module Layout

```
├── backend/
│   ├── app/             # FastAPI main application, routes, Pydantic models, and SecureRAGService
│   ├── config/          # Pipeline settings and YAML configurations
│   ├── core/            # Core RAG engine (agent, chunking, embeddings, hyde, memory, reranking, retrieval, vector_store)
│   ├── evaluation/      # RAG metrics engine (faithfulness, relevancy, latency tracking)
│   ├── ingestion/       # Document processing pipeline (PDF loaders, chunk enhancers, metadata extractors)
│   └── observability/   # Live stage trackers and dashboard data structures
├── security/            # Production security guardrails package
├── tests/               # Comprehensive unit and guardrail verification suite
├── data/                # Raw knowledge base PDF documents (*.pdf)
├── requirements.txt     # Python dependencies for deployment
├── pyproject.toml       # Package definition and build metadata
└── render.yaml          # Render web service deployment configuration
```

---

## Quick Start (Local Development)

### Prerequisites
- Python `3.10+`
- A valid Google Gemini API Key (`GEMINI_API_KEY`)

### 1. Installation
Create and activate a virtual environment, then install requirements:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env` and set your `GEMINI_API_KEY`:
```bash
cp .env.example .env
```
In `.env`:
```env
GEMINI_API_KEY=AIzaSy...
```

### 3. Run FastAPI Server
```bash
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
On startup, the backend will verify if `data/chroma_db` exists. If not, it will automatically read the PDFs in `data/`, generate embeddings, and store them locally.

API documentation (`Swagger UI`) will be available at `http://127.0.0.1:8000/docs`.

---

## Testing

Run the automated test suite (`unit` + `security` guardrails) using `pytest`:
```bash
python -m pytest tests/ -v
```

---

## Render Deployment

This repository is pre-configured for instant deployment on **Render** as a Python Web Service via `render.yaml`.

### Steps:
1. Connect this GitHub repository (`Agentic-Placement-RAG-Backend`) to your Render account as a **Web Service**.
2. Render will automatically pick up settings from `render.yaml`:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
3. Under the Render **Environment Variables** settings, add your `GEMINI_API_KEY` secret.
4. Trigger deploy. The backend will start up, automatically index the PDFs inside `data/` on first launch, and serve RAG queries cleanly.
