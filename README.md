<div align="center">

# 🤖 Agentic Placement RAG — Backend

### Production-grade, security-hardened Agentic RAG engine for technical placement prep

*Hybrid retrieval · HyDE query expansion · Cross-encoder reranking · Multi-layer security · Real-time observability*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-8A2BE2)](https://www.trychroma.com/)
[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?logo=render&logoColor=white)](https://agentic-placement-rag.onrender.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#-license)

[🚀 Live Backend](https://agentic-placement-rag.onrender.com/) · [🎨 Frontend Repo](https://github.com/Snigdha-Gayathri/Agentic-Placement-RAG-Frontend) · [📖 Frontend README](../Agentic-Placement-RAG-Frontend/README.md)

</div>

---

## 🎯 Overview

The **Agentic Placement RAG Backend** is a `FastAPI + ChromaDB + Gemini 2.5 Flash` service that powers an AI assistant for technical placement interview prep (DSA, System Design, Behavioral, and Company-Specific Q&A). It combines **hybrid retrieval** (dense + BM25), **HyDE query expansion**, **cross-encoder reranking**, and a **six-layer security pipeline**, streaming every pipeline stage back to the frontend over Server-Sent Events (SSE) for full transparency.

> On first boot, if no vector index exists, the service automatically ingests the PDFs in `data/`, builds embeddings, and initializes the BM25 corpus — **zero manual database setup.**

---

## ✨ Features

<details open>
<summary><b>🤖 Agentic RAG</b></summary>

- Autonomous multi-stage planning cycle per query (rewrite → expand → retrieve → rerank → ground → generate)
- Agent-driven decisions on whether to re-retrieve or expand context
- Modular `core/agent` engine decoupled from transport layer
</details>

<details open>
<summary><b>🔍 Hybrid Retrieval</b></summary>

- Dense semantic search via Gemini `text-embedding-004`
- Sparse keyword search via `rank-bm25`
- Score fusion for precision + recall balance
</details>

<details open>
<summary><b>🧠 HyDE Query Expansion</b></summary>

- Generates a hypothetical answer document to enrich sparse or ambiguous queries before retrieval
- Improves recall on underspecified or jargon-heavy questions
</details>

<details open>
<summary><b>⚡ Cross-Encoder Reranking</b></summary>

- `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks top candidates for peak precision
- Configurable similarity threshold filters low-relevance chunks
</details>

<details open>
<summary><b>🛡️ Security Layer</b></summary>

- Input validation, prompt injection detection, sliding-window rate limiting
- Retrieval guard, context sanitization, grounding/hallucination verification
</details>

<details open>
<summary><b>📊 Developer Dashboard</b></summary>

- Live SSE pipeline-stage tracking (`/api/pipeline-status/{request_id}`)
- Per-request dashboard payload (`/api/dashboard/{request_id}`)
</details>

<details open>
<summary><b>📈 Observability Metrics</b></summary>

- Faithfulness, relevancy, and latency tracking via the `evaluation/` engine
</details>

<details open>
<summary><b>☁️ Google Drive Sync</b></summary>

- Knowledge base PDFs can be ingested from `data/` and re-indexed automatically on redeploy
</details>

---

## 🏗️ Architecture

```
                   User
                     │
                     ▼
        React + Vite Frontend
                     │
                     ▼
            FastAPI Backend
                     │
 ┌───────────────────┼───────────────────┐
 │                   │                   │
 ▼                   ▼                   ▼
Security         Agent Router      Dashboard
 │                   │                   │
 └──────────────┬────┴──────────────┬────┘
                ▼
        Hybrid Retriever
        Dense + BM25 + HyDE
                │
                ▼
       Cross-Encoder Reranker
                │
                ▼
          Gemini 2.5 Flash
                │
                ▼
          Streaming Response
```

---

## 🧩 Folder Structure

```
Agentic-Placement-RAG-Backend/
├── backend/
│   ├── app/             # FastAPI app, routes, Pydantic models, SecureRAGService
│   ├── config/          # Pipeline settings & YAML configuration
│   ├── core/            # RAG engine: agent, chunking, embeddings, hyde,
│   │                    #   memory, reranking, retrieval, vector_store
│   ├── evaluation/      # Faithfulness / relevancy / latency metrics engine
│   ├── ingestion/       # PDF loaders, chunk enhancers, metadata extractors
│   └── observability/   # Live stage trackers & dashboard data structures
├── security/            # Production security guardrail package
├── tests/               # Unit + guardrail verification suite
├── data/                # Raw knowledge base PDFs
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Package metadata
└── render.yaml          # Render deployment config
```

---

## 🛠️ Tech Stack

**AI / ML**

![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-4285F4?logo=googlegemini&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-8A2BE2)
![BM25](https://img.shields.io/badge/rank--bm25-Sparse_Retrieval-orange)
![CrossEncoder](https://img.shields.io/badge/Cross--Encoder-MiniLM--L6--v2-yellow)

**Backend**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-Validation-E92063)
![SSE](https://img.shields.io/badge/SSE-Streaming-red)

**Security**

![Guardrails](https://img.shields.io/badge/Security-6_Layer_Guardrails-critical)
![RateLimit](https://img.shields.io/badge/Rate_Limiting-Sliding_Window-lightgrey)

**Deployment**

![Render](https://img.shields.io/badge/Render-Web_Service-46E3B7?logo=render&logoColor=white)

**Tooling**

![Pytest](https://img.shields.io/badge/Testing-Pytest-0A9EDC?logo=pytest&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI-499848)

---

## ⚙️ RAG Pipeline

```
User Query
      │
      ▼
Input Validation
      │
      ▼
Prompt Injection Detection
      │
      ▼
Query Rewriting
      │
      ▼
HyDE Expansion
      │
      ▼
Hybrid Retrieval
(Dense + BM25)
      │
      ▼
Cross-Encoder Reranking
      │
      ▼
Context Grounding
      │
      ▼
Gemini 2.5 Flash
      │
      ▼
Output Validation
      │
      ▼
Streaming Response
```

---

## 🔐 Security Pipeline

| # | Guard | Purpose |
|---|-------|---------|
| 1 | **Input Validator** | Normalizes queries, limits token length, filters forbidden patterns |
| 2 | **Prompt Injection Detector** | Blocks jailbreak vectors and heuristic injection attempts |
| 3 | **Sliding Window Rate Limiter** | Protects endpoints against DDoS and excessive token consumption |
| 4 | **Retrieval Guard** | Filters out non-relevant candidate chunks via similarity threshold |
| 5 | **Context Sanitizer** | Strips prompt-injection payloads embedded in retrieved documents |
| 6 | **Grounding & Hallucination Guard** | Verifies LLM output against retrieved chunks for factual fidelity |

Every guard emits a pass/fail verdict visible on the frontend's Security Events panel in real time.

---

## 📈 Metrics

| Metric | Description | Source |
|--------|-------------|--------|
| Faithfulness | How well the answer is grounded in retrieved chunks | `evaluation/` |
| Relevancy | Semantic alignment between query and answer | `evaluation/` |
| Retrieval Latency | Time spent in hybrid retrieval + reranking | `observability/` |
| Generation Latency | Time spent in Gemini generation | `observability/` |
| Guardrail Verdicts | Pass/fail per security layer | `security/` |
| Context Window Usage | Tokens consumed vs. budget | `observability/` |

---

## 🚀 Deployment

```
React (Render Static)
          │
          ▼
        HTTPS
          │
          ▼
FastAPI (Render Web Service)
          │
          ▼
      Gemini API
          │
          ▼
    Google Drive / data/
```

Pre-configured for Render via `render.yaml`:

1. Connect this repo to Render as a **Web Service**.
2. Render applies:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
3. Set `GEMINI_API_KEY` under Environment Variables.
4. Deploy — the backend indexes `data/` PDFs on first boot and serves RAG queries immediately.

---

## 📚 API Documentation

| Endpoint | Method | Description |
|----------|--------|--------------|
| `/api/query` | `POST` | Submit a query, returns streamed agentic RAG response |
| `/api/pipeline-status/{request_id}` | `GET` (SSE) | Live pipeline stage updates for a given request |
| `/api/dashboard/{request_id}` | `GET` | Full observability payload (latencies, scores, verdicts) |
| `/docs` | `GET` | Interactive Swagger UI (auto-generated by FastAPI) |

> Full request/response schemas are available via the auto-generated Swagger UI at `/docs` once the server is running.

---

## 💻 Local Development

**Prerequisites:** Python `3.10+`, a valid `GEMINI_API_KEY`

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# then set GEMINI_API_KEY=AIzaSy... in .env

# 4. Run the server
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Swagger UI: `http://127.0.0.1:8000/docs`
On startup, if `data/chroma_db` doesn't exist, the backend automatically indexes PDFs in `data/`.

---

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

| Suite | Covers |
|-------|--------|
| **Security tests** | Guardrail behavior — injection detection, rate limiting, sanitization |
| **API tests** | Endpoint contracts and response schemas |
| **Retrieval tests** | Hybrid search accuracy, reranking, HyDE expansion |

---

## 🔗 Related

- 🎨 [Frontend README](https://github.com/Snigdha-Gayathri/Agentic-Placement-RAG-Frontend) — UI, dashboard, and developer experience
- 🌐 [Live Demo](https://agentic-placement-rag.onrender.com/)

---

## 📄 License

MIT — see `LICENSE` for details.
