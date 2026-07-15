"""FastAPI entrypoint for secure Agentic RAG backend with developer dashboard observability."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .models import (
    ChatRequest,
    ChatResponse,
    FeatureToggleConfig,
    HealthResponse,
    ReindexResponse,
    SecurityStatusResponse,
)
from .service import DASHBOARD_STORE, TRACKER_STORE, SecureRAGService, build_request_context

app = FastAPI(title="Placement RAG Agent Secure & Observable API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = SecureRAGService()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return service.health()


@app.get("/security/status", response_model=SecurityStatusResponse)
async def security_status() -> SecurityStatusResponse:
    return service.security_status()


@app.post("/reindex", response_model=ReindexResponse)
async def reindex() -> ReindexResponse:
    try:
        return service.reindex()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Index rebuild failed.") from exc


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, raw_request: Request) -> ChatResponse:
    ctx = build_request_context(raw_request.client.host if raw_request.client else None, request.request_id)
    try:
        return await service.process(request, ctx)
    except PermissionError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal processing error.") from exc


@app.get("/pipeline-status/{request_id}")
async def pipeline_status(request_id: str) -> StreamingResponse:
    """Server-Sent Events (SSE) streaming live RAG pipeline progress stages."""

    async def event_generator():
        tracker = TRACKER_STORE.get(request_id)
        if not tracker:
            # If tracker not found or completed quickly, emit completed event
            yield f"data: {json.dumps({'status': 'completed', 'stages': []})}\n\n"
            return

        last_count = 0
        while True:
            stages = [s.to_dict() for s in tracker.get_stages()]
            if len(stages) != last_count or any(s["status"] == "running" for s in stages):
                payload = json.dumps({"request_id": request_id, "stages": stages})
                yield f"data: {payload}\n\n"
                last_count = len(stages)
            if any(s["name"] == "response_complete" and s["status"] == "completed" for s in stages):
                break
            await asyncio.sleep(0.15)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/dashboard/{request_id}")
async def get_dashboard(request_id: str) -> dict[str, Any]:
    """Retrieve comprehensive developer dashboard data for an executed query."""
    data = DASHBOARD_STORE.get(request_id)
    if not data:
        raise HTTPException(status_code=404, detail="Dashboard data not found for request ID.")
    return data.to_dict()


@app.get("/config")
async def get_config() -> dict[str, bool]:
    """Retrieve current feature toggles."""
    return service.get_feature_toggles()


@app.post("/config")
async def update_config(toggles: FeatureToggleConfig) -> dict[str, bool]:
    """Update active feature toggles dynamically without restart."""
    return service.update_feature_toggles(toggles.model_dump())


@app.get("/vector-db/stats")
async def vector_db_stats() -> dict[str, Any]:
    """Retrieve statistics about the underlying vector database."""
    stats = service.chroma_store.get_stats()
    return stats.to_dict()


@app.get("/metrics/{request_id}")
@app.get("/api/metrics/{request_id}")
async def get_metrics(request_id: str) -> dict[str, Any]:
    """Retrieve computed RAG observability metrics for a request."""
    data = DASHBOARD_STORE.get(request_id)
    if not data:
        raise HTTPException(status_code=404, detail="Metrics not found for request ID.")
    return {
        "request_id": request_id,
        "total_latency_ms": data.total_latency_ms,
        "retrieval_latency_ms": data.retrieval_info.retrieval_latency_ms,
        "reranking_latency_ms": data.reranking_info.reranking_latency_ms,
        "generation_latency_ms": data.generation_info.generation_latency_ms,
    }


# Add /api aliases for all main routes so frontend calls to /api/* work seamlessly
@app.get("/api/health", response_model=HealthResponse)
async def api_health() -> HealthResponse:
    return health()

@app.get("/api/security/status", response_model=SecurityStatusResponse)
async def api_security_status() -> SecurityStatusResponse:
    return security_status()

@app.post("/api/reindex", response_model=ReindexResponse)
async def api_reindex() -> ReindexResponse:
    return await reindex()

@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(request: ChatRequest, raw_request: Request) -> ChatResponse:
    return await chat(request, raw_request)

@app.get("/api/pipeline-status/{request_id}")
async def api_pipeline_status(request_id: str) -> StreamingResponse:
    return await pipeline_status(request_id)

@app.get("/api/dashboard/{request_id}")
async def api_get_dashboard(request_id: str) -> dict[str, Any]:
    return await get_dashboard(request_id)

@app.get("/api/config")
async def api_get_config() -> dict[str, bool]:
    return await get_config()

@app.post("/api/config")
async def api_update_config(toggles: FeatureToggleConfig) -> dict[str, bool]:
    return await update_config(toggles)

@app.get("/api/vector-db/stats")
async def api_get_vector_db_stats() -> dict[str, Any]:
    return await vector_db_stats()
