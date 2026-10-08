"""FastAPI entrypoint for secure Agentic RAG backend with developer dashboard observability."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Ensure project root is present in sys.path for deterministic module resolution across environments
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from security import AuthContext, Role, get_auth_context, require_admin, require_user
from .models import (
    ChatRequest,
    ChatResponse,
    FeatureToggleConfig,
    HealthResponse,
    ReindexResponse,
    SecurityStatusResponse,
)
from .service import DASHBOARD_STORE, SESSION_HISTORY, TRACKER_STORE, SecureRAGService, build_request_context
from backend.evaluation.ablation_runner import AblationRunner
from backend.evaluation.benchmark_dataset import get_benchmark_dataset

app = FastAPI(title="Placement RAG Agent Secure & Observable API", version="2.5.0")

# Restrict CORS to configured origins or allow permissive local dev
_allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if _allowed_origins_env:
    _origins = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

service = SecureRAGService()
ablation_runner = AblationRunner(service)


class DynamicThresholdConfig(BaseModel):
    top_k: int = Field(default=15, ge=1, le=50)
    reranker_top_n: int = Field(default=8, ge=1, le=25)
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.12, ge=0.0, le=1.0)
    reranker_threshold: float = Field(default=0.15, ge=0.0, le=1.0)


class CustomEvaluationRequest(BaseModel):
    config_name: str = "Custom Configuration"
    sample_size: int = Field(default=15, ge=1, le=50)
    toggles: dict[str, bool] = Field(default_factory=dict)
    numeric_params: dict[str, Any] = Field(default_factory=dict)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Public healthcheck endpoint."""
    return service.health()


@app.get("/security/status", response_model=SecurityStatusResponse)
async def security_status() -> SecurityStatusResponse:
    """Public security guardrail status."""
    return service.security_status()


@app.get("/api/decision/status")
async def decision_status() -> dict[str, Any]:
    """Safe, read-only status of the pluggable Jev / Decision Backend integration."""
    return service.get_decision_status()



@app.post("/reindex", response_model=ReindexResponse)
async def reindex(auth: AuthContext = Depends(require_admin)) -> ReindexResponse:
    """Protected admin endpoint to trigger vector store and BM25 reindexing."""
    try:
        return await service.reindex()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Index rebuild failed.") from exc


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    raw_request: Request,
    auth: AuthContext = Depends(require_user),
) -> ChatResponse:
    """Protected endpoint for authenticated users to execute RAG queries."""
    client_ip = raw_request.client.host if raw_request.client else "127.0.0.1"
    ctx = build_request_context(
        client_ip=client_ip,
        request_id=request.request_id,
        user_id=auth.user_id,
        role=auth.role,
    )
    try:
        return await service.process(request, ctx)
    except PermissionError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal processing error.") from exc


@app.post("/chat/stream")
@app.post("/api/chat/stream")
async def chat_stream(
    request: ChatRequest,
    raw_request: Request,
    auth: AuthContext = Depends(require_user),
) -> StreamingResponse:
    """Server-Sent Events (SSE) streaming live pipeline execution stages, progressive answer tokens, and final dashboard payload."""
    client_ip = raw_request.client.host if raw_request.client else "127.0.0.1"
    ctx = build_request_context(
        client_ip=client_ip,
        request_id=request.request_id,
        user_id=auth.user_id,
        role=auth.role,
    )

    async def event_generator():
        try:
            async for event in service.process_stream(request, ctx):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/pipeline-status/{request_id}")
async def pipeline_status(
    request_id: str,
    auth: AuthContext = Depends(require_user),
) -> StreamingResponse:
    """Server-Sent Events (SSE) streaming live RAG pipeline progress stages."""

    async def event_generator():
        tracker = TRACKER_STORE.get(request_id)
        if not tracker:
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
async def get_dashboard(
    request_id: str,
    auth: AuthContext = Depends(require_user),
) -> dict[str, Any]:
    """Retrieve developer dashboard data for an authorized caller."""
    data = DASHBOARD_STORE.get_authorized(
        request_id,
        requester_id=auth.user_id,
        is_admin=(auth.role == Role.ADMIN),
    )
    if not data:
        rec = SESSION_HISTORY.get_query(request_id)
        if rec:
            return rec
        raise HTTPException(status_code=404, detail="Dashboard data not found for request ID.")
    return data.to_dict()


@app.get("/config")
async def get_config(auth: AuthContext = Depends(require_user)) -> dict[str, bool]:
    """Retrieve active feature toggles."""
    return service.get_feature_toggles()


@app.post("/config")
async def update_config(
    toggles: FeatureToggleConfig,
    auth: AuthContext = Depends(require_admin),
) -> dict[str, bool]:
    """Admin-only endpoint to update active feature toggles dynamically."""
    return service.update_feature_toggles(toggles.model_dump())


@app.get("/vector-db/stats")
async def vector_db_stats(auth: AuthContext = Depends(require_admin)) -> dict[str, Any]:
    """Admin-only endpoint to retrieve vector database statistics."""
    stats = service.chroma_store.get_stats()
    return stats.to_dict()


@app.get("/metrics/{request_id}")
async def get_metrics(
    request_id: str,
    auth: AuthContext = Depends(require_user),
) -> dict[str, Any]:
    """Retrieve computed RAG observability metrics for an authorized request."""
    data = DASHBOARD_STORE.get_authorized(
        request_id,
        requester_id=auth.user_id,
        is_admin=(auth.role == Role.ADMIN),
    )
    if not data:
        raise HTTPException(status_code=404, detail="Metrics not found for request ID.")
    return {
        "request_id": request_id,
        "total_latency_ms": data.total_latency_ms,
        "retrieval_latency_ms": data.retrieval_info.retrieval_latency_ms,
        "reranking_latency_ms": data.reranking_info.reranking_latency_ms,
        "generation_latency_ms": data.generation_info.generation_latency_ms,
        "metrics": data.computed_metrics,
    }


# ── OBSERVABILITY & BENCHMARK API ENDPOINTS ──────────────────────────────────

@app.get("/api/dashboard/history")
async def get_dashboard_history(limit: int = 50, auth: AuthContext = Depends(require_user)) -> list[dict[str, Any]]:
    """Retrieve recent session query history table."""
    return SESSION_HISTORY.get_history_summary(limit=limit)


@app.get("/api/dashboard/query/{request_id}")
async def get_dashboard_query_record(request_id: str, auth: AuthContext = Depends(require_user)) -> dict[str, Any]:
    """Retrieve full dashboard data for an individual historical query from session history."""
    rec = SESSION_HISTORY.get_query(request_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Query record not found in session history.")
    return rec


@app.get("/api/evaluation/cumulative")
async def get_evaluation_cumulative(auth: AuthContext = Depends(require_user)) -> dict[str, Any]:
    """Retrieve live cumulative retrieval evaluation metrics and security analytics."""
    return {
        "cumulative_metrics": SESSION_HISTORY.get_cumulative_metrics(),
        "security_analytics": SESSION_HISTORY.get_security_analytics(),
    }


@app.get("/api/dashboard/aggregate")
async def get_dashboard_aggregate(auth: AuthContext = Depends(require_user)) -> dict[str, Any]:
    """Retrieve aggregate telemetry metrics, percentiles (P50/P90/P95/P99), route distribution, and efficiency stats."""
    return SESSION_HISTORY.get_aggregate_metrics()


@app.post("/api/evaluation/ablation")
async def run_ablation(sample_size: int = 15, auth: AuthContext = Depends(require_user)) -> list[dict[str, Any]]:
    """Run full 6-configuration ablation study against the benchmark dataset."""
    return await ablation_runner.run_ablation_study(sample_size=sample_size)


@app.post("/api/evaluation/run")
async def run_evaluation_custom(req: CustomEvaluationRequest, auth: AuthContext = Depends(require_user)) -> dict[str, Any]:
    """Run custom configuration benchmark against the benchmark dataset."""
    return await ablation_runner.evaluate_custom_config(
        config_name=req.config_name,
        toggles=req.toggles,
        numeric_params=req.numeric_params,
        sample_size=req.sample_size,
    )


@app.get("/api/config/thresholds")
async def get_thresholds(auth: AuthContext = Depends(require_user)) -> dict[str, Any]:
    """Get active numeric thresholds (Top-K, Reranker Top-N, Dense/BM25 weights, thresholds)."""
    return service.dynamic_params


@app.post("/api/config/thresholds")
async def update_thresholds(
    thresholds: DynamicThresholdConfig,
    auth: AuthContext = Depends(require_user),
) -> dict[str, Any]:
    """Update dynamic numeric thresholds on the live backend service."""
    return service.update_dynamic_params(thresholds.model_dump())


@app.get("/api/corpus/stats")
async def get_corpus_stats(auth: AuthContext = Depends(require_user)) -> dict[str, Any]:
    """Inspect the 59 indexed companies and corpus statistics."""
    from backend.core.retrieval.router import QueryRouter
    companies = sorted(set(QueryRouter.CANONICAL_COMPANIES.values()))
    topics = sorted({t.title() for t in QueryRouter.TOPICS})
    db_stats = service.chroma_store.get_stats()
    return {
        "total_companies": len(companies),
        "companies": companies,
        "total_topics": len(topics),
        "topics": topics,
        "total_indexed_chunks": db_stats.total_chunks,
        "collection_name": db_stats.collection_name,
        "embedding_dimension": db_stats.dimension,
    }


# ── ALIASES FOR FRONTEND COMPATIBILITY ────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def api_health() -> HealthResponse:
    return await health()


@app.get("/api/security/status", response_model=SecurityStatusResponse)
async def api_security_status() -> SecurityStatusResponse:
    return await security_status()


@app.post("/api/reindex", response_model=ReindexResponse)
async def api_reindex(auth: AuthContext = Depends(require_admin)) -> ReindexResponse:
    return await reindex(auth=auth)


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(
    request: ChatRequest,
    raw_request: Request,
    auth: AuthContext = Depends(require_user),
) -> ChatResponse:
    return await chat(request, raw_request, auth=auth)


@app.get("/api/pipeline-status/{request_id}")
async def api_pipeline_status(
    request_id: str,
    auth: AuthContext = Depends(require_user),
) -> StreamingResponse:
    return await pipeline_status(request_id, auth=auth)


@app.get("/api/dashboard/{request_id}")
async def api_get_dashboard(
    request_id: str,
    auth: AuthContext = Depends(require_user),
) -> dict[str, Any]:
    return await get_dashboard(request_id, auth=auth)


@app.get("/api/config")
async def api_get_config(auth: AuthContext = Depends(require_user)) -> dict[str, bool]:
    return await get_config(auth=auth)


@app.post("/api/config")
async def api_update_config(
    toggles: FeatureToggleConfig,
    auth: AuthContext = Depends(require_admin),
) -> dict[str, bool]:
    return await update_config(toggles, auth=auth)


@app.get("/api/vector-db/stats")
async def api_get_vector_db_stats(auth: AuthContext = Depends(require_admin)) -> dict[str, Any]:
    return await vector_db_stats(auth=auth)


@app.get("/api/metrics/{request_id}")
async def api_get_metrics(
    request_id: str,
    auth: AuthContext = Depends(require_user),
) -> dict[str, Any]:
    return await get_metrics(request_id, auth=auth)
