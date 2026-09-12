"""End-to-End Production Acceptance Test Suite for Agentic Placement RAG."""

import uuid
import pytest

from backend.app.models import ChatRequest
from backend.app.service import SecureRAGService, RequestContext, build_request_context
from backend.core.agent.tools import DecomposeQueryTool, EvaluateEvidenceTool
from backend.core.retrieval.router import QueryRouter
from backend.evaluation.ablation_runner import AblationRunner
from backend.evaluation.benchmark_dataset import get_benchmark_dataset
from backend.evaluation.metrics import LatencyMetrics, RAGMetrics
from backend.observability.dashboard import (
    DASHBOARD_STORE,
    SESSION_HISTORY,
    DashboardData,
    QueryInfo,
    SessionHistoryStore,
)


@pytest.mark.anyio
async def test_deterministic_routing_metadata_direct():
    """Verify metadata queries are answered deterministically with 0 LLM calls."""
    router = QueryRouter()
    res = router.route("What companies are indexed in the knowledge base?")

    assert res.route_type == "METADATA_DIRECT"
    assert res.llm_required is False
    assert res.direct_response is not None
    assert "OpenAI" in res.direct_response
    assert "Anthropic" in res.direct_response
    assert "Google" in res.direct_response


@pytest.mark.anyio
async def test_deterministic_routing_out_of_domain():
    """Verify non-interview questions are rejected deterministically with 0 LLM calls."""
    router = QueryRouter()
    res = router.route("What is the weather forecast for tomorrow in Seattle?")

    assert res.route_type == "OUT_OF_DOMAIN"
    assert res.llm_required is False
    assert res.direct_response is not None
    assert "outside the scope" in res.direct_response.lower()


@pytest.mark.anyio
async def test_multi_company_query_decomposition():
    """Verify multi-company comparison query decomposition creates targeted subqueries."""
    tool = DecomposeQueryTool()
    res = tool.execute(
        "Compare the technical interview focus of OpenAI, Anthropic and Databricks.",
        detected_companies=["OpenAI", "Anthropic", "Databricks"],
    )

    assert len(res.output) >= 3
    assert any("OpenAI" in q for q in res.output)
    assert any("Anthropic" in q for q in res.output)
    assert any("Databricks" in q for q in res.output)


@pytest.mark.anyio
async def test_evidence_sufficiency_evaluation_unsupported_company():
    """Verify evidence evaluator marks unsupported company as UNSUPPORTED and not sufficient."""
    tool = EvaluateEvidenceTool()
    res = tool.execute(
        query="What coding questions does Figma ask?",
        chunks=[],
        target_company="Figma",
    )

    assert res.output.status == "UNSUPPORTED"
    assert res.output.is_sufficient is False


@pytest.mark.anyio
async def test_session_history_and_percentiles():
    """Verify SessionHistoryStore computes P50, P90, P95, and bypass statistics correctly."""
    store = SessionHistoryStore(max_history=50)

    # Insert 10 mock queries
    for i in range(1, 11):
        data = DashboardData(
            request_id=f"req_{i}",
            total_latency_ms=i * 20.0,
            query_info=QueryInfo(original_query=f"Query {i}", route_type="SINGLE_HOP" if i <= 7 else "METADATA_DIRECT"),
        )
        data.agent_info.llm_invoked = i <= 7
        data.agent_info.hops_count = 2 if i % 2 == 0 else 1
        store.record_query(data)

    agg = store.get_aggregate_metrics()
    assert agg["total_queries"] == 10
    assert agg["llm_calls_bypassed"] == 3
    assert agg["llm_bypass_rate_pct"] == 30.0
    assert agg["latency"]["p50_ms"] > 0
    assert agg["latency"]["p95_ms"] > agg["latency"]["p50_ms"]


@pytest.mark.anyio
async def test_benchmark_dataset_structure():
    """Verify curated benchmark dataset contains valid test cases across categories."""
    dataset = get_benchmark_dataset()
    assert len(dataset) >= 30
    assert any(c.category == "Simple Factual" for c in dataset)
    assert any(c.category == "Multi-Company Comparison" for c in dataset)
    assert any(c.category == "Unsupported Company" for c in dataset)
    assert any(c.category == "Out of Domain" for c in dataset)


@pytest.mark.anyio
async def test_ablation_runner_structure():
    """Verify AblationRunner defines 6 distinct pipeline configurations."""
    service = SecureRAGService()
    runner = AblationRunner(service)
    assert len(runner.ABLATION_CONFIGS) == 6
    config_names = [c["id"] for c in runner.ABLATION_CONFIGS]
    assert "dense_only" in config_names
    assert "bm25_only" in config_names
    assert "hybrid" in config_names
    assert "hybrid_rerank" in config_names
    assert "full_pipeline" in config_names


@pytest.mark.anyio
async def test_e2e_service_llm_bypass_metadata():
    """Verify SecureRAGService handles metadata query with 0 LLM calls and returns ChatResponse."""
    service = SecureRAGService()
    ctx = build_request_context(user_id="test_user")
    req = ChatRequest(query="What companies are in the database?")

    resp = await service.process(req, ctx)
    assert resp.pipeline_data.agent_info["llm_invoked"] is False
    assert "OpenAI" in resp.answer or "Amazon" in resp.answer
    assert resp.meta.chunk_count == 0


@pytest.mark.anyio
async def test_e2e_service_llm_bypass_out_of_domain():
    """Verify SecureRAGService handles out-of-domain query with 0 LLM calls."""
    service = SecureRAGService()
    ctx = build_request_context(user_id="test_user")
    req = ChatRequest(query="What is the weather in New York?")

    resp = await service.process(req, ctx)
    assert resp.pipeline_data.agent_info["llm_invoked"] is False
    assert "outside the scope" in resp.answer.lower() or "interview" in resp.answer.lower()
