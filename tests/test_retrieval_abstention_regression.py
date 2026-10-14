"""Regression test suite for retrieval regression, fallback preservation, and abstention deduplication.

Tests covered:
1. Supported-company list deduplication (canonical aliases like 'meta' and 'facebook' produce exactly one 'Meta')
2. An explicitly unsupported company (e.g. Palantir when no records exist, verifying clean abstention and notice)
3. Supported company query: "What ML questions does Meta ask?" (grounded, non-abstaining)
4. Supported company query: "What does JP Morgan ask in its interviews?" (grounded, non-abstaining)
5. Supported company query: "What does Google ask about System Design?" (grounded, non-abstaining)
6. Empty primary retrieval followed by successful company-filtered fallback (recovers evidence, grounds answer)
7. Empty primary retrieval and empty fallback (must abstain, must NOT accept General legacy chunks)
8. Fallback results with mismatched company metadata (must NOT be accepted as company evidence)
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from backend.app.models import ChatRequest
from backend.app.service import SecureRAGService, build_request_context
from backend.core.agent.executor import AgentExecutionResult
from backend.core.agent.planner import ReasoningTrace
from backend.core.retrieval.base import RetrievalResult
from backend.core.retrieval.router import QueryRouter
from backend.core.vector_store.base import ScoredChunk


async def _mock_grounded_generator(query: str, context: str, **kwargs) -> str:
    """Mock answer generator ensuring deterministic offline tests without external API delays."""
    return f"Based on verified interview records for {query}:\n{context[:300]}"


# ==============================================================================
# 1. Supported Company List Deduplication Tests
# ==============================================================================

def test_canonical_companies_deduplication_displays_at_most_15_unique():
    """Verify canonical company deduplication: 'Meta' must appear exactly once and at most 15 unique."""
    unique_companies = sorted(list(dict.fromkeys(QueryRouter.CANONICAL_COMPANIES.values()))[:15])

    assert len(unique_companies) <= 15
    assert len(unique_companies) == len(set(unique_companies)), "Duplicate companies detected in displayed list!"
    assert unique_companies.count("Meta") == 1, f"Expected Meta to appear exactly once, got {unique_companies.count('Meta')}"
    assert unique_companies == sorted(unique_companies), "Displayed company list must be sorted deterministically"
    assert "Google" in unique_companies
    assert "JP Morgan" in unique_companies


# ==============================================================================
# 2. Unsupported Company Abstention Tests
# ==============================================================================

@pytest.mark.anyio
async def test_unsupported_company_abstains_with_deduplicated_notice():
    """An unsupported company with 0 records (e.g. Palantir) must trigger Evidence Sufficiency Notice with deduplicated companies."""
    service = SecureRAGService()
    ctx = build_request_context(user_id="test_user")
    req = ChatRequest(query="What does Palantir ask in its interviews?")

    mock_empty_exec = AgentExecutionResult(
        chunks=[],
        trace=ReasoningTrace(),
        hops=[],
        sufficiency=None,
        total_latency_ms=10.0,
        state={},
        answer="",
    )

    # When no chunks exist in primary or fallback for Palantir
    with patch.object(service.agent_executor, "execute", new_callable=AsyncMock) as mock_exec, \
         patch.object(service.bm25_retriever, "retrieve", new_callable=AsyncMock) as mock_bm25, \
         patch.object(service.dense_retriever, "retrieve", new_callable=AsyncMock) as mock_dense:
        mock_exec.return_value = mock_empty_exec
        mock_bm25.return_value = RetrievalResult(chunks=[], retriever_type="bm25", latency_ms=1.0)
        mock_dense.return_value = RetrievalResult(chunks=[], retriever_type="dense", latency_ms=1.0)

        resp = await service.process(req, ctx)

    assert resp is not None
    assert "### Evidence Sufficiency Notice" in resp.answer
    assert "Palantir" in resp.answer
    assert "Supported indexed companies include" in resp.answer

    # Check that Meta appears only once in the abstention notice text
    notice_text = resp.answer
    supported_line = [line for line in notice_text.splitlines() if "Supported indexed companies include" in line]
    assert len(supported_line) > 0
    assert supported_line[0].count("Meta") == 1, "Meta appeared multiple times in the supported companies list!"


# ==============================================================================
# 3. Supported Company Retrieval & Answer Grounding Tests
# ==============================================================================

@pytest.mark.anyio
async def test_meta_ml_query_retrieves_and_synthesizes_grounded_answer():
    """'What ML questions does Meta ask?' must retrieve Meta chunks and synthesize without abstaining."""
    service = SecureRAGService()
    ctx = build_request_context(user_id="test_user")
    req = ChatRequest(query="What ML questions does Meta ask?")

    with patch.object(service.gemini, "generate_answer", side_effect=_mock_grounded_generator):
        resp = await service.process(req, ctx)

    assert resp is not None
    assert "### Evidence Sufficiency Notice" not in resp.answer
    assert resp.pipeline_data is not None

    retrieval_info = resp.pipeline_data.retrieval_info
    chunks = retrieval_info.get("retrieved_chunks", [])
    assert len(chunks) > 0, "Expected non-zero retrieved chunks for Meta"
    # Ensure all chunks are Meta-specific
    for c in chunks:
        comp = c.get("company") if isinstance(c, dict) else getattr(c, "company", "")
        assert comp.lower() == "meta", f"Expected Meta chunk, got company={comp}"
    # Verify answer grounding in retrieved source content
    assert "Meta" in resp.answer


@pytest.mark.anyio
async def test_jp_morgan_interview_query_retrieves_and_synthesizes():
    """'What does JP Morgan ask in its interviews?' must retrieve JP Morgan chunks and not abstain."""
    service = SecureRAGService()
    ctx = build_request_context(user_id="test_user")
    req = ChatRequest(query="What does JP Morgan ask in its interviews?")

    with patch.object(service.gemini, "generate_answer", side_effect=_mock_grounded_generator):
        resp = await service.process(req, ctx)

    assert resp is not None
    assert "### Evidence Sufficiency Notice" not in resp.answer
    assert resp.pipeline_data is not None

    retrieval_info = resp.pipeline_data.retrieval_info
    chunks = retrieval_info.get("retrieved_chunks", [])
    assert len(chunks) > 0, "Expected non-zero retrieved chunks for JP Morgan"
    for c in chunks:
        comp = c.get("company") if isinstance(c, dict) else getattr(c, "company", "")
        assert comp.lower() == "jp morgan", f"Expected JP Morgan chunk, got company={comp}"
    assert "JP Morgan" in resp.answer


@pytest.mark.anyio
async def test_google_system_design_query_retrieves_and_synthesizes():
    """'What does Google ask about System Design?' must retrieve Google chunks and not abstain."""
    service = SecureRAGService()
    ctx = build_request_context(user_id="test_user")
    req = ChatRequest(query="What does Google ask about System Design?")

    with patch.object(service.gemini, "generate_answer", side_effect=_mock_grounded_generator):
        resp = await service.process(req, ctx)

    assert resp is not None
    assert "### Evidence Sufficiency Notice" not in resp.answer
    assert resp.pipeline_data is not None

    retrieval_info = resp.pipeline_data.retrieval_info
    chunks = retrieval_info.get("retrieved_chunks", [])
    assert len(chunks) > 0, "Expected non-zero retrieved chunks for Google"
    for c in chunks:
        comp = c.get("company") if isinstance(c, dict) else getattr(c, "company", "")
        assert comp.lower() == "google", f"Expected Google chunk, got company={comp}"
    assert "Google" in resp.answer


# ==============================================================================
# 4. Empty Primary Retrieval followed by Successful Fallback
# ==============================================================================

@pytest.mark.anyio
async def test_empty_primary_retrieval_triggers_company_filtered_fallback():
    """When primary agent executor returns 0 candidates, fallback recovers company chunks and grounds answer."""
    service = SecureRAGService()
    ctx = build_request_context(user_id="test_user")
    req = ChatRequest(query="What ML questions does Meta ask?")

    # Simulate primary agent executor returning 0 chunks
    mock_empty_exec = AgentExecutionResult(
        chunks=[],
        trace=ReasoningTrace(),
        hops=[],
        sufficiency=None,
        total_latency_ms=10.0,
        state={},
        answer="",
    )

    with patch.object(service.agent_executor, "execute", new_callable=AsyncMock) as mock_exec, \
         patch.object(service.gemini, "generate_answer", side_effect=_mock_grounded_generator):
        mock_exec.return_value = mock_empty_exec
        resp = await service.process(req, ctx)

    # Primary returned 0, but BM25 fallback should have recovered Meta chunks
    assert resp is not None
    assert "### Evidence Sufficiency Notice" not in resp.answer
    retrieval_info = resp.pipeline_data.retrieval_info
    chunks = retrieval_info.get("retrieved_chunks", [])
    assert len(chunks) > 0
    for c in chunks:
        comp = c.get("company") if isinstance(c, dict) else getattr(c, "company", "")
        assert comp.lower() == "meta"
    assert "Meta" in resp.answer


# ==============================================================================
# 5. Empty Primary Retrieval and Empty Fallback (Must Abstain)
# ==============================================================================

@pytest.mark.anyio
async def test_empty_primary_and_empty_fallback_must_abstain_without_general_chunks():
    """When primary and company fallback both return 0 chunks, system must abstain and NOT use General chunks."""
    service = SecureRAGService()
    ctx = build_request_context(user_id="test_user")
    req = ChatRequest(query="What does Palantir ask in its interviews?")

    mock_empty_exec = AgentExecutionResult(
        chunks=[],
        trace=ReasoningTrace(),
        hops=[],
        sufficiency=None,
        total_latency_ms=10.0,
        state={},
        answer="",
    )

    with patch.object(service.agent_executor, "execute", new_callable=AsyncMock) as mock_exec, \
         patch.object(service.bm25_retriever, "retrieve", new_callable=AsyncMock) as mock_bm25, \
         patch.object(service.dense_retriever, "retrieve", new_callable=AsyncMock) as mock_dense:
        mock_exec.return_value = mock_empty_exec
        mock_bm25.return_value = RetrievalResult(chunks=[], retriever_type="bm25", latency_ms=1.0)
        mock_dense.return_value = RetrievalResult(chunks=[], retriever_type="dense", latency_ms=1.0)

        resp = await service.process(req, ctx)

    assert resp is not None
    assert "### Evidence Sufficiency Notice" in resp.answer
    assert "Palantir" in resp.answer
    # Ensure no General chunks leaked into retrieved chunks
    retrieval_info = resp.pipeline_data.retrieval_info
    chunks = retrieval_info.get("retrieved_chunks", [])
    assert len(chunks) == 0, "No chunks should be present when abstaining on unsupported company"


# ==============================================================================
# 6. Fallback with Mismatched Company Metadata (Must Be Rejected)
# ==============================================================================

@pytest.mark.anyio
async def test_fallback_with_mismatched_company_metadata_is_rejected():
    """If fallback produces chunks labeled 'General' or another company, strict post-filter must reject them."""
    service = SecureRAGService()
    ctx = build_request_context(user_id="test_user")
    req = ChatRequest(query="What does Meta ask in its interviews?")

    mock_empty_exec = AgentExecutionResult(
        chunks=[],
        trace=ReasoningTrace(),
        hops=[],
        sufficiency=None,
        total_latency_ms=10.0,
        state={},
        answer="",
    )

    # Mock BM25 retriever returning mismatched 'General' chunks
    general_chunk = ScoredChunk(
        chunk_id="gen_01",
        text="General software engineering interview guidelines.",
        metadata={"company": "General", "topic": "general"},
        score=0.8,
    )
    mock_bm25_res = RetrievalResult(
        chunks=[general_chunk],
        retriever_type="bm25",
        latency_ms=5.0,
    )

    with patch.object(service.agent_executor, "execute", new_callable=AsyncMock) as mock_exec, \
         patch.object(service.bm25_retriever, "retrieve", new_callable=AsyncMock) as mock_bm25, \
         patch.object(service.dense_retriever, "retrieve", new_callable=AsyncMock) as mock_dense:
        mock_exec.return_value = mock_empty_exec
        mock_bm25.return_value = mock_bm25_res
        mock_dense.return_value = RetrievalResult(chunks=[], retriever_type="dense", latency_ms=5.0)

        resp = await service.process(req, ctx)

    # General chunk must NOT be accepted as Meta evidence -> must abstain
    assert resp is not None
    assert "### Evidence Sufficiency Notice" in resp.answer
    assert "Meta" in resp.answer
