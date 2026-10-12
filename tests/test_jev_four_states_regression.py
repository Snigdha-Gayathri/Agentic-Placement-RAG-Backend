import pytest
from unittest.mock import AsyncMock
from backend.core.vector_store.base import ScoredChunk
from backend.core.decision.jev import JevDecisionBackend
from backend.core.decision.circuit_breaker import CircuitBreaker, CircuitState
from backend.core.decision.models import (
    RetrievalMode,
    HopMode,
    RouteType,
)
from typesafe_sdk import SystemOneResponse, ChoiceAnswer, Usage


@pytest.mark.anyio
async def test_jev_state_1_configured_credentials_success():
    """State 1: Valid credentials, successful System One response.
    Verify backend, requested_backend, fallback telemetry, and log records.
    """
    mock_resp = SystemOneResponse(
        model="jev-1.13-free",
        usage=Usage(input_tokens=150, output_tokens=25),
        answers={
            "retrieval_mode": ChoiceAnswer(choice="hybrid", confidence=0.92, probabilities={"hybrid": 0.92, "bm25": 0.05, "dense": 0.03}),
            "hop_mode": ChoiceAnswer(choice="single_hop", confidence=0.95, probabilities={"single_hop": 0.95, "multi_hop": 0.05}),
            "query_transformation": ChoiceAnswer(choice="none", confidence=0.99, probabilities={"none": 0.99, "rewrite": 0.01, "hyde": 0.0}),
            "route_type": ChoiceAnswer(choice="retrieval_required", confidence=0.98, probabilities={"retrieval_required": 0.98, "metadata_direct": 0.02, "out_of_domain": 0.0}),
        },
    )

    mock_client = AsyncMock()
    mock_client.system_one = AsyncMock(return_value=mock_resp)

    backend = JevDecisionBackend(api_key="mock_api_key_123")
    backend._client = mock_client

    plan = await backend.plan_retrieval(
        query="Google SDE interview rounds",
        profile={"detected_companies": ["Google"]},
        query_id="q1",
    )
    assert plan.backend == "jev"
    assert plan.requested_backend == "jev"
    assert plan.fallback_occurred is False
    assert plan.fallback_reason is None
    assert plan.confidence == 0.92
    assert plan.retrieval_mode == RetrievalMode.HYBRID
    assert plan.hop_mode == HopMode.SINGLE_HOP
    assert plan.route_type == RouteType.SINGLE_HOP
    assert plan.log_record is not None
    assert plan.log_record.success is True
    assert plan.log_record.input_tokens == 150
    assert plan.log_record.output_tokens == 25


@pytest.mark.anyio
async def test_jev_state_2_missing_credentials_fallback():
    """State 2: Missing API key.
    Verify graceful delegation to heuristic with UNCONFIGURED_CREDENTIALS.
    """
    backend = JevDecisionBackend(api_key="")
    assert backend.is_configured is False

    # 1. Retrieval planning
    plan = await backend.plan_retrieval(query="Google SDE interview rounds", query_id="q2")
    assert plan.backend == "heuristic"
    assert plan.requested_backend == "jev"
    assert plan.fallback_occurred is True
    assert plan.fallback_reason == "UNCONFIGURED_CREDENTIALS"
    assert plan.confidence == 0.0

    # 2. Evidence sufficiency
    fake_chunk = ScoredChunk(chunk_id="c1", text="Google interview guide SDE rounds", metadata={"company": "Google"}, score=1.0)
    suff = await backend.evaluate_evidence_sufficiency(
        query="Google SDE interview rounds",
        chunks=[fake_chunk],
        target_company="Google",
        query_id="q2",
    )
    assert suff.backend == "heuristic"
    assert suff.requested_backend == "jev"
    assert suff.fallback_occurred is True
    assert suff.fallback_reason == "UNCONFIGURED_CREDENTIALS"

    # 3. Security
    sec = await backend.evaluate_security(query="Google interview rounds", query_id="q2")
    assert sec.backend in ["jev", "heuristic"]
    assert sec.fallback_occurred is True
    assert "not configured" in sec.fallback_reason or "UNCONFIGURED" in sec.fallback_reason


@pytest.mark.anyio
async def test_jev_state_3_open_circuit_breaker_fast_fail():
    """State 3: Circuit breaker tripped to OPEN.
    Verify fast-fail delegation to heuristic without calling client.
    """
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
    cb.record_failure("HTTP 503 Service Unavailable")
    cb.record_failure("HTTP 503 Service Unavailable")
    assert cb.state == CircuitState.OPEN

    mock_client = AsyncMock()
    backend = JevDecisionBackend(api_key="mock_key", circuit_breaker=cb)
    backend._client = mock_client

    plan = await backend.plan_retrieval(query="Amazon leadership principles", query_id="q3")
    mock_client.system_one.assert_not_called()

    assert plan.backend == "heuristic"
    assert plan.requested_backend == "jev"
    assert plan.fallback_occurred is True
    assert "Circuit breaker is OPEN" in plan.fallback_reason
    assert plan.confidence == 0.0

    fake_chunk = ScoredChunk(chunk_id="c1", text="Amazon leadership customer obsession", metadata={"company": "Amazon"}, score=1.0)
    suff = await backend.evaluate_evidence_sufficiency(
        query="Amazon leadership principles",
        chunks=[fake_chunk],
        target_company="Amazon",
        query_id="q3",
    )
    mock_client.system_one.assert_not_called()
    assert suff.backend == "heuristic"
    assert suff.requested_backend == "jev"
    assert suff.fallback_occurred is True
    assert "Circuit breaker is OPEN" in suff.fallback_reason


@pytest.mark.anyio
async def test_jev_state_4_runtime_failure_circuit_breaker_recording():
    """State 4: Runtime API exception.
    Verify error categorization, graceful fallback, and circuit breaker recording.
    """
    mock_client = AsyncMock()
    mock_client.system_one.side_effect = TimeoutError("Request timed out after 10.0s")

    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
    backend = JevDecisionBackend(api_key="mock_key", circuit_breaker=cb)
    backend._client = mock_client

    plan = await backend.plan_retrieval(query="Meta system design", query_id="q4")
    assert plan.backend == "heuristic"
    assert plan.requested_backend == "jev"
    assert plan.fallback_occurred is True
    assert "API_TIMEOUT" in plan.fallback_reason

    # Circuit breaker should record the failure
    assert cb._failure_count == 1
    assert cb.state == CircuitState.CLOSED  # Threshold is 3
