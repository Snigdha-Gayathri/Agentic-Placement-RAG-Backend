"""Typed models for structured decision-making backends in Agentic Placement RAG."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator


class RetrievalMode(str, Enum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


class HopMode(str, Enum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"


class QueryTransformation(str, Enum):
    NONE = "none"
    REWRITE = "rewrite"
    HYDE = "hyde"


class RouteType(str, Enum):
    SINGLE_HOP = "SINGLE_HOP"
    MULTI_HOP = "MULTI_HOP"
    METADATA_DIRECT = "METADATA_DIRECT"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    CONCEPTUAL = "CONCEPTUAL"


class DecisionLogRecord(BaseModel):
    """Structured telemetry record for every decision made by any backend."""

    query_id: str = ""
    decision_backend: str  # "heuristic", "jev"
    requested_backend: str = "heuristic"  # "heuristic", "jev"
    decision_type: str  # "retrieval_routing", "evidence_sufficiency", "security"
    selected_option: str
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = 1.0
    latency_ms: float = 0.0
    success: bool = True
    fallback_occurred: bool = False
    fallback_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


class RetrievalStrategyDecision(BaseModel):
    """Calibrated decision on how the retrieval system should operate."""

    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    hop_mode: HopMode = HopMode.SINGLE_HOP
    query_transformation: QueryTransformation = QueryTransformation.NONE
    route_type: RouteType = RouteType.SINGLE_HOP

    # Granular operation flags (compatible with AgentPlanner)
    bm25_needed: bool = True
    dense_needed: bool = True
    hybrid_needed: bool = True
    rrf_needed: bool = True
    embedding_needed: bool = True
    multi_hop_needed: bool = False
    rewrite_needed: bool = False
    hyde_needed: bool = False
    rerank_needed: bool = True
    chunk_enhancement_needed: bool = False

    metadata_filter_needed: bool = False
    metadata_filters: dict[str, Any] = Field(default_factory=dict)

    # Explanation strings
    reasoning: str = ""
    planned_operations: list[str] = Field(default_factory=list)
    skipped_operations: list[dict[str, str]] = Field(default_factory=list)

    # Probabilistic and telemetry metadata
    backend: str = "heuristic"
    requested_backend: str = "heuristic"
    confidence: float = 1.0
    probabilities: dict[str, float] = Field(default_factory=dict)
    latency_ms: float = 0.0
    fallback_occurred: bool = False
    fallback_reason: str | None = None
    log_record: DecisionLogRecord | None = None

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence {v} must be between 0.0 and 1.0")
        return round(v, 4)

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, v: dict[str, float]) -> dict[str, float]:
        for k, p in v.items():
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"Probability for {k} ({p}) must be in [0.0, 1.0]")
        return v


class SufficiencyDecision(BaseModel):
    """Calibrated decision on whether retrieved evidence is sufficient to answer."""

    is_sufficient: bool = True
    status: str = "SUFFICIENT"  # SUFFICIENT, PARTIALLY_SUFFICIENT, INSUFFICIENT, UNSUPPORTED
    score: float = 1.0
    confidence: float = 1.0
    probabilities: dict[str, float] = Field(default_factory=dict)
    missing_aspects: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    backend: str = "heuristic"
    requested_backend: str = "heuristic"
    latency_ms: float = 0.0
    fallback_occurred: bool = False
    fallback_reason: str | None = None
    log_record: DecisionLogRecord | None = None


class SecurityDecision(BaseModel):
    """Calibrated decision on whether a query contains prompt injection or adversarial intent."""

    is_safe: bool = True
    action: str = "allow"  # "allow", "block"
    category: str = "BENIGN"
    severity: str = "LOW"
    risk_score: float = 0.0
    confidence: float = 1.0
    probabilities: dict[str, float] = Field(default_factory=dict)
    user_safe_message: str = ""

    backend: str = "heuristic"
    requested_backend: str = "heuristic"
    latency_ms: float = 0.0
    fallback_occurred: bool = False
    fallback_reason: str | None = None
    log_record: DecisionLogRecord | None = None
