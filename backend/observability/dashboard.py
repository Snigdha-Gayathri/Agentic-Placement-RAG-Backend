"""Dashboard data aggregation for developer observability."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryInfo:
    """Information about the incoming query and its transformations."""

    original_query: str = ""
    standalone_query: str = ""
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    routing_decision: str = ""
    is_followup: bool = False


@dataclass
class RetrievedChunkInfo:
    """A single retrieved chunk with metadata and scores."""

    text: str = ""
    source: str = ""
    similarity_score: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)
    rank: int = 0
    was_rejected: bool = False
    rejection_reason: str = ""


@dataclass
class RetrievalInfo:
    """Retrieval stage details."""

    retriever_used: str = "dense"
    retrieval_latency_ms: float = 0.0
    retrieved_chunks: list[RetrievedChunkInfo] = field(default_factory=list)
    rejected_chunks: list[RetrievedChunkInfo] = field(default_factory=list)
    total_candidates: int = 0
    final_count: int = 0


@dataclass
class RerankingInfo:
    """Reranking stage details showing rank changes."""

    enabled: bool = False
    reranker_model: str = ""
    reranking_latency_ms: float = 0.0
    rank_changes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentInfo:
    """Agent reasoning trace for agentic RAG mode."""

    enabled: bool = False
    reasoning_trace: list[str] = field(default_factory=list)
    tool_selections: list[str] = field(default_factory=list)
    planning_steps: list[str] = field(default_factory=list)
    iterations: int = 0


@dataclass
class GenerationInfo:
    """LLM generation stage details."""

    system_prompt: str = ""
    user_prompt: str = ""
    raw_response: str = ""
    final_response: str = ""
    token_usage: dict[str, int] = field(default_factory=dict)
    context_window_chars: int = 0
    generation_latency_ms: float = 0.0


@dataclass
class SecurityInfo:
    """Security pipeline details."""

    injection_score: float = 0.0
    injection_action: str = "allow"
    discarded_chunks: int = 0
    rate_limited: bool = False
    output_safe: bool = True
    grounding_passed: bool = True
    hallucination_detected: bool = False


@dataclass
class HyDEInfo:
    """Hypothetical Document Embedding (HyDE) stage details."""

    enabled: bool = False
    hypothetical_document: str = ""
    generation_latency_ms: float = 0.0


@dataclass
class MemoryInfo:
    """Conversation memory details."""

    enabled: bool = False
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    rewritten_query: str = ""
    history_length: int = 0


@dataclass
class FeatureToggles:
    """Feature toggle snapshot for the request."""

    dense_retrieval: bool = True
    bm25_retrieval: bool = True
    hybrid_retrieval: bool = True
    cross_encoder_reranking: bool = True
    metadata_filtering: bool = True
    hyde: bool = False
    conversation_memory: bool = True
    query_rewriting: bool = True
    agent_planning: bool = True
    multi_hop_retrieval: bool = False
    chunk_enhancement: bool = True
    hybrid_search: bool = True
    reranking: bool = True
    agent_mode: bool = True


@dataclass
class DashboardData:
    """Aggregated data for the developer dashboard for a single request.

    Captures every stage's details: query analysis, retrieval, reranking,
    generation, security, and more. Keyed by request_id.
    """

    request_id: str = ""
    query_info: QueryInfo = field(default_factory=QueryInfo)
    retrieval_info: RetrievalInfo = field(default_factory=RetrievalInfo)
    reranking_info: RerankingInfo = field(default_factory=RerankingInfo)
    agent_info: AgentInfo = field(default_factory=AgentInfo)
    generation_info: GenerationInfo = field(default_factory=GenerationInfo)
    security_info: SecurityInfo = field(default_factory=SecurityInfo)
    hyde_info: HyDEInfo = field(default_factory=HyDEInfo)
    memory_info: MemoryInfo = field(default_factory=MemoryInfo)
    feature_toggles: FeatureToggles = field(default_factory=FeatureToggles)
    pipeline_stages: list[dict[str, Any]] = field(default_factory=list)
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize all dashboard data for JSON output."""
        return {
            "request_id": self.request_id,
            "total_latency_ms": self.total_latency_ms,
            "query_info": {
                "original_query": self.query_info.original_query,
                "standalone_query": self.query_info.standalone_query,
                "metadata_filters": self.query_info.metadata_filters,
                "routing_decision": self.query_info.routing_decision,
                "is_followup": self.query_info.is_followup,
            },
            "retrieval_info": {
                "retriever_used": self.retrieval_info.retriever_used,
                "retrieval_latency_ms": self.retrieval_info.retrieval_latency_ms,
                "total_candidates": self.retrieval_info.total_candidates,
                "final_count": self.retrieval_info.final_count,
                "retrieved_chunks": [
                    {
                        "text": c.text[:200],
                        "source": c.source,
                        "similarity_score": c.similarity_score,
                        "score": c.similarity_score,
                        "metadata": c.metadata,
                        "rank": c.rank,
                        "company": c.metadata.get("company", getattr(c, "company", "General")) if isinstance(c.metadata, dict) else getattr(c, "company", "General"),
                    }
                    for c in self.retrieval_info.retrieved_chunks
                ],
                "rejected_chunks": [
                    {
                        "text": c.text[:100],
                        "source": c.source,
                        "rejection_reason": c.rejection_reason,
                    }
                    for c in self.retrieval_info.rejected_chunks
                ],
            },
            "reranking_info": {
                "enabled": self.reranking_info.enabled,
                "reranker_model": self.reranking_info.reranker_model,
                "reranking_latency_ms": self.reranking_info.reranking_latency_ms,
                "rank_changes": self.reranking_info.rank_changes,
            },
            "agent_info": {
                "enabled": self.agent_info.enabled,
                "reasoning_trace": self.agent_info.reasoning_trace,
                "tool_selections": self.agent_info.tool_selections,
                "planning_steps": self.agent_info.planning_steps,
                "iterations": self.agent_info.iterations,
            },
            "generation_info": {
                "system_prompt": self.generation_info.system_prompt,
                "user_prompt": self.generation_info.user_prompt[:500],
                "context_window_chars": self.generation_info.context_window_chars,
                "generation_latency_ms": self.generation_info.generation_latency_ms,
                "token_usage": self.generation_info.token_usage,
            },
            "security_info": {
                "injection_score": self.security_info.injection_score,
                "injection_action": self.security_info.injection_action,
                "discarded_chunks": self.security_info.discarded_chunks,
                "output_safe": self.security_info.output_safe,
                "grounding_passed": self.security_info.grounding_passed,
                "hallucination_detected": self.security_info.hallucination_detected,
            },
            "hyde_info": {
                "enabled": self.hyde_info.enabled,
                "hypothetical_document": self.hyde_info.hypothetical_document[:300],
                "generation_latency_ms": self.hyde_info.generation_latency_ms,
            },
            "memory_info": {
                "enabled": self.memory_info.enabled,
                "history_length": self.memory_info.history_length,
                "rewritten_query": self.memory_info.rewritten_query,
            },
            "feature_toggles": {
                "dense_retrieval": self.feature_toggles.dense_retrieval,
                "bm25_retrieval": self.feature_toggles.bm25_retrieval,
                "hybrid_retrieval": self.feature_toggles.hybrid_retrieval,
                "cross_encoder_reranking": self.feature_toggles.cross_encoder_reranking,
                "metadata_filtering": self.feature_toggles.metadata_filtering,
                "hyde": self.feature_toggles.hyde,
                "conversation_memory": self.feature_toggles.conversation_memory,
                "query_rewriting": self.feature_toggles.query_rewriting,
                "agent_planning": self.feature_toggles.agent_planning,
                "multi_hop_retrieval": self.feature_toggles.multi_hop_retrieval,
                "chunk_enhancement": self.feature_toggles.chunk_enhancement,
                "hybrid_search": self.feature_toggles.hybrid_search,
                "reranking": self.feature_toggles.reranking,
                "agent_mode": self.feature_toggles.agent_mode,
            },
            "metrics": {
                "total_latency_ms": self.total_latency_ms,
                "retrieval_latency_ms": self.retrieval_info.retrieval_latency_ms,
                "reranking_latency_ms": self.reranking_info.reranking_latency_ms,
                "generation_latency_ms": self.generation_info.generation_latency_ms,
                "hyde_latency_ms": self.hyde_info.generation_latency_ms,
                "total_candidates": self.retrieval_info.total_candidates,
                "final_count": self.retrieval_info.final_count,
                "discarded_chunks": self.security_info.discarded_chunks,
                "injection_score": self.security_info.injection_score,
                "iterations": self.agent_info.iterations,
            },
            "pipeline_stages": self.pipeline_stages,
        }


class DashboardStore:
    """Thread-safe in-memory store for dashboard data, keyed by request_id.

    Maintains a bounded cache of recent request data. Older entries are
    evicted when the maximum size is exceeded.
    """

    def __init__(self, max_size: int = 500) -> None:
        self._store: dict[str, DashboardData] = {}
        self._order: list[str] = []
        self._max_size = max_size
        self._lock = threading.Lock()

    def store(self, request_id: str, data: DashboardData) -> None:
        """Store dashboard data for a request.

        Args:
            request_id: The unique request identifier.
            data: The aggregated dashboard data.
        """
        with self._lock:
            if request_id in self._store:
                self._store[request_id] = data
                return

            if len(self._order) >= self._max_size:
                oldest = self._order.pop(0)
                self._store.pop(oldest, None)

            self._store[request_id] = data
            self._order.append(request_id)

    def get(self, request_id: str) -> DashboardData | None:
        """Retrieve dashboard data for a request.

        Args:
            request_id: The request ID to look up.

        Returns:
            DashboardData if found, None otherwise.
        """
        with self._lock:
            return self._store.get(request_id)

    def list_requests(self, limit: int = 50) -> list[str]:
        """List recent request IDs.

        Args:
            limit: Maximum number of IDs to return.

        Returns:
            List of request IDs, most recent first.
        """
        with self._lock:
            return list(reversed(self._order[-limit:]))

    @property
    def size(self) -> int:
        """Current number of stored requests."""
        with self._lock:
            return len(self._store)
