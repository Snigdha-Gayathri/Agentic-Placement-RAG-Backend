"""Dashboard data aggregation and telemetry store for developer observability."""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryInfo:
    """Information about the incoming query and its transformations."""

    original_query: str = ""
    standalone_query: str = ""
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    routing_decision: str = ""
    route_type: str = "SINGLE_HOP"
    is_followup: bool = False
    detected_companies: list[str] = field(default_factory=list)
    detected_topics: list[str] = field(default_factory=list)


@dataclass
class RetrievedChunkInfo:
    """A single retrieved chunk with metadata and ranking scores."""

    text: str = ""
    source: str = ""
    chunk_id: str = ""
    similarity_score: float = 0.0
    cross_encoder_score: float = 0.0
    original_rank: int = 0
    new_rank: int = 0
    rank_change: int = 0
    retrieval_source: str = "dense"
    is_selected: bool = True
    in_context: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    was_rejected: bool = False
    rejection_reason: str = ""

    def __post_init__(self) -> None:
        if self.text and len(self.text) > 300:
            self.text = self.text[:300] + "..."


@dataclass
class RetrievalInfo:
    """Retrieval stage details."""

    retriever_used: str = "hybrid"
    retrieval_latency_ms: float = 0.0
    retrieved_chunks: list[RetrievedChunkInfo] = field(default_factory=list)
    rejected_chunks: list[RetrievedChunkInfo] = field(default_factory=list)
    total_candidates: int = 0
    final_count: int = 0
    pre_reranker_candidates: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RerankingInfo:
    """Reranking stage details showing rank changes and score improvements."""

    enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranking_latency_ms: float = 0.0
    rank_changes: list[dict[str, Any]] = field(default_factory=list)
    pre_reranker_mrr: float | None = None
    post_reranker_mrr: float | None = None
    reranker_improvement: float | None = None


@dataclass
class AgentInfo:
    """Agent reasoning trace and multi-hop execution telemetry."""

    enabled: bool = True
    route_type: str = "SINGLE_HOP"
    llm_invoked: bool = True
    llm_invocation_reason: str = "Synthesis required"
    llm_bypass_reason: str = ""
    hops_count: int = 1
    hops_data: list[dict[str, Any]] = field(default_factory=list)
    reasoning_trace: list[dict[str, Any]] = field(default_factory=list)
    tool_selections: list[str] = field(default_factory=list)
    planning_steps: list[str] = field(default_factory=list)
    iterations: int = 1


@dataclass
class GenerationInfo:
    """LLM generation stage details with native streaming telemetry."""

    system_prompt: str = ""
    user_prompt: str = ""
    raw_response: str = ""
    final_response: str = ""
    token_usage: dict[str, int] = field(default_factory=dict)
    tokens_generated: int = 0
    generation_latency_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    streamed_chunks_count: int = 0

    def __post_init__(self) -> None:
        if self.user_prompt and len(self.user_prompt) > 500:
            self.user_prompt = self.user_prompt[:500] + "..."
        if self.system_prompt and len(self.system_prompt) > 300:
            self.system_prompt = self.system_prompt[:300] + "..."
        if self.raw_response and len(self.raw_response) > 2000:
            self.raw_response = self.raw_response[:2000] + "..."


@dataclass
class SecurityInfo:
    """Security pipeline details and grounding verifications."""

    injection_score: float = 0.0
    injection_action: str = "allow"
    discarded_chunks: int = 0
    rate_limited: bool = False
    output_safe: bool = True
    grounding_passed: bool = True
    hallucination_detected: bool = False
    evidence_sufficiency: str = "SUFFICIENT"
    abstained: bool = False
    abstention_reason: str = ""


@dataclass
class FeatureToggles:
    """Feature toggle snapshot for the request."""

    dense_retrieval: bool = True
    bm25_retrieval: bool = True
    hybrid_retrieval: bool = True
    cross_encoder_reranking: bool = True
    metadata_filtering: bool = True
    conversation_memory: bool = True
    query_rewriting: bool = True
    agent_planning: bool = True
    multi_hop_retrieval: bool = True
    hyde: bool = False
    chunk_enhancement: bool = True
    jev_decision: bool = False


@dataclass
class DashboardData:
    """Complete structured telemetry trace for an individual query."""

    owner_id: str = "anonymous"
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)
    formatted_time: str = ""
    total_latency_ms: float = 0.0
    query_info: QueryInfo = field(default_factory=QueryInfo)
    retrieval_info: RetrievalInfo = field(default_factory=RetrievalInfo)
    reranking_info: RerankingInfo = field(default_factory=RerankingInfo)
    agent_info: AgentInfo = field(default_factory=AgentInfo)
    generation_info: GenerationInfo = field(default_factory=GenerationInfo)
    security_info: SecurityInfo = field(default_factory=SecurityInfo)
    feature_toggles: FeatureToggles = field(default_factory=FeatureToggles)
    pipeline_stages: list[dict[str, Any]] = field(default_factory=list)
    latency_breakdown: dict[str, float] = field(default_factory=dict)
    computed_metrics: dict[str, Any] = field(default_factory=dict)
    cumulative_metrics: dict[str, Any] = field(default_factory=dict)
    failure_category: str = "none"
    failure_diagnosis: str = ""
    evidence_utilization_notes: str = ""
    retrieval_plan: list[str] = field(default_factory=list)
    executed_operations: list[str] = field(default_factory=list)
    skipped_operations: list[dict[str, str]] = field(default_factory=list)
    decision_flow: list[dict[str, Any]] = field(default_factory=list)
    rank_movements: list[dict[str, Any]] = field(default_factory=list)
    security_event: dict[str, Any] = field(default_factory=dict)
    rrf_details: dict[str, Any] = field(default_factory=dict)
    candidate_count_before_reranking: int = 0
    candidate_count_after_reranking: int = 0
    budgets: dict[str, Any] = field(default_factory=dict)
    decision_metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "SUCCESS"
    is_blocked: bool = False

    def __post_init__(self):
        if not self.formatted_time:
            self.formatted_time = time.strftime("%H:%M:%S", time.localtime(self.timestamp))

    def to_dict(self) -> dict[str, Any]:
        """Serialize all dashboard data for JSON API output."""
        return {
            "candidate_count_before_reranking": self.candidate_count_before_reranking,
            "candidate_count_after_reranking": self.candidate_count_after_reranking,
            "budgets": self.budgets,
            "status": self.status,
            "is_blocked": self.is_blocked,
            "owner_id": self.owner_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "formatted_time": self.formatted_time,
            "total_latency_ms": self.total_latency_ms,
            "latency_breakdown": self.latency_breakdown,
            "failure_category": self.failure_category,
            "failure_diagnosis": self.failure_diagnosis,
            "evidence_utilization_notes": self.evidence_utilization_notes,
            "retrieval_plan": self.retrieval_plan,
            "executed_operations": self.executed_operations,
            "skipped_operations": self.skipped_operations,
            "decision_flow": self.decision_flow,
            "rank_movements": self.rank_movements,
            "security_event": self.security_event,
            "rrf_details": self.rrf_details,
            "cumulative_metrics": self.cumulative_metrics,
            "query_info": {
                "original_query": self.query_info.original_query,
                "standalone_query": self.query_info.standalone_query,
                "metadata_filters": self.query_info.metadata_filters,
                "routing_decision": self.query_info.routing_decision,
                "route_type": self.query_info.route_type,
                "is_followup": self.query_info.is_followup,
                "detected_companies": self.query_info.detected_companies,
                "detected_topics": self.query_info.detected_topics,
            },
            "retrieval_info": {
                "retriever_used": self.retrieval_info.retriever_used,
                "retrieval_latency_ms": self.retrieval_info.retrieval_latency_ms,
                "total_candidates": self.retrieval_info.total_candidates,
                "final_count": self.retrieval_info.final_count,
                "retrieved_chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "text": c.text[:250],
                        "source": c.source,
                        "similarity_score": c.similarity_score,
                        "cross_encoder_score": c.cross_encoder_score,
                        "original_rank": c.original_rank,
                        "new_rank": c.new_rank,
                        "rank_change": c.rank_change,
                        "retrieval_source": c.retrieval_source,
                        "is_selected": c.is_selected,
                        "in_context": c.in_context,
                        "metadata": c.metadata,
                        "company": c.metadata.get("company", "General") if isinstance(c.metadata, dict) else "General",
                    }
                    for c in self.retrieval_info.retrieved_chunks
                ],
                "rejected_chunks": [
                    {
                        "text": c.text[:200],
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
                "rank_changes": self.rank_movements or self.reranking_info.rank_changes,
                "pre_reranker_mrr": self.reranking_info.pre_reranker_mrr,
                "post_reranker_mrr": self.reranking_info.post_reranker_mrr,
                "reranker_improvement": self.reranking_info.reranker_improvement,
            },
            "agent_info": {
                "enabled": self.agent_info.enabled,
                "route_type": self.agent_info.route_type,
                "llm_invoked": self.agent_info.llm_invoked,
                "llm_invocation_reason": self.agent_info.llm_invocation_reason,
                "llm_bypass_reason": self.agent_info.llm_bypass_reason,
                "hops_count": self.agent_info.hops_count,
                "hops_data": self.agent_info.hops_data,
                "reasoning_trace": self.agent_info.reasoning_trace,
                "tool_selections": self.agent_info.tool_selections,
                "planning_steps": self.agent_info.planning_steps,
                "iterations": self.agent_info.iterations,
            },
            "generation_info": {
                "system_prompt": self.generation_info.system_prompt,
                "user_prompt": self.generation_info.user_prompt,
                "raw_response": self.generation_info.raw_response,
                "final_response": self.generation_info.final_response,
                "tokens_generated": self.generation_info.tokens_generated,
                "generation_latency_ms": self.generation_info.generation_latency_ms,
                "time_to_first_token_ms": self.generation_info.time_to_first_token_ms,
                "throughput_tokens_per_sec": self.generation_info.throughput_tokens_per_sec,
                "streamed_chunks_count": self.generation_info.streamed_chunks_count,
            },
            "security_info": {
                "injection_score": self.security_info.injection_score,
                "injection_action": self.security_info.injection_action,
                "discarded_chunks": self.security_info.discarded_chunks,
                "rate_limited": self.security_info.rate_limited,
                "output_safe": self.security_info.output_safe,
                "grounding_passed": self.security_info.grounding_passed,
                "hallucination_detected": self.security_info.hallucination_detected,
                "evidence_sufficiency": self.security_info.evidence_sufficiency,
                "abstained": self.security_info.abstained,
                "abstention_reason": self.security_info.abstention_reason,
                "security_event": self.security_event,
            },
            "feature_toggles": {
                "dense_retrieval": self.feature_toggles.dense_retrieval,
                "bm25_retrieval": self.feature_toggles.bm25_retrieval,
                "hybrid_retrieval": self.feature_toggles.hybrid_retrieval,
                "cross_encoder_reranking": self.feature_toggles.cross_encoder_reranking,
                "metadata_filtering": self.feature_toggles.metadata_filtering,
                "conversation_memory": self.feature_toggles.conversation_memory,
                "query_rewriting": self.feature_toggles.query_rewriting,
                "agent_planning": self.feature_toggles.agent_planning,
                "multi_hop_retrieval": self.feature_toggles.multi_hop_retrieval,
                "chunk_enhancement": self.feature_toggles.chunk_enhancement,
                "jev_decision": getattr(self.feature_toggles, "jev_decision", False),
            },
            "decision_metadata": self.decision_metadata,
            "metrics": self.computed_metrics,
            "pipeline_stages": self.pipeline_stages,
        }


class SessionHistoryStore:
    """Thread-safe rolling query history and real-time aggregate statistics engine."""

    def __init__(self, max_history: int = 25) -> None:
        self.max_history = max_history
        self._history: list[DashboardData] = []
        self._lock = threading.RLock()

    def record_query(self, data: DashboardData) -> None:
        with self._lock:
            self._history.insert(0, data)
            if len(self._history) > self.max_history:
                self._history.pop()

    def get_query(self, request_id: str) -> dict[str, Any] | None:
        """Inspect a single historical query in the current session."""
        with self._lock:
            for d in self._history:
                if d.request_id == request_id:
                    return d.to_dict()
            return None

    def get_cumulative_metrics(self) -> dict[str, Any]:
        """Compute live running average of metrics ONLY over queries with valid ground truth."""
        with self._lock:
            mrr_vals = []
            ndcg_vals = []
            prec_vals = []
            rec_vals = []
            hit_vals = []
            for d in self._history:
                m = d.computed_metrics
                # Exclude queries without ground truth and blocked queries
                if m.get("has_ground_truth") is True or (isinstance(m.get("mrr"), (int, float)) and not getattr(d, "is_blocked", False)):
                    if isinstance(m.get("mrr"), (int, float)):
                        mrr_vals.append(float(m["mrr"]))
                    if isinstance(m.get("ndcg"), (int, float)):
                        ndcg_vals.append(float(m["ndcg"]))
                    if isinstance(m.get("precision_at_k"), (int, float)):
                        prec_vals.append(float(m["precision_at_k"]))
                    if isinstance(m.get("recall_at_k"), (int, float)):
                        rec_vals.append(float(m["recall_at_k"]))
                    if isinstance(m.get("hit_rate"), (int, float)):
                        hit_vals.append(float(m["hit_rate"]))

            valid_count = len(mrr_vals)
            return {
                "valid_ground_truth_queries": valid_count,
                "cumulative_mrr": round(statistics.mean(mrr_vals), 3) if mrr_vals else "N/A",
                "cumulative_ndcg": round(statistics.mean(ndcg_vals), 3) if ndcg_vals else "N/A",
                "cumulative_precision_at_k": round(statistics.mean(prec_vals), 3) if prec_vals else "N/A",
                "cumulative_recall_at_k": round(statistics.mean(rec_vals), 3) if rec_vals else "N/A",
                "cumulative_hit_rate": round(statistics.mean(hit_vals), 3) if hit_vals else "N/A",
            }

    def get_history_summary(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            items = self._history[:limit]
            return [
                {
                    "request_id": d.request_id,
                    "timestamp": d.timestamp,
                    "formatted_time": d.formatted_time,
                    "query": d.query_info.original_query,
                    "route": d.query_info.route_type,
                    "llm_invoked": d.agent_info.llm_invoked,
                    "llm_bypass_reason": d.agent_info.llm_bypass_reason,
                    "hops": d.agent_info.hops_count,
                    "retrieved_count": d.retrieval_info.total_candidates,
                    "final_count": d.retrieval_info.final_count,
                    "sufficiency": d.security_info.evidence_sufficiency,
                    "abstained": d.security_info.abstained,
                    "ttft_ms": d.generation_info.time_to_first_token_ms,
                    "total_latency_ms": d.total_latency_ms,
                    "mrr": d.computed_metrics.get("mrr", "N/A"),
                    "ndcg": d.computed_metrics.get("ndcg", "N/A"),
                    "faithfulness": d.computed_metrics.get("faithfulness", "N/A"),
                    "failure_category": d.failure_category,
                    "security_status": d.security_event.get("status", "ALLOWED"),
                    "security_category": d.security_event.get("category", "NORMAL"),
                    "security_severity": d.security_event.get("severity", "LOW"),
                    "executed_operations": d.executed_operations,
                    "decision_flow": d.decision_flow,
                }
                for d in items
            ]

    def get_aggregate_metrics(self) -> dict[str, Any]:
        with self._lock:
            if not self._history:
                return {
                    "total_queries": 0,
                    "llm_calls_total": 0,
                    "llm_calls_bypassed": 0,
                    "llm_bypass_rate_pct": 0.0,
                    "abstentions_count": 0,
                    "abstention_rate_pct": 0.0,
                    "multi_hop_queries_count": 0,
                    "multi_hop_rate_pct": 0.0,
                    "avg_hops_per_query": 1.0,
                    "latency": {
                        "avg_total_ms": 0.0,
                        "p50_ms": 0.0,
                        "p90_ms": 0.0,
                        "p95_ms": 0.0,
                        "p99_ms": 0.0,
                        "avg_retrieval_ms": 0.0,
                        "avg_reranking_ms": 0.0,
                        "avg_ttft_ms": 0.0,
                        "avg_generation_ms": 0.0,
                    },
                    "routes_distribution": {},
                    "failure_distribution": {},
                    "average_quality": {
                        "faithfulness": 0.0,
                        "response_relevancy": 0.0,
                        "groundedness": 0.0,
                    },
                }

            n = len(self._history)
            llm_calls_bypassed = sum(1 for d in self._history if not d.agent_info.llm_invoked)
            llm_calls_total = sum(1 for d in self._history if d.agent_info.llm_invoked)
            abstentions = sum(1 for d in self._history if d.security_info.abstained)
            multi_hops = sum(1 for d in self._history if d.agent_info.hops_count > 1)
            total_hops = sum(d.agent_info.hops_count for d in self._history)

            total_latencies = sorted(d.total_latency_ms for d in self._history)
            ret_latencies = [d.retrieval_info.retrieval_latency_ms for d in self._history]
            rerank_latencies = [d.reranking_info.reranking_latency_ms for d in self._history]
            ttft_latencies = [d.generation_info.time_to_first_token_ms for d in self._history if d.generation_info.time_to_first_token_ms > 0]
            gen_latencies = [d.generation_info.generation_latency_ms for d in self._history if d.generation_info.generation_latency_ms > 0]

            def percentile(sorted_list: list[float], pct: float) -> float:
                if not sorted_list:
                    return 0.0
                k = (len(sorted_list) - 1) * pct
                f = int(k)
                c = min(f + 1, len(sorted_list) - 1)
                d0 = sorted_list[f] * (c - k)
                d1 = sorted_list[c] * (k - f)
                return round(d0 + d1, 2)

            routes_count: dict[str, int] = {}
            failures_count: dict[str, int] = {}
            for d in self._history:
                r = d.query_info.route_type
                routes_count[r] = routes_count.get(r, 0) + 1
                fc = d.failure_category
                if fc != "none":
                    failures_count[fc] = failures_count.get(fc, 0) + 1

            # Quality metrics averages
            faith_scores = [d.computed_metrics.get("faithfulness", 0.0) for d in self._history if isinstance(d.computed_metrics.get("faithfulness"), (int, float))]
            rel_scores = [d.computed_metrics.get("response_relevancy", 0.0) for d in self._history if isinstance(d.computed_metrics.get("response_relevancy"), (int, float))]
            ground_scores = [d.computed_metrics.get("groundedness", 0.0) for d in self._history if isinstance(d.computed_metrics.get("groundedness"), (int, float))]

            return {
                "total_queries": n,
                "llm_calls_total": llm_calls_total,
                "llm_calls_bypassed": llm_calls_bypassed,
                "llm_bypass_rate_pct": round((llm_calls_bypassed / n) * 100, 1),
                "abstentions_count": abstentions,
                "abstention_rate_pct": round((abstentions / n) * 100, 1),
                "multi_hop_queries_count": multi_hops,
                "multi_hop_rate_pct": round((multi_hops / n) * 100, 1),
                "avg_hops_per_query": round(total_hops / n, 2),
                "latency": {
                    "avg_total_ms": round(statistics.mean(total_latencies), 1),
                    "p50_ms": percentile(total_latencies, 0.50),
                    "p90_ms": percentile(total_latencies, 0.90),
                    "p95_ms": percentile(total_latencies, 0.95),
                    "p99_ms": percentile(total_latencies, 0.99),
                    "avg_retrieval_ms": round(statistics.mean(ret_latencies), 1) if ret_latencies else 0.0,
                    "avg_reranking_ms": round(statistics.mean(rerank_latencies), 1) if rerank_latencies else 0.0,
                    "avg_ttft_ms": round(statistics.mean(ttft_latencies), 1) if ttft_latencies else 0.0,
                    "avg_generation_ms": round(statistics.mean(gen_latencies), 1) if gen_latencies else 0.0,
                },
                "routes_distribution": routes_count,
                "failure_distribution": failures_count,
                "average_quality": {
                    "faithfulness": round(statistics.mean(faith_scores), 3) if faith_scores else 0.0,
                    "response_relevancy": round(statistics.mean(rel_scores), 3) if rel_scores else 0.0,
                    "groundedness": round(statistics.mean(ground_scores), 3) if ground_scores else 0.0,
                },
                "cumulative_evaluation": self.get_cumulative_metrics(),
                "security_analytics": {
                    "total_queries": n,
                    "safe_queries": sum(1 for d in self._history if d.security_event.get("status") != "BLOCKED" and d.security_info.injection_action != "block"),
                    "blocked_queries": sum(1 for d in self._history if d.security_event.get("status") == "BLOCKED" or d.security_info.injection_action == "block"),
                    "prompt_injection_attempts": sum(1 for d in self._history if d.security_event.get("category") == "PROMPT_INJECTION"),
                    "secret_exfiltration_attempts": sum(1 for d in self._history if d.security_event.get("category") == "SECRET_EXFILTRATION"),
                    "unauthorized_tool_attempts": sum(1 for d in self._history if d.security_event.get("category") == "UNAUTHORIZED_TOOL_USE"),
                    "security_bypass_attempts": sum(1 for d in self._history if d.security_event.get("category") == "SECURITY_BYPASS"),
                },
            }

    def get_security_analytics(self) -> dict[str, Any]:
        """Aggregate breakdown of security events and block categories across the session."""
        with self._lock:
            n = len(self._history)
            blocked = sum(
                1 for d in self._history
                if d.is_blocked
                or (isinstance(d.security_event, dict) and d.security_event.get("status") == "BLOCKED")
                or (isinstance(d.security_event, dict) and d.security_event.get("is_blocked") is True)
                or getattr(d.security_info, "injection_action", "") == "block"
            )
            return {
                "total_queries": n,
                "safe_queries": n - blocked,
                "blocked_queries": blocked,
                "prompt_injection_attempts": sum(
                    1 for d in self._history
                    if isinstance(d.security_event, dict) and d.security_event.get("category") == "PROMPT_INJECTION"
                ),
                "secret_exfiltration_attempts": sum(
                    1 for d in self._history
                    if isinstance(d.security_event, dict) and d.security_event.get("category") == "SECRET_EXFILTRATION"
                ),
                "unauthorized_tool_attempts": sum(
                    1 for d in self._history
                    if isinstance(d.security_event, dict) and d.security_event.get("category") == "UNAUTHORIZED_TOOL_USE"
                ),
                "security_bypass_attempts": sum(
                    1 for d in self._history
                    if isinstance(d.security_event, dict) and d.security_event.get("category") == "SECURITY_BYPASS"
                ),
            }


class DashboardStore:
    """Thread-safe bounded in-memory store for individual dashboard requests."""

    def __init__(self, max_size: int = 50) -> None:
        self._store: dict[str, DashboardData] = {}
        self._order: list[str] = []
        self._max_size = max_size
        self._lock = threading.Lock()

    def store(self, request_id: str, data: DashboardData) -> None:
        with self._lock:
            if request_id in self._store:
                self._store[request_id] = data
                return
            if len(self._order) >= self._max_size:
                oldest = self._order.pop(0)
                self._store.pop(oldest, None)
            self._store[request_id] = data
            self._order.append(request_id)

    def get(self, request_id: str, default: Any = None) -> DashboardData | None:
        with self._lock:
            return self._store.get(request_id, default)

    def get_authorized(
        self, request_id: str, requester_id: str, is_admin: bool = False
    ) -> DashboardData | None:
        with self._lock:
            data = self._store.get(request_id)
            if not data:
                return None
            if is_admin or data.owner_id == requester_id or data.owner_id == "anonymous":
                return data
            return None

    def __setitem__(self, request_id: str, data: DashboardData) -> None:
        self.store(request_id, data)

    def __getitem__(self, request_id: str) -> DashboardData:
        with self._lock:
            return self._store[request_id]

    def __contains__(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._store

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# Global instances
DASHBOARD_STORE = DashboardStore(max_size=500)
SESSION_HISTORY = SessionHistoryStore(max_history=100)
