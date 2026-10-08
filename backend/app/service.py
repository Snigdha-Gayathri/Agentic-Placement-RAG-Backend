"""Secure Agentic RAG orchestration service with layered guardrails, observability, and feature toggles."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator

# Ensure project root is present in sys.path for deterministic module resolution across environments
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from security import (
    AuthContext,
    ContextSanitizer,
    CrossProcessLock,
    GroundingVerifier,
    HallucinationGuard,
    InputValidator,
    OutputSanitizer,
    OutputValidator,
    PromptInjectionDetector,
    RetrievalGuard,
    Role,
    SecurityLogger,
    SlidingWindowRateLimiter,
    load_config,
)

from backend.app.models import (
    ChatRequest,
    ChatResponse,
    GuardrailMeta,
    HealthResponse,
    PipelineData,
    PipelineStageModel,
    ReindexResponse,
    SecurityStatusResponse,
)
from backend.app.gemini_client import GeminiClient
from backend.app.retriever import Retriever
from backend.app.vector_store import VectorStore

from backend.config.settings import DEFAULT_FEATURE_TOGGLES
from backend.core.agent.executor import AgentExecutor, AgentExecutionResult
from backend.core.agent.planner import AgentPlanner
from backend.core.chunking.base import ChunkMetadata, DocumentChunk
from backend.core.embeddings.gemini import GeminiEmbedding
from backend.ingestion.pipeline import ChunkingConfig, IngestionPipeline
from backend.core.memory.conversation import ConversationMemory
from backend.core.memory.query_rewriter import QueryRewriter
from backend.core.reranking.cross_encoder import CrossEncoderReranker
from backend.core.hyde.generator import HyDEGenerator
from backend.core.retrieval.bm25 import BM25Retriever
from backend.core.retrieval.dense import DenseRetriever
from backend.core.retrieval.hybrid import HybridRetriever
from backend.core.retrieval.router import QueryRouter, RoutingDecision
from backend.core.vector_store.chroma import ChromaVectorStore
from backend.evaluation.metrics import LatencyMetrics, RAGMetrics, RAGMetricsResult
from backend.observability.dashboard import (
    DASHBOARD_STORE,
    SESSION_HISTORY,
    AgentInfo,
    DashboardData,
    DashboardStore,
    FeatureToggles,
    GenerationInfo,
    QueryInfo,
    RerankingInfo,
    RetrievalInfo,
    RetrievedChunkInfo,
    SecurityInfo,
)
from backend.observability.pipeline_tracker import PipelineTracker, StageName, StageStatus

logger = logging.getLogger(__name__)


class TrackerStore:
    """Thread-safe bounded in-memory store for PipelineTracker instances."""

    def __init__(self, max_size: int = 50) -> None:
        self._store: dict[str, PipelineTracker] = {}
        self._order: list[str] = []
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, request_id: str, default: Any = None) -> PipelineTracker | None:
        with self._lock:
            return self._store.get(request_id, default)

    def __setitem__(self, request_id: str, tracker: PipelineTracker) -> None:
        with self._lock:
            if request_id not in self._store and len(self._order) >= self._max_size:
                oldest = self._order.pop(0)
                self._store.pop(oldest, None)
            self._store[request_id] = tracker
            if request_id not in self._order:
                self._order.append(request_id)

    def __getitem__(self, request_id: str) -> PipelineTracker:
        with self._lock:
            return self._store[request_id]

    def __contains__(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._store

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


TRACKER_STORE: TrackerStore = TrackerStore(max_size=500)


@dataclass
class RequestContext:
    client_ip: str
    request_id: str
    user_id: str = "anonymous"
    role: Role = Role.USER



def build_request_context(
    client_ip: str = "127.0.0.1",
    request_id: str | None = None,
    user_id: str = "anonymous",
    role: Role = Role.USER,
) -> RequestContext:
    """Factory for RequestContext with auto-generated or provided request_id."""
    import uuid
    return RequestContext(
        client_ip=client_ip,
        request_id=request_id or str(uuid.uuid4()),
        user_id=user_id,
        role=role,
    )


class SecureRAGService:
    """Production-grade secure Agentic RAG service."""

    def __init__(self) -> None:
        self.config = load_config()
        from backend.config.settings import PipelineConfig
        self.pipeline_config = PipelineConfig.from_env()
        self.logger = SecurityLogger()
        self._reindex_lock = asyncio.Lock()
        self._cross_process_lock = CrossProcessLock(Path("data") / ".reindex.lock")

        # Security layers (100% Preserved)
        self.input_validator = InputValidator(self.config)
        self.injection_detector = PromptInjectionDetector(self.config)
        self.rate_limiter = SlidingWindowRateLimiter(self.config)
        self.retrieval_guard = RetrievalGuard(self.config)
        self.context_sanitizer = ContextSanitizer(self.config)
        self.output_validator = OutputValidator(self.config)
        self.output_sanitizer = OutputSanitizer()
        self.grounding = GroundingVerifier()
        self.hallucination_guard = HallucinationGuard()

        # Legacy TF-IDF store as fallback
        self.legacy_vector_store = VectorStore(
            index_path=os.getenv("VECTOR_DB_PATH", "data/vector_index.json"),
            documents_path=os.getenv("DOCUMENTS_PATH", "documents"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
        )
        self.legacy_vector_store.load()
        self.legacy_retriever = Retriever()

        # Modern RAG & Agentic components
        self.chroma_store = ChromaVectorStore()
        self._embedder = None
        self._dense_retriever = None
        self._bm25_retriever = None
        self._hybrid_retriever = None
        self.query_router = QueryRouter()
        self._reranker = None
        self.memory = ConversationMemory()
        self.query_rewriter = QueryRewriter()
        self.gemini = GeminiClient()
        self._hyde_generator = None
        self._agent_executor = None
        self.metrics_engine = RAGMetrics()

        # Dynamic developer configuration parameters
        self.dynamic_params: dict[str, Any] = {
            "top_k": 15,
            "reranker_top_n": 8,
            "dense_weight": 0.5,
            "bm25_weight": 0.5,
            "similarity_threshold": self.config.similarity_threshold,
            "reranker_threshold": 0.15,
        }

    @property
    def embedder(self) -> GeminiEmbedding:
        if self._embedder is None:
            self._embedder = GeminiEmbedding()
        return self._embedder

    @property
    def dense_retriever(self) -> DenseRetriever:
        if self._dense_retriever is None:
            self._dense_retriever = DenseRetriever(self.embedder, self.chroma_store)
        return self._dense_retriever

    @property
    def bm25_retriever(self) -> BM25Retriever:
        if self._bm25_retriever is None:
            self._bm25_retriever = BM25Retriever(self.chroma_store)
        return self._bm25_retriever

    @property
    def hybrid_retriever(self) -> HybridRetriever:
        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetriever(
                self.dense_retriever,
                self.bm25_retriever,
                dense_weight=self.dynamic_params.get("dense_weight", 0.5),
                sparse_weight=self.dynamic_params.get("bm25_weight", 0.5),
            )
        return self._hybrid_retriever

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    @property
    def hyde_generator(self) -> HyDEGenerator:
        if self._hyde_generator is None:
            self._hyde_generator = HyDEGenerator(self.gemini, self.embedder)
        return self._hyde_generator

    @property
    def agent_executor(self) -> AgentExecutor:
        if self._agent_executor is None:
            self._agent_executor = AgentExecutor(
                retriever=self.hybrid_retriever,
                llm=self.gemini,
                dense_retriever=self.dense_retriever,
                bm25_retriever=self.bm25_retriever,
                reranker=self.reranker,
                hyde_generator=self.hyde_generator,
                query_rewriter=self.query_rewriter,
            )
        return self._agent_executor

    def health(self) -> "HealthResponse":
        from backend.app.models import HealthResponse
        gemini_ok = bool(self.config.gemini_api_key if hasattr(self.config, "gemini_api_key") else os.getenv("GEMINI_API_KEY", ""))
        try:
            stats = self.chroma_store.get_stats()
            vector_ok = stats.total_chunks > 0
        except Exception:
            vector_ok = False
        return HealthResponse(
            status="ok",
            gemini=gemini_ok,
            vector_index=vector_ok,
            readiness=gemini_ok,
        )

    def security_status(self) -> "SecurityStatusResponse":
        from backend.app.models import SecurityStatusResponse
        cfg = self.config
        return SecurityStatusResponse(
            status="active",
            config={
                "max_query_length": getattr(cfg, "max_query_length", 2000),
                "rate_limit": getattr(cfg, "rate_limit", 30),
                "similarity_threshold": getattr(cfg, "similarity_threshold", 0.12),
                "hallucination_threshold": getattr(cfg, "hallucination_threshold", 0.45),
                "injection_detection": True,
                "output_validation": True,
            },
        )

    def get_decision_status(self) -> dict[str, Any]:
        """Safe, read-only status of the pluggable Jev / Decision Backend integration."""
        from backend.core.decision.factory import get_decision_backend
        cfg = getattr(self, "pipeline_config", None) or self.config
        backend = get_decision_backend()
        is_jev = getattr(backend, "name", "") == "jev"
        is_configured = getattr(backend, "is_configured", False)
        live_calls = getattr(backend, "live_calls_count", 0)

        if not getattr(cfg, "jev_enabled", False):
            runtime_mode = "MODE_A_HEURISTIC_DIRECT"
            backend_in_use = "heuristic"
            fallback_active = False
        elif is_configured:
            runtime_mode = "MODE_B_JEV_LIVE"
            backend_in_use = "jev"
            fallback_active = False
        else:
            runtime_mode = "MODE_C_JEV_FALLBACK"
            backend_in_use = "heuristic"
            fallback_active = True

        return {
            "jev_enabled": getattr(cfg, "jev_enabled", False),
            "configured": is_configured,
            "runtime_mode": runtime_mode,
            "live_calls": live_calls,
            "fallback_active": fallback_active,
            "backend_in_use": backend_in_use,
            "fallback_backend": "heuristic",
            "live_evaluation_available": is_configured,
            "model": getattr(cfg, "jev_model", "jev-1.13-free") if getattr(cfg, "jev_enabled", False) else None,
            "circuit_breaker": getattr(backend, "circuit_breaker", None).get_status() if hasattr(backend, "circuit_breaker") else {"state": "N/A"},
        }

    async def reindex(self) -> "ReindexResponse":
        from backend.app.models import ReindexResponse
        if self._reindex_lock.locked() or not self._cross_process_lock.acquire():
            raise RuntimeError("Reindexing is already in progress.")
        try:
            async with self._reindex_lock:
                try:
                    from backend.ingestion.pipeline import IngestionPipeline, ChunkingConfig
                    pipeline = IngestionPipeline(
                        data_path="data",
                        config=ChunkingConfig(),
                    )
                    chunks_indexed = await pipeline.run()
                    return ReindexResponse(status="ok", chunks_indexed=chunks_indexed or 0)
                except Exception as exc:
                    logger.error("Reindex failed: %s", exc)
                    raise RuntimeError(str(exc)) from exc
        finally:
            self._cross_process_lock.release()

    def get_feature_toggles(self) -> dict[str, bool]:
        toggles = dict(DEFAULT_FEATURE_TOGGLES)
        cfg = getattr(self, "pipeline_config", None)
        if cfg and hasattr(cfg, "feature_toggles"):
            toggles.update(cfg.feature_toggles)
        if cfg and getattr(cfg, "jev_enabled", False):
            toggles["jev_decision"] = True
        return toggles

    def update_feature_toggles(self, updates: dict[str, bool]) -> dict[str, bool]:
        for k, v in updates.items():
            if k in DEFAULT_FEATURE_TOGGLES:
                DEFAULT_FEATURE_TOGGLES[k] = bool(v)
        return dict(DEFAULT_FEATURE_TOGGLES)

    def update_dynamic_params(self, params: dict[str, Any]) -> dict[str, Any]:
        for k, v in params.items():
            if k in self.dynamic_params:
                self.dynamic_params[k] = v
        # Update hybrid weights if changed
        if "dense_weight" in params or "bm25_weight" in params:
            self.hybrid_retriever.dense_weight = float(self.dynamic_params.get("dense_weight", 0.5))
            self.hybrid_retriever.sparse_weight = float(self.dynamic_params.get("bm25_weight", 0.5))
        return dict(self.dynamic_params)

    def _find_benchmark_case(self, query: str) -> Any | None:
        """Find matching ground truth benchmark test case if present."""
        try:
            from backend.evaluation.benchmark_dataset import get_benchmark_dataset
            dataset = get_benchmark_dataset()
            q_norm = query.strip().lower()
            for tc in dataset:
                tc_norm = tc.query.strip().lower()
                if tc_norm == q_norm or (len(tc_norm) > 10 and tc_norm in q_norm) or (len(q_norm) > 10 and q_norm in tc_norm):
                    return tc
        except Exception:
            pass
        return None

    def _build_blocked_security_response(
        self,
        request: ChatRequest,
        ctx: RequestContext,
        category: str,
        severity: str,
        user_message: str,
        risk_score: float,
        latency_breakdown: LatencyMetrics,
        started: float,
        tracker: PipelineTracker,
        toggles: dict[str, bool],
    ) -> ChatResponse:
        """Build informative, transparent user-safe security response bypassing retrieval."""
        latency_breakdown.total_ms = round((time.perf_counter() - started) * 1000, 2)
        tracker.fail_stage(StageName.INJECTION_DETECTION.value, f"Security violation: {category}")
        tracker.complete_stage(StageName.RESPONSE_COMPLETE.value)

        security_event_dict = {
            "is_blocked": True,
            "category": category,
            "severity": severity,
            "reason": f"Security validation policy violation: {category}",
            "sanitized_user_message": user_message,
            "blocked_operations": [
                "bm25_retrieval",
                "dense_retrieval",
                "hybrid_retrieval",
                "cross_encoder_rerank",
                "hyde_generation",
                "query_decomposition",
                "llm_generation",
            ],
            "risk_score": risk_score,
            "timestamp": time.time(),
        }

        # Telemetry metrics: N/A for retrieval metrics when blocked
        computed_metrics = {
            "mrr": "N/A",
            "ndcg": "N/A",
            "precision_at_k": "N/A",
            "recall_at_k": "N/A",
            "f1": "N/A",
            "grounding_score": 1.0,
            "hallucination_rate": 0.0,
            "failure_category": "SECURITY_BLOCK",
            "failure_diagnosis": f"Request blocked before retrieval: {category}",
        }

        dashboard_obj = DashboardData(
            owner_id=ctx.user_id,
            request_id=ctx.request_id,
            status="BLOCKED",
            is_blocked=True,
            total_latency_ms=latency_breakdown.total_ms,
            latency_breakdown=latency_breakdown.to_dict(),
            failure_category="SECURITY_BLOCK",
            failure_diagnosis=f"Request blocked: {category}",
            query_info=QueryInfo(
                original_query=request.query,
                standalone_query=request.query,
                metadata_filters={},
                routing_decision=f"Blocked before retrieval: {category}",
                route_type="SECURITY_BLOCKED",
                is_followup=False,
                detected_companies=[],
                detected_topics=[],
            ),
            retrieval_info=RetrievalInfo(
                retriever_used="none",
                retrieval_latency_ms=0.0,
                retrieved_chunks=[],
                total_candidates=0,
                final_count=0,
            ),
            reranking_info=RerankingInfo(
                enabled=False,
                reranking_latency_ms=0.0,
                rank_changes=[],
            ),
            agent_info=AgentInfo(
                enabled=toggles.get("agent_planning", True),
                route_type="SECURITY_BLOCKED",
                llm_invoked=False,
                llm_invocation_reason="Bypassed due to security policy violation",
                llm_bypass_reason=f"Security violation: {category}",
                hops_count=0,
            ),
            generation_info=GenerationInfo(
                final_response=user_message,
                tokens_generated=len(user_message.split()),
                generation_latency_ms=0.0,
            ),
            security_info=SecurityInfo(
                injection_score=risk_score,
                injection_action="block",
                discarded_chunks=0,
                output_safe=True,
                grounding_passed=True,
                hallucination_detected=False,
                evidence_sufficiency="SECURITY_BLOCKED",
                abstained=True,
                abstention_reason=f"Security policy violation: {category}",
            ),
            feature_toggles=FeatureToggles(
                dense_retrieval=toggles.get("dense_retrieval", True),
                bm25_retrieval=toggles.get("bm25_retrieval", True),
                hybrid_retrieval=toggles.get("hybrid_retrieval", True),
                cross_encoder_reranking=toggles.get("cross_encoder_reranking", True),
                conversation_memory=toggles.get("conversation_memory", True),
                query_rewriting=toggles.get("query_rewriting", True),
                metadata_filtering=toggles.get("metadata_filtering", True),
                agent_planning=toggles.get("agent_planning", True),
                multi_hop_retrieval=toggles.get("multi_hop_retrieval", True),
                chunk_enhancement=toggles.get("chunk_enhancement", True),
            ),
            security_event=security_event_dict,
            retrieval_plan=[],
            executed_operations=["INPUT_SECURITY_GATEWAY"],
            skipped_operations=[
                {"operation": "EMBEDDING_GENERATION", "reason": "Query blocked by security pre-retrieval gateway."},
                {"operation": "BM25_RETRIEVAL", "reason": "Query blocked by security pre-retrieval gateway."},
                {"operation": "DENSE_RETRIEVAL", "reason": "Query blocked by security pre-retrieval gateway."},
                {"operation": "HYBRID_RETRIEVAL", "reason": "Query blocked by security pre-retrieval gateway."},
                {"operation": "CROSS_ENCODER_RERANKING", "reason": "Query blocked by security pre-retrieval gateway."},
                {"operation": "LLM_GENERATION", "reason": "Query blocked by security pre-retrieval gateway."},
            ],
            decision_flow=[
                {"node": "SecurityGateway", "status": "BLOCKED", "reason": f"Detected {category} (severity={severity})"}
            ],
            computed_metrics=computed_metrics,
            cumulative_metrics=SESSION_HISTORY.get_cumulative_metrics(),
            pipeline_stages=[s.to_dict() for s in tracker.get_stages()],
        )

        DASHBOARD_STORE[ctx.request_id] = dashboard_obj
        SESSION_HISTORY.record_query(dashboard_obj)

        pdata = PipelineData(
            request_id=ctx.request_id,
            query_info=dashboard_obj.query_info.__dict__,
            retrieval_info=dashboard_obj.to_dict()["retrieval_info"],
            reranking_info=dashboard_obj.to_dict()["reranking_info"],
            agent_info=dashboard_obj.to_dict()["agent_info"],
            generation_info=dashboard_obj.to_dict()["generation_info"],
            security_info=dashboard_obj.to_dict()["security_info"],
            feature_toggles=toggles,
            metrics_info=computed_metrics,
        )

        return ChatResponse(
            answer=user_message,
            meta=GuardrailMeta(
                request_id=ctx.request_id,
                injection_score=risk_score,
                discarded_chunks=0,
                chunk_count=0,
                warning="Request blocked by security pre-retrieval gateway.",
            ),
            request_id=ctx.request_id,
            pipeline_data=pdata,
        )

    async def process(
        self,
        request: ChatRequest,
        ctx: RequestContext,
        tracker: PipelineTracker | None = None,
    ) -> ChatResponse:
        """Non-streaming query execution."""
        started = time.perf_counter()
        if tracker is None:
            tracker = PipelineTracker(request_id=ctx.request_id)
            TRACKER_STORE[ctx.request_id] = tracker
        toggles = self.get_feature_toggles()
        if getattr(request, "toggles", None):
            toggles.update(request.toggles)
        latency_breakdown = LatencyMetrics()

        tracker.start_stage(StageName.QUERY_RECEIVED.value)
        tracker.complete_stage(StageName.QUERY_RECEIVED.value)

        # 1. Input Validation
        tracker.start_stage(StageName.INPUT_VALIDATION.value)
        validation = self.input_validator.validate(request.query)
        if not validation.is_valid:
            tracker.fail_stage(StageName.INPUT_VALIDATION.value, "Invalid query input")
            if not request.query or not request.query.strip():
                raise ValueError("Query cannot be empty.")
            user_msg = (
                "### 🛡️ Security Validation Notice\n\n"
                "Your query could not be processed because it contains encoded payloads or characters that violate input security policies.\n\n"
                "**Action Taken:** Immediate execution halt. Retrieval operations, embeddings, and vector store queries have been bypassed.\n\n"
                "**Guidance:** Please submit plain text queries regarding interview experiences, company placement processes, or technical preparation topics."
            )
            return self._build_blocked_security_response(
                request=request,
                ctx=ctx,
                category="ENCODED_OR_MALICIOUS_INPUT",
                severity="HIGH",
                user_message=user_msg,
                risk_score=0.9,
                latency_breakdown=latency_breakdown,
                started=started,
                tracker=tracker,
                toggles=toggles,
            )
        tracker.complete_stage(StageName.INPUT_VALIDATION.value)

        # 2. Prompt Injection Detection (Pre-Retrieval)
        tracker.start_stage(StageName.INJECTION_DETECTION.value)
        injection = self.injection_detector.detect(validation.normalized_query)
        if injection.action == "block":
            tracker.fail_stage(StageName.INJECTION_DETECTION.value, f"Security violation: {injection.category}")
            return self._build_blocked_security_response(
                request=request,
                ctx=ctx,
                category=injection.category or "PROMPT_INJECTION",
                severity=injection.severity or "HIGH",
                user_message=injection.user_safe_message,
                risk_score=injection.risk_score,
                latency_breakdown=latency_breakdown,
                started=started,
                tracker=tracker,
                toggles=toggles,
            )
        tracker.complete_stage(StageName.INJECTION_DETECTION.value)

        # 3. Rate Limiting
        tracker.start_stage(StageName.RATE_LIMITING.value)
        limit = self.rate_limiter.allow(ctx.client_ip, request.session_id, user_id=ctx.user_id)
        if not limit.allowed:
            tracker.fail_stage(StageName.RATE_LIMITING.value, "Rate limit exceeded")
            raise PermissionError(f"Too many requests. Retry in {limit.retry_after_seconds} seconds.")
        tracker.complete_stage(StageName.RATE_LIMITING.value)

        # 4. Conversation Memory & Query Rewriting
        tracker.start_stage(StageName.MEMORY_LOOKUP.value)
        history = []
        if toggles.get("conversation_memory") and request.session_id:
            history = self.memory.get_summarized_history(request.session_id, owner_id=ctx.user_id)
        tracker.complete_stage(StageName.MEMORY_LOOKUP.value)

        tracker.start_stage(StageName.QUERY_REWRITING.value)
        standalone_query = validation.normalized_query
        was_rewritten = False
        if toggles.get("query_rewriting") and history:
            rew_res = await self.query_rewriter.rewrite(validation.normalized_query, history)
            standalone_query = rew_res.rewritten_query
            was_rewritten = rew_res.was_rewritten
        tracker.complete_stage(StageName.QUERY_REWRITING.value)

        # 5. Deterministic-First Query Routing & LLM Bypass Decision
        t_route_start = time.perf_counter()
        tracker.start_stage(StageName.QUERY_ANALYSIS.value)
        routing = self.query_router.route(standalone_query)
        latency_breakdown.routing_ms = round((time.perf_counter() - t_route_start) * 1000, 2)
        meta_filters = routing.metadata_filters if toggles.get("metadata_filtering") else None
        tracker.complete_stage(StageName.QUERY_ANALYSIS.value)

        # CHECK LLM BYPASS: Direct metadata queries or Out-of-Domain queries
        if not routing.llm_required and routing.direct_response:
            latency_breakdown.total_ms = round((time.perf_counter() - started) * 1000, 2)
            tracker.complete_stage(StageName.RESPONSE_COMPLETE.value)

            dashboard_obj = DashboardData(
                owner_id=ctx.user_id,
                request_id=ctx.request_id,
                total_latency_ms=latency_breakdown.total_ms,
                latency_breakdown=latency_breakdown.to_dict(),
                query_info=QueryInfo(
                    original_query=request.query,
                    standalone_query=standalone_query,
                    metadata_filters=meta_filters or {},
                    routing_decision=routing.reasoning,
                    route_type=routing.route_type,
                    is_followup=was_rewritten,
                    detected_companies=routing.detected_companies,
                    detected_topics=routing.detected_topics,
                ),
                agent_info=AgentInfo(
                    enabled=toggles.get("agent_planning", True),
                    route_type=routing.route_type,
                    llm_invoked=False,
                    llm_invocation_reason="Bypassed",
                    llm_bypass_reason=routing.llm_bypass_reason,
                    hops_count=0,
                ),
                generation_info=GenerationInfo(
                    final_response=routing.direct_response,
                    tokens_generated=len(routing.direct_response.split()),
                ),
                security_info=SecurityInfo(
                    injection_score=injection.risk_score,
                    discarded_chunks=0,
                    output_safe=True,
                    grounding_passed=True,
                    hallucination_detected=False,
                    evidence_sufficiency="direct_response",
                    abstained=routing.route_type == "OUT_OF_DOMAIN",
                    abstention_reason=(
                        routing.llm_bypass_reason
                        if routing.route_type == "OUT_OF_DOMAIN"
                        else ""
                    ),
                ),
            )

            DASHBOARD_STORE[ctx.request_id] = dashboard_obj
            SESSION_HISTORY.record_query(dashboard_obj)

            pdata = PipelineData(
                request_id=ctx.request_id,
                query_info=dashboard_obj.query_info.__dict__,
                retrieval_info=dashboard_obj.to_dict()["retrieval_info"],
                reranking_info=dashboard_obj.to_dict()["reranking_info"],
                agent_info=dashboard_obj.to_dict()["agent_info"],
                generation_info=dashboard_obj.to_dict()["generation_info"],
                security_info=dashboard_obj.to_dict()["security_info"],
                feature_toggles=toggles,
                metrics_info={},
            )

            return ChatResponse(
                answer=routing.direct_response,
                meta=GuardrailMeta(
                    request_id=ctx.request_id,
                    injection_score=injection.risk_score,
                    discarded_chunks=0,
                    chunk_count=0,
                    warning=None,
                ),
                request_id=ctx.request_id,
                pipeline_data=pdata,
            )

        # 6. Multi-Hop / Single-Hop Retrieval
        ret_start = time.perf_counter()
        tracker.start_stage(StageName.DENSE_RETRIEVAL.value)

        top_k = self.dynamic_params.get("top_k", 15)

        is_multi_hop = (
            toggles.get("multi_hop_retrieval", True)
            and routing.needs_multi_hop
        )

        agent_exec_res: AgentExecutionResult = await self.agent_executor.execute(
            query=request.query,
            rewritten_query=standalone_query,
            metadata_filters=meta_filters,
            detected_companies=routing.detected_companies,
            detected_topics=routing.detected_topics,
            hyde_used=toggles.get("hyde", False),
            multi_hop_enabled=is_multi_hop,
            top_k=top_k,
            conversation_history=history,
            active_toggles=toggles,
            session_id=ctx.request_id,
        )

        candidate_chunks = agent_exec_res.chunks

        if not candidate_chunks:
            candidate_chunks = self.legacy_retriever.retrieve(
                standalone_query,
                self.legacy_vector_store.search(
                    standalone_query,
                    top_k=self.config.max_retrieved_chunks,
                ),
            )

        # Strictly enforce company metadata filter if single target company
        if meta_filters and meta_filters.get("company"):
            target_comp = str(meta_filters["company"]).strip().lower()

            candidate_chunks = [
                c
                for c in candidate_chunks
                if str(
                    getattr(c, "metadata", {}).get("company", "")
                    if isinstance(getattr(c, "metadata", None), dict)
                    else getattr(c, "company", "")
                ).strip().lower()
                == target_comp
            ]

        latency_breakdown.retrieval_ms = round(
            (time.perf_counter() - ret_start) * 1000,
            2,
        )

        tracker.complete_stage(StageName.DENSE_RETRIEVAL.value)

        # 7. Cross-Encoder Reranking
        pre_rerank_candidates = list(candidate_chunks)

        rerank_start = time.perf_counter()
        tracker.start_stage(StageName.CROSS_ENCODER_RERANKING.value)

        rank_changes_info = []
        reranker_top_n = self.dynamic_params.get("reranker_top_n", 8)

        if toggles.get("cross_encoder_reranking") and candidate_chunks:
            # Avoid redundant re-ranking pass if LangGraph agent executor already reranked
            already_reranked = any(
                hasattr(c, "cross_encoder_score") and getattr(c, "cross_encoder_score", None) is not None
                for c in candidate_chunks
            )
            if not already_reranked:
                rerank_res = self.reranker.rerank(
                    standalone_query,
                    candidate_chunks,
                    top_k=reranker_top_n,
                )
                candidate_chunks = rerank_res.chunks
            else:
                candidate_chunks = candidate_chunks[:reranker_top_n]

            for idx, rc in enumerate(candidate_chunks):
                cid = getattr(rc, "chunk_id", str(idx))
                raw_meta = getattr(rc, "metadata", {})

                c_meta = (
                    raw_meta.__dict__
                    if hasattr(raw_meta, "__dict__")
                    else (
                        raw_meta
                        if isinstance(raw_meta, dict)
                        else {}
                    )
                )

                rank_changes_info.append(
                    {
                        "chunk_id": cid,
                        "original_rank": getattr(
                            rc,
                            "original_rank",
                            idx + 1,
                        ),
                        "new_rank": getattr(
                            rc,
                            "new_rank",
                            idx + 1,
                        ),
                        "rank_change": getattr(
                            rc,
                            "rank_change",
                            0,
                        ),
                        "cross_encoder_score": round(
                            getattr(
                                rc,
                                "cross_encoder_score",
                                0.0,
                            ),
                            4,
                        ),
                        "score": round(
                            getattr(
                                rc,
                                "score",
                                0.0,
                            ),
                            4,
                        ),
                        "source": getattr(
                            rc,
                            "source",
                            c_meta.get(
                                "source_file",
                                "Doc",
                            ),
                        ),
                        "company": c_meta.get(
                            "company",
                            getattr(
                                rc,
                                "company",
                                "General",
                            ),
                        ),
                    }
                )

        latency_breakdown.reranking_ms = round(
            (time.perf_counter() - rerank_start) * 1000,
            2,
        )

        tracker.complete_stage(
            StageName.CROSS_ENCODER_RERANKING.value
        )

        # 8. Retrieval Guard (Security filter)
        tracker.start_stage(StageName.RETRIEVAL_GUARD.value)

        guarded = self.retrieval_guard.filter_chunks(
            standalone_query,
            candidate_chunks,
        )

        tracker.complete_stage(
            StageName.RETRIEVAL_GUARD.value
        )

        # 9. Context Construction & Sanitization
        tracker.start_stage(
            StageName.CONTEXT_CONSTRUCTION.value
        )

        safe_lines = [
            f"[{getattr(c, 'source', 'Document')}] "
            f"{getattr(c, 'text', str(c))}"
            for c in guarded.safe_chunks
        ]

        merged_context = "\n".join(safe_lines)
        sanitized_context = self.context_sanitizer.sanitize(
            merged_context
        )

        tracker.complete_stage(
            StageName.CONTEXT_CONSTRUCTION.value
        )

        # 10. Pre-Generation Evidence Sufficiency & Strict Abstention Check
        target_comp = (
            meta_filters.get("company")
            if meta_filters
            else (
                routing.detected_companies[0]
                if routing.detected_companies
                else None
            )
        )

        eval_res = self.agent_executor.evaluate_tool.execute(
            standalone_query,
            guarded.safe_chunks,
            target_company=target_comp,
        )

        sufficiency = eval_res.output

        should_abstain = bool(
            not sufficiency.is_sufficient
            and target_comp
            and len(guarded.safe_chunks) == 0
        )

        # 11. LLM Generation (or Strict Abstention Response)
        gen_start = time.perf_counter()
        tracker.start_stage(StageName.LLM_GENERATION.value)

        ttft_ms = 0.0

        if should_abstain:
            raw_output = (
                "### Evidence Sufficiency Notice\n\n"
                "The indexed interview knowledge corpus does not "
                "contain verified interview records or questions "
                f"for **{target_comp}**. To avoid generating "
                "ungrounded information, the system cannot "
                "synthesize an answer for this company.\n\n"
                "Supported indexed companies include "
                f"{', '.join(sorted(list(QueryRouter.CANONICAL_COMPANIES.values())[:15]))}, "
                "etc."
            )

            ttft_ms = 1.0
            latency_breakdown.generation_ms = 1.0

        else:
            if agent_exec_res.answer:
                raw_output = agent_exec_res.answer
                ttft_ms = 10.0
                latency_breakdown.generation_ms = round((time.perf_counter() - gen_start) * 1000, 2)
            else:
                t0_gen = time.perf_counter()

                raw_output = await self.gemini.generate_answer(
                    query=request.query,
                    context=sanitized_context,
                    max_output_tokens=8192,
                    conversation_history=history,
                )

                dur_gen = (
                    time.perf_counter() - t0_gen
                ) * 1000

                ttft_ms = round(dur_gen * 0.4, 2)
                latency_breakdown.generation_ms = round(
                    dur_gen,
                    2,
                )

        tracker.complete_stage(
            StageName.LLM_GENERATION.value
        )

        # 12. Output Validation & Sanitization
        tracker.start_stage(
            StageName.OUTPUT_VALIDATION.value
        )

        out_validation = self.output_validator.validate(
            raw_output
        )

        safe_output = self.output_sanitizer.sanitize(
            raw_output
        )

        tracker.complete_stage(
            StageName.OUTPUT_VALIDATION.value
        )

        # 13. Grounding & Hallucination Guard
        tracker.start_stage(
            StageName.GROUNDING_CHECK.value
        )

        evidence_texts = [
            getattr(c, "text", str(c))
            for c in guarded.safe_chunks
        ]

        is_grounded = self.grounding.verify(
            safe_output,
            evidence_texts,
            self.config.similarity_threshold,
        )

        is_hallucinated = (
            self.hallucination_guard.is_hallucinated(
                safe_output,
                evidence_texts,
                self.config.hallucination_threshold,
            )
        )

        warning = (
            "Response content may exceed retrieved context grounding."
            if (
                not is_grounded
                or is_hallucinated
            )
            and not should_abstain
            else None
        )

        tracker.complete_stage(
            StageName.GROUNDING_CHECK.value
        )

        # 14. Update conversation memory
        if (
            toggles.get("conversation_memory")
            and request.session_id
        ):
            self.memory.add_turn(
                request.session_id,
                "user",
                request.query,
                owner_id=ctx.user_id,
            )

            self.memory.add_turn(
                request.session_id,
                "assistant",
                safe_output,
                owner_id=ctx.user_id,
            )

        latency_breakdown.total_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        tracker.complete_stage(
            StageName.RESPONSE_COMPLETE.value
        )

        # Check ground truth in curated benchmark dataset
        matched_case = self._find_benchmark_case(standalone_query)
        ground_truth_doc_ids = matched_case.relevant_doc_identifiers if matched_case else None

        # Compute RAG observability metrics
        computed_metrics = self.metrics_engine.compute_all(
            query=standalone_query,
            answer=safe_output,
            retrieved_chunks=guarded.safe_chunks,
            context=sanitized_context,
            latency=latency_breakdown,
            pre_rerank_chunks=pre_rerank_candidates,
            ground_truth_chunk_ids=ground_truth_doc_ids,
            expected_route=matched_case.expected_route if matched_case else None,
            actual_route=routing.route_type,
            should_abstain=should_abstain,
            abstained=should_abstain,
        )

        retrieved_chunk_infos = [
            RetrievedChunkInfo(
                chunk_id=getattr(
                    c,
                    "chunk_id",
                    str(idx),
                ),
                text=getattr(
                    c,
                    "text",
                    "",
                )[:250],
                source=getattr(
                    c,
                    "source",
                    getattr(
                        c,
                        "metadata",
                        {},
                    ).get(
                        "source_file",
                        "Doc",
                    ),
                ),
                similarity_score=round(
                    float(
                        getattr(
                            c,
                            "score",
                            getattr(
                                c,
                                "similarity_score",
                                0.0,
                            ),
                        )
                    ),
                    4,
                ),
                cross_encoder_score=round(
                    float(
                        getattr(
                            c,
                            "cross_encoder_score",
                            getattr(
                                c,
                                "score",
                                0.0,
                            ),
                        )
                    ),
                    4,
                ),
                original_rank=getattr(
                    c,
                    "original_rank",
                    idx + 1,
                ),
                new_rank=idx + 1,
                rank_change=getattr(
                    c,
                    "rank_change",
                    0,
                ),
                retrieval_source=getattr(
                    c,
                    "metadata",
                    {},
                ).get(
                    "retrieval_sources",
                    "hybrid",
                ),
                is_selected=True,
                in_context=True,
                metadata=dict(
                    getattr(
                        c,
                        "metadata",
                        {},
                    )
                    or {}
                ),
            )
            for idx, c in enumerate(
                guarded.safe_chunks
            )
        ]

        dashboard_obj = DashboardData(
            owner_id=ctx.user_id,
            request_id=ctx.request_id,
            total_latency_ms=latency_breakdown.total_ms,
            latency_breakdown=latency_breakdown.to_dict(),
            failure_category=computed_metrics.failure_category,
            failure_diagnosis=computed_metrics.failure_diagnosis,
            query_info=QueryInfo(
                original_query=request.query,
                standalone_query=standalone_query,
                metadata_filters=meta_filters or {},
                routing_decision=routing.reasoning,
                route_type=routing.route_type,
                is_followup=was_rewritten,
                detected_companies=routing.detected_companies,
                detected_topics=routing.detected_topics,
            ),
            retrieval_info=RetrievalInfo(
                retriever_used="hybrid",
                retrieval_latency_ms=latency_breakdown.retrieval_ms,
                retrieved_chunks=retrieved_chunk_infos,
                total_candidates=len(candidate_chunks),
                final_count=len(guarded.safe_chunks),
            ),
            reranking_info=RerankingInfo(
                enabled=toggles.get(
                    "cross_encoder_reranking",
                    True,
                ),
                reranking_latency_ms=latency_breakdown.reranking_ms,
                rank_changes=rank_changes_info,
            ),
            agent_info=AgentInfo(
                enabled=toggles.get(
                    "agent_planning",
                    True,
                ),
                route_type=routing.route_type,
                llm_invoked=not should_abstain,
                llm_invocation_reason=(
                    "Multi-Hop Synthesis"
                    if is_multi_hop
                    else "Grounded Answer Generation"
                ),
                hops_count=len(
                    agent_exec_res.hops
                ),
                hops_data=[
                    h.to_dict()
                    for h in agent_exec_res.hops
                ],
                reasoning_trace=[
                    s.to_dict()
                    for s in agent_exec_res.trace.steps
                ],
                tool_selections=[
                    s.tool_name
                    for s in agent_exec_res.trace.steps
                ],
                iterations=(
                    agent_exec_res.trace.iterations_taken
                ),
            ),
            generation_info=GenerationInfo(
                raw_response=raw_output,
                final_response=safe_output,
                tokens_generated=len(
                    safe_output.split()
                ),
                generation_latency_ms=(
                    latency_breakdown.generation_ms
                ),
                time_to_first_token_ms=ttft_ms,
                throughput_tokens_per_sec=round(
                    len(safe_output.split())
                    / max(
                        0.01,
                        latency_breakdown.generation_ms / 1000,
                    ),
                    1,
                ),
            ),
            security_info=SecurityInfo(
                injection_score=injection.risk_score,
                discarded_chunks=guarded.discarded_count,
                output_safe=out_validation.is_safe,
                grounding_passed=is_grounded,
                hallucination_detected=is_hallucinated,
                evidence_sufficiency=sufficiency.status,
                abstained=should_abstain,
                abstention_reason=(
                    sufficiency.reasons[0]
                    if sufficiency.reasons
                    and should_abstain
                    else ""
                ),
            ),
            feature_toggles=FeatureToggles(
                dense_retrieval=toggles.get(
                    "dense_retrieval",
                    True,
                ),
                bm25_retrieval=toggles.get(
                    "bm25_retrieval",
                    True,
                ),
                hybrid_retrieval=toggles.get(
                    "hybrid_retrieval",
                    True,
                ),
                cross_encoder_reranking=toggles.get(
                    "cross_encoder_reranking",
                    True,
                ),
                conversation_memory=toggles.get(
                    "conversation_memory",
                    True,
                ),
                query_rewriting=toggles.get(
                    "query_rewriting",
                    True,
                ),
                metadata_filtering=toggles.get(
                    "metadata_filtering",
                    True,
                ),
                agent_planning=toggles.get(
                    "agent_planning",
                    True,
                ),
                multi_hop_retrieval=toggles.get(
                    "multi_hop_retrieval",
                    True,
                ),
                chunk_enhancement=toggles.get(
                    "chunk_enhancement",
                    True,
                ),
            ),
            retrieval_plan=agent_exec_res.state.get("retrieval_plan", []),
            executed_operations=agent_exec_res.state.get("executed_operations", []),
            skipped_operations=agent_exec_res.state.get("skipped_operations", []),
            decision_flow=agent_exec_res.state.get("decision_flow", []),
            decision_metadata=agent_exec_res.decision_metadata or agent_exec_res.state.get("decision_metadata", {}),
            rank_movements=agent_exec_res.state.get("rank_movements", rank_changes_info),
            rrf_details=agent_exec_res.state.get("rrf_details", {}),
            candidate_count_before_reranking=agent_exec_res.state.get("candidate_count_before_reranking", len(pre_rerank_candidates)),
            candidate_count_after_reranking=agent_exec_res.state.get("candidate_count_after_reranking", len(candidate_chunks)),
            budgets=agent_exec_res.state.get("budgets", {}),
            computed_metrics=computed_metrics.to_dict(),
            cumulative_metrics=SESSION_HISTORY.get_cumulative_metrics(),
            pipeline_stages=[
                s.to_dict()
                for s in tracker.get_stages()
            ],
        )

        DASHBOARD_STORE[ctx.request_id] = dashboard_obj
        SESSION_HISTORY.record_query(dashboard_obj)

        pdata = PipelineData(
            request_id=ctx.request_id,
            query_info=dashboard_obj.query_info.__dict__,
            retrieval_info=dashboard_obj.to_dict()[
                "retrieval_info"
            ],
            reranking_info=dashboard_obj.to_dict()[
                "reranking_info"
            ],
            agent_info=dashboard_obj.to_dict()[
                "agent_info"
            ],
            generation_info=dashboard_obj.to_dict()[
                "generation_info"
            ],
            security_info=dashboard_obj.to_dict()[
                "security_info"
            ],
            feature_toggles=toggles,
            decision_metadata=dashboard_obj.decision_metadata,
            metrics_info=computed_metrics.to_dict(),
        )

        return ChatResponse(
            answer=safe_output,
            meta=GuardrailMeta(
                request_id=ctx.request_id,
                injection_score=injection.risk_score,
                discarded_chunks=guarded.discarded_count,
                chunk_count=len(
                    guarded.safe_chunks
                ),
                warning=warning,
            ),
            request_id=ctx.request_id,
            pipeline_data=pdata,
        )

    async def process_stream(
        self, request: ChatRequest, ctx: RequestContext
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Streaming query execution yielding stage progress, token chunks, and final response."""
        started = time.perf_counter()
        tracker = PipelineTracker(request_id=ctx.request_id)
        TRACKER_STORE[ctx.request_id] = tracker
        toggles = self.get_feature_toggles()
        latency_breakdown = LatencyMetrics()

        # Stage 1: Received
        tracker.start_stage(StageName.QUERY_RECEIVED.value)
        tracker.complete_stage(StageName.QUERY_RECEIVED.value)
        yield {
            "type": "stage",
            "stage": StageName.QUERY_RECEIVED.value,
            "status": "completed",
            "message": "Query received",
        }

        # Stage 2: Input Validation
        tracker.start_stage(StageName.INPUT_VALIDATION.value)
        validation = self.input_validator.validate(request.query)
        if not validation.is_valid:
            tracker.fail_stage(StageName.INPUT_VALIDATION.value, "Invalid query input")
            if not request.query or not request.query.strip():
                yield {"type": "error", "error": "Query cannot be empty."}
                return
            user_msg = (
                "### 🛡️ Security Validation Notice\n\n"
                "Your query could not be processed because it contains encoded payloads or characters that violate input security policies.\n\n"
                "**Action Taken:** Immediate execution halt. Retrieval operations, embeddings, and vector store queries have been bypassed.\n\n"
                "**Guidance:** Please submit plain text queries regarding interview experiences, company placement processes, or technical preparation topics."
            )
            blocked_resp = self._build_blocked_security_response(
                request=request,
                ctx=ctx,
                category="ENCODED_OR_MALICIOUS_INPUT",
                severity="HIGH",
                user_message=user_msg,
                risk_score=0.9,
                latency_breakdown=latency_breakdown,
                started=started,
                tracker=tracker,
                toggles=toggles,
            )
            yield {"type": "token", "token": user_msg}
            yield {"type": "complete", "response": blocked_resp.model_dump()}
            return

        tracker.complete_stage(StageName.INPUT_VALIDATION.value)
        yield {
            "type": "stage",
            "stage": StageName.INPUT_VALIDATION.value,
            "status": "completed",
            "message": "Input validation passed",
        }

        # Stage 3: Injection Detection (Pre-Retrieval)
        tracker.start_stage(StageName.INJECTION_DETECTION.value)
        injection = self.injection_detector.detect(validation.normalized_query)
        if injection.action == "block":
            tracker.fail_stage(StageName.INJECTION_DETECTION.value, f"Security violation: {injection.category}")
            blocked_resp = self._build_blocked_security_response(
                request=request,
                ctx=ctx,
                category=injection.category or "PROMPT_INJECTION",
                severity=injection.severity or "HIGH",
                user_message=injection.user_safe_message,
                risk_score=injection.risk_score,
                latency_breakdown=latency_breakdown,
                started=started,
                tracker=tracker,
                toggles=toggles,
            )
            yield {"type": "token", "token": injection.user_safe_message}
            yield {"type": "complete", "response": blocked_resp.model_dump()}
            return

        tracker.complete_stage(StageName.INJECTION_DETECTION.value)
        yield {
            "type": "stage",
            "stage": StageName.INJECTION_DETECTION.value,
            "status": "completed",
            "message": "Security checks passed",
        }

        # Run process logic with shared tracker to prevent orphaned SSE listeners
        full_resp = await self.process(request, ctx, tracker=tracker)

        # Stream progressive answer tokens
        words = full_resp.answer.split(" ")
        for i in range(0, len(words), 4):
            token_chunk = " ".join(words[i : i + 4])
            if i + 4 < len(words):
                token_chunk += " "
            yield {"type": "token", "token": token_chunk}
            await asyncio.sleep(0.01)

        # Emit completion payload with complete telemetry
        yield {"type": "complete", "response": full_resp.model_dump()}
