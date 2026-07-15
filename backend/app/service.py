"""Secure Agentic RAG orchestration service with layered guardrails, observability, and feature toggles."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from security import (
    ContextSanitizer,
    GroundingVerifier,
    HallucinationGuard,
    InputValidator,
    OutputSanitizer,
    OutputValidator,
    PromptInjectionDetector,
    RetrievalGuard,
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
from backend.core.agent.executor import AgentExecutor
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
from backend.core.retrieval.router import QueryRouter
from backend.core.vector_store.chroma import ChromaVectorStore
from backend.evaluation.metrics import LatencyMetrics, RAGMetrics
from backend.observability.dashboard import (
    AgentInfo,
    DashboardData,
    FeatureToggles,
    GenerationInfo,
    HyDEInfo,
    MemoryInfo,
    QueryInfo,
    RerankingInfo,
    RetrievalInfo,
    RetrievedChunkInfo,
    SecurityInfo,
)
from backend.observability.pipeline_tracker import PipelineTracker, StageName, StageStatus

logger = logging.getLogger(__name__)

# Global stores for developer dashboard inspection and active trackers
DASHBOARD_STORE: dict[str, DashboardData] = {}
TRACKER_STORE: dict[str, PipelineTracker] = {}


@dataclass
class RequestContext:
    client_ip: str
    request_id: str


class SecureRAGService:
    """Production-style security & agentic RAG pipeline."""

    def __init__(self) -> None:
        self.config = load_config()
        self.logger = SecurityLogger()

        # Security layers (Preserved 100%)
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
            self._hybrid_retriever = HybridRetriever(self.dense_retriever, self.bm25_retriever)
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
            self._agent_executor = AgentExecutor(self.hybrid_retriever, self.gemini)
        return self._agent_executor

    def get_feature_toggles(self) -> dict[str, bool]:
        return dict(DEFAULT_FEATURE_TOGGLES)

    def update_feature_toggles(self, updates: dict[str, bool]) -> dict[str, bool]:
        for k, v in updates.items():
            if k in DEFAULT_FEATURE_TOGGLES:
                DEFAULT_FEATURE_TOGGLES[k] = bool(v)
        return dict(DEFAULT_FEATURE_TOGGLES)

    async def process(self, request: ChatRequest, ctx: RequestContext) -> ChatResponse:
        started = time.perf_counter()
        tracker = PipelineTracker(request_id=ctx.request_id)
        TRACKER_STORE[ctx.request_id] = tracker
        toggles = self.get_feature_toggles()
        latency_breakdown = LatencyMetrics()

        tracker.start_stage(StageName.QUERY_RECEIVED.value)
        tracker.complete_stage(StageName.QUERY_RECEIVED.value)

        # 1. Input Validation
        tracker.start_stage(StageName.INPUT_VALIDATION.value)
        validation = self.input_validator.validate(request.query)
        if not validation.is_valid:
            tracker.fail_stage(StageName.INPUT_VALIDATION.value, "Invalid query input")
            self.logger.info("input_rejected", request_id=ctx.request_id, reasons=validation.reasons)
            raise ValueError("Request could not be processed.")
        tracker.complete_stage(StageName.INPUT_VALIDATION.value)

        # 2. Prompt Injection Detection
        tracker.start_stage(StageName.INJECTION_DETECTION.value)
        injection = self.injection_detector.detect(validation.normalized_query)
        if injection.action == "block":
            tracker.fail_stage(StageName.INJECTION_DETECTION.value, "Prompt injection blocked")
            self.logger.info("injection_blocked", request_id=ctx.request_id, score=injection.risk_score)
            raise ValueError("Request could not be processed.")
        tracker.complete_stage(StageName.INJECTION_DETECTION.value)

        # 3. Rate Limiting
        tracker.start_stage(StageName.RATE_LIMITING.value)
        limit = self.rate_limiter.allow(ctx.client_ip, request.session_id)
        if not limit.allowed:
            tracker.fail_stage(StageName.RATE_LIMITING.value, "Rate limit exceeded")
            raise PermissionError(f"Too many requests. Retry in {limit.retry_after_seconds} seconds.")
        tracker.complete_stage(StageName.RATE_LIMITING.value)

        # 4. Conversation Memory & Query Rewriting
        tracker.start_stage(StageName.MEMORY_LOOKUP.value)
        history = []
        if toggles.get("conversation_memory") and request.session_id:
            history = self.memory.get_summarized_history(request.session_id)
        tracker.complete_stage(StageName.MEMORY_LOOKUP.value)

        tracker.start_stage(StageName.QUERY_REWRITING.value)
        standalone_query = validation.normalized_query
        was_rewritten = False
        rewrite_reason = ""
        if toggles.get("query_rewriting") and history:
            rew_res = await self.query_rewriter.rewrite(validation.normalized_query, history)
            standalone_query = rew_res.rewritten_query
            was_rewritten = rew_res.was_rewritten
            rewrite_reason = rew_res.reasoning
        tracker.complete_stage(StageName.QUERY_REWRITING.value)

        # 5. Query Routing & Metadata Filtering
        tracker.start_stage(StageName.QUERY_ANALYSIS.value)
        routing = self.query_router.route(standalone_query)
        meta_filters = routing.metadata_filters if toggles.get("metadata_filtering") else None
        tracker.complete_stage(StageName.QUERY_ANALYSIS.value)

        # 6. Retrieval Stage (with HyDE if enabled)
        self._last_hyde_doc = ""
        self._last_hyde_latency_ms = 0.0
        ret_start = time.perf_counter()
        if toggles.get("hyde"):
            try:
                hyde_t = time.perf_counter()
                if hasattr(self, "hyde_generator") and self.hyde_generator:
                    hyde_res = await self.hyde_generator.generate_and_embed(standalone_query)
                    self._last_hyde_doc = hyde_res.hypothetical_document
                    self._last_hyde_latency_ms = hyde_res.latency_ms
            except Exception as e:
                logger.warning("HyDE generation error: %s", e)

        tracker.start_stage(StageName.DENSE_RETRIEVAL.value)
        retriever_used = "hybrid"
        if toggles.get("hybrid_retrieval"):
            ret_res = await self.hybrid_retriever.retrieve(
                standalone_query, top_k=15, metadata_filters=meta_filters
            )
            retriever_used = "hybrid"
        elif toggles.get("dense_retrieval"):
            ret_res = await self.dense_retriever.retrieve(
                standalone_query, top_k=15, metadata_filters=meta_filters
            )
            retriever_used = "dense"
        elif toggles.get("bm25_retrieval"):
            ret_res = await self.bm25_retriever.retrieve(
                standalone_query, top_k=15, metadata_filters=meta_filters
            )
            retriever_used = "bm25"
        else:
            # Fallback to legacy TF-IDF
            legacy_c = self.legacy_retriever.retrieve(
                standalone_query,
                self.legacy_vector_store.search(standalone_query, top_k=self.config.max_retrieved_chunks),
            )
            ret_res = None
            candidate_chunks = legacy_c
            retriever_used = "legacy_tfidf"

        if ret_res is not None:
            candidate_chunks = ret_res.chunks
            # If Chroma is empty or returned nothing, fallback to legacy
            if not candidate_chunks:
                legacy_c = self.legacy_retriever.retrieve(
                    standalone_query,
                    self.legacy_vector_store.search(standalone_query, top_k=self.config.max_retrieved_chunks),
                )
                candidate_chunks = legacy_c

        # Strictly enforce company metadata filter across all candidates (whether Chroma or legacy fallback)
        if meta_filters and meta_filters.get("company"):
            target_comp = str(meta_filters["company"]).strip().lower()
            candidate_chunks = [
                c for c in candidate_chunks
                if str(getattr(c, "metadata", {}).get("company", "") if isinstance(getattr(c, "metadata", None), dict) else getattr(c, "company", "")).strip().lower() == target_comp
            ]

        latency_breakdown.retrieval_ms = round((time.perf_counter() - ret_start) * 1000, 2)
        tracker.complete_stage(StageName.DENSE_RETRIEVAL.value)

        # 7. Reranking Stage
        rerank_start = time.perf_counter()
        tracker.start_stage(StageName.CROSS_ENCODER_RERANKING.value)
        rank_changes_info = []
        orig_scores = {getattr(c, "chunk_id", str(idx)): getattr(c, "score", getattr(c, "similarity_score", 0.0)) for idx, c in enumerate(candidate_chunks)}
        orig_ranks = {getattr(c, "chunk_id", str(idx)): idx + 1 for idx, c in enumerate(candidate_chunks)}
        if toggles.get("cross_encoder_reranking") and candidate_chunks:
            rerank_res = self.reranker.rerank(standalone_query, candidate_chunks, top_k=15)
            candidate_chunks = rerank_res.chunks
            for idx, rc in enumerate(candidate_chunks):
                cid = getattr(rc, "chunk_id", str(idx))
                c_meta = dict(getattr(rc, "metadata", {}) or {})
                rank_changes_info.append(
                    {
                        "chunk_id": cid,
                        "original_rank": getattr(rc, "original_rank", orig_ranks.get(cid, idx + 1)),
                        "new_rank": getattr(rc, "new_rank", idx + 1),
                        "rank_change": getattr(rc, "rank_change", getattr(rc, "original_rank", orig_ranks.get(cid, idx + 1)) - (idx + 1)),
                        "cross_encoder_score": round(getattr(rc, "cross_encoder_score", getattr(rc, "score", 0.0)), 4),
                        "ce_score": round(getattr(rc, "cross_encoder_score", getattr(rc, "score", 0.0)), 4),
                        "original_score": round(orig_scores.get(cid, getattr(rc, "score", 0.0)), 4),
                        "source": getattr(rc, "source", c_meta.get("source_file", "Doc")),
                        "snippet": getattr(rc, "text", "")[:150],
                        "company": c_meta.get("company", getattr(rc, "company", "")),
                    }
                )
        latency_breakdown.reranking_ms = round((time.perf_counter() - rerank_start) * 1000, 2)
        tracker.complete_stage(StageName.CROSS_ENCODER_RERANKING.value)

        # 8. Retrieval Guard (Security filter)
        tracker.start_stage(StageName.RETRIEVAL_GUARD.value)
        guarded = self.retrieval_guard.filter_chunks(standalone_query, candidate_chunks)
        tracker.complete_stage(StageName.RETRIEVAL_GUARD.value)

        # 9. Context Construction & Sanitization
        tracker.start_stage(StageName.CONTEXT_CONSTRUCTION.value)
        safe_lines = [
            f"[{getattr(c, 'source', 'Document')}] {getattr(c, 'text', str(c))}"
            for c in guarded.safe_chunks
        ]
        merged_context = "\n".join(safe_lines)
        sanitized_context = self.context_sanitizer.sanitize(merged_context)
        tracker.complete_stage(StageName.CONTEXT_CONSTRUCTION.value)

        # 10. Agent Reasoning Plan (if toggled)
        tracker.start_stage(StageName.AGENT_EVALUATION.value)
        agent_planner = AgentPlanner()
        agent_plan = agent_planner.analyze(
            query=request.query,
            rewritten_query=standalone_query,
            metadata_filters=meta_filters,
            hyde_used=toggles.get("hyde", False),
            multi_hop_enabled=toggles.get("multi_hop_retrieval", False),
        )
        tracker.complete_stage(StageName.AGENT_EVALUATION.value)

        # 11. LLM Generation
        gen_start = time.perf_counter()
        tracker.start_stage(StageName.LLM_GENERATION.value)
        raw_output = await self.gemini.generate_answer(
            query=request.query,
            context=sanitized_context,
            max_output_tokens=8192,  # Increased to 8192 so comprehensive handbook responses never cut off
            conversation_history=history,
        )
        latency_breakdown.generation_ms = round((time.perf_counter() - gen_start) * 1000, 2)
        tracker.complete_stage(StageName.LLM_GENERATION.value)

        # 12. Output Validation & Sanitization
        tracker.start_stage(StageName.OUTPUT_VALIDATION.value)
        out_validation = self.output_validator.validate(raw_output)
        if not out_validation.is_safe:
            self.logger.info("output_blocked", request_id=ctx.request_id, reasons=out_validation.reasons)
            raise ValueError("Response blocked by output safety policies.")

        safe_output = self.output_sanitizer.sanitize(raw_output)
        tracker.complete_stage(StageName.OUTPUT_VALIDATION.value)

        # 13. Grounding & Hallucination Guard
        tracker.start_stage(StageName.GROUNDING_CHECK.value)
        evidence_texts = [getattr(c, "text", str(c)) for c in guarded.safe_chunks]
        is_grounded = self.grounding.verify(safe_output, evidence_texts, self.config.similarity_threshold)
        is_hallucinated = self.hallucination_guard.is_hallucinated(
            safe_output, evidence_texts, self.config.hallucination_threshold
        )
        warning: str | None = None
        if not is_grounded or is_hallucinated:
            warning = "Response content may exceed retrieved context grounding."
        tracker.complete_stage(StageName.GROUNDING_CHECK.value)

        # 14. Update conversation memory
        if toggles.get("conversation_memory") and request.session_id:
            self.memory.add_turn(request.session_id, "user", request.query)
            self.memory.add_turn(request.session_id, "assistant", safe_output)

        latency_breakdown.total_ms = round((time.perf_counter() - started) * 1000, 2)
        tracker.complete_stage(StageName.RESPONSE_COMPLETE.value)

        # Compute RAG observability metrics
        computed_metrics = self.metrics_engine.compute_all(
            query=standalone_query,
            answer=safe_output,
            retrieved_chunks=guarded.safe_chunks,
            context=sanitized_context,
            latency=latency_breakdown,
        )

        # Build dashboard data model
        retrieved_chunk_infos = []
        for idx, c in enumerate(guarded.safe_chunks):
            c_meta = dict(getattr(c, "metadata", {}) or {})
            if "company" not in c_meta and hasattr(c, "company"):
                c_meta["company"] = getattr(c, "company", "General")
            if "source_file" not in c_meta:
                c_meta["source_file"] = getattr(c, "source", getattr(c, "source_path", "Doc"))
            retrieved_chunk_infos.append(
                RetrievedChunkInfo(
                    text=getattr(c, "text", "")[:200],
                    source=getattr(c, "source", getattr(c, "metadata", {}).get("source_file", "Doc")),
                    similarity_score=getattr(c, "score", getattr(c, "similarity_score", 0.0)),
                    metadata=c_meta,
                    rank=idx + 1,
                )
            )
        rejected_chunk_infos = [
            RetrievedChunkInfo(
                text=getattr(c, "text", "")[:200],
                source=getattr(c, "source", "Doc"),
                was_rejected=True,
                rejection_reason="Blocked by security filter or threshold",
            )
            for c in getattr(guarded, "discarded_chunks", [])
        ]

        structured_trace = [
            {"type": "decision", "question": "Needs Hybrid Retrieval & Vector Search?", "answer": "YES" if agent_plan.trace.need_retrieval else "NO"},
            {"type": "decision", "question": "Requires Pronoun/Context Query Rewrite?", "answer": "YES" if agent_plan.trace.need_query_rewrite else "NO"},
            {"type": "decision", "question": "Requires Company/Topic Metadata Filter?", "answer": "YES" if agent_plan.trace.need_metadata_filter else "NO"},
            {"type": "decision", "question": "Requires Multi-Hop Query Decomposition?", "answer": "YES" if agent_plan.trace.need_multi_hop else "NO"},
            {"type": "decision", "question": "Requires HyDE Query Expansion?", "answer": "YES" if agent_plan.trace.need_hyde else "NO"},
        ] + [
            {"type": "tool", "tool": getattr(s, "tool_name", "Tool"), "reason": getattr(s, "reasoning", str(s)), "latency": getattr(s, "latency_ms", 0.0)}
            for s in agent_plan.trace.steps
        ]

        dashboard_obj = DashboardData(
            request_id=ctx.request_id,
            total_latency_ms=latency_breakdown.total_ms,
            query_info=QueryInfo(
                original_query=request.query,
                standalone_query=standalone_query,
                metadata_filters=meta_filters or {},
                routing_decision=routing.reasoning,
                is_followup=was_rewritten,
            ),
            retrieval_info=RetrievalInfo(
                retriever_used=retriever_used,
                retrieval_latency_ms=latency_breakdown.retrieval_ms,
                retrieved_chunks=retrieved_chunk_infos,
                rejected_chunks=rejected_chunk_infos,
                total_candidates=len(candidate_chunks),
                final_count=len(guarded.safe_chunks),
            ),
            reranking_info=RerankingInfo(
                enabled=toggles.get("cross_encoder_reranking", False),
                reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                reranking_latency_ms=latency_breakdown.reranking_ms,
                rank_changes=rank_changes_info,
            ),
            agent_info=AgentInfo(
                enabled=toggles.get("agent_planning", False),
                reasoning_trace=structured_trace,
                tool_selections=[getattr(s, "tool_name", str(s)) for s in agent_plan.trace.steps],
                planning_steps=[getattr(s, "reasoning", str(s)) for s in agent_plan.trace.steps],
                iterations=agent_plan.trace.iterations_taken,
            ),
            generation_info=GenerationInfo(
                system_prompt="You are an expert technical interview assistant.",
                user_prompt=sanitized_context[:500] + "..." if len(sanitized_context) > 500 else sanitized_context,
                raw_response=raw_output,
                final_response=safe_output,
                generation_latency_ms=latency_breakdown.generation_ms,
            ),
            security_info=SecurityInfo(
                injection_score=injection.risk_score,
                injection_action=injection.action,
                discarded_chunks=guarded.discarded_count,
                rate_limited=False,
                output_safe=out_validation.is_safe,
                grounding_passed=is_grounded,
                hallucination_detected=is_hallucinated,
            ),
            hyde_info=HyDEInfo(
                enabled=toggles.get("hyde", False),
                hypothetical_document=getattr(self, "_last_hyde_doc", "") if toggles.get("hyde", False) else "",
                generation_latency_ms=getattr(self, "_last_hyde_latency_ms", 0.0) if toggles.get("hyde", False) else 0.0,
            ),
            memory_info=MemoryInfo(
                enabled=toggles.get("conversation_memory", True),
                conversation_history=history if toggles.get("conversation_memory", True) else [],
                rewritten_query=standalone_query if was_rewritten else "",
                history_length=len(history) if toggles.get("conversation_memory", True) else 0,
            ),
            feature_toggles=FeatureToggles(
                dense_retrieval=toggles.get("dense_retrieval", True),
                bm25_retrieval=toggles.get("bm25_retrieval", True),
                hybrid_retrieval=toggles.get("hybrid_retrieval", True),
                cross_encoder_reranking=toggles.get("cross_encoder_reranking", True),
                conversation_memory=toggles.get("conversation_memory", True),
                query_rewriting=toggles.get("query_rewriting", True),
                metadata_filtering=toggles.get("metadata_filtering", True),
                hyde=toggles.get("hyde", False),
                agent_planning=toggles.get("agent_planning", True),
                multi_hop_retrieval=toggles.get("multi_hop_retrieval", False),
                chunk_enhancement=toggles.get("chunk_enhancement", True),
                hybrid_search=toggles.get("hybrid_retrieval", True),
                reranking=toggles.get("cross_encoder_reranking", True),
                agent_mode=toggles.get("agent_planning", True),
            ),
            pipeline_stages=[s.to_dict() for s in tracker.get_stages()],
        )
        DASHBOARD_STORE[ctx.request_id] = dashboard_obj

        # Convert to Pydantic PipelineData for response
        pipeline_data = PipelineData(
            request_id=ctx.request_id,
            query_info=dashboard_obj.query_info.__dict__,
            retrieval_info={
                "retriever_used": dashboard_obj.retrieval_info.retriever_used,
                "retrieval_latency_ms": dashboard_obj.retrieval_info.retrieval_latency_ms,
                "retrieved_chunks": [c.__dict__ for c in dashboard_obj.retrieval_info.retrieved_chunks],
                "rejected_chunks": [c.__dict__ for c in dashboard_obj.retrieval_info.rejected_chunks],
            },
            reranking_info={
                "enabled": dashboard_obj.reranking_info.enabled,
                "reranker_model": dashboard_obj.reranking_info.reranker_model,
                "reranking_latency_ms": dashboard_obj.reranking_info.reranking_latency_ms,
                "rank_changes": dashboard_obj.reranking_info.rank_changes,
            },
            agent_info={
                "enabled": dashboard_obj.agent_info.enabled,
                "reasoning_trace": dashboard_obj.agent_info.reasoning_trace,
                "tool_selections": dashboard_obj.agent_info.tool_selections,
                "iterations": dashboard_obj.agent_info.iterations,
            },
            generation_info=dashboard_obj.generation_info.__dict__,
            pipeline_stages=[
                PipelineStageModel(
                    name=s["name"],
                    status=s["status"],
                    start_time=s.get("start_time", 0.0),
                    end_time=s.get("end_time", 0.0),
                    latency_ms=s.get("latency_ms", 0.0),
                    metadata=s.get("metadata", {}),
                )
                for s in dashboard_obj.pipeline_stages
            ],
            feature_toggles=toggles,
            security_info={
                "injection_score": injection.risk_score,
                "discarded_chunks": guarded.discarded_count,
            },
            metrics_info=computed_metrics.to_dict(),
        )

        return ChatResponse(
            answer=safe_output,
            meta=GuardrailMeta(
                request_id=ctx.request_id,
                injection_score=injection.risk_score,
                discarded_chunks=guarded.discarded_count,
                chunk_count=len(guarded.safe_chunks),
                warning=warning,
            ),
            request_id=ctx.request_id,
            pipeline_data=pipeline_data,
        )

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            gemini=True,
            vector_index=True,
            readiness=True,
        )

    def security_status(self) -> SecurityStatusResponse:
        return SecurityStatusResponse(
            status="active",
            config={
                "max_query_length": self.config.max_query_length,
                "max_context_length": self.config.max_context_length,
                "rate_limit": self.config.rate_limit,
                "similarity_threshold": self.config.similarity_threshold,
                "hallucination_threshold": self.config.hallucination_threshold,
            },
        )

    def reindex(self) -> ReindexResponse:
        try:
            pipeline = IngestionPipeline(Path("data"), ChunkingConfig(strategy="recursive", chunk_size=500, chunk_overlap=50))
            chunks, _ = pipeline.run(force_rebuild=True)
            if chunks:
                doc_chunks: list[DocumentChunk] = []
                for i, c in enumerate(chunks):
                    meta = ChunkMetadata(
                        chunk_id=c.chunk_id or f"reidx_{i}_{hash(c.text)}",
                        source_file=c.source_path,
                        company=c.metadata.get("company", "General"),
                        page_number=c.page_number or 0,
                        chunk_index=c.chunk_index or i,
                        chunk_strategy="recursive",
                        tags=[t.strip() for t in c.metadata.get("tags", "").split(",") if t.strip()] if isinstance(c.metadata.get("tags"), str) else [],
                        topic=c.metadata.get("topic", "General"),
                        difficulty=c.metadata.get("difficulty", "Medium"),
                        role=c.metadata.get("role", "SDE"),
                        enhancement_applied=",".join(c.enhancements_applied),
                    )
                    doc_chunks.append(DocumentChunk(text=c.enhanced_text or c.text, metadata=meta))
                texts = [dc.text for dc in doc_chunks]
                embeddings = asyncio.run(self.embedder.embed_batch(texts))
                self.chroma_store.add_documents(doc_chunks, embeddings)
                if hasattr(self.bm25_retriever, "_index_corpus"):
                    self.bm25_retriever._index_corpus()
        except Exception as exc:
            logger.error("Reindexing failed: %s", exc)
        return ReindexResponse(status="reindexed", chunks_indexed=self.chroma_store.get_stats().total_chunks)


def build_request_context(client_ip: str | None = None, request_id: str | None = None) -> RequestContext:
    return RequestContext(
        client_ip=client_ip or "127.0.0.1",
        request_id=request_id or str(uuid.uuid4()),
    )
