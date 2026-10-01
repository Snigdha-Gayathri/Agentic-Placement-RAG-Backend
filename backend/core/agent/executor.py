"""Iterative agent executor orchestrating retrieval, evaluation, and multi-hop reasoning via LangGraph."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .graph import RetrievalGraphBuilder
from .planner import AgentPlanner, PlanStep, ReasoningTrace
from .state import AgenticRAGState
from .tools import (
    BM25RetrieveTool,
    ChunkEnhanceTool,
    DecomposeQueryTool,
    DenseRetrieveTool,
    EvaluateEvidenceTool,
    EvidenceSufficiencyResult,
    HybridRetrieveTool,
    HyDETool,
    RerankTool,
)

logger = logging.getLogger(__name__)


@dataclass
class HopRecord:
    """Telemetry record for an individual retrieval hop."""

    hop_index: int
    subquery: str
    chunks_retrieved: int
    latency_ms: float
    status: str
    target_filter: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hop_index": self.hop_index,
            "subquery": self.subquery,
            "chunks_retrieved": self.chunks_retrieved,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "target_filter": self.target_filter,
        }


@dataclass
class AgentExecutionResult:
    """Complete result from AgentExecutor and LangGraph retrieval agent."""

    chunks: list[Any]
    trace: ReasoningTrace
    hops: list[HopRecord]
    sufficiency: EvidenceSufficiencyResult
    total_latency_ms: float
    state: dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    disclosure: str = ""
    decision_metadata: dict[str, Any] = field(default_factory=dict)


class AgentExecutor:
    """Executes the compiled LangGraph StateGraph retrieval agent."""

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        max_iterations: int = 3,
        dense_retriever: Any | None = None,
        bm25_retriever: Any | None = None,
        reranker: Any | None = None,
        hyde_generator: Any | None = None,
        query_rewriter: Any | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.max_iterations = max_iterations

        # Extract sub-retrievers if passed hybrid retriever
        self.dense_retriever = dense_retriever or getattr(retriever, "dense_retriever", None)
        self.bm25_retriever = bm25_retriever or getattr(retriever, "bm25_retriever", None)
        self.vector_store = getattr(self.dense_retriever, "vector_store", None) or getattr(retriever, "vector_store", None)

        # Fallback mocks/defaults for tools if components are omitted
        if self.bm25_retriever is not None:
            self.bm25_tool = BM25RetrieveTool(self.bm25_retriever)
        else:
            self.bm25_tool = BM25RetrieveTool(retriever)

        if self.dense_retriever is not None:
            self.dense_tool = DenseRetrieveTool(self.dense_retriever)
        else:
            self.dense_tool = DenseRetrieveTool(retriever)

        self.hybrid_tool = HybridRetrieveTool(retriever)

        if hyde_generator is not None and self.vector_store is not None:
            self.hyde_tool = HyDETool(hyde_generator, self.vector_store)
        else:
            # Simple fallback wrapper
            class _DummyHyDE:
                async def execute(self, query: str, top_k: int = 10, metadata_filters: dict | None = None):
                    return await retriever.retrieve(query, top_k=top_k, metadata_filters=metadata_filters)
            self.hyde_tool = HyDETool(_DummyHyDE(), self.vector_store)

        if reranker is not None:
            self.rerank_tool = RerankTool(reranker)
        else:
            from backend.core.reranking.cross_encoder import CrossEncoderReranker
            self.rerank_tool = RerankTool(CrossEncoderReranker())

        self.chunk_enhance_tool = ChunkEnhanceTool()
        self.decompose_tool = DecomposeQueryTool()
        self.evaluate_tool = EvaluateEvidenceTool()
        self.query_rewriter = query_rewriter
        self.planner = AgentPlanner()

        # Build and compile LangGraph agent
        self.graph_builder = RetrievalGraphBuilder(
            bm25_tool=self.bm25_tool,
            dense_tool=self.dense_tool,
            hybrid_tool=self.hybrid_tool,
            hyde_tool=self.hyde_tool,
            rerank_tool=self.rerank_tool,
            chunk_enhance_tool=self.chunk_enhance_tool,
            decompose_tool=self.decompose_tool,
            evaluate_tool=self.evaluate_tool,
            query_rewriter=self.query_rewriter,
            gemini_client=self.llm,
            planner=self.planner,
            max_hops=max_iterations,
        )
        self.graph = self.graph_builder.create_graph()

    async def execute(
        self,
        query: str,
        rewritten_query: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
        detected_companies: list[str] | None = None,
        detected_topics: list[str] | None = None,
        hyde_used: bool = False,
        multi_hop_enabled: bool = False,
        top_k: int = 15,
        conversation_history: list[dict[str, Any]] | None = None,
        is_blocked: bool = False,
        security_event: dict[str, Any] | None = None,
        active_toggles: dict[str, bool] | None = None,
        session_id: str | None = None,
        decision_backend: Any | None = None,
    ) -> AgentExecutionResult:
        """Run the LangGraph agent for the given query and parameters."""
        start_exec = time.perf_counter()

        if decision_backend is not None:
            self.planner.decision_backend = decision_backend

        # Initial state setup
        initial_state: AgenticRAGState = {
            "original_query": query,
            "rewritten_query": rewritten_query or query,
            "conversation_history": conversation_history or [],
            "metadata_filter": metadata_filters or {},
            "metadata_filter_used": bool(metadata_filters),
            "target_company": detected_companies[0] if detected_companies else None,
            "detected_topics": detected_topics or [],
            "is_blocked": is_blocked,
            "security_event": security_event or {},
            "active_toggles": active_toggles or {},
            "session_id": session_id or "",
            "decision_metadata": {},
            "retrieval_trace": [],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "final_context_chunks": [],
            "executed_operations": [],
            "skipped_operations": [],
            "escalation_count": 0,
            "budgets": {
                "max_hops": self.max_iterations,
                "max_retrieval_calls": 6,
                "calls_made": 0,
            },
        }

        # Execute through LangGraph StateGraph
        try:
            final_state = await self.graph.ainvoke(initial_state)
        except Exception as exc:
            logger.error("LangGraph agent execution error: %s", exc)
            # Safe deterministic fallback
            fb_res = await self.hybrid_tool.execute(query, top_k=top_k, metadata_filters=metadata_filters)
            final_state = dict(initial_state)
            final_state["retrieved_chunks"] = fb_res.output
            final_state["final_context_chunks"] = fb_res.output
            final_state["executed_operations"] = ["HYBRID_RETRIEVAL_FALLBACK"]
            final_state["evidence_assessment"] = {
                "status": "PARTIALLY_SUFFICIENT",
                "is_sufficient": len(fb_res.output) > 0,
                "score": 0.5,
                "reasons": ["Fallback retrieval executed."],
                "missing_aspects": [],
            }

        total_lat = round((time.perf_counter() - start_exec) * 1000, 2)

        # Build backward-compatible trace and hops
        trace_steps = [
            PlanStep(
                tool_name=s.get("tool_name", "Tool"),
                reasoning=s.get("reasoning", ""),
                inputs=s.get("inputs", {}),
                output_summary=s.get("output_summary", ""),
                latency_ms=s.get("latency_ms", 0.0),
                status=s.get("status", "SUCCESS"),
            )
            for s in final_state.get("retrieval_trace", [])
        ]

        trace = ReasoningTrace(
            need_retrieval=True,
            need_query_rewrite=final_state.get("rewrite_used", False),
            need_metadata_filter=final_state.get("metadata_filter_used", False),
            need_multi_hop=final_state.get("multi_hop_used", False),
            need_hyde=final_state.get("hyde_used", False),
            need_rerank=final_state.get("reranking_used", False),
            need_chunk_enhancement=final_state.get("chunk_enhancement_used", False),
            enough_evidence=final_state.get("evidence_assessment", {}).get("is_sufficient", False),
            steps=trace_steps,
            iterations_taken=max(1, final_state.get("escalation_count", 0) + 1),
        )

        hops: list[HopRecord] = [
            HopRecord(
                hop_index=h.get("hop", i + 1),
                subquery=h.get("query", ""),
                chunks_retrieved=h.get("retrieved", 0),
                latency_ms=h.get("latency_ms", 0.0),
                status="SUCCESS",
            )
            for i, h in enumerate(final_state.get("hop_results", []))
        ]

        ev = final_state.get("evidence_assessment", {})
        sufficiency = EvidenceSufficiencyResult(
            status=ev.get("status", "SUFFICIENT"),
            is_sufficient=ev.get("is_sufficient", True),
            score=ev.get("score", 1.0),
            reasons=ev.get("reasons", []),
            missing_aspects=ev.get("missing_aspects", []),
            chunk_count=len(final_state.get("final_context_chunks", [])),
        )

        return AgentExecutionResult(
            chunks=final_state.get("final_context_chunks", []),
            trace=trace,
            hops=hops,
            sufficiency=sufficiency,
            total_latency_ms=total_lat,
            state=final_state,
            answer=final_state.get("generated_answer", ""),
            disclosure=final_state.get("answer_disclosure", ""),
            decision_metadata=final_state.get("decision_metadata", {}),
        )
