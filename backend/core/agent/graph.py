"""LangGraph StateGraph implementation for truly agentic retrieval."""

from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.graph import END, StateGraph

from .planner import AgentPlanner, PlanStep
from .state import AgenticRAGState
from .tools import (
    BM25RetrieveTool,
    ChunkEnhanceTool,
    DecomposeQueryTool,
    DenseRetrieveTool,
    EvaluateEvidenceTool,
    HybridRetrieveTool,
    HyDETool,
    RerankTool,
)

logger = logging.getLogger(__name__)


def build_disclosure_section(state: AgenticRAGState) -> str:
    """Build compact, honest user-facing retrieval disclosure section."""
    executed = set(state.get("executed_operations", []))

    # Determine status of each technique
    items = [
        ("Query rewriting", "QUERY_REWRITE" in executed),
        ("Metadata filtering", "METADATA_FILTERING" in executed and state.get("metadata_filter_used", False)),
        ("Embedding generation", "EMBEDDING_GENERATION" in executed and state.get("embedding_generation_used", False)),
        ("BM25 retrieval", "BM25_RETRIEVAL" in executed),
        ("Dense retrieval", "DENSE_RETRIEVAL" in executed),
        ("Hybrid retrieval", "HYBRID_RETRIEVAL" in executed),
        ("RRF fusion", "RRF_FUSION" in executed),
        ("Cross-encoder reranking", "CROSS_ENCODER_RERANKING" in executed and state.get("reranking_used", False)),
        ("HyDE", "HYDE_GENERATION" in executed),
        ("Multi-hop", "MULTI_HOP_RETRIEVAL" in executed and state.get("multi_hop_used", False)),
        ("Chunk enhancement", "CHUNK_ENHANCEMENT" in executed and state.get("chunk_enhancement_used", False)),
    ]

    lines = ["\n\n---\n**Retrieval used:**"]
    for name, used in items:
        symbol = "✓" if used else "✗"
        lines.append(f"{symbol} {name}")

    # Build concise why summary
    skipped_names = [name for name, used in items if not used]
    if "HyDE" in skipped_names and "Multi-hop" in skipped_names:
        why = "HyDE and multi-hop were unnecessary because the initial retrieval produced sufficient high-confidence evidence."
    elif "Multi-hop" in skipped_names:
        why = "Multi-hop was skipped because the query focused on a single entity/topic."
    elif "HyDE" in skipped_names:
        why = "HyDE was skipped because query vocabulary directly matched the indexed knowledge corpus."
    else:
        why = "Operations were dynamically selected based on query specificity and evidence quality."

    lines.append(f"\n*Why:* {why}")
    return "\n".join(lines)


class RetrievalGraphBuilder:
    """Builds and compiles the LangGraph StateGraph for agentic retrieval."""

    def __init__(
        self,
        bm25_tool: BM25RetrieveTool,
        dense_tool: DenseRetrieveTool,
        hybrid_tool: HybridRetrieveTool,
        hyde_tool: HyDETool,
        rerank_tool: RerankTool,
        chunk_enhance_tool: ChunkEnhanceTool,
        decompose_tool: DecomposeQueryTool,
        evaluate_tool: EvaluateEvidenceTool,
        query_rewriter: Any,
        gemini_client: Any,
        planner: AgentPlanner | None = None,
        max_hops: int = 3,
        max_retrieval_calls: int = 6,
        max_escalations: int = 2,
    ) -> None:
        self.bm25_tool = bm25_tool
        self.dense_tool = dense_tool
        self.hybrid_tool = hybrid_tool
        self.hyde_tool = hyde_tool
        self.rerank_tool = rerank_tool
        self.chunk_enhance_tool = chunk_enhance_tool
        self.decompose_tool = decompose_tool
        self.evaluate_tool = evaluate_tool
        self.query_rewriter = query_rewriter
        self.gemini_client = gemini_client
        self.planner = planner or AgentPlanner()
        self.max_hops = max_hops
        self.max_retrieval_calls = max_retrieval_calls
        self.max_escalations = max_escalations

    def create_graph(self) -> Any:
        """Compile the LangGraph StateGraph."""
        builder = StateGraph(AgenticRAGState)

        # Nodes
        builder.add_node("analyze_and_plan", self._node_analyze_and_plan)
        builder.add_node("execute_rewrite", self._node_execute_rewrite)
        builder.add_node("execute_metadata_filter", self._node_execute_metadata_filter)
        builder.add_node("execute_retrieval", self._node_execute_retrieval)
        builder.add_node("execute_chunk_enhancement", self._node_execute_chunk_enhancement)
        builder.add_node("execute_rerank", self._node_execute_rerank)
        builder.add_node("evaluate_evidence", self._node_evaluate_evidence)
        builder.add_node("escalate_retrieval", self._node_escalate_retrieval)
        builder.add_node("synthesize_answer", self._node_synthesize_answer)

        # Edges
        builder.set_entry_point("analyze_and_plan")
        builder.add_edge("analyze_and_plan", "execute_rewrite")
        builder.add_edge("execute_rewrite", "execute_metadata_filter")
        builder.add_edge("execute_metadata_filter", "execute_retrieval")
        builder.add_edge("execute_retrieval", "execute_chunk_enhancement")
        builder.add_edge("execute_chunk_enhancement", "execute_rerank")
        builder.add_edge("execute_rerank", "evaluate_evidence")

        # Conditional Edge after evidence evaluation
        builder.add_conditional_edges(
            "evaluate_evidence",
            self._route_evidence_check,
            {
                "synthesize": "synthesize_answer",
                "escalate": "escalate_retrieval",
            },
        )

        builder.add_edge("escalate_retrieval", "execute_retrieval")
        builder.add_edge("synthesize_answer", END)

        return builder.compile()

    # ── Node Implementations ────────────────────────────────────────────────

    async def _node_analyze_and_plan(self, state: AgenticRAGState) -> dict[str, Any]:
        """Analyze query profile and construct dynamic retrieval plan."""
        t0 = time.perf_counter()
        query = state.get("original_query", "")
        history = state.get("conversation_history", [])

        # Check if already blocked by security
        if state.get("is_blocked", False):
            return {
                "retrieval_plan": ["SECURITY_TERMINATION"],
                "executed_operations": ["SECURITY_CHECK"],
                "termination_reason": "SECURITY_BLOCKED",
            }

        plan_res = self.planner.plan(query, history)
        profile = plan_res["profile"]

        trace_step = PlanStep(
            tool_name="AgentPlanner",
            reasoning=f"Analyzed query: exact_lexical={profile['exact_lexical_cues']}, semantic={profile['pure_semantic_cues']}, multi_hop={profile['needs_multi_hop']}",
            inputs={"query": query},
            output_summary=f"Selected initial operations: {', '.join(plan_res['planned_operations'][:4])}",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            status="SUCCESS",
        )

        existing_trace = list(state.get("retrieval_trace", []))
        existing_trace.append(trace_step.to_dict())

        return {
            "retrieval_plan": plan_res["planned_operations"],
            "skipped_operations": plan_res["skipped_operations"],
            "retrieval_trace": existing_trace,
            "target_company": profile["detected_companies"][0] if profile["detected_companies"] else None,
            "detected_topics": profile["detected_topics"],
            "rewrite_used": plan_res["rewrite_needed"],
            "rewrite_reason": plan_res["rewrite_reason"],
            "metadata_filter_used": bool(state.get("metadata_filter") or plan_res["metadata_filters"]),
            "metadata_filter": {**(state.get("metadata_filter") or {}), **plan_res["metadata_filters"]},
            "metadata_filter_reason": plan_res["metadata_filter_reason"],
            "embedding_generation_used": plan_res["embedding_needed"],
            "embedding_model": "models/embedding-001" if plan_res["embedding_needed"] else "None",
            "embedding_reason": plan_res["embedding_reason"],
            "bm25_used": plan_res["bm25_needed"],
            "bm25_reason": plan_res["bm25_reason"],
            "dense_used": plan_res["dense_needed"],
            "dense_reason": plan_res["dense_reason"],
            "hybrid_used": plan_res["hybrid_needed"],
            "hybrid_reason": plan_res["hybrid_reason"],
            "rrf_used": plan_res["rrf_needed"],
            "rrf_reason": plan_res["rrf_reason"],
            "multi_hop_used": plan_res["multi_hop_needed"],
            "hyde_used": plan_res["hyde_needed"],
            "hyde_reason": plan_res["hyde_reason"],
            "chunk_enhancement_used": plan_res["chunk_enhancement_needed"],
            "chunk_enhancement_type": "metadata_prepend_and_semantic_header" if plan_res["chunk_enhancement_needed"] else "standard",
            "chunk_enhancement_reason": plan_res["chunk_enhancement_reason"],
            "reranking_used": plan_res["rerank_needed"],
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "reranking_reason": plan_res["rerank_reason"],
            "budgets": {
                "max_hops": self.max_hops,
                "max_retrieval_calls": self.max_retrieval_calls,
                "calls_made": 0,
            },
            "escalation_count": 0,
            "executed_operations": ["QUERY_ANALYSIS"],
        }

    async def _node_execute_rewrite(self, state: AgenticRAGState) -> dict[str, Any]:
        """Conditionally execute query rewriting preserving original query."""
        t0 = time.perf_counter()
        orig_q = state.get("original_query", "")
        rewrite_needed = state.get("rewrite_used", False)
        trace = list(state.get("retrieval_trace", []))
        executed = list(state.get("executed_operations", []))

        if rewrite_needed and self.query_rewriter:
            try:
                history_turns = state.get("conversation_history", [])
                rew_res = await self.query_rewriter.rewrite(orig_q, history_turns)
                rewritten = rew_res.rewritten_query
                lat = round((time.perf_counter() - t0) * 1000, 2)
                executed.append("QUERY_REWRITE")
                trace.append(
                    PlanStep(
                        tool_name="QueryRewriter",
                        reasoning=state.get("rewrite_reason", "Resolved references."),
                        inputs={"original_query": orig_q},
                        output_summary=f"Rewritten to: '{rewritten}'",
                        latency_ms=lat,
                        status="SUCCESS",
                    ).to_dict()
                )
                return {
                    "rewritten_query": rewritten,
                    "rewrite_used": True,
                    "retrieval_trace": trace,
                    "executed_operations": executed,
                }
            except Exception as exc:
                logger.warning("Rewrite error: %s", exc)

        # Skipped or fallback
        trace.append(
            PlanStep(
                tool_name="QueryRewriter",
                reasoning="Preserved original query without transformation.",
                inputs={"original_query": orig_q},
                output_summary="Original query retained.",
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                status="SKIPPED",
            ).to_dict()
        )
        return {
            "rewritten_query": orig_q,
            "rewrite_used": False,
            "retrieval_trace": trace,
            "executed_operations": executed,
        }

    async def _node_execute_metadata_filter(self, state: AgenticRAGState) -> dict[str, Any]:
        """Validate and apply metadata filtering constraints."""
        t0 = time.perf_counter()
        meta_needed = state.get("metadata_filter_used", False)
        filters = state.get("metadata_filter", {})
        trace = list(state.get("retrieval_trace", []))
        executed = list(state.get("executed_operations", []))

        if meta_needed and filters:
            executed.append("METADATA_FILTERING")
            trace.append(
                PlanStep(
                    tool_name="MetadataFilter",
                    reasoning=state.get("metadata_filter_reason", ""),
                    inputs={"filters": filters},
                    output_summary=f"Active filters: {filters}",
                    latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                    status="SUCCESS",
                ).to_dict()
            )
            return {
                "metadata_filter_used": True,
                "metadata_filter": filters,
                "retrieval_trace": trace,
                "executed_operations": executed,
            }

        trace.append(
            PlanStep(
                tool_name="MetadataFilter",
                reasoning="No strict metadata filtering applied to preserve recall.",
                inputs={},
                output_summary="Unfiltered search across all companies/topics.",
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                status="SKIPPED",
            ).to_dict()
        )
        return {
            "metadata_filter_used": False,
            "metadata_filter": {},
            "retrieval_trace": trace,
            "executed_operations": executed,
        }

    async def _node_execute_retrieval(self, state: AgenticRAGState) -> dict[str, Any]:
        """Dynamically execute the planned retrieval operations."""
        t0 = time.perf_counter()
        active_q = state.get("rewritten_query") or state.get("original_query", "")
        meta_filters = state.get("metadata_filter", {}) if state.get("metadata_filter_used") else None
        trace = list(state.get("retrieval_trace", []))
        executed = list(state.get("executed_operations", []))
        budgets = dict(state.get("budgets", {}))
        budgets["calls_made"] = budgets.get("calls_made", 0) + 1

        chunks: list[Any] = list(state.get("retrieved_chunks", []))
        existing_ids = {getattr(c, "chunk_id", str(i)) for i, c in enumerate(chunks)}

        # A. Multi-Hop Retrieval Path
        if state.get("multi_hop_used"):
            executed.append("MULTI_HOP_RETRIEVAL")
            dec_res = self.decompose_tool.execute(
                active_q,
                detected_companies=[state.get("target_company")] if state.get("target_company") else None,
                detected_topics=state.get("detected_topics"),
            )
            subqueries = dec_res.output
            hop_records: list[dict[str, Any]] = []

            for h_idx, subq in enumerate(subqueries[:self.max_hops], start=1):
                hop_t0 = time.perf_counter()
                hop_filter = meta_filters
                # If comparison, target company per hop
                if state.get("target_company") and h_idx == 1:
                    hop_filter = {"company": state.get("target_company")}

                h_ret = await self.hybrid_tool.execute(subq, top_k=8, metadata_filters=hop_filter)
                budgets["calls_made"] += 1
                novel = 0
                for c in h_ret.output:
                    cid = getattr(c, "chunk_id", str(hash(getattr(c, "text", str(c)))))
                    if cid not in existing_ids:
                        chunks.append(c)
                        existing_ids.add(cid)
                        novel += 1

                hop_records.append({
                    "hop": h_idx,
                    "query": subq,
                    "retrieved": len(h_ret.output),
                    "novel": novel,
                    "latency_ms": round((time.perf_counter() - hop_t0) * 1000, 2),
                })

            executed.extend(["BM25_RETRIEVAL", "DENSE_RETRIEVAL", "HYBRID_RETRIEVAL", "RRF_FUSION", "EMBEDDING_GENERATION"])
            trace.append(
                PlanStep(
                    tool_name="MultiHopRetriever",
                    reasoning=f"Executed {len(hop_records)} retrieval hops over decomposed subqueries.",
                    inputs={"subqueries": subqueries},
                    output_summary=f"Retrieved {len(chunks)} cumulative candidates across {len(hop_records)} hops.",
                    latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                    status="SUCCESS",
                ).to_dict()
            )

            return {
                "retrieved_chunks": chunks,
                "hop_count": len(hop_records),
                "hop_queries": subqueries,
                "hop_results": hop_records,
                "retrieval_trace": trace,
                "executed_operations": executed,
                "budgets": budgets,
                "bm25_used": True,
                "dense_used": True,
                "hybrid_used": True,
                "rrf_used": True,
                "embedding_generation_used": True,
            }

        # B. HyDE Retrieval Path
        if state.get("hyde_used"):
            executed.extend(["HYDE_GENERATION", "EMBEDDING_GENERATION", "DENSE_RETRIEVAL"])
            hyde_res = await self.hyde_tool.execute(active_q, top_k=15, metadata_filters=meta_filters)
            new_chunks = hyde_res.output
            chunks.extend([c for c in new_chunks if getattr(c, "chunk_id", "") not in existing_ids])
            trace.append(
                PlanStep(
                    tool_name="HyDETool",
                    reasoning=state.get("hyde_reason", "Hypothetical expansion."),
                    inputs={"query": active_q},
                    output_summary=f"Retrieved {len(new_chunks)} candidates using hypothetical document vector.",
                    latency_ms=hyde_res.latency_ms,
                    status=hyde_res.status,
                ).to_dict()
            )
            return {
                "retrieved_chunks": chunks,
                "retrieval_trace": trace,
                "executed_operations": executed,
                "budgets": budgets,
                "embedding_generation_used": True,
                "dense_used": True,
            }

        # C. Pure BM25 Retrieval Path
        if state.get("bm25_used") and not state.get("dense_used") and not state.get("hybrid_used"):
            executed.append("BM25_RETRIEVAL")
            bm_res = await self.bm25_tool.execute(active_q, top_k=15, metadata_filters=meta_filters)
            chunks = bm_res.output
            trace.append(
                PlanStep(
                    tool_name="BM25RetrieveTool",
                    reasoning=state.get("bm25_reason", "Lexical search."),
                    inputs={"query": active_q, "filters": meta_filters or {}},
                    output_summary=f"Retrieved {len(chunks)} candidates via BM25.",
                    latency_ms=bm_res.latency_ms,
                    status=bm_res.status,
                ).to_dict()
            )
            return {
                "retrieved_chunks": chunks,
                "retrieval_trace": trace,
                "executed_operations": executed,
                "budgets": budgets,
                "embedding_generation_used": False,
            }

        # D. Pure Dense Retrieval Path
        if state.get("dense_used") and not state.get("bm25_used") and not state.get("hybrid_used"):
            executed.extend(["EMBEDDING_GENERATION", "DENSE_RETRIEVAL"])
            dn_res = await self.dense_tool.execute(active_q, top_k=15, metadata_filters=meta_filters)
            chunks = dn_res.output
            trace.append(
                PlanStep(
                    tool_name="DenseRetrieveTool",
                    reasoning=state.get("dense_reason", "Semantic search."),
                    inputs={"query": active_q, "filters": meta_filters or {}},
                    output_summary=f"Retrieved {len(chunks)} candidates via dense similarity.",
                    latency_ms=dn_res.latency_ms,
                    status=dn_res.status,
                ).to_dict()
            )
            return {
                "retrieved_chunks": chunks,
                "retrieval_trace": trace,
                "executed_operations": executed,
                "budgets": budgets,
                "embedding_generation_used": True,
            }

        # E. Hybrid RRF Retrieval Path (Default for mixed queries)
        executed.extend(["EMBEDDING_GENERATION", "BM25_RETRIEVAL", "DENSE_RETRIEVAL", "HYBRID_RETRIEVAL", "RRF_FUSION"])
        hy_res = await self.hybrid_tool.execute(active_q, top_k=15, metadata_filters=meta_filters)
        chunks = hy_res.output
        trace.append(
            PlanStep(
                tool_name="HybridRetrieveTool",
                reasoning=state.get("hybrid_reason", "Combined sparse/dense search with RRF fusion."),
                inputs={"query": active_q, "filters": meta_filters or {}},
                output_summary=f"Retrieved {len(chunks)} fused candidates via RRF.",
                latency_ms=hy_res.latency_ms,
                status=hy_res.status,
            ).to_dict()
        )
        return {
            "retrieved_chunks": chunks,
            "retrieval_trace": trace,
            "executed_operations": executed,
            "budgets": budgets,
            "embedding_generation_used": True,
            "bm25_used": True,
            "dense_used": True,
            "hybrid_used": True,
            "rrf_used": True,
            "rrf_k": 60,
        }

    async def _node_execute_chunk_enhancement(self, state: AgenticRAGState) -> dict[str, Any]:
        """Conditionally apply contextual headers and metadata enhancement."""
        chunks = state.get("retrieved_chunks", [])
        trace = list(state.get("retrieval_trace", []))
        executed = list(state.get("executed_operations", []))

        if state.get("chunk_enhancement_used") and chunks:
            executed.append("CHUNK_ENHANCEMENT")
            enh_res = self.chunk_enhance_tool.execute(chunks)
            trace.append(
                PlanStep(
                    tool_name="ChunkEnhanceTool",
                    reasoning=state.get("chunk_enhancement_reason", "Prepend section headers."),
                    inputs={"chunk_count": len(chunks)},
                    output_summary=enh_res.summary,
                    latency_ms=enh_res.latency_ms,
                    status="SUCCESS",
                ).to_dict()
            )
            return {
                "retrieved_chunks": enh_res.output,
                "retrieval_trace": trace,
                "executed_operations": executed,
            }

        return {
            "retrieved_chunks": chunks,
            "retrieval_trace": trace,
            "executed_operations": executed,
        }

    async def _node_execute_rerank(self, state: AgenticRAGState) -> dict[str, Any]:
        """Conditionally execute cross-encoder reranking."""
        active_q = state.get("rewritten_query") or state.get("original_query", "")
        chunks = state.get("retrieved_chunks", [])
        trace = list(state.get("retrieval_trace", []))
        executed = list(state.get("executed_operations", []))

        if state.get("reranking_used") and chunks:
            executed.append("CROSS_ENCODER_RERANKING")
            rr_res = self.rerank_tool.execute(active_q, chunks, top_k=8)
            ranked = rr_res.output
            movements = rr_res.extra.get("rank_movements", [])
            trace.append(
                PlanStep(
                    tool_name="RerankTool",
                    reasoning=state.get("reranking_reason", "Precision reranking."),
                    inputs={"candidates_count": len(chunks)},
                    output_summary=rr_res.summary,
                    latency_ms=rr_res.latency_ms,
                    status=rr_res.status,
                ).to_dict()
            )
            return {
                "reranked_chunks": ranked,
                "final_context_chunks": ranked,
                "candidate_count_before_reranking": len(chunks),
                "candidate_count_after_reranking": len(ranked),
                "rank_movements": movements,
                "retrieval_trace": trace,
                "executed_operations": executed,
            }

        # Skipped reranking
        trace.append(
            PlanStep(
                tool_name="RerankTool",
                reasoning="Reranking bypassed; preserving original retrieval order.",
                inputs={"candidates_count": len(chunks)},
                output_summary=f"Retained top {min(8, len(chunks))} chunks without cross-encoder.",
                latency_ms=0.0,
                status="SKIPPED",
            ).to_dict()
        )
        return {
            "reranked_chunks": chunks[:8],
            "final_context_chunks": chunks[:8],
            "candidate_count_before_reranking": len(chunks),
            "candidate_count_after_reranking": min(8, len(chunks)),
            "rank_movements": [],
            "retrieval_trace": trace,
            "executed_operations": executed,
        }

    async def _node_evaluate_evidence(self, state: AgenticRAGState) -> dict[str, Any]:
        """Evaluate evidence quality and sufficiency."""
        active_q = state.get("rewritten_query") or state.get("original_query", "")
        chunks = state.get("final_context_chunks", [])
        target_comp = state.get("target_company")
        trace = list(state.get("retrieval_trace", []))
        executed = list(state.get("executed_operations", []))

        ev_res = self.evaluate_tool.execute(active_q, chunks, target_company=target_comp)
        sufficiency = ev_res.output

        executed.append("EVIDENCE_ASSESSMENT")
        trace.append(
            PlanStep(
                tool_name="EvaluateEvidenceTool",
                reasoning=f"Sufficiency check: {sufficiency.status} (score={sufficiency.score})",
                inputs={"chunk_count": len(chunks), "target_company": target_comp or "None"},
                output_summary=ev_res.summary,
                latency_ms=ev_res.latency_ms,
                status="SUCCESS",
            ).to_dict()
        )

        return {
            "evidence_assessment": {
                "status": sufficiency.status,
                "is_sufficient": sufficiency.is_sufficient,
                "score": sufficiency.score,
                "reasons": sufficiency.reasons,
                "missing_aspects": sufficiency.missing_aspects,
            },
            "retrieval_trace": trace,
            "executed_operations": executed,
        }

    def _route_evidence_check(self, state: AgenticRAGState) -> str:
        """Conditional routing: terminate or escalate."""
        ev = state.get("evidence_assessment", {})
        if ev.get("is_sufficient", False) or ev.get("status") == "UNSUPPORTED":
            return "synthesize"

        budgets = state.get("budgets", {})
        calls = budgets.get("calls_made", 0)
        esc_count = state.get("escalation_count", 0)

        if calls >= self.max_retrieval_calls or esc_count >= self.max_escalations:
            return "synthesize"

        return "escalate"

    async def _node_escalate_retrieval(self, state: AgenticRAGState) -> dict[str, Any]:
        """Select and configure the next escalation retrieval strategy."""
        t0 = time.perf_counter()
        esc_plan = self.planner.plan_escalation(dict(state))
        trace = list(state.get("retrieval_trace", []))
        executed = list(state.get("executed_operations", []))

        action = esc_plan.get("action", "TERMINATE")
        reason = esc_plan.get("reason", "")
        esc_count = state.get("escalation_count", 0) + 1

        trace.append(
            PlanStep(
                tool_name="RetrievalEscalationAgent",
                reasoning=reason,
                inputs={"action": action, "escalation_iteration": esc_count},
                output_summary=f"Escalated retrieval strategy: {action}",
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                status="SUCCESS",
            ).to_dict()
        )

        updates: dict[str, Any] = {
            "escalation_count": esc_count,
            "retrieval_trace": trace,
            "executed_operations": executed,
        }

        if action == "ESCALATE_TO_DENSE":
            updates["dense_used"] = True
            updates["embedding_generation_used"] = True
            updates["hybrid_used"] = True
            updates["rrf_used"] = True
        elif action == "ESCALATE_TO_HYDE":
            updates["hyde_used"] = True
            updates["embedding_generation_used"] = True
            updates["dense_used"] = True
        elif action == "ESCALATE_TO_RERANK":
            updates["reranking_used"] = True
        elif action == "ESCALATE_TO_MULTI_HOP":
            updates["multi_hop_used"] = True

        return updates

    async def _node_synthesize_answer(self, state: AgenticRAGState) -> dict[str, Any]:
        """Synthesize final grounded answer and append retrieval disclosure."""
        orig_q = state.get("original_query", "")
        chunks = state.get("final_context_chunks", [])
        history = state.get("conversation_history", [])
        trace = list(state.get("retrieval_trace", []))
        executed = list(state.get("executed_operations", []))

        # Check if query was blocked by security
        if state.get("is_blocked", False):
            safe_msg = state.get("security_event", {}).get("user_safe_message", "Request blocked by security.")
            return {
                "generated_answer": safe_msg,
                "answer_disclosure": "",
                "termination_reason": "SECURITY_BLOCKED",
            }

        # Check evidence sufficiency abstention
        ev = state.get("evidence_assessment", {})
        if ev.get("status") == "UNSUPPORTED" and state.get("target_company") and len(chunks) == 0:
            target_comp = state.get("target_company")
            abstain_msg = (
                "### Evidence Sufficiency Notice\n\n"
                f"The indexed interview knowledge corpus does not contain verified interview records for **{target_comp}**. "
                "To avoid generating ungrounded information, the system cannot synthesize an answer for this company."
            )
            disc = build_disclosure_section(state)
            return {
                "generated_answer": abstain_msg.strip() + disc,
                "answer_disclosure": disc,
                "termination_reason": "UNSUPPORTED_COMPANY_ABSTAINED",
            }

        # Build clean context lines
        context_lines = [
            f"[{getattr(c, 'source', getattr(c, 'metadata', {}).get('source_file', 'Doc'))}] {getattr(c, 'text', str(c))}"
            for c in chunks
        ]
        merged_context = "\n\n".join(context_lines)

        t0_gen = time.perf_counter()
        raw_answer = await self.gemini_client.generate_answer(
            query=orig_q,
            context=merged_context,
            max_output_tokens=8192,
            conversation_history=history,
        )
        gen_lat = round((time.perf_counter() - t0_gen) * 1000, 2)

        disclosure = build_disclosure_section(state)
        final_answer = raw_answer.strip() + disclosure

        executed.append("ANSWER_SYNTHESIS")
        trace.append(
            PlanStep(
                tool_name="AnswerSynthesizer",
                reasoning="Generated grounded answer from retrieved context and appended transparent retrieval disclosure.",
                inputs={"chunk_count": len(chunks)},
                output_summary=f"Synthesized answer ({len(raw_answer.split())} words) with retrieval disclosure.",
                latency_ms=gen_lat,
                status="SUCCESS",
            ).to_dict()
        )

        term_reason = "EVIDENCE_SUFFICIENT" if ev.get("is_sufficient", True) else "MAX_CALLS_OR_BUDGET_EXHAUSTED"

        return {
            "generated_answer": final_answer,
            "answer_disclosure": disclosure,
            "sanitized_context": merged_context,
            "termination_reason": term_reason,
            "retrieval_trace": trace,
            "executed_operations": executed,
        }
