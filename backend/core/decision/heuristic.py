"""Heuristic / Deterministic Decision Backend for Agentic Placement RAG.

Faithfully implements the baseline decision logic to guarantee 100% backward
compatibility and strict invariance when Jev is disabled.
"""

from __future__ import annotations

import time
from typing import Any

from .base import DecisionBackend
from .models import (
    DecisionLogRecord,
    HopMode,
    QueryTransformation,
    RetrievalMode,
    RetrievalStrategyDecision,
    RouteType,
    SecurityDecision,
    SufficiencyDecision,
)


class HeuristicDecisionBackend(DecisionBackend):
    """Deterministic rule-based decision backend mirroring the frozen baseline."""

    @property
    def name(self) -> str:
        return "heuristic"

    async def plan_retrieval(
        self,
        query: str,
        conversation_history: list[dict[str, Any]] | None = None,
        profile: dict[str, Any] | None = None,
        active_toggles: dict[str, bool] | None = None,
        query_id: str = "",
    ) -> RetrievalStrategyDecision:
        """Execute heuristic retrieval planning using established rules."""
        t0 = time.perf_counter()
        toggles = active_toggles or {}

        # If profile not provided, calculate using QueryRouter & AgentPlanner standards
        if profile is None:
            from backend.core.agent.planner import AgentPlanner
            planner = AgentPlanner()
            profile = planner.analyze_query_profile(query, conversation_history)

        planned_ops: list[str] = []
        skipped_ops: list[dict[str, str]] = []

        # 1. Query Rewriting Decision
        rewrite_needed = bool(
            toggles.get("query_rewriting", True)
            and (profile["is_followup"] or profile["has_multi_aspects"] or profile["word_count"] > 25)
        )
        if rewrite_needed:
            planned_ops.append("QUERY_REWRITE")
            rewrite_reason = "Query contains conversational follow-up references, multi-part intent, or complex syntax benefiting from retrieval optimization."
            query_transformation = QueryTransformation.REWRITE
        else:
            skipped_ops.append({
                "operation": "QUERY_REWRITE",
                "reason": "Query is standalone, clear, and unambiguous; rewriting was unnecessary.",
            })
            rewrite_reason = "Query is self-contained; skipping rewrite."
            query_transformation = QueryTransformation.NONE

        # 2. Metadata Filtering Decision
        meta_filters: dict[str, Any] = {}
        filter_needed = False
        filter_reason = ""
        if toggles.get("metadata_filtering", True):
            if profile["detected_companies"] and len(profile["detected_companies"]) == 1:
                meta_filters["company"] = profile["detected_companies"][0]
                filter_needed = True
            if profile["detected_lps"]:
                meta_filters["leadership_principle"] = profile["detected_lps"][0]
                filter_needed = True

        if filter_needed and meta_filters:
            planned_ops.append("METADATA_FILTERING")
            filter_reason = f"Applied precise constraints for verified entities: {meta_filters}"
        else:
            skipped_ops.append({
                "operation": "METADATA_FILTERING",
                "reason": "No restrictive metadata filter applied to preserve maximum recall across relevant corpus sections.",
            })
            filter_reason = "Unfiltered retrieval selected to maximize cross-domain recall."

        # 3. Retrieval Method Decision
        multi_hop_needed = bool(
            toggles.get("multi_hop_retrieval", True) and profile["needs_multi_hop"]
        )

        bm25_needed = False
        dense_needed = False
        hybrid_needed = False
        rrf_needed = False
        embedding_needed = False
        bm25_reason = ""
        dense_reason = ""
        hybrid_reason = ""
        rrf_reason = ""

        if multi_hop_needed:
            retrieval_mode = RetrievalMode.HYBRID
            hop_mode = HopMode.MULTI_HOP
            route_type = RouteType.MULTI_HOP
            planned_ops.append("MULTI_HOP_RETRIEVAL")
            bm25_needed = True
            dense_needed = True
            hybrid_needed = True
            rrf_needed = True
            embedding_needed = True
            bm25_reason = "Multi-hop sub-queries require keyword matching for specific entities and interview rounds."
            dense_reason = "Multi-hop sub-queries require semantic similarity across behavioral criteria."
            hybrid_reason = "Hybrid combination required across multiple evidence hops."
            rrf_reason = "RRF fuses ranking candidates across hops."
        elif profile["exact_lexical_cues"] and not profile["pure_semantic_cues"]:
            retrieval_mode = RetrievalMode.BM25
            hop_mode = HopMode.SINGLE_HOP
            route_type = RouteType.SINGLE_HOP
            bm25_needed = True
            bm25_reason = "Query consists of exact entities, leadership principles, or established technical terms where BM25 provides high lexical precision."
            dense_needed = False
            dense_reason = "Skipped dense search in favor of direct lexical matching."
            hybrid_needed = False
            rrf_needed = False
            embedding_needed = False
            planned_ops.append("BM25_RETRIEVAL")
            skipped_ops.append({
                "operation": "DENSE_RETRIEVAL",
                "reason": "BM25 is optimal for exact leadership principles and keyword terms; dense search skipped to save latency.",
            })
            skipped_ops.append({
                "operation": "EMBEDDING_GENERATION",
                "reason": "Embeddings not required when pure lexical BM25 retrieval is sufficient.",
            })
            skipped_ops.append({
                "operation": "HYBRID_RETRIEVAL",
                "reason": "Lexical evidence sufficient; hybrid fusion not required.",
            })
        elif profile["pure_semantic_cues"] and not profile["exact_lexical_cues"]:
            retrieval_mode = RetrievalMode.DENSE
            hop_mode = HopMode.SINGLE_HOP
            route_type = RouteType.CONCEPTUAL
            dense_needed = True
            dense_reason = "Query is open-ended and conceptual; semantic dense retrieval captures deep behavioral similarity."
            bm25_needed = False
            bm25_reason = "Pure lexical match would fail on abstract behavioral descriptions with low verbatim overlap."
            embedding_needed = True
            hybrid_needed = False
            rrf_needed = False
            planned_ops.append("EMBEDDING_GENERATION")
            planned_ops.append("DENSE_RETRIEVAL")
            skipped_ops.append({
                "operation": "BM25_RETRIEVAL",
                "reason": "Skipped BM25 because query is abstract and lacks exact corpus keyword overlap.",
            })
            skipped_ops.append({
                "operation": "HYBRID_RETRIEVAL",
                "reason": "Single dense retrieval path selected for semantic query.",
            })
        else:
            retrieval_mode = RetrievalMode.HYBRID
            hop_mode = HopMode.SINGLE_HOP
            route_type = RouteType.SINGLE_HOP if profile["detected_companies"] else RouteType.CONCEPTUAL
            bm25_needed = True
            dense_needed = True
            hybrid_needed = True
            rrf_needed = True
            embedding_needed = True
            bm25_reason = "Matches exact terminology, company names, and technical topics."
            dense_reason = "Captures semantic intent and descriptive context."
            hybrid_reason = "Both lexical and semantic retrieval provide complementary recall for mixed query."
            rrf_reason = "Reciprocal Rank Fusion fuses BM25 and dense ranking signals."
            planned_ops.append("EMBEDDING_GENERATION")
            planned_ops.append("BM25_RETRIEVAL")
            planned_ops.append("DENSE_RETRIEVAL")
            planned_ops.append("HYBRID_RETRIEVAL")
            planned_ops.append("RRF_FUSION")

        # 4. HyDE Decision
        hyde_needed = bool(
            toggles.get("hyde", False)
            or (profile["vocabulary_mismatch_likely"] and not profile["exact_lexical_cues"])
        )
        if hyde_needed:
            planned_ops.append("HYDE_GENERATION")
            embedding_needed = True
            hyde_reason = "Query is conceptual with vocabulary mismatch; hypothetical document expansion bridges semantic gap."
            query_transformation = QueryTransformation.HYDE
        else:
            skipped_ops.append({
                "operation": "HYDE_GENERATION",
                "reason": "Query contains sufficient direct retrieval vocabulary; hypothetical document generation skipped.",
            })
            hyde_reason = "Query vocabulary is well-defined; HyDE skipped."

        # 5. Chunk Enhancement Decision
        chunk_enhancement_needed = bool(
            toggles.get("chunk_enhancement", True)
            and (profile["needs_multi_hop"] or profile["pure_semantic_cues"])
        )
        if chunk_enhancement_needed:
            planned_ops.append("CHUNK_ENHANCEMENT")
        else:
            skipped_ops.append({
                "operation": "CHUNK_ENHANCEMENT",
                "reason": "Standard chunks provide sufficient clarity without runtime context expansion.",
            })

        # 6. Cross-Encoder Reranking Decision
        rerank_needed = bool(
            toggles.get("cross_encoder_reranking", True)
            and (hybrid_needed or multi_hop_needed or profile["pure_semantic_cues"] or profile["has_multi_aspects"])
        )
        if rerank_needed:
            planned_ops.append("CROSS_ENCODER_RERANKING")
            rerank_reason = "Candidate set requires precision cross-encoder reranking to optimize top-k context order."
        else:
            skipped_ops.append({
                "operation": "CROSS_ENCODER_RERANKING",
                "reason": "Initial retrieval score margin is clear; cross-encoder reranking bypassed to save latency.",
            })
            rerank_reason = "Reranking skipped for low-ambiguity candidate set."

        planned_ops.append("EVIDENCE_ASSESSMENT")
        planned_ops.append("ANSWER_SYNTHESIS")

        latency_ms = round((time.perf_counter() - t0) * 1000, 3)

        probabilities = {
            "bm25": 1.0 if retrieval_mode == RetrievalMode.BM25 else 0.0,
            "dense": 1.0 if retrieval_mode == RetrievalMode.DENSE else 0.0,
            "hybrid": 1.0 if retrieval_mode == RetrievalMode.HYBRID else 0.0,
        }

        log_rec = DecisionLogRecord(
            query_id=query_id,
            decision_backend="heuristic",
            decision_type="retrieval_routing",
            selected_option=retrieval_mode.value,
            probabilities=probabilities,
            confidence=1.0,
            latency_ms=latency_ms,
            success=True,
            fallback_occurred=False,
            fallback_reason=None,
        )

        return RetrievalStrategyDecision(
            retrieval_mode=retrieval_mode,
            hop_mode=hop_mode,
            query_transformation=query_transformation,
            route_type=route_type,
            bm25_needed=bm25_needed,
            dense_needed=dense_needed,
            hybrid_needed=hybrid_needed,
            rrf_needed=rrf_needed,
            embedding_needed=embedding_needed,
            multi_hop_needed=multi_hop_needed,
            rewrite_needed=rewrite_needed,
            hyde_needed=hyde_needed,
            rerank_needed=rerank_needed,
            chunk_enhancement_needed=chunk_enhancement_needed,
            metadata_filter_needed=filter_needed,
            metadata_filters=meta_filters,
            reasoning=f"Heuristic rules evaluated: lexical={profile['exact_lexical_cues']}, semantic={profile['pure_semantic_cues']}, multi_hop={profile['needs_multi_hop']}",
            planned_operations=planned_ops,
            skipped_operations=skipped_ops,
            backend="heuristic",
            confidence=1.0,
            probabilities=probabilities,
            latency_ms=latency_ms,
            fallback_occurred=False,
            fallback_reason=None,
            log_record=log_rec,
        )

    async def evaluate_evidence_sufficiency(
        self,
        query: str,
        chunks: list[Any],
        target_company: str | None = None,
        query_id: str = "",
    ) -> SufficiencyDecision:
        """Heuristic evidence sufficiency assessment matching baseline tool."""
        t0 = time.perf_counter()
        from backend.core.agent.tools import EvaluateEvidenceTool

        tool = EvaluateEvidenceTool()
        tool_res = tool.execute(query=query, chunks=chunks, target_company=target_company)
        out = tool_res.output

        latency_ms = round((time.perf_counter() - t0) * 1000, 3)

        probabilities = {
            "sufficient": 1.0 if out.is_sufficient else 0.0,
            "insufficient": 0.0 if out.is_sufficient else 1.0,
        }

        log_rec = DecisionLogRecord(
            query_id=query_id,
            decision_backend="heuristic",
            decision_type="evidence_sufficiency",
            selected_option=out.status,
            probabilities=probabilities,
            confidence=1.0,
            latency_ms=latency_ms,
            success=True,
            fallback_occurred=False,
            fallback_reason=None,
        )

        return SufficiencyDecision(
            is_sufficient=out.is_sufficient,
            status=out.status,
            score=out.score,
            confidence=1.0,
            probabilities=probabilities,
            missing_aspects=out.missing_aspects,
            reasons=out.reasons,
            backend="heuristic",
            latency_ms=latency_ms,
            fallback_occurred=False,
            fallback_reason=None,
            log_record=log_rec,
        )

    async def evaluate_security(
        self,
        query: str,
        query_id: str = "",
    ) -> SecurityDecision:
        """Heuristic security evaluation using deterministic regex."""
        t0 = time.perf_counter()
        from security import PromptInjectionDetector, load_config

        detector = PromptInjectionDetector(load_config())
        inj_res = detector.detect(query)

        latency_ms = round((time.perf_counter() - t0) * 1000, 3)

        is_safe = inj_res.action != "block"
        probabilities = {
            "benign": 0.0 if not is_safe else 1.0,
            "adversarial": 1.0 if not is_safe else 0.0,
        }

        log_rec = DecisionLogRecord(
            query_id=query_id,
            decision_backend="heuristic",
            decision_type="security",
            selected_option="benign" if is_safe else "adversarial",
            probabilities=probabilities,
            confidence=1.0,
            latency_ms=latency_ms,
            success=True,
            fallback_occurred=False,
            fallback_reason=None,
        )

        return SecurityDecision(
            is_safe=is_safe,
            action=inj_res.action,
            category=inj_res.category or "BENIGN",
            severity=inj_res.severity or "LOW",
            risk_score=inj_res.risk_score,
            confidence=1.0,
            probabilities=probabilities,
            user_safe_message=inj_res.user_safe_message,
            backend="heuristic",
            latency_ms=latency_ms,
            fallback_occurred=False,
            fallback_reason=None,
            log_record=log_rec,
        )
