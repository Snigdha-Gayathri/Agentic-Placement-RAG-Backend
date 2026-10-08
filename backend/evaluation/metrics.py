"""RAG evaluation metrics computation engine with ground-truth separation and failure classification."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\b\w+\b", text) if len(w) > 2}


def _jaccard(s1: set[str], s2: set[str]) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))


@dataclass
class LatencyMetrics:
    """Breakdown of pipeline timing."""

    query_analysis_ms: float = 0.0
    routing_ms: float = 0.0
    decomposition_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    verification_ms: float = 0.0
    ttft_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "query_analysis_ms": self.query_analysis_ms,
            "routing_ms": self.routing_ms,
            "decomposition_ms": self.decomposition_ms,
            "retrieval_ms": self.retrieval_ms,
            "reranking_ms": self.reranking_ms,
            "verification_ms": self.verification_ms,
            "ttft_ms": self.ttft_ms,
            "generation_ms": self.generation_ms,
            "total_ms": self.total_ms,
        }


@dataclass
class RAGMetricsResult:
    """Comprehensive RAG observability metrics with explicit provenance."""

    has_ground_truth: bool = False
    context_precision: float | str = "N/A"
    context_recall: float | str = "N/A"
    mrr: float | str = "N/A"
    ndcg: float | str = "N/A"
    precision_at_k: float | str = "N/A"
    recall_at_k: float | str = "N/A"
    hit_rate: float | str = "N/A"
    pre_reranker_mrr: float | str = "N/A"
    post_reranker_mrr: float | str = "N/A"
    reranker_improvement: float | str = "N/A"

    # Reference-free estimated generation scores
    faithfulness: float = 1.0
    response_relevancy: float = 1.0
    semantic_similarity: float = 1.0
    groundedness: float = 1.0

    # Failure diagnosis
    failure_category: str = "none"
    failure_diagnosis: str = "Pipeline executed normally."
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_ground_truth": self.has_ground_truth,
            "context_precision": self.context_precision if isinstance(self.context_precision, str) else round(self.context_precision, 3),
            "context_recall": self.context_recall if isinstance(self.context_recall, str) else round(self.context_recall, 3),
            "mrr": self.mrr if isinstance(self.mrr, str) else round(self.mrr, 3),
            "ndcg": self.ndcg if isinstance(self.ndcg, str) else round(self.ndcg, 3),
            "precision_at_k": self.precision_at_k if isinstance(self.precision_at_k, str) else round(self.precision_at_k, 3),
            "recall_at_k": self.recall_at_k if isinstance(self.recall_at_k, str) else round(self.recall_at_k, 3),
            "hit_rate": self.hit_rate if isinstance(self.hit_rate, str) else round(self.hit_rate, 3),
            "pre_reranker_mrr": self.pre_reranker_mrr if isinstance(self.pre_reranker_mrr, str) else round(self.pre_reranker_mrr, 3),
            "post_reranker_mrr": self.post_reranker_mrr if isinstance(self.post_reranker_mrr, str) else round(self.post_reranker_mrr, 3),
            "reranker_improvement": self.reranker_improvement if isinstance(self.reranker_improvement, str) else round(self.reranker_improvement, 3),
            "faithfulness": round(self.faithfulness, 3),
            "response_relevancy": round(self.response_relevancy, 3),
            "semantic_similarity": round(self.semantic_similarity, 3),
            "groundedness": round(self.groundedness, 3),
            "failure_category": self.failure_category,
            "failure_diagnosis": self.failure_diagnosis,
            "latency": self.latency.to_dict(),
        }


class RAGMetrics:
    """Calculates retrieval, generation, ranking, and failure diagnosis metrics."""

    def compute_all(
        self,
        query: str,
        answer: str,
        retrieved_chunks: list[Any],
        context: str,
        latency: LatencyMetrics,
        pre_rerank_chunks: list[Any] | None = None,
        ground_truth_chunk_ids: list[str] | None = None,
        expected_route: str | None = None,
        actual_route: str = "SINGLE_HOP",
        should_abstain: bool = False,
        abstained: bool = False,
        k: int = 5,
    ) -> RAGMetricsResult:
        q_tok = _tokens(query)
        a_tok = _tokens(answer)
        c_tok = _tokens(context)

        # 1. Ground-Truth Backed Ranking Metrics (when reference labels provided)
        has_gt = bool(ground_truth_chunk_ids and len(ground_truth_chunk_ids) > 0)
        mrr: float | str = "N/A"
        ndcg: float | str = "N/A"
        prec_k: float | str = "N/A"
        rec_k: float | str = "N/A"
        hit_rate: float | str = "N/A"
        c_precision: float | str = "N/A"
        c_recall: float | str = "N/A"
        pre_mrr: float | str = "N/A"
        post_mrr: float | str = "N/A"
        rerank_impr: float | str = "N/A"

        if has_gt:
            target_ids = set(ground_truth_chunk_ids)
            rel_flags: list[int] = []
            for c in retrieved_chunks:
                cid = getattr(c, "chunk_id", "")
                source = getattr(c, "source", "")
                is_hit = 1 if (cid in target_ids or any(t.lower() in source.lower() for t in target_ids)) else 0
                rel_flags.append(is_hit)

            hits = sum(rel_flags)
            hit_rate = 1.0 if hits > 0 else 0.0
            c_precision = hits / max(1, len(rel_flags))
            c_recall = hits / max(1, len(target_ids))

            # MRR
            first_hit_rank = next((i + 1 for i, r in enumerate(rel_flags) if r == 1), 0)
            post_mrr = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0
            mrr = post_mrr

            # NDCG@K
            dcg = sum((r / math.log2(i + 2)) for i, r in enumerate(rel_flags[:k]))
            idcg = sum((1.0 / math.log2(i + 2)) for i in range(min(hits, k)))
            ndcg = (dcg / idcg) if idcg > 0 else 0.0

            # @K
            top_k_flags = rel_flags[:k]
            prec_k = sum(top_k_flags) / max(1, len(top_k_flags))
            rec_k = sum(top_k_flags) / max(1, len(target_ids))

            # Pre-reranker MRR if pre-rerank candidates provided
            if pre_rerank_chunks:
                pre_flags = [
                    1 if (getattr(c, "chunk_id", "") in target_ids or any(t.lower() in getattr(c, "source", "").lower() for t in target_ids)) else 0
                    for c in pre_rerank_chunks
                ]
                pre_first_hit = next((i + 1 for i, r in enumerate(pre_flags) if r == 1), 0)
                pre_mrr = 1.0 / pre_first_hit if pre_first_hit > 0 else 0.0
                rerank_impr = round(post_mrr - pre_mrr, 3)

        # 2. Generation Grounding & Relevancy
        a_sents = [s.strip() for s in re.split(r"[.!?]\s+", answer) if len(s.strip()) > 15]
        if a_sents and c_tok:
            supported = sum(1 for s in a_sents if _jaccard(_tokens(s), c_tok) >= 0.12)
            faithfulness = supported / max(1, len(a_sents))
        else:
            faithfulness = 1.0 if abstained else (0.0 if not context.strip() else 0.8)

        groundedness = faithfulness
        resp_relevancy = min(1.0, len(q_tok.intersection(a_tok)) / max(1, min(len(q_tok), len(a_tok)))) if a_tok else 0.0
        sem_sim = _jaccard(q_tok, a_tok)

        # 3. Failure Classification
        failure_cat = "none"
        diagnosis = "Query processed successfully."

        if should_abstain and not abstained and not has_gt:
            failure_cat = "abstention_failure"
            diagnosis = "System attempted to answer an unsupported query without sufficient evidence."
        elif not should_abstain and abstained and len(retrieved_chunks) >= 3:
            failure_cat = "over_abstention"
            diagnosis = "System abstained despite sufficient retrieved candidate chunks."
        elif expected_route and expected_route != actual_route:
            failure_cat = "routing_failure"
            diagnosis = f"Query routed to {actual_route} instead of expected {expected_route}."
        elif not retrieved_chunks and not should_abstain:
            failure_cat = "retrieval_failure"
            diagnosis = "Zero candidate chunks retrieved from the knowledge corpus."
        elif has_gt and isinstance(hit_rate, float) and hit_rate == 0.0:
            failure_cat = "retrieval_failure"
            diagnosis = "Relevant ground-truth chunks were not retrieved in candidate set."
        elif has_gt and isinstance(post_mrr, float) and post_mrr < 0.25 and hit_rate > 0:
            failure_cat = "ranking_failure"
            diagnosis = "Relevant chunk was retrieved but ranked low by the reranker."
        elif faithfulness < 0.5 and not abstained:
            failure_cat = "generation_failure"
            diagnosis = "Generated answer contains claims not grounded in the retrieved context."

        return RAGMetricsResult(
            has_ground_truth=has_gt,
            context_precision=c_precision,
            context_recall=c_recall,
            mrr=mrr,
            ndcg=ndcg,
            precision_at_k=prec_k,
            recall_at_k=rec_k,
            hit_rate=hit_rate,
            pre_reranker_mrr=pre_mrr,
            post_reranker_mrr=post_mrr,
            reranker_improvement=rerank_impr,
            faithfulness=faithfulness,
            response_relevancy=resp_relevancy,
            semantic_similarity=sem_sim,
            groundedness=groundedness,
            failure_category=failure_cat,
            failure_diagnosis=diagnosis,
            latency=latency,
        )
