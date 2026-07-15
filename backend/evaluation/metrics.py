"""RAG evaluation metrics computation engine."""

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

    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    vector_search_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "retrieval_ms": self.retrieval_ms,
            "reranking_ms": self.reranking_ms,
            "generation_ms": self.generation_ms,
            "total_ms": self.total_ms,
            "vector_search_ms": self.vector_search_ms,
        }


@dataclass
class RAGMetricsResult:
    """Comprehensive RAG observability metrics."""

    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    response_relevancy: float = 0.0
    semantic_similarity: float = 0.0
    groundedness: float = 0.0
    mrr: float = 0.0
    map_score: float = 0.0
    ndcg: float = 0.0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    top_k_recall: float = 0.0
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_precision": round(self.context_precision, 3),
            "context_recall": round(self.context_recall, 3),
            "faithfulness": round(self.faithfulness, 3),
            "response_relevancy": round(self.response_relevancy, 3),
            "semantic_similarity": round(self.semantic_similarity, 3),
            "groundedness": round(self.groundedness, 3),
            "mrr": round(self.mrr, 3),
            "map_score": round(self.map_score, 3),
            "ndcg": round(self.ndcg, 3),
            "precision_at_k": round(self.precision_at_k, 3),
            "recall_at_k": round(self.recall_at_k, 3),
            "top_k_recall": round(self.top_k_recall, 3),
            "latency": self.latency.to_dict(),
        }


class RAGMetrics:
    """Computes RAG evaluation metrics without external dependencies."""

    def compute_all(
        self,
        query: str,
        answer: str,
        retrieved_chunks: list[Any],
        context: str,
        latency: LatencyMetrics,
        relevant_threshold: float = 0.15,
        k: int = 5,
    ) -> RAGMetricsResult:
        q_tok = _tokens(query)
        a_tok = _tokens(answer)
        c_tok = _tokens(context)

        # Context Precision & Recall
        rel_flags: list[int] = []
        for c in retrieved_chunks:
            s = getattr(c, "score", 0.0)
            c_text = getattr(c, "text", "")
            # Relevant if score above threshold or overlaps with answer keywords
            if s >= relevant_threshold or _jaccard(a_tok, _tokens(c_text)) >= 0.1:
                rel_flags.append(1)
            else:
                rel_flags.append(0)

        total_ret = len(rel_flags)
        rel_count = sum(rel_flags)
        c_precision = rel_count / max(1, total_ret)
        c_recall = min(1.0, len(a_tok.intersection(c_tok)) / max(1, len(a_tok)))

        # Faithfulness / Groundedness
        a_sents = [s.strip() for s in re.split(r"[.!?]\s+", answer) if len(s.strip()) > 15]
        supported = 0
        for sent in a_sents:
            s_tok = _tokens(sent)
            if _jaccard(s_tok, c_tok) >= 0.15:
                supported += 1
        faithfulness = supported / max(1, len(a_sents))
        groundedness = faithfulness

        # Response Relevancy
        resp_relevancy = min(1.0, len(q_tok.intersection(a_tok)) / max(1, min(len(q_tok), len(a_tok))))
        sem_sim = _jaccard(q_tok, a_tok)

        # Ranking metrics: MRR, MAP, NDCG
        mrr = 0.0
        for i, rel in enumerate(rel_flags):
            if rel == 1:
                mrr = 1.0 / (i + 1)
                break

        map_score = 0.0
        cum_prec = 0.0
        hits = 0
        for i, rel in enumerate(rel_flags):
            if rel == 1:
                hits += 1
                cum_prec += hits / (i + 1)
        if hits > 0:
            map_score = cum_prec / hits

        dcg = sum((rel / math.log2(i + 2)) for i, rel in enumerate(rel_flags))
        idcg = sum((1.0 / math.log2(i + 2)) for i in range(hits))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        # @K metrics
        top_k_flags = rel_flags[:k]
        prec_at_k = sum(top_k_flags) / max(1, len(top_k_flags))
        rec_at_k = sum(top_k_flags) / max(1, hits)
        top_k_rec = 1.0 if sum(top_k_flags) > 0 else 0.0

        return RAGMetricsResult(
            context_precision=c_precision,
            context_recall=c_recall,
            faithfulness=faithfulness,
            response_relevancy=resp_relevancy,
            semantic_similarity=sem_sim,
            groundedness=groundedness,
            mrr=mrr,
            map_score=map_score,
            ndcg=ndcg,
            precision_at_k=prec_at_k,
            recall_at_k=rec_at_k,
            top_k_recall=top_k_rec,
            latency=latency,
        )
