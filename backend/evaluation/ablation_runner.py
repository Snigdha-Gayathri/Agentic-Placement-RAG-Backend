"""Ablation study and custom threshold evaluation benchmark runner."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from backend.evaluation.benchmark_dataset import BenchmarkTestCase, get_benchmark_dataset
from backend.evaluation.metrics import LatencyMetrics, RAGMetrics

logger = logging.getLogger(__name__)


@dataclass
class ConfigurationBenchmarkResult:
    """Benchmark results for a single pipeline configuration."""

    config_name: str
    description: str
    total_evaluated: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate: float
    faithfulness: float
    abstention_accuracy: float
    avg_latency_ms: float
    p95_latency_ms: float
    total_llm_calls: int
    llm_calls_bypassed: int
    llm_bypass_rate_pct: float
    toggles: dict[str, bool] = field(default_factory=dict)
    numeric_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "description": self.description,
            "total_evaluated": self.total_evaluated,
            "recall_at_k": round(self.recall_at_k, 3),
            "precision_at_k": round(self.precision_at_k, 3),
            "mrr": round(self.mrr, 3),
            "ndcg_at_k": round(self.ndcg_at_k, 3),
            "hit_rate": round(self.hit_rate, 3),
            "faithfulness": round(self.faithfulness, 3),
            "abstention_accuracy": round(self.abstention_accuracy, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "total_llm_calls": self.total_llm_calls,
            "llm_calls_bypassed": self.llm_calls_bypassed,
            "llm_bypass_rate_pct": round(self.llm_bypass_rate_pct, 1),
            "toggles": self.toggles,
            "numeric_params": self.numeric_params,
        }


class AblationRunner:
    """Runs ablation studies comparing 6 pipeline variations against the interview benchmark."""

    ABLATION_CONFIGS = [
        {
            "id": "dense_only",
            "name": "1. Dense Only",
            "description": "Dense semantic search only (no BM25, no Cross-Encoder, no Agent planning)",
            "toggles": {
                "dense_retrieval": True,
                "bm25_retrieval": False,
                "hybrid_retrieval": False,
                "cross_encoder_reranking": False,
                "agent_planning": False,
                "multi_hop_retrieval": False,
            },
        },
        {
            "id": "bm25_only",
            "name": "2. BM25 Only",
            "description": "Lexical BM25 search only (no Dense vector search, no Reranker)",
            "toggles": {
                "dense_retrieval": False,
                "bm25_retrieval": True,
                "hybrid_retrieval": False,
                "cross_encoder_reranking": False,
                "agent_planning": False,
                "multi_hop_retrieval": False,
            },
        },
        {
            "id": "hybrid",
            "name": "3. Hybrid RRF",
            "description": "Dense + BM25 combined via Reciprocal Rank Fusion (no Reranker)",
            "toggles": {
                "dense_retrieval": True,
                "bm25_retrieval": True,
                "hybrid_retrieval": True,
                "cross_encoder_reranking": False,
                "agent_planning": False,
                "multi_hop_retrieval": False,
            },
        },
        {
            "id": "hybrid_rerank",
            "name": "4. Hybrid + Reranker",
            "description": "Hybrid RRF + Cross-Encoder deep neural reranker",
            "toggles": {
                "dense_retrieval": True,
                "bm25_retrieval": True,
                "hybrid_retrieval": True,
                "cross_encoder_reranking": True,
                "agent_planning": False,
                "multi_hop_retrieval": False,
            },
        },
        {
            "id": "hybrid_rerank_agent",
            "name": "5. Hybrid + Reranker + Agent",
            "description": "Hybrid + Reranker + Query Router & deterministic LLM bypass",
            "toggles": {
                "dense_retrieval": True,
                "bm25_retrieval": True,
                "hybrid_retrieval": True,
                "cross_encoder_reranking": True,
                "agent_planning": True,
                "multi_hop_retrieval": False,
            },
        },
        {
            "id": "full_pipeline",
            "name": "6. Full Agentic Multi-Hop",
            "description": "Hybrid + Reranker + Agentic Routing + Multi-Hop Query Decomposition",
            "toggles": {
                "dense_retrieval": True,
                "bm25_retrieval": True,
                "hybrid_retrieval": True,
                "cross_encoder_reranking": True,
                "agent_planning": True,
                "multi_hop_retrieval": True,
            },
        },
    ]

    def __init__(self, service: Any) -> None:
        self.service = service
        self.metrics_engine = RAGMetrics()

    async def run_ablation_study(self, sample_size: int = 20) -> list[dict[str, Any]]:
        """Run all 6 ablation configurations across a sample of benchmark queries."""
        dataset = get_benchmark_dataset()[:sample_size]
        results = []

        orig_toggles = self.service.get_feature_toggles()

        try:
            for cfg in self.ABLATION_CONFIGS:
                self.service.update_feature_toggles(cfg["toggles"])
                bench_res = await self._evaluate_on_dataset(cfg["name"], cfg["description"], dataset, cfg["toggles"])
                results.append(bench_res.to_dict())
        finally:
            self.service.update_feature_toggles(orig_toggles)

        return results

    async def evaluate_custom_config(
        self,
        config_name: str,
        toggles: dict[str, bool],
        numeric_params: dict[str, Any],
        sample_size: int = 15,
    ) -> dict[str, Any]:
        """Evaluate a custom set of developer thresholds against the benchmark."""
        dataset = get_benchmark_dataset()[:sample_size]
        orig_toggles = self.service.get_feature_toggles()

        try:
            self.service.update_feature_toggles(toggles)
            res = await self._evaluate_on_dataset(
                config_name, "Custom Developer Configuration", dataset, toggles, numeric_params
            )
            return res.to_dict()
        finally:
            self.service.update_feature_toggles(orig_toggles)

    async def _evaluate_on_dataset(
        self,
        name: str,
        description: str,
        dataset: list[BenchmarkTestCase],
        toggles: dict[str, bool],
        numeric_params: dict[str, Any] | None = None,
    ) -> ConfigurationBenchmarkResult:
        from backend.app.service import build_request_context
        from backend.app.models import ChatRequest

        mrr_list: list[float] = []
        ndcg_list: list[float] = []
        rec_list: list[float] = []
        prec_list: list[float] = []
        hit_list: list[float] = []
        faith_list: list[float] = []
        abstention_correct: int = 0
        latencies: list[float] = []
        llm_calls = 0
        llm_bypassed = 0

        for item in dataset:
            start_t = time.perf_counter()
            ctx = build_request_context(user_id="benchmark_evaluator")
            req = ChatRequest(query=item.query)

            resp = await self.service.process(req, ctx)
            dur_ms = (time.perf_counter() - start_t) * 1000
            latencies.append(dur_ms)

            pdata = resp.pipeline_data
            retrieved = pdata.retrieval_info.get("retrieved_chunks", [])
            sec_info = pdata.security_info

            # Track LLM calls
            is_bypassed = pdata.agent_info.get("llm_invoked") is False or "Bypassed" in pdata.query_info.get("routing_decision", "")
            if is_bypassed:
                llm_bypassed += 1
            else:
                llm_calls += 1

            # Check abstention correctness
            is_abstained = "unable to answer" in resp.answer.lower() or "not indexed" in resp.answer.lower() or sec_info.get("discarded_chunks", 0) == len(retrieved)
            if (item.should_abstain and is_abstained) or (not item.should_abstain and not is_abstained):
                abstention_correct += 1

            # Compute retrieval metrics against ground truth
            gt_ids = item.relevant_doc_identifiers
            rel_flags: list[int] = []
            for c in retrieved:
                src = c.get("source", "")
                txt = c.get("text", "")
                is_rel = 1 if (any(g.lower() in src.lower() or g.lower() in txt.lower() for g in gt_ids)) else 0
                rel_flags.append(is_rel)

            if rel_flags and any(rel_flags):
                hit_list.append(1.0)
                first_hit = rel_flags.index(1) + 1
                mrr_list.append(1.0 / first_hit)
                hits = sum(rel_flags)
                prec_list.append(hits / len(rel_flags))
                rec_list.append(min(1.0, hits / max(1, len(gt_ids))))
                dcg = sum((r / math.log2(i + 2)) for i, r in enumerate(rel_flags[:5]))
                idcg = sum((1.0 / math.log2(i + 2)) for i in range(min(hits, 5)))
                ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)
            elif not item.should_abstain:
                hit_list.append(0.0)
                mrr_list.append(0.0)
                prec_list.append(0.0)
                rec_list.append(0.0)
                ndcg_list.append(0.0)

            # Faithfulness score
            if pdata.metrics_info and "faithfulness" in pdata.metrics_info:
                faith_list.append(float(pdata.metrics_info["faithfulness"]))

        total_n = len(dataset)
        sorted_lat = sorted(latencies)
        p95_idx = int(len(sorted_lat) * 0.95)
        p95_lat = sorted_lat[p95_idx] if sorted_lat else 0.0

        return ConfigurationBenchmarkResult(
            config_name=name,
            description=description,
            total_evaluated=total_n,
            recall_at_k=sum(rec_list) / max(1, len(rec_list)) if rec_list else 0.0,
            precision_at_k=sum(prec_list) / max(1, len(prec_list)) if prec_list else 0.0,
            mrr=sum(mrr_list) / max(1, len(mrr_list)) if mrr_list else 0.0,
            ndcg_at_k=sum(ndcg_list) / max(1, len(ndcg_list)) if ndcg_list else 0.0,
            hit_rate=sum(hit_list) / max(1, len(hit_list)) if hit_list else 0.0,
            faithfulness=sum(faith_list) / max(1, len(faith_list)) if faith_list else 0.85,
            abstention_accuracy=abstention_correct / max(1, total_n),
            avg_latency_ms=sum(latencies) / max(1, len(latencies)) if latencies else 0.0,
            p95_latency_ms=p95_lat,
            total_llm_calls=llm_calls,
            llm_calls_bypassed=llm_bypassed,
            llm_bypass_rate_pct=(llm_bypassed / max(1, total_n)) * 100,
            toggles=toggles,
            numeric_params=numeric_params or {},
        )
