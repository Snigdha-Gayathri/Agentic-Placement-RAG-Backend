"""Hybrid retriever fusing Dense and BM25 search via Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import time
from typing import Any

from backend.core.vector_store.base import ScoredChunk
from .base import BaseRetriever, RetrievalResult


class HybridRetriever(BaseRetriever):
    """Combines Dense and BM25 retrievers using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        start_t = time.perf_counter()
        dense_res = await self.dense_retriever.retrieve(
            query, top_k=top_k * 2, metadata_filters=metadata_filters
        )
        bm25_res = await self.bm25_retriever.retrieve(
            query, top_k=top_k * 2, metadata_filters=metadata_filters
        )

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, ScoredChunk] = {}
        source_map: dict[str, list[str]] = {}

        for rank, chunk in enumerate(dense_res.chunks):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            score = self.dense_weight * (1.0 / (self.rrf_k + rank + 1))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score
            source_map.setdefault(cid, []).append("dense")

        for rank, chunk in enumerate(bm25_res.chunks):
            cid = chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = chunk
            score = self.sparse_weight * (1.0 / (self.rrf_k + rank + 1))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score
            source_map.setdefault(cid, []).append("bm25")

        sorted_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]
        max_rrf = rrf_scores[sorted_ids[0]] if sorted_ids else 1.0

        fused_chunks: list[ScoredChunk] = []
        for cid in sorted_ids:
            original = chunk_map[cid]
            norm_score = rrf_scores[cid] / max_rrf
            meta = dict(original.metadata)
            meta["retrieval_sources"] = ",".join(source_map.get(cid, []))
            fused_chunks.append(
                ScoredChunk(
                    chunk_id=cid,
                    text=original.text,
                    metadata=meta,
                    score=round(norm_score, 4),
                )
            )

        if metadata_filters:
            filtered_fused = []
            for fc in fused_chunks:
                match = all(
                    str(fc.metadata.get(k, "")).strip().lower() == str(v).strip().lower()
                    for k, v in metadata_filters.items() if v
                )
                if match:
                    filtered_fused.append(fc)
            fused_chunks = filtered_fused

        latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
        return RetrievalResult(
            chunks=fused_chunks,
            retriever_type="hybrid",
            latency_ms=latency_ms,
            metadata_filters_applied=metadata_filters,
        )
