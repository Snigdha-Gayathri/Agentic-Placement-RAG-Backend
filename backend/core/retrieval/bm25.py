"""BM25 sparse retriever over corpus chunks."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from backend.core.vector_store.base import BaseVectorStore, ScoredChunk
from .base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)

SDE_ALIASES = {
    "sde",
    "swe",
    "software development engineer",
    "software engineer",
    "sde-2",
    "sde 1",
    "sde 2",
    "sde i",
    "sde ii",
}
PM_ALIASES = {
    "pm",
    "product manager",
    "product management",
}


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"\b\w+\b", text) if len(w) > 1]


def _metadata_matches(chunk_meta: dict[str, Any] | None, filters: dict[str, Any]) -> bool:
    """Safe metadata matcher that handles missing metadata, None values, and role aliases."""
    if not isinstance(chunk_meta, dict):
        return False
    for k, v in filters.items():
        if not v:
            continue
        val = str(chunk_meta.get(k, "")).strip().lower()
        req = str(v).strip().lower()
        if k == "role":
            if req in SDE_ALIASES and val in SDE_ALIASES:
                continue
            if req in PM_ALIASES and val in PM_ALIASES:
                continue
        if val != req:
            return False
    return True


class BM25Retriever(BaseRetriever):
    """Sparse BM25 retriever over all corpus chunks."""

    def __init__(self, vector_store: BaseVectorStore) -> None:
        self.vector_store = vector_store
        self._bm25 = None
        self._all_chunks: list[ScoredChunk] = []
        self._index_corpus()

    def _index_corpus(self) -> None:
        try:
            self._all_chunks = self.vector_store.get_all_chunks()
            if not self._all_chunks:
                return
            tokenized = [_tokenize(c.text) for c in self._all_chunks]
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(tokenized)
            logger.info("Indexed %d chunks for BM25 retrieval.", len(self._all_chunks))
        except Exception as exc:
            logger.warning("BM25 index initialization warning/fallback: %s", exc)
            self._bm25 = None

    def _score_candidates(
        self, query_tokens: list[str], metadata_filters: dict[str, Any] | None
    ) -> list[ScoredChunk]:
        """Score candidate chunks using BM25 with optional metadata filtering."""
        scored: list[ScoredChunk] = []
        if self._bm25 and query_tokens:
            doc_scores = self._bm25.get_scores(query_tokens)
            max_s = max(doc_scores) if len(doc_scores) > 0 and max(doc_scores) > 0 else 0.0
            if max_s > 0:
                for chunk, s in zip(self._all_chunks, doc_scores):
                    if metadata_filters and not _metadata_matches(chunk.metadata, metadata_filters):
                        continue
                    norm_score = float(s / max_s)
                    if norm_score > 0.01:
                        scored.append(
                            ScoredChunk(
                                chunk_id=chunk.chunk_id,
                                text=chunk.text,
                                metadata=chunk.metadata or {},
                                score=round(norm_score, 4),
                            )
                        )

        # Fallback to keyword overlap if BM25 index unavailable or yielded zero scores (e.g. tiny test corpus IDF=0)
        if not scored and query_tokens:
            q_set = set(query_tokens)
            for chunk in self._all_chunks:
                if metadata_filters and not _metadata_matches(chunk.metadata, metadata_filters):
                    continue
                c_tokens = set(_tokenize(chunk.text))
                overlap = len(q_set.intersection(c_tokens))
                if overlap > 0:
                    scored.append(
                        ScoredChunk(
                            chunk_id=chunk.chunk_id,
                            text=chunk.text,
                            metadata=chunk.metadata or {},
                            score=round(overlap / max(1, len(q_set)), 4),
                        )
                    )
        return scored

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
        fallback_unfiltered: bool = True,
    ) -> RetrievalResult:
        start_t = time.perf_counter()
        if not self._all_chunks:
            self._index_corpus()

        if not self._all_chunks:
            return RetrievalResult(chunks=[], retriever_type="bm25", latency_ms=0.0)

        query_tokens = _tokenize(query)
        scored: list[ScoredChunk] = []
        applied_filters = metadata_filters

        if query_tokens:
            scored = self._score_candidates(query_tokens, metadata_filters)

            # Hierarchical fallback if filtered set returned nothing
            if metadata_filters and len(scored) == 0 and fallback_unfiltered:
                # 1. Hierarchical fallback to company-only if compound filter had 0 hits
                if "company" in metadata_filters and len(metadata_filters) > 1:
                    comp_filter = {"company": metadata_filters["company"]}
                    scored = self._score_candidates(query_tokens, comp_filter)
                    if scored:
                        applied_filters = comp_filter

                # 2. Only if company fallback was also 0, fall back to completely unfiltered
                if len(scored) == 0 and fallback_unfiltered:
                    scored = self._score_candidates(query_tokens, None)
                    applied_filters = None

        scored.sort(key=lambda x: x.score, reverse=True)
        top_chunks = scored[:top_k]
        latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
        return RetrievalResult(
            chunks=top_chunks,
            retriever_type="bm25",
            latency_ms=latency_ms,
            metadata_filters_applied=applied_filters,
        )
