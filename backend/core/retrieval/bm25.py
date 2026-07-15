"""BM25 sparse retriever over corpus chunks."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from backend.core.vector_store.base import BaseVectorStore, ScoredChunk
from .base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"\b\w+\b", text) if len(w) > 1]


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

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        start_t = time.perf_counter()
        if not self._all_chunks:
            self._index_corpus()

        if not self._all_chunks:
            return RetrievalResult(chunks=[], retriever_type="bm25", latency_ms=0.0)

        query_tokens = _tokenize(query)
        scored: list[ScoredChunk] = []

        if self._bm25 and query_tokens:
            doc_scores = self._bm25.get_scores(query_tokens)
            max_s = max(doc_scores) if len(doc_scores) > 0 and max(doc_scores) > 0 else 1.0
            for chunk, s in zip(self._all_chunks, doc_scores):
                if metadata_filters:
                    match = all(
                        str(chunk.metadata.get(k, "")).strip().lower() == str(v).strip().lower()
                        for k, v in metadata_filters.items() if v
                    )
                    if not match:
                        continue
                norm_score = float(s / max_s)
                if norm_score > 0.01:
                    scored.append(
                        ScoredChunk(
                            chunk_id=chunk.chunk_id,
                            text=chunk.text,
                            metadata=chunk.metadata,
                            score=round(norm_score, 4),
                        )
                    )
        else:
            # Fallback keyword match
            q_set = set(query_tokens)
            for chunk in self._all_chunks:
                if metadata_filters:
                    match = all(
                        str(chunk.metadata.get(k, "")).strip().lower() == str(v).strip().lower()
                        for k, v in metadata_filters.items() if v
                    )
                    if not match:
                        continue
                c_tokens = set(_tokenize(chunk.text))
                overlap = len(q_set.intersection(c_tokens))
                if overlap > 0:
                    scored.append(
                        ScoredChunk(
                            chunk_id=chunk.chunk_id,
                            text=chunk.text,
                            metadata=chunk.metadata,
                            score=round(overlap / max(1, len(q_set)), 4),
                        )
                    )

        scored.sort(key=lambda x: x.score, reverse=True)
        top_chunks = scored[:top_k]
        latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
        return RetrievalResult(
            chunks=top_chunks,
            retriever_type="bm25",
            latency_ms=latency_ms,
            metadata_filters_applied=metadata_filters,
        )
