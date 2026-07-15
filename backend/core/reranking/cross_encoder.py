"""Cross-encoder reranker using sentence-transformers."""

from __future__ import annotations

import logging
import time
from typing import Any

from backend.core.vector_store.base import ScoredChunk
from .base import BaseReranker, RankedChunk, RerankingResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Reranker using cross-encoder/ms-marco-MiniLM-L-6-v2."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._init_model()

    def _init_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            logger.info("Loaded CrossEncoder model: %s", self.model_name)
        except Exception as exc:
            logger.warning("Could not initialize CrossEncoder (%s): %s. Will fallback to original score.", self.model_name, exc)
            self._model = None

    def rerank(
        self, query: str, chunks: list[ScoredChunk], top_k: int = 5
    ) -> RerankingResult:
        start_t = time.perf_counter()
        if not chunks:
            return RerankingResult(chunks=[], latency_ms=0.0)

        pairs = [[query, c.text] for c in chunks]
        ranked: list[RankedChunk] = []

        if self._model:
            try:
                scores = self._model.predict(pairs)
                # Normalize cross encoder logits/scores roughly to [0, 1] via sigmoid or minmax
                for orig_rank_0, (chunk, s) in enumerate(zip(chunks, scores)):
                    orig_rank = orig_rank_0 + 1
                    orig_score = getattr(chunk, "score", getattr(chunk, "similarity_score", 0.0))
                    ranked.append(
                        RankedChunk(
                            chunk_id=getattr(chunk, "chunk_id", str(orig_rank_0)),
                            text=getattr(chunk, "text", str(chunk)),
                            metadata=getattr(chunk, "metadata", {}) or {},
                            original_score=round(float(orig_score), 4),
                            cross_encoder_score=round(float(s), 4),
                            original_rank=orig_rank,
                            new_rank=orig_rank,
                            rank_change=0,
                        )
                    )
                ranked.sort(key=lambda x: x.cross_encoder_score, reverse=True)
            except Exception as exc:
                logger.warning("Error predicting cross-encoder scores: %s", exc)
                ranked = self._fallback_rank(chunks)
        else:
            ranked = self._fallback_rank(chunks)

        top_ranked = ranked[:top_k]
        for new_r_0, item in enumerate(top_ranked):
            new_r = new_r_0 + 1
            item.new_rank = new_r
            item.rank_change = item.original_rank - new_r

        latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
        return RerankingResult(chunks=top_ranked, latency_ms=latency_ms)

    def _fallback_rank(self, chunks: list[ScoredChunk]) -> list[RankedChunk]:
        return [
            RankedChunk(
                chunk_id=getattr(c, "chunk_id", str(i)),
                text=getattr(c, "text", str(c)),
                metadata=getattr(c, "metadata", {}) or {},
                original_score=round(float(getattr(c, "score", getattr(c, "similarity_score", 0.0))), 4),
                cross_encoder_score=round(float(getattr(c, "score", getattr(c, "similarity_score", 0.0))), 4),
                original_rank=i + 1,
                new_rank=i + 1,
                rank_change=0,
            )
            for i, c in enumerate(chunks)
        ]
