"""Cross-encoder reranker using sentence-transformers with memory-constrained lifecycle."""

from __future__ import annotations

import gc
import logging
import math
import os
import threading
import time
from typing import Any

from backend.core.vector_store.base import ScoredChunk
from .base import BaseReranker, RankedChunk, RerankingResult

logger = logging.getLogger(__name__)

_SHARED_CROSS_ENCODER: Any = None
_SHARED_CROSS_ENCODER_LOCK = threading.Lock()


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class CrossEncoderReranker(BaseReranker):
    """Memory-optimized reranker using cross-encoder/ms-marco-MiniLM-L-6-v2."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name

    def _get_model(self) -> Any:
        """Lazily load shared CrossEncoder singleton with strictly bounded thread and RAM footprint."""
        global _SHARED_CROSS_ENCODER
        if _SHARED_CROSS_ENCODER is not None:
            return _SHARED_CROSS_ENCODER if _SHARED_CROSS_ENCODER is not False else None

        low_mem_env = os.getenv("RERANKER_LOW_MEMORY", "").strip().lower()
        if low_mem_env in ("1", "true", "yes", "on"):
            logger.info("RERANKER_LOW_MEMORY is enabled; using zero-memory calibrated ranking fallback.")
            with _SHARED_CROSS_ENCODER_LOCK:
                _SHARED_CROSS_ENCODER = False
            return None

        with _SHARED_CROSS_ENCODER_LOCK:
            if _SHARED_CROSS_ENCODER is not None:
                return _SHARED_CROSS_ENCODER if _SHARED_CROSS_ENCODER is not False else None

            try:
                # Constrain PyTorch thread pools to prevent multithread memory explosions
                os.environ["OMP_NUM_THREADS"] = "1"
                os.environ["MKL_NUM_THREADS"] = "1"
                os.environ["OPENBLAS_NUM_THREADS"] = "1"
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                import torch
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
                torch.set_grad_enabled(False)

                from sentence_transformers import CrossEncoder

                logger.info("Initializing shared CrossEncoder: %s (CPU, single-thread, max_length=128)", self.model_name)
                _SHARED_CROSS_ENCODER = CrossEncoder(
                    self.model_name,
                    device="cpu",
                    max_length=128,
                    model_kwargs={"low_cpu_mem_usage": True},
                )
                logger.info("Successfully loaded shared CrossEncoder model: %s", self.model_name)
            except Exception as exc:
                logger.warning(
                    "Could not initialize CrossEncoder (%s): %s. Will fallback to calibrated rank scoring.",
                    self.model_name,
                    exc,
                )
                _SHARED_CROSS_ENCODER = False

        return _SHARED_CROSS_ENCODER if _SHARED_CROSS_ENCODER is not False else None

    def rerank(
        self, query: str, chunks: list[ScoredChunk], top_k: int = 5
    ) -> RerankingResult:
        start_t = time.perf_counter()
        if not chunks:
            return RerankingResult(chunks=[], latency_ms=0.0)

        model = self._get_model()
        ranked: list[RankedChunk] = []

        if model:
            try:
                import torch
                # Truncate text pairs to bounded size for fast evaluation & minimal tensor RAM
                pairs = [[query, getattr(c, "text", str(c))[:300]] for c in chunks]
                with torch.inference_mode():
                    scores = model.predict(pairs, batch_size=4, show_progress_bar=False)

                for orig_rank_0, (chunk, s) in enumerate(zip(chunks, scores)):
                    orig_rank = orig_rank_0 + 1
                    orig_score = getattr(chunk, "score", getattr(chunk, "similarity_score", 0.0))
                    norm_score = _sigmoid(float(s))
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
                            score=round(max(norm_score, float(orig_score)), 4),
                        )
                    )
                ranked.sort(key=lambda x: x.cross_encoder_score, reverse=True)
                gc.collect()
            except Exception as exc:
                logger.warning("Error predicting cross-encoder scores: %s. Falling back to calibrated rank scoring.", exc)
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

