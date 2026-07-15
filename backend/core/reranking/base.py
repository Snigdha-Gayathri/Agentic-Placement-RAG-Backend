"""Abstract base class and data structures for reranking."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from backend.core.vector_store.base import ScoredChunk


@dataclass
class RankedChunk:
    """Chunk that has been reranked with score and rank movement tracking."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    original_score: float
    cross_encoder_score: float
    original_rank: int
    new_rank: int
    rank_change: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "original_score": self.original_score,
            "cross_encoder_score": self.cross_encoder_score,
            "original_rank": self.original_rank,
            "new_rank": self.new_rank,
            "rank_change": self.rank_change,
        }


@dataclass
class RerankingResult:
    """Output returned by any reranker."""

    chunks: list[RankedChunk] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "latency_ms": self.latency_ms,
        }


class BaseReranker(abc.ABC):
    """Abstract interface for reranking candidate chunks."""

    @abc.abstractmethod
    def rerank(
        self, query: str, chunks: list[ScoredChunk], top_k: int = 5
    ) -> RerankingResult:
        """Rerank candidate chunks for a query."""
        ...
