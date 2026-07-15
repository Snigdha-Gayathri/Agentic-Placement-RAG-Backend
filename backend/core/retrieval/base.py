"""Abstract base class for retrieval components."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from backend.core.vector_store.base import ScoredChunk


@dataclass
class RetrievalResult:
    """Standardized output returned by any retriever."""

    chunks: list[ScoredChunk] = field(default_factory=list)
    retriever_type: str = "dense"
    latency_ms: float = 0.0
    metadata_filters_applied: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "retriever_type": self.retriever_type,
            "latency_ms": self.latency_ms,
            "metadata_filters_applied": self.metadata_filters_applied or {},
        }


class BaseRetriever(abc.ABC):
    """Abstract interface for document retrievers."""

    @abc.abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a given query."""
        ...
