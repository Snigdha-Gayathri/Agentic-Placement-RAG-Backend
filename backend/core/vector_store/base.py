"""Abstract base class and data structures for vector stores."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from backend.core.chunking.base import DocumentChunk


@dataclass
class ScoredChunk:
    """A retrieved chunk along with its similarity score and metadata."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "score": self.score,
        }


@dataclass
class VectorDBStats:
    """Statistics about the underlying vector database index."""

    total_chunks: int = 0
    embedding_dimension: int = 0
    collection_name: str = ""
    metadata_distribution: dict[str, Any] = field(default_factory=dict)
    avg_chunk_length: float = 0.0
    index_size_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_chunks": self.total_chunks,
            "embedding_dimension": self.embedding_dimension,
            "collection_name": self.collection_name,
            "metadata_distribution": self.metadata_distribution,
            "avg_chunk_length": self.avg_chunk_length,
            "index_size_mb": self.index_size_mb,
        }


class BaseVectorStore(abc.ABC):
    """Abstract interface for dense vector storage and retrieval."""

    @abc.abstractmethod
    def add_documents(
        self, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> None:
        """Add chunks and their embeddings to the store."""
        ...

    @abc.abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Search the vector store by dense embedding similarity."""
        ...

    @abc.abstractmethod
    def get_all_chunks(self) -> list[ScoredChunk]:
        """Retrieve all stored chunks (useful for BM25 indexing or fallback)."""
        ...

    @abc.abstractmethod
    def get_stats(self) -> VectorDBStats:
        """Return statistics on the stored collection."""
        ...

    @abc.abstractmethod
    def delete_collection(self) -> None:
        """Delete all stored data in the collection."""
        ...
