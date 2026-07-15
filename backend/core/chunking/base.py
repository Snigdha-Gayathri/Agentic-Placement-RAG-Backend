"""Base abstractions for document chunking."""
from __future__ import annotations

import abc
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkMetadata:
    """Rich metadata attached to every document chunk.

    Attributes are populated from the source document's known properties.
    Unknown fields default to empty strings.
    """

    chunk_id: str = ""
    source_file: str = ""
    company: str = ""
    page_number: int = 0
    chunk_index: int = 0
    chunk_strategy: str = ""
    tags: list[str] = field(default_factory=list)
    topic: str = ""
    difficulty: str = ""
    role: str = ""
    year: str = ""
    interview_round: str = ""
    document_version: str = ""
    parent_chunk_id: str | None = None
    enhancement_applied: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary suitable for vector-store metadata."""
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "company": self.company,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "chunk_strategy": self.chunk_strategy,
            "tags": ",".join(self.tags) if self.tags else "",
            "topic": self.topic,
            "difficulty": self.difficulty,
            "role": self.role,
            "year": self.year,
            "interview_round": self.interview_round,
            "document_version": self.document_version,
            "parent_chunk_id": self.parent_chunk_id or "",
            "enhancement_applied": self.enhancement_applied or "",
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChunkMetadata:
        """Deserialise from a plain dictionary."""
        tags_raw = d.get("tags", "")
        tags = tags_raw.split(",") if isinstance(tags_raw, str) and tags_raw else (
            tags_raw if isinstance(tags_raw, list) else []
        )
        return cls(
            chunk_id=d.get("chunk_id", ""),
            source_file=d.get("source_file", ""),
            company=d.get("company", ""),
            page_number=int(d.get("page_number", 0)),
            chunk_index=int(d.get("chunk_index", 0)),
            chunk_strategy=d.get("chunk_strategy", ""),
            tags=tags,
            topic=d.get("topic", ""),
            difficulty=d.get("difficulty", ""),
            role=d.get("role", ""),
            year=d.get("year", ""),
            interview_round=d.get("interview_round", ""),
            document_version=d.get("document_version", ""),
            parent_chunk_id=d.get("parent_chunk_id") or None,
            enhancement_applied=d.get("enhancement_applied") or None,
        )


@dataclass
class DocumentChunk:
    """A chunk of text with its associated metadata."""

    text: str
    metadata: ChunkMetadata


class BaseChunker(abc.ABC):
    """Abstract interface for document chunking strategies."""

    @abc.abstractmethod
    def chunk(self, text: str, metadata_base: dict[str, Any]) -> list[DocumentChunk]:
        """Split *text* into chunks.

        Args:
            text: The full document text.
            metadata_base: Shared metadata fields (source_file, company, etc.).

        Returns:
            A list of :class:`DocumentChunk` instances.
        """
        ...

    @staticmethod
    def _generate_chunk_id() -> str:
        """Create a unique chunk identifier."""
        return uuid.uuid4().hex[:16]

    @staticmethod
    def _build_metadata(
        metadata_base: dict[str, Any],
        *,
        chunk_index: int,
        chunk_strategy: str,
        parent_chunk_id: str | None = None,
    ) -> ChunkMetadata:
        """Build a :class:`ChunkMetadata` from base values and per-chunk overrides."""
        return ChunkMetadata(
            chunk_id=uuid.uuid4().hex[:16],
            source_file=metadata_base.get("source_file", ""),
            company=metadata_base.get("company", ""),
            page_number=int(metadata_base.get("page_number", 0)),
            chunk_index=chunk_index,
            chunk_strategy=chunk_strategy,
            tags=metadata_base.get("tags", []),
            topic=metadata_base.get("topic", ""),
            difficulty=metadata_base.get("difficulty", ""),
            role=metadata_base.get("role", ""),
            year=metadata_base.get("year", ""),
            interview_round=metadata_base.get("interview_round", ""),
            document_version=metadata_base.get("document_version", ""),
            parent_chunk_id=parent_chunk_id,
            enhancement_applied=metadata_base.get("enhancement_applied"),
        )


class ChunkerFactory:
    """Factory that returns a chunker instance for a given strategy name.

    Supported strategies: ``fixed``, ``recursive``, ``semantic``,
    ``sliding_window``, ``sentence_window``, ``parent_child``.
    """

    _registry: dict[str, type[BaseChunker]] = {}

    @classmethod
    def register(cls, name: str, chunker_cls: type[BaseChunker]) -> None:
        """Register a chunker class under a strategy name."""
        cls._registry[name] = chunker_cls

    @classmethod
    def create(cls, strategy: str, **kwargs: Any) -> BaseChunker:
        """Instantiate a chunker for *strategy*.

        Args:
            strategy: One of the registered strategy names.
            **kwargs: Forwarded to the chunker constructor.

        Returns:
            A :class:`BaseChunker` instance.

        Raises:
            ValueError: If *strategy* is not registered.
        """
        # Lazy-import to ensure all strategy modules have registered
        cls._ensure_registered()
        if strategy not in cls._registry:
            raise ValueError(
                f"Unknown chunk strategy '{strategy}'. "
                f"Available: {sorted(cls._registry)}"
            )
        return cls._registry[strategy](**kwargs)

    @classmethod
    def _ensure_registered(cls) -> None:
        """Import strategy modules so they can self-register."""
        if cls._registry:
            return
        # Side-effect imports trigger registration via module-level calls
        from backend.core.chunking import (  # noqa: F401
            fixed,
            parent_child,
            recursive,
            semantic,
            sentence_window,
            sliding_window,
        )
