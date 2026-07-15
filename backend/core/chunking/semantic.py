"""Semantic chunker — uses embedding similarity to find natural breakpoints."""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from backend.core.chunking.base import (
    BaseChunker,
    ChunkerFactory,
    DocumentChunk,
)

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticChunker(BaseChunker):
    """Split text using embedding-similarity breakpoint detection.

    Sentences with low similarity to the running group are treated as
    natural breakpoints, starting a new chunk.

    Requires an embedding model to be passed during construction.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.5,
        embedding_model: Any | None = None,
    ) -> None:
        self._chunk_size = max(50, chunk_size)
        self._overlap = max(0, min(chunk_overlap, self._chunk_size - 1))
        self._threshold = similarity_threshold
        self._embedding = embedding_model

    def chunk(self, text: str, metadata_base: dict[str, Any]) -> list[DocumentChunk]:
        """Split *text* at semantic breakpoints detected via embeddings.

        If no embedding model is available, falls back to sentence-level
        fixed-size grouping.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        if self._embedding is None:
            logger.warning(
                "No embedding model provided for SemanticChunker — "
                "falling back to sentence grouping."
            )
            return self._fallback_chunk(sentences, metadata_base)

        # Run the async embedding in a sync context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an event loop — create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                embeddings = pool.submit(
                    asyncio.run,
                    self._embedding.embed_batch(sentences),
                ).result()
        else:
            embeddings = asyncio.run(self._embedding.embed_batch(sentences))

        groups = self._find_breakpoints(sentences, embeddings)

        result: list[DocumentChunk] = []
        for idx, group in enumerate(groups):
            chunk_text = " ".join(group).strip()
            if not chunk_text:
                continue
            meta = self._build_metadata(
                metadata_base,
                chunk_index=idx,
                chunk_strategy="semantic",
            )
            result.append(DocumentChunk(text=chunk_text, metadata=meta))
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _find_breakpoints(
        self,
        sentences: list[str],
        embeddings: list[list[float]],
    ) -> list[list[str]]:
        """Group sentences by detecting cosine-similarity drops."""
        if len(sentences) <= 1:
            return [sentences]

        groups: list[list[str]] = []
        current_group: list[str] = [sentences[0]]
        current_embedding = embeddings[0]
        current_word_count = len(sentences[0].split())

        for i in range(1, len(sentences)):
            sim = _cosine_similarity(current_embedding, embeddings[i])
            word_count = len(sentences[i].split())

            if sim < self._threshold or current_word_count + word_count > self._chunk_size:
                groups.append(current_group)
                current_group = [sentences[i]]
                current_embedding = embeddings[i]
                current_word_count = word_count
            else:
                current_group.append(sentences[i])
                # Update running embedding as average
                current_embedding = [
                    (a * len(current_group) + b) / (len(current_group) + 1)
                    for a, b in zip(current_embedding, embeddings[i])
                ]
                current_word_count += word_count

        if current_group:
            groups.append(current_group)
        return groups

    def _fallback_chunk(
        self,
        sentences: list[str],
        metadata_base: dict[str, Any],
    ) -> list[DocumentChunk]:
        """Group sentences by word count without embeddings."""
        chunks: list[DocumentChunk] = []
        current: list[str] = []
        current_len = 0
        idx = 0

        for sentence in sentences:
            words = len(sentence.split())
            if current_len + words > self._chunk_size and current:
                meta = self._build_metadata(
                    metadata_base,
                    chunk_index=idx,
                    chunk_strategy="semantic",
                )
                chunks.append(DocumentChunk(text=" ".join(current), metadata=meta))
                idx += 1
                current = []
                current_len = 0
            current.append(sentence)
            current_len += words

        if current:
            meta = self._build_metadata(
                metadata_base,
                chunk_index=idx,
                chunk_strategy="semantic",
            )
            chunks.append(DocumentChunk(text=" ".join(current), metadata=meta))
        return chunks


# Auto-register with the factory
ChunkerFactory.register("semantic", SemanticChunker)
