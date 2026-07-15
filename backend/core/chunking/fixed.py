"""Fixed-size word-count chunker with overlap."""
from __future__ import annotations

import re
from typing import Any

from backend.core.chunking.base import (
    BaseChunker,
    ChunkerFactory,
    DocumentChunk,
)


class FixedChunker(BaseChunker):
    """Split text into fixed-size chunks by word count with configurable overlap.

    This mirrors the original ``_chunk_text`` logic from
    ``backend/app/vector_store.py`` but emits rich :class:`DocumentChunk`
    objects.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self._chunk_size = max(50, chunk_size)
        self._overlap = max(0, min(chunk_overlap, self._chunk_size - 1))

    def chunk(self, text: str, metadata_base: dict[str, Any]) -> list[DocumentChunk]:
        """Split *text* into fixed-size word chunks."""
        normalised = re.sub(r"\s+", " ", text).strip()
        if not normalised:
            return []

        words = normalised.split(" ")
        step = max(1, self._chunk_size - self._overlap)
        chunks: list[DocumentChunk] = []

        for idx, start in enumerate(range(0, len(words), step)):
            chunk_text = " ".join(words[start : start + self._chunk_size]).strip()
            if not chunk_text:
                continue
            meta = self._build_metadata(
                metadata_base,
                chunk_index=idx,
                chunk_strategy="fixed",
            )
            chunks.append(DocumentChunk(text=chunk_text, metadata=meta))
        return chunks


# Auto-register with the factory
ChunkerFactory.register("fixed", FixedChunker)
