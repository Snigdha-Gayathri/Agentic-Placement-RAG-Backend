"""Sliding-window chunker with configurable window size and stride."""
from __future__ import annotations

import re
from typing import Any

from backend.core.chunking.base import (
    BaseChunker,
    ChunkerFactory,
    DocumentChunk,
)


class SlidingWindowChunker(BaseChunker):
    """Produce overlapping chunks using a sliding window.

    The window advances by ``stride`` words on each step, producing
    chunks of exactly ``window_size`` words (except possibly the last one).
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        window_size: int | None = None,
        stride: int | None = None,
    ) -> None:
        self._window = window_size or max(50, chunk_size)
        self._stride = stride or max(1, self._window - max(0, chunk_overlap))

    def chunk(self, text: str, metadata_base: dict[str, Any]) -> list[DocumentChunk]:
        """Split *text* using a sliding window."""
        normalised = re.sub(r"\s+", " ", text).strip()
        if not normalised:
            return []

        words = normalised.split()
        chunks: list[DocumentChunk] = []

        for idx, start in enumerate(range(0, len(words), self._stride)):
            window_words = words[start : start + self._window]
            if not window_words:
                break
            chunk_text = " ".join(window_words)
            meta = self._build_metadata(
                metadata_base,
                chunk_index=idx,
                chunk_strategy="sliding_window",
            )
            chunks.append(DocumentChunk(text=chunk_text, metadata=meta))

            # Stop if we've reached the end
            if start + self._window >= len(words):
                break
        return chunks


# Auto-register with the factory
ChunkerFactory.register("sliding_window", SlidingWindowChunker)
