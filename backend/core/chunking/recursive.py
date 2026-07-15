"""Recursive chunker that tries to preserve semantic units."""
from __future__ import annotations

import re
from typing import Any

from backend.core.chunking.base import (
    BaseChunker,
    ChunkerFactory,
    DocumentChunk,
)


class RecursiveChunker(BaseChunker):
    """Recursively split text by paragraph → sentence → word boundaries.

    At each level, the chunker tries to keep semantic units together.  If a
    unit exceeds ``chunk_size`` words it is recursively split at the next
    finer granularity.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self._chunk_size = max(50, chunk_size)
        self._overlap = max(0, min(chunk_overlap, self._chunk_size - 1))

    def chunk(self, text: str, metadata_base: dict[str, Any]) -> list[DocumentChunk]:
        """Split *text* recursively into semantically coherent chunks."""
        raw_chunks = self._recursive_split(text, level=0)

        # Merge small adjacent chunks and apply overlap
        merged = self._merge_chunks(raw_chunks)

        result: list[DocumentChunk] = []
        for idx, chunk_text in enumerate(merged):
            meta = self._build_metadata(
                metadata_base,
                chunk_index=idx,
                chunk_strategy="recursive",
            )
            result.append(DocumentChunk(text=chunk_text, metadata=meta))
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    _SEPARATORS = [
        r"\n\n+",        # paragraph breaks
        r"\n",            # single newlines
        r"(?<=[.!?])\s+", # sentence endings
        r"\s+",           # whitespace (word level)
    ]

    def _recursive_split(self, text: str, level: int) -> list[str]:
        """Split *text* at the current separator level, recurse if needed."""
        text = text.strip()
        if not text:
            return []

        word_count = len(text.split())
        if word_count <= self._chunk_size:
            return [text]

        if level >= len(self._SEPARATORS):
            # Fallback: hard split by word count
            return self._hard_split(text)

        pattern = self._SEPARATORS[level]
        segments = re.split(pattern, text)
        segments = [s.strip() for s in segments if s.strip()]

        chunks: list[str] = []
        for segment in segments:
            if len(segment.split()) <= self._chunk_size:
                chunks.append(segment)
            else:
                chunks.extend(self._recursive_split(segment, level + 1))
        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """Word-level hard split as final fallback."""
        words = text.split()
        step = max(1, self._chunk_size - self._overlap)
        result: list[str] = []
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + self._chunk_size])
            if chunk:
                result.append(chunk)
        return result

    def _merge_chunks(self, chunks: list[str]) -> list[str]:
        """Merge adjacent small chunks and apply overlap between them."""
        if not chunks:
            return []

        merged: list[str] = []
        buffer: list[str] = []
        buffer_len = 0

        for chunk in chunks:
            words = chunk.split()
            if buffer_len + len(words) <= self._chunk_size:
                buffer.append(chunk)
                buffer_len += len(words)
            else:
                if buffer:
                    merged.append(" ".join(buffer))
                buffer = [chunk]
                buffer_len = len(words)

        if buffer:
            merged.append(" ".join(buffer))

        # Apply overlap: prepend tail of previous chunk to next
        if self._overlap <= 0 or len(merged) <= 1:
            return merged

        overlapped: list[str] = [merged[0]]
        for i in range(1, len(merged)):
            prev_words = merged[i - 1].split()
            overlap_words = prev_words[-self._overlap :] if len(prev_words) >= self._overlap else prev_words
            overlapped.append(" ".join(overlap_words) + " " + merged[i])
        return overlapped


# Auto-register with the factory
ChunkerFactory.register("recursive", RecursiveChunker)
