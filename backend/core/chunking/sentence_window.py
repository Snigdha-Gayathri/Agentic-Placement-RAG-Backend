"""Sentence-window chunker — each chunk is a sentence with surrounding context."""
from __future__ import annotations

import re
from typing import Any

from backend.core.chunking.base import (
    BaseChunker,
    ChunkerFactory,
    DocumentChunk,
)


class SentenceWindowChunker(BaseChunker):
    """Create one chunk per sentence, enriched with N surrounding sentences.

    Each chunk's primary content is a single sentence, but the text
    includes ``window_size`` sentences before and after for context.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        window_size: int = 2,
    ) -> None:
        self._chunk_size = chunk_size  # reserved for future length guard
        self._window = max(0, window_size)
        _ = chunk_overlap  # not used but accepted for factory compat

    def chunk(self, text: str, metadata_base: dict[str, Any]) -> list[DocumentChunk]:
        """Split *text* into sentence-window chunks."""
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: list[DocumentChunk] = []
        for idx, _sentence in enumerate(sentences):
            start = max(0, idx - self._window)
            end = min(len(sentences), idx + self._window + 1)
            window_text = " ".join(sentences[start:end]).strip()
            if not window_text:
                continue
            meta = self._build_metadata(
                metadata_base,
                chunk_index=idx,
                chunk_strategy="sentence_window",
            )
            chunks.append(DocumentChunk(text=window_text, metadata=meta))
        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using punctuation boundaries."""
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if s.strip()]


# Auto-register with the factory
ChunkerFactory.register("sentence_window", SentenceWindowChunker)
