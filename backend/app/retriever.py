"""Retriever that ranks provided RAG items by semantic overlap."""

from __future__ import annotations

from typing import Iterable

from security import RetrievedChunk
from security.utils import jaccard_similarity, normalize_text


class Retriever:
    """Backend retriever adapter that accepts scored chunks from VectorStore.

    The VectorStore returns ScoredChunk objects; this adapter converts them
    into RetrievedChunk instances compatible with the rest of the guard pipeline.
    """

    def retrieve(self, query: str, items: Iterable[object]) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for item in items:
            text = normalize_text(item.text)
            similarity = float(item.score)
            chunks.append(RetrievedChunk(source=item.source_path, text=text, similarity=similarity))
        return sorted(chunks, key=lambda c: c.similarity, reverse=True)
