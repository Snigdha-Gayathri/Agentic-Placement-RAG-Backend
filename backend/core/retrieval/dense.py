"""Dense embedding retriever."""

from __future__ import annotations

import time
from typing import Any

from backend.core.embeddings.base import BaseEmbedding
from backend.core.vector_store.base import BaseVectorStore
from .base import BaseRetriever, RetrievalResult


class DenseRetriever(BaseRetriever):
    """Dense vector similarity search using embeddings."""

    def __init__(self, embedder: BaseEmbedding, vector_store: BaseVectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        start_t = time.perf_counter()
        query_emb = await self.embedder.embed_text(query)
        chunks = self.vector_store.search(
            query_embedding=query_emb,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )
        latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
        return RetrievalResult(
            chunks=chunks,
            retriever_type="dense",
            latency_ms=latency_ms,
            metadata_filters_applied=metadata_filters,
        )
