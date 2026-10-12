"""ChromaDB implementation of BaseVectorStore."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from backend.core.chunking.base import DocumentChunk
from .base import BaseVectorStore, ScoredChunk, VectorDBStats

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Persistent ChromaDB vector store implementation supporting metadata filters."""

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "placement_rag",
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._init_db()

    def _init_db(self) -> None:
        try:
            import chromadb

            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Initialized ChromaDB collection %r at %s", self.collection_name, self.persist_directory)
        except Exception as exc:
            logger.warning("Could not initialize ChromaDB PersistentClient: %s. Falling back to EphemeralClient.", exc)
            try:
                import chromadb
                self._client = chromadb.EphemeralClient()
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as inner_exc:
                logger.error("Failed to initialize ChromaDB: %s", inner_exc)
                self._client = None
                self._collection = None

    def add_documents(
        self, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> None:
        if not self._collection or not chunks:
            return
        ids = []
        documents = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            cid = chunk.metadata.chunk_id or f"chunk_{i}_{hash(chunk.text)}"
            ids.append(cid)
            documents.append(chunk.text)
            meta = chunk.metadata.to_dict()
            # Ensure primitive values for Chroma metadata
            clean_meta = {}
            for k, v in meta.items():
                if v is None:
                    clean_meta[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            metadatas.append(clean_meta)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
        fallback_unfiltered: bool = True,
    ) -> list[ScoredChunk]:
        if not self._collection:
            return []
        where_filter = None
        valid_filters = {}
        if metadata_filters:
            for k, v in metadata_filters.items():
                if not v:
                    continue
                # Normalize canonical role aliases to ChromaDB stored metadata schema ('SDE')
                if k == "role" and str(v).strip().lower() in ("software development engineer", "sde", "swe", "software engineer", "sde 1", "sde 2", "sde i", "sde ii"):
                    valid_filters[k] = "SDE"
                else:
                    valid_filters[k] = v

            if "company" in valid_filters:
                where_filter = {"company": valid_filters["company"]}
            elif len(valid_filters) == 1:
                k, v = next(iter(valid_filters.items()))
                where_filter = {k: v}
            elif len(valid_filters) > 1:
                where_filter = {"$and": [{k: v} for k, v in valid_filters.items()]}

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("Chroma search error with filters %r: %s. Retrying without filter.", where_filter, exc)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

        docs = results.get("documents", [[]])[0]
        # Hierarchical fallback: try company-only filter before going completely unfiltered
        if where_filter and (not docs or len(docs) == 0) and fallback_unfiltered:
            if "company" in valid_filters and len(valid_filters) > 1:
                logger.info("Chroma search with compound filter %r returned 0 results; falling back to company filter.", where_filter)
                try:
                    results = self._collection.query(
                        query_embeddings=[query_embedding],
                        n_results=top_k,
                        where={"company": valid_filters["company"]},
                        include=["documents", "metadatas", "distances"],
                    )
                    docs = results.get("documents", [[]])[0]
                except Exception:
                    pass
            if not docs or len(docs) == 0:
                logger.info("Chroma search with filter %r returned 0 results; falling back to unfiltered search.", where_filter)
                results = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
                docs = results.get("documents", [[]])[0]

        scored = []
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            # Cosine distance in Chroma is 1 - cosine similarity
            sim = max(0.0, min(1.0, 1.0 - dist))
            scored.append(
                ScoredChunk(
                    chunk_id=cid,
                    text=doc,
                    metadata=meta or {},
                    score=sim,
                )
            )

        if metadata_filters:
            filtered_scored = []
            sde_aliases = {"sde", "swe", "software development engineer", "software engineer", "sde-2", "sde 1", "sde i", "sde ii"}
            for sc in scored:
                def _val_match(k: str, target_v: Any) -> bool:
                    actual_v = str(sc.metadata.get(k, "")).strip().lower()
                    tv = str(target_v).strip().lower()
                    if actual_v == tv:
                        return True
                    if k == "role" and actual_v in sde_aliases and tv in sde_aliases:
                        return True
                    return False

                match = all(_val_match(k, v) for k, v in metadata_filters.items() if v)
                if match:
                    filtered_scored.append(sc)
            if filtered_scored:
                scored = filtered_scored
            elif fallback_unfiltered:
                logger.info("Chroma post-filter %r yielded 0 hits; retaining unfiltered candidates.", metadata_filters)
            else:
                scored = []

        return scored

    def get_all_chunks(self) -> list[ScoredChunk]:
        if not self._collection:
            return []
        data = self._collection.get(include=["documents", "metadatas"])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        ids = data.get("ids", [])
        return [
            ScoredChunk(
                chunk_id=cid,
                text=doc,
                metadata=meta or {},
                score=1.0,
            )
            for cid, doc, meta in zip(ids, docs, metas)
        ]

    def get_stats(self) -> VectorDBStats:
        if not self._collection:
            return VectorDBStats()
        data = self._collection.get(include=["documents", "metadatas"])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        total = len(docs)
        avg_len = sum(len(d) for d in docs) / total if total > 0 else 0.0

        company_counts: dict[str, int] = {}
        topic_counts: dict[str, int] = {}
        for m in metas:
            comp = m.get("company", "General") or "General"
            company_counts[comp] = company_counts.get(comp, 0) + 1
            top = m.get("topic", "General") or "General"
            topic_counts[top] = topic_counts.get(top, 0) + 1

        size_mb = 0.0
        try:
            if os.path.exists(self.persist_directory):
                size_bytes = sum(
                    f.stat().st_size for f in Path(self.persist_directory).rglob("*") if f.is_file()
                )
                size_mb = round(size_bytes / (1024 * 1024), 2)
        except Exception:
            pass

        return VectorDBStats(
            total_chunks=total,
            embedding_dimension=768,
            collection_name=self.collection_name,
            metadata_distribution={"companies": company_counts, "topics": topic_counts},
            avg_chunk_length=round(avg_len, 1),
            index_size_mb=size_mb,
        )

    def delete_collection(self) -> None:
        if self._client and self._collection:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def delete_by_source(self, source_file: str) -> None:
        if self._collection:
            try:
                self._collection.delete(where={"source_file": source_file})
                logger.info("Deleted old ChromaDB chunks for source: %s", source_file)
            except Exception as exc:
                logger.error("Failed to delete chunks for %s: %s", source_file, exc)
