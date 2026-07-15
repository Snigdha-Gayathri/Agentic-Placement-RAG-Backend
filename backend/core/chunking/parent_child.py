"""Parent-child chunker — large parent chunks with smaller child chunks."""
from __future__ import annotations

import re
import uuid
from typing import Any

from backend.core.chunking.base import (
    BaseChunker,
    ChunkerFactory,
    DocumentChunk,
)


class ParentChildChunker(BaseChunker):
    """Create large *parent* chunks and smaller *child* chunks linked via metadata.

    Parent chunks are used for context, child chunks for retrieval.
    Each child stores its ``parent_chunk_id`` in metadata so the pipeline
    can retrieve the parent for broader context when needed.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        parent_size: int | None = None,
        child_size: int | None = None,
    ) -> None:
        self._parent_size = parent_size or max(200, chunk_size * 3)
        self._child_size = child_size or max(50, chunk_size)
        self._child_overlap = max(0, min(chunk_overlap, self._child_size - 1))

    def chunk(self, text: str, metadata_base: dict[str, Any]) -> list[DocumentChunk]:
        """Split *text* into parent and child chunks.

        Returns all chunks (parents first, then children) so they can be
        indexed together.  Children carry ``parent_chunk_id`` in metadata.
        """
        normalised = re.sub(r"\s+", " ", text).strip()
        if not normalised:
            return []

        words = normalised.split()
        parent_step = max(1, self._parent_size)
        all_chunks: list[DocumentChunk] = []
        global_child_idx = 0

        for p_start in range(0, len(words), parent_step):
            parent_words = words[p_start : p_start + self._parent_size]
            parent_text = " ".join(parent_words)
            parent_id = uuid.uuid4().hex[:16]

            # Parent chunk
            parent_meta = self._build_metadata(
                metadata_base,
                chunk_index=p_start // parent_step,
                chunk_strategy="parent_child_parent",
            )
            parent_meta.chunk_id = parent_id
            all_chunks.append(DocumentChunk(text=parent_text, metadata=parent_meta))

            # Child chunks within this parent
            child_step = max(1, self._child_size - self._child_overlap)
            for c_start in range(0, len(parent_words), child_step):
                child_words = parent_words[c_start : c_start + self._child_size]
                child_text = " ".join(child_words).strip()
                if not child_text:
                    continue
                child_meta = self._build_metadata(
                    metadata_base,
                    chunk_index=global_child_idx,
                    chunk_strategy="parent_child_child",
                    parent_chunk_id=parent_id,
                )
                all_chunks.append(DocumentChunk(text=child_text, metadata=child_meta))
                global_child_idx += 1

        return all_chunks


# Auto-register with the factory
ChunkerFactory.register("parent_child", ParentChildChunker)
