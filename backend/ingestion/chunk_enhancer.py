"""Chunk enhancement strategies for improved retrieval quality."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

logger = logging.getLogger(__name__)


class EnhancementStrategy(str, Enum):
    """Available chunk enhancement strategies."""

    METADATA_PREPEND = "metadata_prepend"
    TITLE_INJECTION = "title_injection"
    CONTEXT_EXPANSION = "context_expansion"
    SEMANTIC_HEADER = "semantic_header"


@dataclass
class EnhancedChunk:
    """A text chunk with applied enhancements and metadata."""

    chunk_id: str
    text: str
    enhanced_text: str
    source_path: str
    metadata: dict[str, str] = field(default_factory=dict)
    enhancements_applied: list[str] = field(default_factory=list)
    page_number: int | None = None
    chunk_index: int = 0


class ChunkEnhancer:
    """Apply configurable enhancement strategies to document chunks.

    Strategies can be combined. Each adds contextual information to the chunk
    text to improve retrieval relevance and LLM response quality.

    Args:
        strategies: List of enhancement strategies to apply (in order).
    """

    # Section title patterns to detect in content
    SECTION_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE),  # Markdown headers
        re.compile(r"^([A-Z][A-Za-z\s]{3,50})$", re.MULTILINE),  # Title-case lines
        re.compile(r"^(?:Problem|Question|Exercise)\s*(?:#?\d+)?[:\s](.+)", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^(?:Chapter|Section|Part)\s+\d+[:\s]*(.+)", re.MULTILINE | re.IGNORECASE),
    ]

    def __init__(
        self,
        strategies: list[EnhancementStrategy] | None = None,
    ) -> None:
        self._strategies = strategies or [EnhancementStrategy.METADATA_PREPEND]

    def enhance(
        self,
        chunk: EnhancedChunk,
        *,
        neighboring_chunks: list[str] | None = None,
    ) -> EnhancedChunk:
        """Apply all configured enhancement strategies to a chunk.

        Args:
            chunk: The chunk to enhance.
            neighboring_chunks: Optional list of neighboring chunk texts for
                context expansion (previous, next).

        Returns:
            EnhancedChunk with enhanced_text populated.
        """
        enhanced_text = chunk.text
        applied: list[str] = []

        for strategy in self._strategies:
            try:
                if strategy == EnhancementStrategy.METADATA_PREPEND:
                    enhanced_text = self._metadata_prepend(enhanced_text, chunk.metadata)
                    applied.append(strategy.value)
                elif strategy == EnhancementStrategy.TITLE_INJECTION:
                    enhanced_text = self._title_injection(enhanced_text)
                    applied.append(strategy.value)
                elif strategy == EnhancementStrategy.CONTEXT_EXPANSION:
                    enhanced_text = self._context_expansion(
                        enhanced_text,
                        neighboring_chunks or [],
                    )
                    applied.append(strategy.value)
                elif strategy == EnhancementStrategy.SEMANTIC_HEADER:
                    enhanced_text = self._semantic_header(enhanced_text)
                    applied.append(strategy.value)
            except Exception as exc:
                logger.warning(
                    "Enhancement strategy %s failed for chunk %s: %s",
                    strategy.value,
                    chunk.chunk_id,
                    exc,
                )

        chunk.enhanced_text = enhanced_text
        chunk.enhancements_applied = applied
        return chunk

    def enhance_batch(
        self,
        chunks: list[EnhancedChunk],
    ) -> list[EnhancedChunk]:
        """Enhance a batch of chunks, providing neighboring context.

        Args:
            chunks: Ordered list of chunks from the same document.

        Returns:
            List of enhanced chunks.
        """
        texts = [c.text for c in chunks]
        enhanced: list[EnhancedChunk] = []

        for i, chunk in enumerate(chunks):
            neighbors: list[str] = []
            if i > 0:
                neighbors.append(texts[i - 1])
            else:
                neighbors.append("")
            if i < len(texts) - 1:
                neighbors.append(texts[i + 1])
            else:
                neighbors.append("")

            enhanced.append(self.enhance(chunk, neighboring_chunks=neighbors))

        return enhanced

    @staticmethod
    def _metadata_prepend(text: str, metadata: dict[str, str]) -> str:
        """Prepend structured metadata as a text header.

        Example output:
            [Company: Amazon | Topic: DSA | Source: Amazon_DSA.pdf]
            <chunk text>
        """
        parts: list[str] = []
        if metadata.get("company"):
            parts.append(f"Company: {metadata['company']}")
        if metadata.get("topic"):
            parts.append(f"Topic: {metadata['topic']}")
        if metadata.get("source"):
            parts.append(f"Source: {metadata['source']}")
        if metadata.get("difficulty"):
            parts.append(f"Difficulty: {metadata['difficulty']}")

        if parts:
            header = f"[{' | '.join(parts)}]"
            return f"{header}\n{text}"
        return text

    def _title_injection(self, text: str) -> str:
        """Add an inferred section title from content patterns."""
        for pattern in self.SECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                title = match.group(1).strip()
                if 3 <= len(title) <= 80:
                    # Don't duplicate if already at the start
                    if not text.strip().startswith(title):
                        return f"## {title}\n{text}"
                    return text
        return text

    @staticmethod
    def _context_expansion(
        text: str,
        neighboring_chunks: list[str],
    ) -> str:
        """Add first/last sentences from neighboring chunks for context.

        Args:
            text: The current chunk text.
            neighboring_chunks: [previous_chunk, next_chunk] texts.
        """
        prefix = ""
        suffix = ""

        if len(neighboring_chunks) >= 1 and neighboring_chunks[0]:
            prev_text = neighboring_chunks[0].strip()
            # Get last sentence of previous chunk
            sentences = re.split(r"(?<=[.!?])\s+", prev_text)
            if sentences:
                last_sentence = sentences[-1].strip()
                if len(last_sentence) > 10:
                    prefix = f"[...{last_sentence}] "

        if len(neighboring_chunks) >= 2 and neighboring_chunks[1]:
            next_text = neighboring_chunks[1].strip()
            # Get first sentence of next chunk
            sentences = re.split(r"(?<=[.!?])\s+", next_text)
            if sentences:
                first_sentence = sentences[0].strip()
                if len(first_sentence) > 10:
                    suffix = f" [{first_sentence}...]"

        return f"{prefix}{text}{suffix}"

    @staticmethod
    def _semantic_header(text: str) -> str:
        """Generate a one-line summary header using the first meaningful sentence."""
        # Find the first sentence that's long enough to be meaningful
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for sentence in sentences:
            cleaned = sentence.strip()
            # Skip very short or header-like content
            if len(cleaned) > 20 and not cleaned.startswith("["):
                # Truncate if too long
                header = cleaned[:100].rstrip(".")
                return f"Summary: {header}\n{text}"
        return text
