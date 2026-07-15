"""Ingestion pipeline: scan → load → extract metadata → chunk → enhance → store."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .chunk_enhancer import ChunkEnhancer, EnhancedChunk, EnhancementStrategy
from .document_loader import DocumentLoader, RawDocument
from .metadata_extractor import DocumentMetadata, MetadataExtractor

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    strategy: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50
    enhancement_strategies: list[str] = field(
        default_factory=lambda: ["metadata_prepend"],
    )


@dataclass
class IngestionStats:
    """Summary statistics for an ingestion run."""

    documents_found: int = 0
    documents_loaded: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    total_chunks: int = 0
    total_enhanced_chunks: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "documents_found": self.documents_found,
            "documents_loaded": self.documents_loaded,
            "documents_skipped": self.documents_skipped,
            "documents_failed": self.documents_failed,
            "total_chunks": self.total_chunks,
            "total_enhanced_chunks": self.total_enhanced_chunks,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "errors": self.errors,
        }


class IngestionPipeline:
    """Orchestrate the full ingestion pipeline from raw files to indexed chunks.

    Supports incremental indexing by tracking file hashes. Documents already
    indexed (by SHA-256 hash) are skipped unless ``force_rebuild`` is True.

    Args:
        data_path: Directory containing source documents.
        config: Chunking and enhancement configuration.
        index_state_path: Path to the hash state file for incremental indexing.
    """

    def __init__(
        self,
        data_path: str | Path,
        config: ChunkingConfig | None = None,
        index_state_path: str | Path | None = None,
    ) -> None:
        self._data_path = Path(data_path)
        self._config = config or ChunkingConfig()
        self._state_path = Path(index_state_path) if index_state_path else (
            self._data_path / ".ingestion_state.json"
        )

        self._loader = DocumentLoader()
        self._extractor = MetadataExtractor()

        # Parse enhancement strategies
        strategies = [
            EnhancementStrategy(s) for s in self._config.enhancement_strategies
            if s in {e.value for e in EnhancementStrategy}
        ]
        self._enhancer = ChunkEnhancer(strategies=strategies or [EnhancementStrategy.METADATA_PREPEND])

        self._indexed_hashes: dict[str, str] = {}  # filename -> hash
        self._load_state()

    def run(
        self,
        *,
        force_rebuild: bool = False,
        dry_run: bool = False,
    ) -> tuple[list[EnhancedChunk], IngestionStats]:
        """Execute the full ingestion pipeline.

        Args:
            force_rebuild: If True, re-process all documents regardless of hash.
            dry_run: If True, scan and report but don't produce chunks.

        Returns:
            Tuple of (list of enhanced chunks, ingestion statistics).
        """
        start_time = time.perf_counter()
        stats = IngestionStats()

        # Stage 1: Scan for files
        files = self._loader.scan_directory(self._data_path)
        stats.documents_found = len(files)
        logger.info(
            "Ingestion started: found %d files in %s",
            stats.documents_found,
            self._data_path,
        )

        if dry_run:
            stats.elapsed_seconds = time.perf_counter() - start_time
            logger.info("Dry run complete: %d files found", stats.documents_found)
            return [], stats

        all_chunks: list[EnhancedChunk] = []

        for file_path in files:
            try:
                # Stage 2: Load document
                doc = self._loader.load_document(file_path)

                # Stage 3: Check for incremental skip
                if not force_rebuild and self._is_already_indexed(doc):
                    stats.documents_skipped += 1
                    logger.debug("Skipping (already indexed): %s", doc.filename)
                    continue

                stats.documents_loaded += 1
                logger.info(
                    "Processing [%d/%d]: %s (%d chars)",
                    stats.documents_loaded,
                    stats.documents_found,
                    doc.filename,
                    len(doc.text),
                )

                # Stage 4: Extract metadata
                page_count = len(doc.page_texts) if doc.page_texts else 1
                metadata = self._extractor.extract(
                    filename=doc.filename,
                    text=doc.text,
                    page_count=page_count,
                )

                # Stage 5: Chunk the document
                raw_chunks = self._chunk_document(doc, metadata)
                stats.total_chunks += len(raw_chunks)

                # Stage 6: Enhance chunks
                enhanced = self._enhancer.enhance_batch(raw_chunks)
                stats.total_enhanced_chunks += len(enhanced)
                all_chunks.extend(enhanced)

                # Update index state
                self._indexed_hashes[doc.filename] = doc.file_hash

            except Exception as exc:
                stats.documents_failed += 1
                error_msg = f"{file_path.name}: {exc}"
                stats.errors.append(error_msg)
                logger.error("Failed to process %s: %s", file_path.name, exc)
                continue

        # Save state for incremental indexing
        self._save_state()

        stats.elapsed_seconds = time.perf_counter() - start_time
        logger.info(
            "Ingestion complete: %d docs → %d chunks in %.1fs "
            "(skipped: %d, failed: %d)",
            stats.documents_loaded,
            stats.total_enhanced_chunks,
            stats.elapsed_seconds,
            stats.documents_skipped,
            stats.documents_failed,
        )

        return all_chunks, stats

    def _chunk_document(
        self,
        doc: RawDocument,
        metadata: DocumentMetadata,
    ) -> list[EnhancedChunk]:
        """Split a document into chunks using the configured strategy.

        Args:
            doc: The loaded document.
            metadata: Extracted metadata for the document.

        Returns:
            List of EnhancedChunk objects (with text set, enhanced_text empty).
        """
        if self._config.strategy == "page":
            return self._chunk_by_page(doc, metadata)
        return self._chunk_recursive(doc, metadata)

    def _chunk_recursive(
        self,
        doc: RawDocument,
        metadata: DocumentMetadata,
    ) -> list[EnhancedChunk]:
        """Chunk text using recursive word-level splitting with overlap."""
        normalized = re.sub(r"\s+", " ", doc.text).strip()
        if not normalized:
            return []

        words = normalized.split(" ")
        chunk_size = self._config.chunk_size
        overlap = self._config.chunk_overlap
        step = max(1, chunk_size - overlap)

        chunks: list[EnhancedChunk] = []
        meta_dict = self._build_meta_dict(metadata, doc)

        for idx, start in enumerate(range(0, len(words), step)):
            chunk_text = " ".join(words[start: start + chunk_size]).strip()
            if not chunk_text:
                continue

            chunk_id = hashlib.sha256(
                f"{doc.file_path}::{idx}::{chunk_text[:100]}".encode("utf-8"),
            ).hexdigest()

            chunks.append(
                EnhancedChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    enhanced_text="",
                    source_path=doc.file_path,
                    metadata=dict(meta_dict),
                    chunk_index=idx,
                ),
            )

        return chunks

    def _chunk_by_page(
        self,
        doc: RawDocument,
        metadata: DocumentMetadata,
    ) -> list[EnhancedChunk]:
        """Chunk text page-by-page, sub-chunking large pages."""
        chunks: list[EnhancedChunk] = []
        meta_dict = self._build_meta_dict(metadata, doc)
        idx = 0

        for page_num, page_text in doc.page_texts:
            page_normalized = re.sub(r"\s+", " ", page_text).strip()
            if not page_normalized:
                continue

            words = page_normalized.split(" ")
            chunk_size = self._config.chunk_size

            if len(words) <= chunk_size:
                chunk_id = hashlib.sha256(
                    f"{doc.file_path}::page{page_num}::{idx}".encode("utf-8"),
                ).hexdigest()
                chunks.append(
                    EnhancedChunk(
                        chunk_id=chunk_id,
                        text=page_normalized,
                        enhanced_text="",
                        source_path=doc.file_path,
                        metadata=dict(meta_dict),
                        page_number=page_num,
                        chunk_index=idx,
                    ),
                )
                idx += 1
            else:
                step = max(1, chunk_size - self._config.chunk_overlap)
                for start in range(0, len(words), step):
                    chunk_text = " ".join(words[start: start + chunk_size]).strip()
                    if not chunk_text:
                        continue
                    chunk_id = hashlib.sha256(
                        f"{doc.file_path}::page{page_num}::{idx}".encode("utf-8"),
                    ).hexdigest()
                    chunks.append(
                        EnhancedChunk(
                            chunk_id=chunk_id,
                            text=chunk_text,
                            enhanced_text="",
                            source_path=doc.file_path,
                            metadata=dict(meta_dict),
                            page_number=page_num,
                            chunk_index=idx,
                        ),
                    )
                    idx += 1

        return chunks

    @staticmethod
    def _build_meta_dict(
        metadata: DocumentMetadata,
        doc: RawDocument,
    ) -> dict[str, str]:
        """Build a metadata dict for chunk storage."""
        return {
            "company": metadata.company,
            "topic": metadata.topic,
            "difficulty": metadata.difficulty,
            "source_type": metadata.source_type,
            "source": doc.filename,
            "file_hash": doc.file_hash,
            "tags": ",".join(metadata.tags),
        }

    def _is_already_indexed(self, doc: RawDocument) -> bool:
        """Check if a document has already been indexed by its hash."""
        stored_hash = self._indexed_hashes.get(doc.filename)
        return stored_hash == doc.file_hash

    def _load_state(self) -> None:
        """Load incremental indexing state from disk."""
        if self._state_path.exists():
            try:
                data = json.loads(self._state_path.read_text(encoding="utf-8"))
                self._indexed_hashes = data.get("hashes", {})
                logger.debug(
                    "Loaded ingestion state: %d indexed files",
                    len(self._indexed_hashes),
                )
            except Exception as exc:
                logger.warning("Failed to load ingestion state: %s", exc)
                self._indexed_hashes = {}

    def _save_state(self) -> None:
        """Persist incremental indexing state to disk."""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"hashes": self._indexed_hashes}
            self._state_path.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
            logger.debug("Saved ingestion state for %d files", len(self._indexed_hashes))
        except Exception as exc:
            logger.warning("Failed to save ingestion state: %s", exc)
