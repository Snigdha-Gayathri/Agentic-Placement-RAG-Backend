"""CLI entrypoint for building the document index.

Usage:
    python -m backend.ingestion.build_index --data-path data/ --strategy recursive --chunk-size 500 --chunk-overlap 50
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from backend.core.chunking.base import ChunkMetadata, DocumentChunk
from backend.core.embeddings.gemini import GeminiEmbedding
from backend.core.vector_store.chroma import ChromaVectorStore
from .pipeline import ChunkingConfig, IngestionPipeline

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Set up structured logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    """Build the document index from source documents.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        prog="build_index",
        description="Build the RAG document index with metadata extraction and chunk enhancement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Strategies:
  recursive  — Word-level sliding window (default)
  page       — Page-level chunks, sub-chunking large pages

Enhancement Strategies (comma-separated):
  metadata_prepend  — Add [Company: X | Topic: Y] header (default)
  title_injection   — Add inferred section titles
  context_expansion — Add neighboring chunk context
  semantic_header   — Add first-sentence summary

Examples:
  python -m backend.ingestion.build_index --data-path data/
  python -m backend.ingestion.build_index --data-path data/ --strategy page --chunk-size 300
  python -m backend.ingestion.build_index --data-path data/ --force-rebuild --enhance metadata_prepend,semantic_header
""",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help="Path to directory containing source documents (default: data)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["recursive", "page"],
        default="recursive",
        help="Chunking strategy (default: recursive)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of words per chunk (default: 500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Word overlap between chunks (default: 50)",
    )
    parser.add_argument(
        "--enhance",
        type=str,
        default="metadata_prepend",
        help="Comma-separated enhancement strategies (default: metadata_prepend)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Re-process all documents even if already indexed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report without producing chunks",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args(argv)

    # Validate inputs
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be greater than 0")
    if args.chunk_overlap < 0:
        parser.error("--chunk-overlap must be >= 0")
    if args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap must be less than --chunk-size")

    data_path = Path(args.data_path)
    if not data_path.is_dir():
        print(f"ERROR: Data directory '{data_path}' not found.", file=sys.stderr)
        return 1

    _configure_logging(args.verbose)

    # Parse enhancement strategies
    enhancement_strategies = [s.strip() for s in args.enhance.split(",") if s.strip()]

    config = ChunkingConfig(
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        enhancement_strategies=enhancement_strategies,
    )

    print("=" * 60)
    print("  Placement RAG Agent — Index Builder")
    print("=" * 60)
    print(f"  Data path:      {data_path}")
    print(f"  Strategy:       {config.strategy}")
    print(f"  Chunk size:     {config.chunk_size} words")
    print(f"  Chunk overlap:  {config.chunk_overlap} words")
    print(f"  Enhancements:   {', '.join(config.enhancement_strategies)}")
    print(f"  Force rebuild:  {args.force_rebuild}")
    print(f"  Dry run:        {args.dry_run}")
    print("=" * 60)
    print()

    try:
        pipeline = IngestionPipeline(
            data_path=data_path,
            config=config,
        )

        chunks, stats = pipeline.run(
            force_rebuild=args.force_rebuild,
            dry_run=args.dry_run,
        )

        # Print summary
        print()
        print("=" * 60)
        print("  Ingestion Summary")
        print("=" * 60)
        print(f"  Documents found:    {stats.documents_found}")
        print(f"  Documents loaded:   {stats.documents_loaded}")
        print(f"  Documents skipped:  {stats.documents_skipped}")
        print(f"  Documents failed:   {stats.documents_failed}")
        print(f"  Total chunks:       {stats.total_chunks}")
        print(f"  Enhanced chunks:    {stats.total_enhanced_chunks}")
        print(f"  Elapsed time:       {stats.elapsed_seconds:.1f}s")
        print("=" * 60)

        if stats.errors:
            print()
            print("  Errors:")
            for error in stats.errors:
                print(f"    [ERROR] {error}")

        if not args.dry_run and chunks:
            print()
            print("  Embedding and storing chunks into ChromaDB...")
            embedder = GeminiEmbedding()
            store = ChromaVectorStore()
            doc_chunks: list[DocumentChunk] = []
            for i, c in enumerate(chunks):
                meta = ChunkMetadata(
                    chunk_id=c.chunk_id or f"ingest_{i}_{hash(c.text)}",
                    source_file=c.source_path,
                    company=c.metadata.get("company", "General"),
                    page_number=c.page_number or 0,
                    chunk_index=c.chunk_index or i,
                    chunk_strategy=config.strategy,
                    tags=[t.strip() for t in c.metadata.get("tags", "").split(",") if t.strip()] if isinstance(c.metadata.get("tags"), str) else [],
                    topic=c.metadata.get("topic", "General"),
                    difficulty=c.metadata.get("difficulty", "Medium"),
                    role=c.metadata.get("role", "SDE"),
                    enhancement_applied=",".join(c.enhancements_applied),
                )
                doc_chunks.append(DocumentChunk(text=c.enhanced_text or c.text, metadata=meta))
            texts = [dc.text for dc in doc_chunks]
            embeddings = asyncio.run(embedder.embed_batch(texts))
            store.add_documents(doc_chunks, embeddings)
            print(f"  [SUCCESS] Stored {len(doc_chunks)} chunks into ChromaDB ({store.collection_name})")

            # Collect company distribution
            companies: dict[str, int] = {}
            for chunk in chunks:
                company = chunk.metadata.get("company", "Unknown")
                companies[company] = companies.get(company, 0) + 1
            print()
            print("  Chunks by Company:")
            for company, count in sorted(companies.items(), key=lambda x: -x[1]):
                print(f"    {company:20s}: {count:5d} chunks")

        print()
        if stats.documents_failed == 0:
            print("  [SUCCESS] Ingestion completed successfully")
        else:
            print(f"  [WARNING] Ingestion completed with {stats.documents_failed} error(s)")

        return 0 if stats.documents_failed == 0 else 1

    except Exception as exc:
        print(f"\nERROR: Ingestion failed: {exc}", file=sys.stderr)
        logger.exception("Ingestion pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
