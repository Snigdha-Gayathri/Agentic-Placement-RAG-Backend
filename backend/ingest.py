#!/usr/bin/env python
"""CLI for ingesting documents into the vector store.

Usage:
    python backend/ingest.py [--documents-path DOCUMENTS_PATH] [--vector-db-path VECTOR_DB_PATH]

Example:
    python backend/ingest.py --documents-path documents/ --vector-db-path data/vector_index.json
"""

import argparse
import sys
from pathlib import Path

try:
    from .app.vector_store import VectorStore
except ImportError:  # pragma: no cover - supports `python backend/ingest.py`
    from app.vector_store import VectorStore


def main(argv: list[str] | None = None) -> int:
    """Ingest documents and rebuild the vector index."""
    parser = argparse.ArgumentParser(
        description="Ingest documents and rebuild the vector store index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported formats:
  - .txt (plain text)
  - .md (markdown)
  - .pdf (requires pypdf)
  - .docx (requires python-docx)

Examples:
  python backend/ingest.py --documents-path documents/ --vector-db-path data/vector_index.json
  python -m backend.ingest --documents-path documents/ --vector-db-path data/vector_index.json
        """,
    )
    parser.add_argument(
        "--documents-path",
        type=str,
        default="documents",
        help="Path to directory containing documents to ingest. Default: documents",
    )
    parser.add_argument(
        "--vector-db-path",
        type=str,
        default="data/vector_index.json",
        help="Path where vector index will be written. Default: data/vector_index.json",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of words per chunk. Default: 500",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Number of overlapping words between chunks. Default: 50",
    )

    args = parser.parse_args(argv)
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be greater than 0")
    if args.chunk_overlap < 0:
        parser.error("--chunk-overlap must be greater than or equal to 0")
    if args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap must be smaller than --chunk-size")

    docs_path = Path(args.documents_path)
    if not docs_path.is_dir():
        print(f"ERROR: Documents directory '{docs_path}' not found.", file=sys.stderr)
        return 1

    vector_store = VectorStore(
        index_path=args.vector_db_path,
        documents_path=args.documents_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"Reading documents from: {args.documents_path}")
    print(f"Chunk size: {args.chunk_size} words")
    print(f"Chunk overlap: {args.chunk_overlap} words")
    print()

    try:
        chunks = vector_store.build_chunks_from_documents()
        if not chunks:
            print("WARNING: No documents found to ingest.")
            return 0

        print(f"Built {len(chunks)} chunks from documents.")
        print("Rebuilding vector index...")
        vector_store.rebuild(chunks)

        print(f"[OK] Vector index written to: {args.vector_db_path}")
        print(f"[OK] Ready for RAG retrieval with {len(chunks)} indexed chunks.")
        return 0
    except Exception as exc:
        print(f"ERROR: Ingestion failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
