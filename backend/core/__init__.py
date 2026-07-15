"""Core modules for the Agentic RAG pipeline.

Sub-packages:
    llm         — Language model abstractions and Gemini implementation
    embeddings  — Text embedding abstractions and Gemini implementation
    chunking    — Document chunking strategies
    vector_store — Vector database abstractions and ChromaDB implementation
    retrieval   — Dense, BM25, and hybrid retrievers
    reranking   — Cross-encoder reranking
    agent       — Agentic planning and tool execution
    memory      — Conversation memory and query rewriting
    hyde        — Hypothetical Document Embedding
"""
from __future__ import annotations
