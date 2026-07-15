"""Retrieval strategies: Dense, BM25, Hybrid RRF, and Query Router."""

from __future__ import annotations

from .base import BaseRetriever, RetrievalResult
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .hybrid import HybridRetriever
from .router import QueryRouter, RoutingDecision

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "QueryRouter",
    "RoutingDecision",
]
