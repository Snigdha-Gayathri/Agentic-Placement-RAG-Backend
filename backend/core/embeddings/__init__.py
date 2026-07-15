"""Text embedding abstractions and implementations."""
from __future__ import annotations

from backend.core.embeddings.base import BaseEmbedding
from backend.core.embeddings.gemini import GeminiEmbedding

__all__ = ["BaseEmbedding", "GeminiEmbedding"]
