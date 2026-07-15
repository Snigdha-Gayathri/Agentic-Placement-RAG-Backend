"""Abstract base class for text embedding providers."""
from __future__ import annotations

import abc


class BaseEmbedding(abc.ABC):
    """Abstract interface for text-embedding providers.

    Subclasses must implement :meth:`embed_text` and :meth:`embed_batch`.
    """

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors produced."""
        ...

    @abc.abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: The input text.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...

    @abc.abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings.

        Args:
            texts: A list of input texts.

        Returns:
            A list of embedding vectors, one per input text.
        """
        ...
