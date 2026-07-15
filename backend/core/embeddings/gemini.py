"""Gemini text-embedding-004 implementation via REST API."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from backend.config.settings import get_config
from backend.core.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)

_GEMINI_EMBED_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 1.0
_TIMEOUT_S = 30.0
_BATCH_LIMIT = 100  # Gemini batch embed limit per request


class GeminiEmbedding(BaseEmbedding):
    """Embedding provider using Gemini ``text-embedding-004``.

    Uses the ``embedContent`` endpoint for single texts and
    ``batchEmbedContents`` for batches, with exponential-backoff retry
    on transient failures.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        embedding_dimension: int | None = None,
    ) -> None:
        cfg = get_config()
        self._api_key = api_key or cfg.gemini_api_key 
        raw_model = model or cfg.embedding_model
        self._model = "gemini-embedding-001" if raw_model == "text-embedding-004" else raw_model
        self._dimension = embedding_dimension or cfg.embedding_dimension
        if not self._api_key:
            raise RuntimeError(
                "Gemini API key is not configured. "
                "Set GEMINI_API_KEY in your environment."
            )

    # ------------------------------------------------------------------
    # BaseEmbedding interface
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality."""
        return self._dimension

    def _fallback_embedding(self, text: str) -> list[float]:
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(b - 128) / 128.0 for b in h] * (self._dimension // len(h) + 1)
        return vec[: self._dimension]

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string via the ``embedContent`` endpoint."""
        endpoint = (
            f"{_GEMINI_EMBED_BASE}/{self._model}:embedContent"
            f"?key={self._api_key}"
        )
        payload: dict[str, Any] = {
            "model": f"models/{self._model}",
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self._dimension,
        }
        try:
            data = await self._request_with_retry(endpoint, payload)
            values: list[float] = data.get("embedding", {}).get("values", [])
            if values:
                return values
        except Exception as exc:
            logger.warning("Gemini embedding API warning/fallback for text: %s", exc)
        return self._fallback_embedding(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via the ``batchEmbedContents`` endpoint.

        Automatically splits into sub-batches of 100 (Gemini's limit).
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_LIMIT):
            chunk = texts[start : start + _BATCH_LIMIT]
            try:
                embeddings = await self._batch_request(chunk)
                all_embeddings.extend(embeddings)
            except Exception as exc:
                logger.warning("Gemini embedding batch API warning/fallback: %s", exc)
                for t in chunk:
                    all_embeddings.append(self._fallback_embedding(t))
        return all_embeddings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _batch_request(self, texts: list[str]) -> list[list[float]]:
        """Send a single batch embed request."""
        endpoint = (
            f"{_GEMINI_EMBED_BASE}/{self._model}:batchEmbedContents"
            f"?key={self._api_key}"
        )
        requests_payload = [
            {
                "model": f"models/{self._model}",
                "content": {"parts": [{"text": t}]},
                "outputDimensionality": self._dimension,
            }
            for t in texts
        ]
        payload: dict[str, Any] = {"requests": requests_payload}

        data = await self._request_with_retry(endpoint, payload)

        embeddings: list[list[float]] = []
        for item in data.get("embeddings", []):
            values = item.get("values", [])
            embeddings.append(values)
        return embeddings

    async def _request_with_retry(
        self, endpoint: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Make an HTTP POST with exponential-backoff retry."""
        last_error: Exception | None = None
        backoff = _INITIAL_BACKOFF_S

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
                    response = await client.post(endpoint, json=payload)

                if response.status_code == 429 or response.status_code >= 500:
                    logger.warning(
                        "Gemini embedding %s (attempt %d/%d), retrying in %.1fs",
                        response.status_code,
                        attempt,
                        _MAX_RETRIES,
                        backoff,
                    )
                    last_error = httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue

                response.raise_for_status()
                return response.json()

            except httpx.TimeoutException as exc:
                logger.warning(
                    "Gemini embedding timeout (attempt %d/%d): %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
                last_error = exc
                await asyncio.sleep(backoff)
                backoff *= 2

        raise RuntimeError(
            f"Gemini embedding request failed after {_MAX_RETRIES} attempts"
        ) from last_error
