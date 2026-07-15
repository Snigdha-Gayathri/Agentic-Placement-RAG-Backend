"""Google Gemini LLM implementation with retry logic and token tracking."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from backend.config.settings import get_config
from backend.core.llm.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 1.0
_TIMEOUT_S = 60.0


class GeminiLLM(BaseLLM):
    """Gemini ``generateContent`` implementation of :class:`BaseLLM`.

    Uses ``httpx`` for async HTTP calls with exponential-backoff retry.
    Token usage is extracted from the ``usageMetadata`` response field.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        cfg = get_config()
        self._api_key = api_key or cfg.gemini_api_key 
        self._model = model or cfg.llm_model
        if not self._api_key:
            raise RuntimeError(
                "Gemini API key is not configured. "
                "Set GEMINI_API_KEY in your environment."
            )

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Call the Gemini ``generateContent`` endpoint.

        Retries up to 3 times with exponential backoff on transient errors.
        """
        endpoint = (
            f"{_GEMINI_BASE}/{self._model}:generateContent"
            f"?key={self._api_key}"
        )

        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }

        last_error: Exception | None = None
        backoff = _INITIAL_BACKOFF_S

        for attempt in range(1, _MAX_RETRIES + 1):
            t0 = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
                    response = await client.post(endpoint, json=payload)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                if response.status_code == 429 or response.status_code >= 500:
                    # Transient — retry
                    logger.warning(
                        "Gemini %s (attempt %d/%d), retrying in %.1fs",
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
                data: dict[str, Any] = response.json()
                return self._parse_response(data, latency_ms)

            except httpx.TimeoutException as exc:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                logger.warning(
                    "Gemini timeout (attempt %d/%d): %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
                last_error = exc
                await asyncio.sleep(backoff)
                backoff *= 2

            except httpx.HTTPStatusError as exc:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                if exc.response.status_code in (429, 500, 502, 503):
                    logger.warning(
                        "Gemini transient error (attempt %d/%d): %s",
                        attempt,
                        _MAX_RETRIES,
                        exc,
                    )
                    last_error = exc
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    raise

        raise RuntimeError(
            f"Gemini generate failed after {_MAX_RETRIES} attempts"
        ) from last_error

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_response(self, data: dict[str, Any], latency_ms: float) -> LLMResponse:
        """Extract text and usage metadata from the raw API response."""
        candidates = data.get("candidates", [])
        if not candidates:
            return LLMResponse(
                text="I encountered an issue processing your request.",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                model=self._model,
            )

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "\n".join(
            part.get("text", "") for part in parts if part.get("text")
        )

        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)

        return LLMResponse(
            text=text or "I encountered an issue processing your request.",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model=self._model,
        )
