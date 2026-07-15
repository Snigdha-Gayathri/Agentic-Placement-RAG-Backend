"""Hypothetical Document Generator for HyDE retrieval."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class HyDEResult:
    """Output of HyDE generation."""

    hypothetical_document: str
    embedding: list[float]
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothetical_document": self.hypothetical_document,
            "embedding_dimension": len(self.embedding),
            "latency_ms": self.latency_ms,
        }


class HyDEGenerator:
    """Generates a hypothetical ideal answer document and embeds it for dense retrieval."""

    def __init__(self, llm: Any, embedder: Any) -> None:
        self.llm = llm
        self.embedder = embedder

    async def generate_and_embed(self, query: str) -> HyDEResult:
        start_t = time.perf_counter()
        system_p = (
            "You are a technical interview expert. Write a concise, factual technical passage "
            "that accurately answers the prompt. Do not include introductory filler."
        )
        resp = await self.llm.generate(system_prompt=system_p, user_prompt=query, max_tokens=300)
        hypo_doc = resp.text.strip()
        emb = await self.embedder.embed_text(hypo_doc)
        latency = round((time.perf_counter() - start_t) * 1000, 2)
        return HyDEResult(
            hypothetical_document=hypo_doc,
            embedding=emb,
            latency_ms=latency,
        )
