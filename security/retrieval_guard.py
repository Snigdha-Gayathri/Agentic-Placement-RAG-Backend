"""Retrieved-chunk validation for poisoning and relevance controls."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .config import SecurityConfig
from .utils import jaccard_similarity, normalize_text


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    text: str
    similarity: float


@dataclass(frozen=True)
class RetrievalGuardResult:
    safe_chunks: list[RetrievedChunk]
    discarded_count: int
    reasons: list[str]


class RetrievalGuard:
    """Filter poisoned or irrelevant chunks before context construction."""

    def __init__(self, config: SecurityConfig) -> None:
        self._config = config

    def filter_chunks(self, query: str, chunks: list[Any]) -> RetrievalGuardResult:
        safe: list[Any] = []
        seen: set[str] = set()
        reasons: list[str] = []
        discarded = 0

        def _get_sort_score(c: Any) -> float:
            ce = getattr(c, "cross_encoder_score", None)
            if ce is not None:
                return float(ce)
            return float(getattr(c, "similarity", getattr(c, "score", getattr(c, "original_score", getattr(c, "similarity_score", 0.0)))))

        for chunk in sorted(chunks, key=_get_sort_score, reverse=True):
            if len(safe) >= self._config.max_retrieved_chunks:
                break

            text = normalize_text(getattr(chunk, "text", str(chunk)))

            if len(text) > self._config.max_context_length:
                discarded += 1
                reasons.append("chunk_too_long")
                continue

            blocked = False
            for pattern in self._config.blocked_retrieval_patterns:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    blocked = True
                    break
            if blocked:
                discarded += 1
                reasons.append("retrieval_block_pattern")
                continue

            orig_s = getattr(chunk, "score", getattr(chunk, "original_score", getattr(chunk, "similarity", 0.0)))
            score = max(float(orig_s), jaccard_similarity(query, text))
            if score < self._config.similarity_threshold:
                discarded += 1
                reasons.append("low_similarity")
                continue

            fingerprint = re.sub(r"\W+", "", text.lower())[:400]
            if fingerprint in seen:
                discarded += 1
                reasons.append("duplicate_chunk")
                continue

            seen.add(fingerprint)
            safe.append(chunk)

        return RetrievalGuardResult(safe_chunks=safe, discarded_count=discarded, reasons=reasons)
