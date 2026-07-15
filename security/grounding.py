"""Grounding verification utilities."""

from __future__ import annotations

import re

from .constants import SAFE_FALLBACK_RESPONSE
from .utils import jaccard_similarity


class GroundingVerifier:
    """Verify whether response is sufficiently grounded in retrieved evidence."""

    def verify(self, answer: str, evidence_chunks: list[str], threshold: float) -> bool:
        if not evidence_chunks:
            return False

        answer_sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer) if len(s.strip()) > 20]
        if not answer_sentences:
            return False

        supported = 0
        for sentence in answer_sentences:
            best = max((jaccard_similarity(sentence, chunk) for chunk in evidence_chunks), default=0.0)
            if best >= threshold:
                supported += 1

        return (supported / len(answer_sentences)) >= 0.5

    def fallback(self) -> str:
        return SAFE_FALLBACK_RESPONSE
