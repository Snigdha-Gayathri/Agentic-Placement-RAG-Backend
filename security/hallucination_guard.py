"""Hallucination guard for unsupported claim detection."""

from __future__ import annotations

import re

from .utils import jaccard_similarity


class HallucinationGuard:
    """Detect unsupported claims and fabricated details in model output."""

    _fabrication_patterns = [
        r"https?://[^\s]+",
        r"\b\d{1,3}%\b",
        r"\baccording\s+to\s+internal\s+policy\b",
        r"\bcontact\s+us\s+at\b",
    ]

    def score(self, answer: str, evidence_chunks: list[str]) -> float:
        if not answer:
            return 1.0

        unsupported_flags = 0
        for pattern in self._fabrication_patterns:
            if re.search(pattern, answer, flags=re.IGNORECASE):
                unsupported_flags += 1

        sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer) if len(s.strip()) > 20]
        if not sentences:
            return min(1.0, unsupported_flags * 0.2)

        unsupported_sentences = 0
        for sentence in sentences:
            best = max((jaccard_similarity(sentence, chunk) for chunk in evidence_chunks), default=0.0)
            if best < 0.18:
                unsupported_sentences += 1

        ratio = unsupported_sentences / max(1, len(sentences))
        return min(1.0, ratio + unsupported_flags * 0.12)

    def is_hallucinated(self, answer: str, evidence_chunks: list[str], threshold: float) -> bool:
        return self.score(answer, evidence_chunks) >= threshold
