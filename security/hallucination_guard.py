"""Hallucination guard for unsupported claim and contradiction detection."""

from __future__ import annotations

import re

from .grounding import (
    FABRICATION_SIGNALS,
    _content_similarity,
    _extract_tokens,
    _has_negation,
)


class HallucinationGuard:
    """Detect unsupported claims, fabricated details, and contradictions in model output."""

    _fabrication_patterns = [
        r"https?://[^\s]+",
        r"\b\d{1,3}%\b",
        r"\baccording\s+to\s+internal\s+policy\b",
        r"\bcontact\s+us\s+at\b",
    ] + FABRICATION_SIGNALS

    def score(self, answer: str, evidence_chunks: list[str]) -> float:
        if not answer:
            return 1.0
        if not evidence_chunks:
            return 1.0

        merged_evidence = " ".join(evidence_chunks).lower()
        evidence_tokens = _extract_tokens(merged_evidence)
        unsupported_flags = 0

        for pattern in self._fabrication_patterns:
            if re.search(pattern, answer, flags=re.IGNORECASE):
                if not re.search(pattern, merged_evidence, flags=re.IGNORECASE):
                    unsupported_flags += 2

        sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer) if len(s.strip()) > 15]
        if not sentences:
            return min(1.0, unsupported_flags * 0.25)

        unsupported_sentences = 0
        for s in sentences:
            s_tokens = _extract_tokens(s)
            if not s_tokens:
                continue

            # Detect negation contradiction: polarity mismatch
            if _has_negation(s) != _has_negation(merged_evidence):
                overlap = s_tokens & evidence_tokens
                if len(overlap) >= 2:
                    unsupported_flags += 3

            overlap_tokens = s_tokens & evidence_tokens
            token_support_ratio = len(overlap_tokens) / len(s_tokens) if s_tokens else 0.0

            if token_support_ratio < 0.35:
                unsupported_sentences += 1

        ratio = unsupported_sentences / max(1, len(sentences))
        return min(1.0, ratio + unsupported_flags * 0.25)

    def is_hallucinated(self, answer: str, evidence_chunks: list[str], threshold: float = 0.45) -> bool:
        return self.score(answer, evidence_chunks) >= threshold
