"""Grounding verification utilities with semantic, negation, and claim validation."""

from __future__ import annotations

import re

from .constants import SAFE_FALLBACK_RESPONSE

STOPWORDS = frozenset([
    "the", "a", "an", "and", "or", "is", "in", "to", "for", "of", "with", "at", "by",
    "on", "from", "that", "this", "it", "as", "are", "be", "was", "were", "have", "has",
    "had", "do", "does", "did", "can", "could", "should", "would", "must", "also",
    "before", "after", "about", "above", "below", "between", "both", "during", "through",
    "candidates", "candidate", "interview", "interviews", "question", "questions",
])

NEGATIONS = frozenset([
    "not", "no", "never", "neither", "nor", "cannot", "cant", "doesnt", "dont",
    "isnt", "arent", "wasnt", "werent", "wont", "hardly", "scarcely", "barely",
])

SYNONYMS: dict[str, str] = {
    "algorithmic": "algorithm",
    "algorithms": "algorithm",
    "architecture": "design",
    "computational": "complexity",
    "reasoning": "problem-solving",
    "proficiency": "evaluation",
    "evaluate": "evaluate",
    "evaluates": "evaluate",
    "testing": "evaluate",
    "tested": "evaluate",
}

FABRICATION_SIGNALS = [
    r"\$\s*\d+",
    r"\b\d+\s*(?:dollars|usd|inr|rupees|fee|registration fee|charge|deposit)\b",
    r"\bmandatory\s+(?:fee|payment|registration|video|recording|document|patent|submission)\b",
    r"\bpay\s+(?:a\s+)?fee\b",
    r"\b\d{1,2}\s+(?:years|yrs)\s+(?:of\s+)?(?:experience|faang)\b",
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b",
    r"\bpatent\s+(?:documentation|filing|submission)\b",
    r"\bvideo\s+recording\b",
    r"\bdeadline\b",
]


def _stem_token(token: str) -> str:
    """Normalize token and apply stem suffix reduction and semantic synonyms."""
    t = token.lower().replace("'", "")
    if t in SYNONYMS:
        return SYNONYMS[t]
    for suffix in ["ing", "tion", "tions", "ities", "ity", "ed", "es", "s", "al", "ic"]:
        if len(t) > len(suffix) + 3 and t.endswith(suffix):
            return t[:-len(suffix)]
    return t


def _extract_tokens(text: str) -> set[str]:
    """Extract lowercase non-stopword normalized content tokens."""
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_\'-]+\b", text.lower())
    clean_tokens = set()
    for t in raw_tokens:
        clean = t.replace("'", "")
        if len(clean) > 2 and clean not in STOPWORDS:
            clean_tokens.add(_stem_token(clean))
    return clean_tokens


def _has_negation(text: str) -> bool:
    """Check whether text contains explicit negation tokens."""
    tokens = set(t.replace("'", "") for t in re.findall(r"\b[a-zA-Z0-9_\'-]+\b", text.lower()))
    return bool(tokens & NEGATIONS)


def _content_similarity(sentence: str, evidence: str) -> float:
    """Compute content-token similarity between sentence and evidence chunk."""
    s_tokens = _extract_tokens(sentence)
    e_tokens = _extract_tokens(evidence)
    if not s_tokens or not e_tokens:
        return 0.0
    return len(s_tokens & e_tokens) / len(s_tokens)


class GroundingVerifier:
    """Verify whether response is factually grounded in retrieved evidence."""

    def verify(self, answer: str, evidence_chunks: list[str], threshold: float = 0.40) -> bool:
        if not evidence_chunks or not answer:
            return False

        merged_evidence = " ".join(evidence_chunks).lower()
        evidence_tokens = _extract_tokens(merged_evidence)

        # 1. Global check for fabricated numbers, fees, dates, and ungrounded constraints
        for pattern in FABRICATION_SIGNALS:
            if re.search(pattern, answer, flags=re.IGNORECASE):
                if not re.search(pattern, merged_evidence, flags=re.IGNORECASE):
                    return False

        sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer) if len(s.strip()) > 15]
        if not sentences:
            return False

        supported_count = 0
        for s in sentences:
            s_tokens = _extract_tokens(s)
            if not s_tokens:
                continue

            # 2. Check for negation contradiction: polarity mismatch between claim and evidence
            s_negated = _has_negation(s)
            e_negated = _has_negation(merged_evidence)
            if s_negated != e_negated:
                overlap = s_tokens & evidence_tokens
                if len(overlap) >= 2:
                    return False

            # 3. Check token support ratio of sentence against retrieved evidence
            overlap_tokens = s_tokens & evidence_tokens
            token_support_ratio = len(overlap_tokens) / len(s_tokens) if s_tokens else 0.0

            if token_support_ratio >= threshold:
                supported_count += 1
            else:
                # If an individual sentence is completely unsupported (< 0.25), reject grounding
                if token_support_ratio < 0.25:
                    return False

        return (supported_count / len(sentences)) >= 0.75

    def fallback(self) -> str:
        return SAFE_FALLBACK_RESPONSE
