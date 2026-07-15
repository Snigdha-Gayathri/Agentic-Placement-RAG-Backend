"""Prompt and jailbreak injection detection using heuristic scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import SecurityConfig
from .constants import PROMPT_INJECTION_PATTERNS


@dataclass(frozen=True)
class InjectionDetectionResult:
    risk_score: float
    confidence: float
    reasons: list[str]
    action: str


class PromptInjectionDetector:
    """Score query risk for direct and indirect prompt injection patterns."""

    def __init__(self, config: SecurityConfig) -> None:
        self._threshold = config.prompt_injection_threshold

    def detect(self, query: str, history: list[str] | None = None) -> InjectionDetectionResult:
        lowered = query.lower()
        reasons: list[str] = []
        score = 0.0

        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                reasons.append(f"pattern:{pattern}")
                score += 0.15

        if "base64" in lowered or "hex" in lowered:
            reasons.append("encoded_payload_indicator")
            score += 0.1

        if "roleplay" in lowered or "pretend" in lowered:
            reasons.append("roleplay_attack_indicator")
            score += 0.12

        if "ignore" in lowered and "instruction" in lowered:
            reasons.append("instruction_hijack_indicator")
            score += 0.18

        if history:
            joined = " ".join(history[-3:]).lower()
            if "ignore previous" in joined and "ignore previous" in lowered:
                reasons.append("multi_turn_escalation")
                score += 0.18

        score = min(score, 1.0)
        confidence = min(1.0, 0.45 + 0.08 * len(reasons))

        if score >= self._threshold:
            action = "block"
        elif score >= max(0.4, self._threshold * 0.6):
            action = "warn"
        else:
            action = "allow"

        return InjectionDetectionResult(
            risk_score=score,
            confidence=confidence,
            reasons=reasons,
            action=action,
        )
