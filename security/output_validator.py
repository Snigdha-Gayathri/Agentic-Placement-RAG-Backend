"""Output validation for prompt leakage and sensitive content."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import SecurityConfig


@dataclass(frozen=True)
class OutputValidationResult:
    is_safe: bool
    reasons: list[str]


class OutputValidator:
    """Detect unsafe output content before response is returned."""

    def __init__(self, config: SecurityConfig) -> None:
        self._patterns = config.blocked_output_patterns

    def validate(self, text: str) -> OutputValidationResult:
        reasons: list[str] = []

        pii_patterns = [
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
        ]

        for pattern in self._patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                reasons.append("blocked_output_pattern")
                break

        for pattern in pii_patterns:
            if re.search(pattern, text):
                reasons.append("pii_detected")
                break

        return OutputValidationResult(is_safe=len(reasons) == 0, reasons=reasons)
