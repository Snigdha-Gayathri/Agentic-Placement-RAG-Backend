"""Input validation layer for user queries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import SecurityConfig
from .utils import looks_like_base64, looks_like_hex_blob, normalize_text, repeated_token_ratio


@dataclass(frozen=True)
class InputValidationResult:
    is_valid: bool
    normalized_query: str
    reasons: list[str]


class InputValidator:
    """Validate and normalize user input before retrieval."""

    def __init__(self, config: SecurityConfig) -> None:
        self._config = config

    def validate(self, query: str) -> InputValidationResult:
        reasons: list[str] = []
        normalized = normalize_text(query)

        if not normalized:
            reasons.append("empty_prompt")

        if len(normalized) > self._config.max_query_length:
            reasons.append("prompt_too_long")

        if re.search(r"\s{8,}", query):
            reasons.append("excessive_whitespace")

        if "\x00" in query:
            reasons.append("null_byte_detected")

        if looks_like_base64(normalized):
            reasons.append("base64_payload_detected")

        if looks_like_hex_blob(normalized):
            reasons.append("hex_payload_detected")

        if repeated_token_ratio(normalized) > 0.45:
            reasons.append("repeated_token_flooding")

        lower = normalized.lower()
        html_js_xml = [r"<script", r"<html", r"<xml", r"javascript:"]
        for marker in html_js_xml:
            if marker in lower:
                reasons.append("embedded_markup_or_script")
                break

        sql_shell_patterns = [
            r"\bunion\s+select\b",
            r"\bdrop\s+table\b",
            r"\b(or|and)\s+1=1\b",
            r"(;|\|\||&&)\s*(bash|sh|cmd|powershell|rm|curl|wget)\b",
        ]
        for pattern in sql_shell_patterns:
            if re.search(pattern, lower):
                reasons.append("command_or_injection_pattern")
                break

        for pattern in self._config.blocked_input_patterns:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                reasons.append("blocked_pattern_match")
                break

        return InputValidationResult(
            is_valid=len(reasons) == 0,
            normalized_query=normalized,
            reasons=reasons,
        )
