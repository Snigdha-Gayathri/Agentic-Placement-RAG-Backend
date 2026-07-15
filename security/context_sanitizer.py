"""Context sanitization prior to model invocation."""

from __future__ import annotations

import re

from .config import SecurityConfig
from .utils import normalize_text, remove_control_characters


class ContextSanitizer:
    """Sanitize retrieved context to prevent context manipulation and overflow."""

    def __init__(self, config: SecurityConfig) -> None:
        self._max_context_length = config.max_context_length

    def sanitize(self, context: str) -> str:
        cleaned = context
        cleaned = re.sub(r"<script[\s\S]*?>[\s\S]*?<\/script>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<style[\s\S]*?>[\s\S]*?<\/style>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = re.sub(r"\{\{[\s\S]*?\}\}", " ", cleaned)
        cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
        cleaned = re.sub(r"<\?xml[\s\S]*?\?>", " ", cleaned, flags=re.IGNORECASE)

        cleaned = remove_control_characters(cleaned)
        cleaned = normalize_text(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)

        if len(cleaned) > self._max_context_length:
            cleaned = cleaned[: self._max_context_length]

        return cleaned.strip()
