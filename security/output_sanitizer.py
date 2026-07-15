"""Output sanitization and redaction."""

from __future__ import annotations

import re

from .utils import redact_secrets


class OutputSanitizer:
    """Sanitize model output before returning to clients."""

    def sanitize(self, text: str) -> str:
        sanitized = redact_secrets(text)
        sanitized = re.sub(r"(?i)authorization\s*:\s*bearer\s+[^\s]+", "Authorization: [REDACTED]", sanitized)
        sanitized = re.sub(r"(?i)set-cookie\s*:\s*[^\n]+", "Set-Cookie: [REDACTED]", sanitized)
        sanitized = re.sub(r"(?i)traceback\s+\(most\s+recent\s+call\s+last\):[\s\S]*", "[REDACTED_ERROR_TRACE]", sanitized)
        lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in sanitized.splitlines()]
        sanitized = "\n".join(lines).strip()
        return sanitized
