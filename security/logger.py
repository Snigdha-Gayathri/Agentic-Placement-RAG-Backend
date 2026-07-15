"""Structured security logging with automatic redaction."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from .utils import redact_secrets


class SecurityLogger:
    """Minimal JSON logger for security decisions and request telemetry."""

    def __init__(self) -> None:
        self._logger = logging.getLogger("rag_security")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def info(self, event: str, **fields: Any) -> None:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
        }
        for key, value in fields.items():
            if isinstance(value, str):
                payload[key] = redact_secrets(value)
            else:
                payload[key] = value
        self._logger.info(json.dumps(payload, ensure_ascii=True))
