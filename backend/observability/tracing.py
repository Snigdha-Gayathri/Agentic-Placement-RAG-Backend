"""Structured request tracing with JSON logging for observability backends."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCategory(str, Enum):
    """Categorized error types for structured logging."""

    RETRIEVAL_ERROR = "retrieval_error"
    LLM_ERROR = "llm_error"
    SECURITY_ERROR = "security_error"
    VALIDATION_ERROR = "validation_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    PIPELINE_ERROR = "pipeline_error"
    UNKNOWN_ERROR = "unknown_error"


class WarningLevel(str, Enum):
    """Warning severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TraceEntry:
    """A single trace log entry."""

    timestamp: float
    stage: str
    status: str
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_category: str | None = None
    warning_level: str | None = None


class _JSONFormatter(logging.Formatter):
    """JSON log formatter compatible with Grafana/Loki/ELK."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields from the record
        extra_keys = {"request_id", "correlation_id", "stage", "status",
                      "latency_ms", "error_category", "metric_name",
                      "metric_value", "metadata"}
        for key in extra_keys:
            value = getattr(record, key, None)
            if value is not None:
                log_data[key] = value

        if record.exc_info and record.exc_info[1]:
            log_data["exception"] = str(record.exc_info[1])

        return json.dumps(log_data, default=str, ensure_ascii=False)


def _get_json_logger(name: str = "rag.tracing") -> logging.Logger:
    """Get or create a JSON-formatted logger.

    Returns a logger with JSON formatting for structured log ingestion.
    Only adds the handler if one doesn't already exist.
    """
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JSONFormatter())
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
        log.propagate = False
    return log


class RequestTracer:
    """Structured request tracer with JSON logging.

    Provides correlation ID propagation, stage-level timing, error
    categorization, and performance metrics. Outputs are compatible
    with Grafana/Loki/ELK log aggregation backends.

    Args:
        request_id: The unique request identifier. Auto-generated if not provided.
        correlation_id: Optional correlation ID for distributed tracing.
    """

    def __init__(
        self,
        request_id: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        self._request_id = request_id or str(uuid.uuid4())
        self._correlation_id = correlation_id or self._request_id
        self._entries: list[TraceEntry] = []
        self._start_time = time.perf_counter()
        self._logger = _get_json_logger()

    @property
    def request_id(self) -> str:
        """The unique request ID."""
        return self._request_id

    @property
    def correlation_id(self) -> str:
        """The correlation ID for distributed tracing."""
        return self._correlation_id

    def log_stage(
        self,
        stage: str,
        status: str,
        latency_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a pipeline stage execution.

        Args:
            stage: Name of the pipeline stage.
            status: Stage status (completed, failed, skipped, etc.).
            latency_ms: Stage execution time in milliseconds.
            metadata: Optional key-value metadata.
        """
        entry = TraceEntry(
            timestamp=time.time(),
            stage=stage,
            status=status,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        self._entries.append(entry)

        self._logger.info(
            "stage_%s: %s (%.1fms)",
            status,
            stage,
            latency_ms or 0,
            extra={
                "request_id": self._request_id,
                "correlation_id": self._correlation_id,
                "stage": stage,
                "status": status,
                "latency_ms": latency_ms,
                "metadata": metadata,
            },
        )

    def log_error(
        self,
        stage: str,
        error: str | Exception,
        category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR,
        warning_level: WarningLevel = WarningLevel.HIGH,
    ) -> None:
        """Log a pipeline error with categorization.

        Args:
            stage: The stage where the error occurred.
            error: Error message or exception.
            category: Error category for routing/alerting.
            warning_level: Severity level.
        """
        error_str = str(error)
        entry = TraceEntry(
            timestamp=time.time(),
            stage=stage,
            status="failed",
            error=error_str,
            error_category=category.value,
            warning_level=warning_level.value,
        )
        self._entries.append(entry)

        self._logger.error(
            "stage_error: %s in %s [%s]",
            error_str,
            stage,
            category.value,
            extra={
                "request_id": self._request_id,
                "correlation_id": self._correlation_id,
                "stage": stage,
                "status": "failed",
                "error_category": category.value,
            },
        )

    def log_metric(
        self,
        name: str,
        value: float | int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a named performance metric.

        Args:
            name: Metric name (e.g., 'retrieval_latency_ms', 'chunk_count').
            value: Metric value.
            metadata: Optional context.
        """
        self._logger.info(
            "metric: %s = %s",
            name,
            value,
            extra={
                "request_id": self._request_id,
                "correlation_id": self._correlation_id,
                "metric_name": name,
                "metric_value": value,
                "metadata": metadata,
            },
        )

    def log_warning(
        self,
        stage: str,
        message: str,
        level: WarningLevel = WarningLevel.MEDIUM,
    ) -> None:
        """Log a warning for a pipeline stage.

        Args:
            stage: The pipeline stage.
            message: Warning message.
            level: Warning severity.
        """
        entry = TraceEntry(
            timestamp=time.time(),
            stage=stage,
            status="warning",
            warning_level=level.value,
            metadata={"message": message},
        )
        self._entries.append(entry)

        self._logger.warning(
            "stage_warning: %s in %s [%s]",
            message,
            stage,
            level.value,
            extra={
                "request_id": self._request_id,
                "correlation_id": self._correlation_id,
                "stage": stage,
                "status": "warning",
            },
        )

    def get_entries(self) -> list[TraceEntry]:
        """Get all trace entries.

        Returns:
            List of TraceEntry objects in chronological order.
        """
        return list(self._entries)

    def get_total_elapsed_ms(self) -> float:
        """Get total elapsed time since tracer creation."""
        return round((time.perf_counter() - self._start_time) * 1000, 2)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full trace for JSON output.

        Returns:
            Dictionary with request_id, correlation_id, entries, and timing.
        """
        return {
            "request_id": self._request_id,
            "correlation_id": self._correlation_id,
            "total_elapsed_ms": self.get_total_elapsed_ms(),
            "entries": [
                {
                    "timestamp": entry.timestamp,
                    "stage": entry.stage,
                    "status": entry.status,
                    "latency_ms": entry.latency_ms,
                    "metadata": entry.metadata if entry.metadata else None,
                    "error": entry.error,
                    "error_category": entry.error_category,
                    "warning_level": entry.warning_level,
                }
                for entry in self._entries
            ],
        }
