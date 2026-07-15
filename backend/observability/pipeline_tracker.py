"""Pipeline stage tracker for request-level observability."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StageStatus(str, Enum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageName(str, Enum):
    """Well-known pipeline stage names."""

    QUERY_RECEIVED = "query_received"
    INPUT_VALIDATION = "input_validation"
    INJECTION_DETECTION = "injection_detection"
    RATE_LIMITING = "rate_limiting"
    QUERY_ANALYSIS = "query_analysis"
    MEMORY_LOOKUP = "memory_lookup"
    QUERY_REWRITING = "query_rewriting"
    METADATA_FILTERING = "metadata_filtering"
    HYDE_GENERATION = "hyde_generation"
    EMBEDDING_GENERATION = "embedding_generation"
    DENSE_RETRIEVAL = "dense_retrieval"
    BM25_RETRIEVAL = "bm25_retrieval"
    HYBRID_FUSION = "hybrid_fusion"
    CROSS_ENCODER_RERANKING = "cross_encoder_reranking"
    RETRIEVAL_GUARD = "retrieval_guard"
    CONTEXT_CONSTRUCTION = "context_construction"
    CONTEXT_SANITIZATION = "context_sanitization"
    AGENT_EVALUATION = "agent_evaluation"
    LLM_GENERATION = "llm_generation"
    OUTPUT_VALIDATION = "output_validation"
    OUTPUT_SANITIZATION = "output_sanitization"
    GROUNDING_CHECK = "grounding_check"
    HALLUCINATION_CHECK = "hallucination_check"
    METRICS_COMPUTATION = "metrics_computation"
    DASHBOARD_COLLECTION = "dashboard_collection"
    RESPONSE_COMPLETE = "response_complete"


@dataclass
class PipelineStage:
    """A single pipeline stage with timing and status information."""

    name: str
    status: str = StageStatus.PENDING.value
    start_time: float | None = None
    end_time: float | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON/SSE output."""
        result: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "latency_ms": self.latency_ms,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        if self.error:
            result["error"] = self.error
        return result


class PipelineTracker:
    """Track pipeline stages for a single request with timing and metadata.

    Thread-safe tracking of stage progression for SSE streaming and
    dashboard data collection.

    Args:
        request_id: Unique request identifier.
    """

    def __init__(self, request_id: str) -> None:
        self._request_id = request_id
        self._stages: dict[str, PipelineStage] = {}
        self._stage_order: list[str] = []
        self._start_time = time.perf_counter()

    @property
    def request_id(self) -> str:
        """The unique request ID being tracked."""
        return self._request_id

    def start_stage(self, stage_name: str, metadata: dict[str, Any] | None = None) -> None:
        """Mark a pipeline stage as running.

        Args:
            stage_name: Name of the stage (use StageName enum values).
            metadata: Optional metadata to attach to the stage.
        """
        stage = PipelineStage(
            name=stage_name,
            status=StageStatus.RUNNING.value,
            start_time=time.perf_counter(),
            metadata=metadata or {},
        )
        self._stages[stage_name] = stage
        if stage_name not in self._stage_order:
            self._stage_order.append(stage_name)

    def complete_stage(
        self,
        stage_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark a pipeline stage as completed.

        Args:
            stage_name: Name of the stage to complete.
            metadata: Optional additional metadata to merge.
        """
        stage = self._stages.get(stage_name)
        if stage is None:
            # Stage wasn't started; create a completed entry
            stage = PipelineStage(
                name=stage_name,
                start_time=time.perf_counter(),
            )
            self._stages[stage_name] = stage
            if stage_name not in self._stage_order:
                self._stage_order.append(stage_name)

        stage.status = StageStatus.COMPLETED.value
        stage.end_time = time.perf_counter()
        if stage.start_time is not None:
            stage.latency_ms = round((stage.end_time - stage.start_time) * 1000, 2)
        if metadata:
            stage.metadata.update(metadata)

    def fail_stage(self, stage_name: str, error: str) -> None:
        """Mark a pipeline stage as failed.

        Args:
            stage_name: Name of the stage that failed.
            error: Error description.
        """
        stage = self._stages.get(stage_name)
        if stage is None:
            stage = PipelineStage(
                name=stage_name,
                start_time=time.perf_counter(),
            )
            self._stages[stage_name] = stage
            if stage_name not in self._stage_order:
                self._stage_order.append(stage_name)

        stage.status = StageStatus.FAILED.value
        stage.end_time = time.perf_counter()
        stage.error = error
        if stage.start_time is not None:
            stage.latency_ms = round((stage.end_time - stage.start_time) * 1000, 2)

    def skip_stage(self, stage_name: str, reason: str = "") -> None:
        """Mark a pipeline stage as skipped.

        Args:
            stage_name: Name of the stage to skip.
            reason: Reason the stage was skipped.
        """
        stage = PipelineStage(
            name=stage_name,
            status=StageStatus.SKIPPED.value,
            metadata={"skip_reason": reason} if reason else {},
        )
        self._stages[stage_name] = stage
        if stage_name not in self._stage_order:
            self._stage_order.append(stage_name)

    def get_stage(self, stage_name: str) -> PipelineStage | None:
        """Get a specific stage by name."""
        return self._stages.get(stage_name)

    def get_stages(self) -> list[PipelineStage]:
        """Get all stages in execution order.

        Returns:
            Ordered list of PipelineStage objects.
        """
        return [self._stages[name] for name in self._stage_order if name in self._stages]

    def get_total_latency_ms(self) -> float:
        """Get total elapsed time since tracker creation."""
        return round((time.perf_counter() - self._start_time) * 1000, 2)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full tracker state for JSON output.

        Returns:
            Dictionary with request_id, total_latency_ms, and ordered stages.
        """
        return {
            "request_id": self._request_id,
            "total_latency_ms": self.get_total_latency_ms(),
            "stages": [stage.to_dict() for stage in self.get_stages()],
        }
