"""Abstract base class for decision backends in Agentic Placement RAG."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import (
    DecisionLogRecord,
    RetrievalStrategyDecision,
    SecurityDecision,
    SufficiencyDecision,
)


class DecisionBackend(ABC):
    """Abstract interface defining the decision surface of the RAG system.

    Decouples the strategic choices (retrieval strategy, evidence sufficiency,
    security evaluation) from the execution engines, enabling clean head-to-head
    comparisons between heuristic/deterministic policies and calibrated models.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the decision backend (e.g. 'heuristic', 'jev')."""
        pass

    @abstractmethod
    async def plan_retrieval(
        self,
        query: str,
        conversation_history: list[dict[str, Any]] | None = None,
        profile: dict[str, Any] | None = None,
        active_toggles: dict[str, bool] | None = None,
        query_id: str = "",
    ) -> RetrievalStrategyDecision:
        """Decide the retrieval mode (BM25/Dense/Hybrid), hop mode, and query transformations."""
        pass

    @abstractmethod
    async def evaluate_evidence_sufficiency(
        self,
        query: str,
        chunks: list[Any],
        target_company: str | None = None,
        query_id: str = "",
    ) -> SufficiencyDecision:
        """Decide whether retrieved evidence is sufficient to answer without hallucinating."""
        pass

    @abstractmethod
    async def evaluate_security(
        self,
        query: str,
        query_id: str = "",
    ) -> SecurityDecision:
        """Decide whether a query represents an adversarial prompt injection or benign request."""
        pass
