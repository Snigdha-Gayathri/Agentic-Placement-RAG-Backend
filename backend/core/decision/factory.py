"""Factory and configuration for DecisionBackend in Agentic Placement RAG."""

from __future__ import annotations

import os
from typing import Literal

from .base import DecisionBackend
from .heuristic import HeuristicDecisionBackend
from .jev import JevDecisionBackend

_ACTIVE_BACKEND: DecisionBackend | None = None


def get_decision_backend(
    name: str | None = None,
    api_key: str | None = None,
    force_new: bool = False,
) -> DecisionBackend:
    """Retrieve or construct the active DecisionBackend based on configuration.

    Args:
        name: Backend name ('heuristic', 'jev'). If None, reads from DECISION_BACKEND env var.
        api_key: Optional TypeSafe API key override.
        force_new: If True, constructs a new instance rather than returning cached singleton.

    Returns:
        DecisionBackend instance (HeuristicDecisionBackend or JevDecisionBackend).
    """
    global _ACTIVE_BACKEND

    if name is not None:
        backend_type = name.strip().lower()
    else:
        jev_enabled_env = os.getenv("JEV_ENABLED", "").strip().lower()
        if jev_enabled_env in ("1", "true", "yes", "on"):
            backend_type = "jev"
        else:
            backend_type = os.getenv("DECISION_BACKEND", "heuristic").strip().lower()

    if not force_new and _ACTIVE_BACKEND is not None and _ACTIVE_BACKEND.name == backend_type:
        return _ACTIVE_BACKEND

    if backend_type == "jev":
        backend = JevDecisionBackend(api_key=api_key)
    else:
        backend = HeuristicDecisionBackend()

    if not force_new:
        _ACTIVE_BACKEND = backend

    return backend


def set_decision_backend(backend: DecisionBackend) -> None:
    """Explicitly set the active decision backend."""
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backend
