"""Configuration system for the Agentic RAG pipeline.

Provides centralized configuration management with environment variable
overrides, feature toggles, and a singleton accessor.
"""
from __future__ import annotations

from backend.config.settings import PipelineConfig, get_config, update_feature_toggle

__all__ = ["PipelineConfig", "get_config", "update_feature_toggle"]
