"""Configuration for secure RAG guardrails."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .constants import (
    DEFAULT_HALLUCINATION_THRESHOLD,
    DEFAULT_MAX_CONTEXT_LENGTH,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_QUERY_LENGTH,
    DEFAULT_MAX_RETRIEVED_CHUNKS,
    DEFAULT_PROMPT_INJECTION_THRESHOLD,
    DEFAULT_RATE_LIMIT,
    DEFAULT_RATE_WINDOW_SECONDS,
    DEFAULT_SIMILARITY_THRESHOLD,
    INPUT_BLOCK_PATTERNS,
    OUTPUT_BLOCK_PATTERNS,
    RETRIEVAL_BLOCK_PATTERNS,
)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class SecurityConfig:
    """Runtime configuration for the security pipeline."""

    max_query_length: int = field(default_factory=lambda: _env_int("MAX_QUERY_LENGTH", DEFAULT_MAX_QUERY_LENGTH))
    max_context_length: int = field(default_factory=lambda: _env_int("MAX_CONTEXT_LENGTH", DEFAULT_MAX_CONTEXT_LENGTH))
    max_output_tokens: int = field(default_factory=lambda: _env_int("MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS))
    max_retrieved_chunks: int = field(default_factory=lambda: _env_int("MAX_RETRIEVED_CHUNKS", DEFAULT_MAX_RETRIEVED_CHUNKS))

    rate_limit: int = field(default_factory=lambda: _env_int("RATE_LIMIT", DEFAULT_RATE_LIMIT))
    rate_window_seconds: int = field(default_factory=lambda: _env_int("RATE_WINDOW_SECONDS", DEFAULT_RATE_WINDOW_SECONDS))

    similarity_threshold: float = field(default_factory=lambda: _env_float("SIMILARITY_THRESHOLD", DEFAULT_SIMILARITY_THRESHOLD))
    hallucination_threshold: float = field(default_factory=lambda: _env_float("HALLUCINATION_THRESHOLD", DEFAULT_HALLUCINATION_THRESHOLD))
    prompt_injection_threshold: float = field(default_factory=lambda: _env_float("PROMPT_INJECTION_THRESHOLD", DEFAULT_PROMPT_INJECTION_THRESHOLD))

    blocked_input_patterns: list[str] = field(default_factory=lambda: INPUT_BLOCK_PATTERNS.copy())
    blocked_retrieval_patterns: list[str] = field(default_factory=lambda: RETRIEVAL_BLOCK_PATTERNS.copy())
    blocked_output_patterns: list[str] = field(default_factory=lambda: OUTPUT_BLOCK_PATTERNS.copy())
    allowed_file_types: list[str] = field(default_factory=lambda: os.getenv("ALLOWED_FILE_TYPES", "txt,md,pdf,docx").split(","))


def load_config() -> SecurityConfig:
    """Load security config from environment with safe defaults."""

    return SecurityConfig()
