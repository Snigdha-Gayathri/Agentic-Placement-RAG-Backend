"""Unit tests for Phase 2 ingestion pipeline, deduplication, schema validation, and attribution."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from backend.ingestion.ingest_placement_chunks import (
    PlacementChunkRecord,
    compute_text_similarity,
    load_company_aliases,
    normalize_company,
)
from backend.config.settings import PipelineConfig, DEFAULT_FEATURE_TOGGLES
from backend.core.agent.graph import build_disclosure_section, AgenticRAGState
from backend.core.vector_store.base import ScoredChunk


def test_schema_validation_accepts_valid_record():
    """Validates that a correctly formatted chunk passes validation."""
    data = {
        "chunk_id": "test_chunk_001",
        "company": "Amazon",
        "role": "SDE-2",
        "round_type": "Behavioral",
        "topic": "behavioral",
        "difficulty": "Medium",
        "year": "2024",
        "text": "Tell me about a time when you had to deal with an ambiguous situation with unclear requirements.",
        "likely_questions": ["How do you handle ambiguity?"],
        "source_repo": "yangshun/tech-interview-handbook",
        "source_file": "behavioral/amazon.md",
        "license": "MIT",
        "reliability": "curated",
        "retrieved_at": "2026-08-20T10:00:00Z",
    }
    rec = PlacementChunkRecord.model_validate(data)
    assert rec.chunk_id == "test_chunk_001"
    assert rec.company == "Amazon"
    assert rec.source_repo == "yangshun/tech-interview-handbook"
    assert rec.license == "MIT"


def test_schema_validation_rejects_empty_company_and_short_text():
    """Validates that empty company and short text raise ValidationError."""
    with pytest.raises(ValidationError):
        PlacementChunkRecord.model_validate({
            "chunk_id": "c1",
            "company": "   ",
            "text": "Valid long text here that exceeds ten chars.",
        })

    with pytest.raises(ValidationError):
        PlacementChunkRecord.model_validate({
            "chunk_id": "c2",
            "company": "Google",
            "text": "short",
        })


def test_company_alias_normalization():
    """Validates canonical mapping of company aliases."""
    alias_map = load_company_aliases()
    assert normalize_company("aws", alias_map) == "Amazon"
    assert normalize_company("AWS", alias_map) == "Amazon"
    assert normalize_company("facebook", alias_map) == "Meta"
    assert normalize_company("FB", alias_map) == "Meta"
    assert normalize_company("alphabet", alias_map) == "Google"
    assert normalize_company("msft", alias_map) == "Microsoft"
    assert normalize_company("UnknownStartup", alias_map) == "UnknownStartup"


def test_text_similarity_deduplication():
    """Near-duplicate texts exceed threshold; distinct texts do not."""
    text1 = "Explain how Amazon DynamoDB achieves high availability using consistent hashing and replication."
    text2 = "Explain how Amazon DynamoDB achieves high availability using consistent hashing and data replication."
    distinct = "What is the difference between TCP and UDP protocols in computer networking?"

    sim_high = compute_text_similarity(text1, text2)
    sim_low = compute_text_similarity(text1, distinct)

    assert sim_high >= 0.80, f"Expected high similarity, got {sim_high}"
    assert sim_low < 0.30, f"Expected low similarity, got {sim_low}"


def test_kb_version_and_attribution_config():
    """Validates PipelineConfig holds kb_version and attribution toggles."""
    cfg = PipelineConfig()
    assert hasattr(cfg, "kb_version")
    assert cfg.kb_version == "all"
    assert hasattr(cfg, "source_attribution_enabled")
    assert cfg.source_attribution_enabled is True
    assert DEFAULT_FEATURE_TOGGLES["source_attribution"] is True


def test_source_attribution_surfaced_in_disclosure():
    """Validates that build_disclosure_section properly formats source_repo and license."""
    chunk = ScoredChunk(
        chunk_id="chunk_test_101",
        text="Sample interview text",
        metadata={
            "source_repo": "yangshun/tech-interview-handbook",
            "source_file": "algorithms/binary_search.md",
            "license": "MIT",
        },
        score=0.95,
    )
    state: AgenticRAGState = {
        "executed_operations": ["HYBRID_RETRIEVAL", "ANSWER_SYNTHESIS"],
        "final_context_chunks": [chunk],
        "active_toggles": {"source_attribution": True},
    }
    disclosure = build_disclosure_section(state)
    assert "**Sources & Attribution:**" in disclosure
    assert "yangshun/tech-interview-handbook" in disclosure
    assert "algorithms/binary_search.md" in disclosure
    assert "(MIT)" in disclosure
