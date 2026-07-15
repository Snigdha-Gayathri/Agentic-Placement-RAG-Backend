"""Pydantic request/response models for the secure Agentic RAG API."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class RetrievedItem(BaseModel):
    company: str = Field(default="", max_length=100)
    question: str = Field(default="", max_length=4000)
    tags: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=12000)
    session_id: str | None = Field(default=None, max_length=128)
    request_id: str | None = Field(default=None, max_length=128)


class GuardrailMeta(BaseModel):
    request_id: str
    injection_score: float
    discarded_chunks: int
    chunk_count: int
    warning: str | None = None


class PipelineStageModel(BaseModel):
    name: str
    status: str
    start_time: float
    end_time: float
    latency_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class PipelineData(BaseModel):
    request_id: str
    query_info: dict[str, Any] = Field(default_factory=dict)
    retrieval_info: dict[str, Any] = Field(default_factory=dict)
    reranking_info: dict[str, Any] = Field(default_factory=dict)
    agent_info: dict[str, Any] = Field(default_factory=dict)
    generation_info: dict[str, Any] = Field(default_factory=dict)
    pipeline_stages: list[PipelineStageModel] = Field(default_factory=list)
    feature_toggles: dict[str, bool] = Field(default_factory=dict)
    security_info: dict[str, Any] = Field(default_factory=dict)
    hyde_info: dict[str, Any] = Field(default_factory=dict)
    memory_info: dict[str, Any] = Field(default_factory=dict)
    metrics_info: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    meta: GuardrailMeta
    request_id: str | None = None
    pipeline_data: PipelineData | None = None


class ReindexResponse(BaseModel):
    status: str
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    gemini: bool
    vector_index: bool
    readiness: bool


class SecurityStatusResponse(BaseModel):
    status: str
    config: dict[str, Any]


class FeatureToggleConfig(BaseModel):
    dense_retrieval: bool = True
    bm25_retrieval: bool = True
    hybrid_retrieval: bool = True
    cross_encoder_reranking: bool = True
    metadata_filtering: bool = True
    hyde: bool = False
    conversation_memory: bool = True
    query_rewriting: bool = True
    agent_planning: bool = True
    multi_hop_retrieval: bool = False
    chunk_enhancement: bool = False


class VectorDBStatsModel(BaseModel):
    total_chunks: int
    embedding_dimension: int
    collection_name: str
    metadata_distribution: dict[str, Any]
    avg_chunk_length: float
    index_size_mb: float
