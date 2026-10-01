"""Typed state model for LangGraph Agentic Retrieval."""

from __future__ import annotations

from typing import Any, TypedDict


class AgenticRAGState(TypedDict, total=False):
    """Structured LangGraph agent state representing the entire retrieval lifecycle."""

    # 1. Query Information
    original_query: str
    rewritten_query: str
    rewrite_used: bool
    rewrite_reason: str

    # 2. Metadata Filtering
    metadata_filter_used: bool
    metadata_filter: dict[str, Any]
    metadata_filter_reason: str
    target_company: str | None
    detected_topics: list[str]

    # 3. Embedding Generation
    embedding_generation_used: bool
    embedding_model: str
    embedding_reason: str

    # 4. Chunk Enhancement
    chunk_enhancement_used: bool
    chunk_enhancement_type: str
    chunk_enhancement_reason: str

    # 5. BM25 Sparse Retrieval
    bm25_used: bool
    bm25_top_k: int
    bm25_results: list[dict[str, Any]]
    bm25_reason: str

    # 6. Dense Vector Retrieval
    dense_used: bool
    dense_top_k: int
    dense_results: list[dict[str, Any]]
    dense_reason: str

    # 7. Hybrid Retrieval
    hybrid_used: bool
    hybrid_reason: str
    bm25_contribution: float
    dense_contribution: float

    # 8. RRF Fusion
    rrf_used: bool
    rrf_k: int
    rrf_input_rankings: dict[str, list[str]]
    rrf_output_ranking: list[str]
    rrf_reason: str

    # 9. HyDE
    hyde_used: bool
    hyde_reason: str
    hyde_generated_representation: str
    hyde_retrieval_result: str

    # 10. Multi-Hop Retrieval
    multi_hop_used: bool
    hop_count: int
    hop_queries: list[str]
    hop_results: list[dict[str, Any]]
    hop_reasons: list[str]

    # 11. Cross-Encoder Reranking
    reranking_used: bool
    reranker_model: str
    reranking_reason: str
    candidate_count_before_reranking: int
    candidate_count_after_reranking: int
    rank_movements: list[dict[str, Any]]

    # 12. Planning & Execution Trace
    retrieval_plan: list[str]
    executed_operations: list[str]
    skipped_operations: list[dict[str, str]]
    retrieval_trace: list[dict[str, Any]]

    # 13. Retrieved Chunks and Context
    retrieved_chunks: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]
    final_context_chunks: list[dict[str, Any]]
    sanitized_context: str

    # 14. Evidence Assessment & Escalation
    evidence_assessment: dict[str, Any]
    termination_reason: str
    escalation_count: int

    # 15. Budgets & Bounds
    budgets: dict[str, Any]
    latency: dict[str, float]
    errors: list[dict[str, Any]]

    # 16. Context & Security
    session_id: str
    conversation_history: list[dict[str, Any]]
    is_blocked: bool
    security_event: dict[str, Any]

    # 17. Final Answer & Disclosure
    generated_answer: str
    answer_disclosure: str

    # 18. Structured Decision Metadata & Telemetry
    decision_metadata: dict[str, Any]
    active_toggles: dict[str, bool]
