"""Agent retrieval, reranking, HyDE, chunk enhancement, and evidence evaluation tools."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from backend.core.vector_store.base import ScoredChunk

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Standard output from an agent tool execution."""

    tool_name: str
    output: Any
    summary: str
    latency_ms: float = 0.0
    status: str = "SUCCESS"
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceSufficiencyResult:
    """Detailed evidence sufficiency classification."""

    status: str  # SUFFICIENT, PARTIALLY_SUFFICIENT, INSUFFICIENT, CONFLICTING, UNSUPPORTED
    is_sufficient: bool
    score: float
    reasons: list[str]
    missing_aspects: list[str]
    chunk_count: int


class BM25RetrieveTool:
    """Executes sparse lexical BM25 retrieval."""

    name = "BM25RetrieveTool"
    description = "Retrieve document chunks matching exact query keywords using BM25."

    def __init__(self, bm25_retriever: Any) -> None:
        self.bm25_retriever = bm25_retriever

    async def execute(
        self,
        query: str,
        top_k: int = 15,
        metadata_filters: dict[str, Any] | None = None,
    ) -> ToolResult:
        start_t = time.perf_counter()
        try:
            res = await self.bm25_retriever.retrieve(query, top_k=top_k, metadata_filters=metadata_filters)
            chunks = res.chunks
            latency = round((time.perf_counter() - start_t) * 1000, 2)
            summary = f"Retrieved {len(chunks)} chunks via BM25 lexical search."
            return ToolResult(
                tool_name=self.name,
                output=chunks,
                summary=summary,
                latency_ms=latency,
                status="SUCCESS",
                extra={"top_k": top_k, "filters": metadata_filters or {}},
            )
        except Exception as exc:
            latency = round((time.perf_counter() - start_t) * 1000, 2)
            logger.warning("BM25 retrieval error: %s", exc)
            return ToolResult(
                tool_name=self.name,
                output=[],
                summary=f"BM25 retrieval failed: {exc}",
                latency_ms=latency,
                status="FALLBACK",
                error=str(exc),
            )


class DenseRetrieveTool:
    """Executes dense vector similarity search using embeddings."""

    name = "DenseRetrieveTool"
    description = "Retrieve document chunks via semantic embedding similarity."

    def __init__(self, dense_retriever: Any) -> None:
        self.dense_retriever = dense_retriever

    async def execute(
        self,
        query: str,
        top_k: int = 15,
        metadata_filters: dict[str, Any] | None = None,
    ) -> ToolResult:
        start_t = time.perf_counter()
        try:
            res = await self.dense_retriever.retrieve(query, top_k=top_k, metadata_filters=metadata_filters)
            chunks = res.chunks
            latency = round((time.perf_counter() - start_t) * 1000, 2)
            summary = f"Retrieved {len(chunks)} chunks via dense vector similarity."
            return ToolResult(
                tool_name=self.name,
                output=chunks,
                summary=summary,
                latency_ms=latency,
                status="SUCCESS",
                extra={"top_k": top_k, "filters": metadata_filters or {}},
            )
        except Exception as exc:
            latency = round((time.perf_counter() - start_t) * 1000, 2)
            logger.warning("Dense retrieval error: %s", exc)
            return ToolResult(
                tool_name=self.name,
                output=[],
                summary=f"Dense retrieval failed: {exc}",
                latency_ms=latency,
                status="FALLBACK",
                error=str(exc),
            )


class HybridRetrieveTool:
    """Combines BM25 and Dense retrieval with Reciprocal Rank Fusion (RRF)."""

    name = "HybridRetrieveTool"
    description = "Execute hybrid sparse + dense retrieval fused via Reciprocal Rank Fusion."

    def __init__(self, hybrid_retriever: Any) -> None:
        self.hybrid_retriever = hybrid_retriever

    async def execute(
        self,
        query: str,
        top_k: int = 15,
        metadata_filters: dict[str, Any] | None = None,
    ) -> ToolResult:
        start_t = time.perf_counter()
        try:
            res = await self.hybrid_retriever.retrieve(query, top_k=top_k, metadata_filters=metadata_filters)
            chunks = res.chunks
            latency = round((time.perf_counter() - start_t) * 1000, 2)
            summary = f"Retrieved {len(chunks)} chunks via Hybrid RRF search."
            return ToolResult(
                tool_name=self.name,
                output=chunks,
                summary=summary,
                latency_ms=latency,
                status="SUCCESS",
                extra={"top_k": top_k, "rrf_k": getattr(self.hybrid_retriever, "rrf_k", 60)},
            )
        except Exception as exc:
            latency = round((time.perf_counter() - start_t) * 1000, 2)
            logger.warning("Hybrid retrieval error: %s", exc)
            return ToolResult(
                tool_name=self.name,
                output=[],
                summary=f"Hybrid retrieval failed: {exc}",
                latency_ms=latency,
                status="FALLBACK",
                error=str(exc),
            )


class HyDETool:
    """Hypothetical Document Embeddings: generates synthetic ideal passage and embeds it."""

    name = "HyDETool"
    description = "Generate a hypothetical document representation to bridge semantic vocabulary gap."

    def __init__(self, hyde_generator: Any, vector_store: Any) -> None:
        self.hyde_generator = hyde_generator
        self.vector_store = vector_store

    async def execute(
        self,
        query: str,
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
    ) -> ToolResult:
        start_t = time.perf_counter()
        try:
            hyde_res = await self.hyde_generator.generate_and_embed(query)
            chunks = self.vector_store.search(
                query_embedding=hyde_res.embedding,
                top_k=top_k,
                metadata_filters=metadata_filters,
            )
            latency = round((time.perf_counter() - start_t) * 1000, 2)
            summary = f"Generated HyDE representation and retrieved {len(chunks)} candidates."
            return ToolResult(
                tool_name=self.name,
                output=chunks,
                summary=summary,
                latency_ms=latency,
                status="SUCCESS",
                extra={
                    "hypothetical_document": hyde_res.hypothetical_document[:200],
                    "embedding_dimension": len(hyde_res.embedding),
                },
            )
        except Exception as exc:
            latency = round((time.perf_counter() - start_t) * 1000, 2)
            logger.warning("HyDE execution error: %s", exc)
            return ToolResult(
                tool_name=self.name,
                output=[],
                summary=f"HyDE generation failed: {exc}",
                latency_ms=latency,
                status="FALLBACK",
                error=str(exc),
            )


class RerankTool:
    """Cross-encoder precision reranker."""

    name = "RerankTool"
    description = "Rerank candidate chunks using cross-encoder for precision relevance scoring."

    def __init__(self, reranker: Any) -> None:
        self.reranker = reranker

    def execute(
        self,
        query: str,
        chunks: list[Any],
        top_k: int = 8,
    ) -> ToolResult:
        start_t = time.perf_counter()
        if not chunks:
            return ToolResult(tool_name=self.name, output=[], summary="No candidates to rerank.", latency_ms=0.0)
        try:
            rerank_res = self.reranker.rerank(query, chunks, top_k=top_k)
            ranked_chunks = rerank_res.chunks
            rank_movements = []
            for rc in ranked_chunks:
                orig_r = getattr(rc, "original_rank", 1)
                new_r = getattr(rc, "new_rank", 1)
                rank_movements.append({
                    "chunk_id": getattr(rc, "chunk_id", ""),
                    "original_rank": orig_r,
                    "new_rank": new_r,
                    "movement": f"{orig_r} → {new_r}",
                    "rank_change": getattr(rc, "rank_change", orig_r - new_r),
                    "original_score": round(getattr(rc, "original_score", 0.0), 4),
                    "cross_encoder_score": round(getattr(rc, "cross_encoder_score", 0.0), 4),
                    "source": getattr(rc, "source", getattr(rc, "metadata", {}).get("source_file", "Doc")),
                    "company": getattr(rc, "metadata", {}).get("company", getattr(rc, "company", "General")),
                })

            latency = round((time.perf_counter() - start_t) * 1000, 2)
            summary = f"Reranked {len(chunks)} candidates down to top {len(ranked_chunks)}."
            return ToolResult(
                tool_name=self.name,
                output=ranked_chunks,
                summary=summary,
                latency_ms=latency,
                status="SUCCESS",
                extra={
                    "candidate_count_before": len(chunks),
                    "candidate_count_after": len(ranked_chunks),
                    "rank_movements": rank_movements,
                },
            )
        except Exception as exc:
            latency = round((time.perf_counter() - start_t) * 1000, 2)
            logger.warning("Reranking error: %s", exc)
            return ToolResult(
                tool_name=self.name,
                output=chunks[:top_k],
                summary=f"Reranking failed; preserving original ranking: {exc}",
                latency_ms=latency,
                status="FALLBACK",
                error=str(exc),
            )


class ChunkEnhanceTool:
    """Enhances chunk representations with contextual metadata headers."""

    name = "ChunkEnhanceTool"
    description = "Enrich candidate chunks with section headers and company metadata."

    def execute(self, chunks: list[Any]) -> ToolResult:
        start_t = time.perf_counter()
        enhanced: list[Any] = []
        for c in chunks:
            comp = getattr(c, "metadata", {}).get("company", getattr(c, "company", "General"))
            topic = getattr(c, "metadata", {}).get("topic", getattr(c, "topic", "Interview Preparation"))
            src = getattr(c, "source", getattr(c, "metadata", {}).get("source_file", "Doc"))
            raw_text = getattr(c, "text", str(c))

            prefix = f"[{comp.upper()} INTERVIEW ARCHIVE | Topic: {topic.title()} | Source: {src}]\n"
            if not raw_text.startswith("["):
                new_text = prefix + raw_text
            else:
                new_text = raw_text

            # Create enhanced chunk copy
            if hasattr(c, "chunk_id"):
                meta = dict(getattr(c, "metadata", {}) or {})
                meta["is_enhanced"] = True
                meta["enhancement_strategy"] = "metadata_prepend_and_section_header"
                enhanced_chunk = ScoredChunk(
                    chunk_id=getattr(c, "chunk_id"),
                    text=new_text,
                    metadata=meta,
                    score=getattr(c, "score", 0.0),
                )
                enhanced.append(enhanced_chunk)
            else:
                enhanced.append(c)

        latency = round((time.perf_counter() - start_t) * 1000, 2)
        summary = f"Applied contextual metadata headers across {len(enhanced)} candidate chunks."
        return ToolResult(
            tool_name=self.name,
            output=enhanced,
            summary=summary,
            latency_ms=latency,
            status="SUCCESS",
        )


class DecomposeQueryTool:
    """Decompose complex multi-hop interview queries into targeted subqueries."""

    name = "DecomposeQueryTool"
    description = "Break a complex comparison or multi-round query into subqueries."

    def execute(
        self,
        query: str,
        detected_companies: list[str] | None = None,
        detected_topics: list[str] | None = None,
    ) -> ToolResult:
        start_t = time.perf_counter()
        subqueries: list[str] = []
        lower = query.lower()

        # 1. Decompose by multiple detected companies (e.g. "OpenAI vs Anthropic vs Databricks")
        if detected_companies and len(detected_companies) >= 2:
            for comp in detected_companies:
                if "system design" in lower:
                    subqueries.append(f"{comp} system design interview questions topics")
                elif "dsa" in lower or "coding" in lower:
                    subqueries.append(f"{comp} DSA coding interview questions")
                elif "agent" in lower or "rag" in lower:
                    subqueries.append(f"{comp} AI agent and RAG interview questions")
                else:
                    subqueries.append(f"{comp} technical interview questions process format")

        # 2. Decompose by behavioral example + leadership principle + evaluation criteria
        elif ("disagree" in lower or "behavioral" in lower) and ("principle" in lower or "evaluation" in lower):
            comp_prefix = f"{detected_companies[0]} " if detected_companies else "Amazon "
            subqueries.append(f"{comp_prefix}behavioral questions disagreement with manager coworker")
            subqueries.append(f"{comp_prefix}leadership principles demonstrate disagree and commit ownership")
            subqueries.append(f"{comp_prefix}behavioral interview evaluation criteria rubric expected answer")

        # 3. Decompose by multiple interview rounds
        elif "round" in lower or "stage" in lower or "interview process" in lower:
            comp_prefix = f"{detected_companies[0]} " if detected_companies else ""
            if "round 1" in lower or "oa" in lower or "online assessment" in lower or "screening" in lower:
                subqueries.append(f"{comp_prefix}round 1 online assessment screening coding questions")
            if "round 2" in lower or "technical" in lower or "dsa" in lower:
                subqueries.append(f"{comp_prefix}round 2 technical coding data structures algorithms")
            if "system design" in lower or "round 3" in lower or "onsite" in lower:
                subqueries.append(f"{comp_prefix}system design architecture interview round")
            if "behavioral" in lower or "hr" in lower or "leadership" in lower:
                subqueries.append(f"{comp_prefix}behavioral leadership principles culture round")

        # 4. Decompose by multiple topics
        elif detected_topics and len(detected_topics) >= 2:
            comp_prefix = f"{detected_companies[0]} " if detected_companies else ""
            for topic in detected_topics:
                subqueries.append(f"{comp_prefix}{topic.replace('_', ' ')} interview questions")

        # 5. Fallback conjunction split
        if not subqueries and (" vs " in lower or " compare " in lower):
            clean_q = re.sub(r"\b(compare|the|focus|between|difference|in|of)\b", "", query, flags=re.IGNORECASE).strip()
            parts = [p.strip() for p in re.split(r"\b(vs|and|,)\b", clean_q, flags=re.IGNORECASE) if len(p.strip()) > 3]
            if len(parts) >= 2:
                subqueries = [f"{p} interview questions and technical topics" for p in parts[:3]]

        if not subqueries:
            subqueries = [query]

        latency = round((time.perf_counter() - start_t) * 1000, 2)
        summary = f"Decomposed query into {len(subqueries)} targeted subqueries."
        return ToolResult(
            tool_name=self.name,
            output=subqueries,
            summary=summary,
            latency_ms=latency,
            status="SUCCESS",
        )


class EvaluateEvidenceTool:
    """Classifies evidence sufficiency into SUFFICIENT, PARTIALLY_SUFFICIENT, INSUFFICIENT, CONFLICTING, or UNSUPPORTED."""

    name = "EvaluateEvidenceTool"
    description = "Evaluate if candidate chunks contain sufficient evidence to answer the query."

    def execute(
        self,
        query: str,
        chunks: list[Any],
        target_company: str | None = None,
    ) -> ToolResult:
        start_t = time.perf_counter()
        reasons: list[str] = []
        missing_aspects: list[str] = []

        if not chunks:
            res = EvidenceSufficiencyResult(
                status="UNSUPPORTED",
                is_sufficient=False,
                score=0.0,
                reasons=["No candidate chunks found in the interview knowledge corpus."],
                missing_aspects=["All company interview information missing"],
                chunk_count=0,
            )
            latency = round((time.perf_counter() - start_t) * 1000, 2)
            return ToolResult(
                tool_name=self.name,
                output=res,
                summary="Evidence status: UNSUPPORTED (0 chunks)",
                latency_ms=latency,
                status="SUCCESS",
            )

        # Check target company presence in retrieved chunk metadata
        if target_company:
            comp_chunks = []
            for c in chunks:
                raw_m = getattr(c, "metadata", None)
                if hasattr(raw_m, "company"):
                    c_comp = raw_m.company
                elif isinstance(raw_m, dict):
                    c_comp = raw_m.get("company", "")
                else:
                    c_comp = getattr(c, "company", "")
                if str(c_comp).strip().lower() == target_company.strip().lower():
                    comp_chunks.append(c)
            if not comp_chunks:
                res = EvidenceSufficiencyResult(
                    status="UNSUPPORTED",
                    is_sufficient=False,
                    score=0.0,
                    reasons=[f"Retrieved chunks do not contain verified interview records for target company '{target_company}'."],
                    missing_aspects=[f"Company '{target_company}' interview records"],
                    chunk_count=0,
                )
                latency = round((time.perf_counter() - start_t) * 1000, 2)
                return ToolResult(
                    tool_name=self.name,
                    output=res,
                    summary=f"Evidence status: UNSUPPORTED ({target_company} not indexed)",
                    latency_ms=latency,
                    status="SUCCESS",
                )

        # Evaluate score distribution
        scores = [getattr(c, "score", getattr(c, "similarity_score", 0.0)) for c in chunks]
        max_score = max(scores) if scores else 0.0

        # Check keyword coverage
        q_words = {w.lower() for w in re.findall(r"\b\w+\b", query) if len(w) > 3}
        all_text = " ".join(getattr(c, "text", str(c)) for c in chunks).lower()
        covered_words = {w for w in q_words if w in all_text}
        coverage_ratio = len(covered_words) / max(1, len(q_words))

        if coverage_ratio < 0.20 and max_score < 0.08:
            status = "INSUFFICIENT"
            is_sufficient = False
            reasons.append(f"Low relevance score ({max_score:.2f}) and minimal query term overlap ({coverage_ratio:.0%}).")
            missing_aspects.append("Key interview question details")
        elif len(chunks) == 1 and max_score < 0.15:
            status = "PARTIALLY_SUFFICIENT"
            is_sufficient = False
            reasons.append("Single candidate chunk with marginal relevance.")
            missing_aspects.append("Multi-perspective interview round evidence")
        elif len(chunks) >= 2 or max_score >= 0.20:
            status = "SUFFICIENT"
            is_sufficient = True
            reasons.append(f"Strong evidence found across {len(chunks)} chunks with max relevance score {max_score:.2f}.")
        else:
            status = "PARTIALLY_SUFFICIENT"
            is_sufficient = True
            reasons.append(f"Acceptable evidence found across {len(chunks)} chunks.")

        res = EvidenceSufficiencyResult(
            status=status,
            is_sufficient=is_sufficient,
            score=round(max_score, 4),
            reasons=reasons,
            missing_aspects=missing_aspects,
            chunk_count=len(chunks),
        )
        latency = round((time.perf_counter() - start_t) * 1000, 2)
        summary = f"Evidence status: {status} (Score: {max_score:.2f}, Chunks: {len(chunks)})"
        return ToolResult(
            tool_name=self.name,
            output=res,
            summary=summary,
            latency_ms=latency,
            status="SUCCESS",
        )


# Backward compatibility aliases
RetrieveTool = HybridRetrieveTool
GenerateAnswerTool = None
