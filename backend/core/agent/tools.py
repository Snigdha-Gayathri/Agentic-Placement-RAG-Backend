"""Agent tools for retrieval, decomposition, evaluation, and answer generation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Standard output from an agent tool execution."""

    tool_name: str
    output: Any
    summary: str
    latency_ms: float = 0.0


class RetrieveTool:
    """Tool wrapper around document retrieval."""

    name = "retrieve"
    description = "Retrieve document chunks relevant to a query."

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    async def execute(
        self, query: str, top_k: int = 10, metadata_filters: dict[str, Any] | None = None
    ) -> ToolResult:
        start_t = time.perf_counter()
        res = await self.retriever.retrieve(query, top_k=top_k, metadata_filters=metadata_filters)
        latency = round((time.perf_counter() - start_t) * 1000, 2)
        summary = f"Retrieved {len(res.chunks)} candidate chunks using {res.retriever_type} search."
        return ToolResult(
            tool_name=self.name,
            output=res.chunks,
            summary=summary,
            latency_ms=latency,
        )


class DecomposeQueryTool:
    """Decompose complex multi-hop queries into subqueries."""

    name = "decompose_query"
    description = "Break a complex comparison query into subqueries."

    def execute(self, query: str) -> ToolResult:
        start_t = time.perf_counter()
        lower = query.lower()
        subqueries = [query]
        if " vs " in lower or "compare" in lower:
            parts = [p.strip() for p in query.replace(" vs ", " and ").split(" and ") if len(p.strip()) > 3]
            if len(parts) >= 2:
                subqueries = parts[:3]
        latency = round((time.perf_counter() - start_t) * 1000, 2)
        summary = f"Decomposed query into {len(subqueries)} sub-queries."
        return ToolResult(
            tool_name=self.name,
            output=subqueries,
            summary=summary,
            latency_ms=latency,
        )


class EvaluateEvidenceTool:
    """Check if retrieved evidence is sufficient to answer the query."""

    name = "evaluate_evidence"
    description = "Evaluate if candidate chunks contain sufficient evidence."

    def execute(self, query: str, chunks: list[Any]) -> ToolResult:
        start_t = time.perf_counter()
        # Heuristic: at least 1 chunk with good score or at least 2 non-empty chunks
        sufficient = len(chunks) >= 2 or (len(chunks) == 1 and getattr(chunks[0], "score", 0.0) >= 0.15)
        latency = round((time.perf_counter() - start_t) * 1000, 2)
        summary = "Evidence sufficient to answer query." if sufficient else "Evidence insufficient; further retrieval recommended."
        return ToolResult(
            tool_name=self.name,
            output={"sufficient": sufficient, "chunk_count": len(chunks)},
            summary=summary,
            latency_ms=latency,
        )


class GenerateAnswerTool:
    """Generate final grounded answer using LLM."""

    name = "generate_answer"
    description = "Generate response given query and retrieved context chunks."

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def execute(self, system_prompt: str, user_prompt: str) -> ToolResult:
        start_t = time.perf_counter()
        resp = await self.llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        latency = round((time.perf_counter() - start_t) * 1000, 2)
        summary = f"Generated grounded answer ({resp.output_tokens} tokens)."
        return ToolResult(
            tool_name=self.name,
            output=resp,
            summary=summary,
            latency_ms=latency,
        )
