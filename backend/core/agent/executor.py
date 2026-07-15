"""Iterative agent executor orchestrating retrieval, evaluation, and multi-hop reasoning."""

from __future__ import annotations

import logging
import time
from typing import Any

from .planner import AgentPlanner, PlanStep, ReasoningTrace
from .tools import (
    DecomposeQueryTool,
    EvaluateEvidenceTool,
    GenerateAnswerTool,
    RetrieveTool,
)

logger = logging.getLogger(__name__)


class AgentExecutor:
    """Executes multi-step reasoning trace over retrieval and synthesis."""

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        max_iterations: int = 3,
    ) -> None:
        self.retrieve_tool = RetrieveTool(retriever)
        self.decompose_tool = DecomposeQueryTool()
        self.evaluate_tool = EvaluateEvidenceTool()
        self.generate_tool = GenerateAnswerTool(llm)
        self.planner = AgentPlanner()
        self.max_iterations = max_iterations

    async def execute(
        self,
        query: str,
        rewritten_query: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
        hyde_used: bool = False,
        multi_hop_enabled: bool = False,
    ) -> tuple[list[Any], ReasoningTrace]:
        plan = self.planner.analyze(
            query=query,
            rewritten_query=rewritten_query,
            metadata_filters=metadata_filters,
            hyde_used=hyde_used,
            multi_hop_enabled=multi_hop_enabled,
        )
        trace = plan.trace
        active_query = rewritten_query or query

        # Step 1: Initial retrieval
        ret_res = await self.retrieve_tool.execute(
            query=active_query, top_k=15, metadata_filters=metadata_filters
        )
        chunks = ret_res.output
        trace.steps.append(
            PlanStep(
                tool_name=ret_res.tool_name,
                reasoning="Initial retrieval pass for query evidence",
                inputs={"query": active_query, "metadata_filters": metadata_filters or {}},
                output_summary=ret_res.summary,
                latency_ms=ret_res.latency_ms,
            )
        )

        # Step 2: Multi-hop subquery retrieval if needed
        if multi_hop_enabled and len(plan.initial_steps) > 1:
            dec_res = self.decompose_tool.execute(active_query)
            trace.steps.append(
                PlanStep(
                    tool_name=dec_res.tool_name,
                    reasoning="Decomposing multi-hop query into targeted subqueries",
                    inputs={"query": active_query},
                    output_summary=dec_res.summary,
                    latency_ms=dec_res.latency_ms,
                )
            )
            subqueries = dec_res.output
            for subq in subqueries[1:]:
                sub_res = await self.retrieve_tool.execute(
                    query=subq, top_k=8, metadata_filters=metadata_filters
                )
                # deduplicate chunks by chunk_id
                existing_ids = {c.chunk_id for c in chunks}
                for c in sub_res.output:
                    if c.chunk_id not in existing_ids:
                        chunks.append(c)
                        existing_ids.add(c.chunk_id)
                trace.steps.append(
                    PlanStep(
                        tool_name=sub_res.tool_name,
                        reasoning=f"Multi-hop subquery retrieval for '{subq}'",
                        inputs={"query": subq},
                        output_summary=sub_res.summary,
                        latency_ms=sub_res.latency_ms,
                    )
                )

        # Step 3: Evaluate evidence
        eval_res = self.evaluate_tool.execute(active_query, chunks)
        trace.enough_evidence = eval_res.output["sufficient"]
        trace.steps.append(
            PlanStep(
                tool_name=eval_res.tool_name,
                reasoning="Assessing candidate chunks against query requirements",
                inputs={"chunk_count": len(chunks)},
                output_summary=eval_res.summary,
                latency_ms=eval_res.latency_ms,
            )
        )

        # Step 4: Iterative second retrieval if evidence insufficient
        iteration = 1
        while not trace.enough_evidence and iteration < self.max_iterations:
            iteration += 1
            trace.iterations_taken = iteration
            expanded_query = f"{active_query} interview questions solutions explanations"
            ret_res2 = await self.retrieve_tool.execute(
                query=expanded_query, top_k=10, metadata_filters=None
            )
            existing_ids = {c.chunk_id for c in chunks}
            for c in ret_res2.output:
                if c.chunk_id not in existing_ids:
                    chunks.append(c)
                    existing_ids.add(c.chunk_id)
            trace.steps.append(
                PlanStep(
                    tool_name=ret_res2.tool_name,
                    reasoning=f"Iteration {iteration}: Second retrieval pass with relaxed query and filters",
                    inputs={"query": expanded_query},
                    output_summary=ret_res2.summary,
                    latency_ms=ret_res2.latency_ms,
                )
            )
            eval_res2 = self.evaluate_tool.execute(active_query, chunks)
            trace.enough_evidence = eval_res2.output["sufficient"]
            trace.steps.append(
                PlanStep(
                    tool_name=eval_res2.tool_name,
                    reasoning=f"Iteration {iteration}: Re-evaluating evidence sufficiency",
                    inputs={"chunk_count": len(chunks)},
                    output_summary=eval_res2.summary,
                    latency_ms=eval_res2.latency_ms,
                )
            )

        return chunks, trace
