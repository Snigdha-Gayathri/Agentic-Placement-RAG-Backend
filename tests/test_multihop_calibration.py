"""Unit and regression tests for calibrated multi-hop query routing and decomposition.

Ensures that single-company questions containing conjunctions ("and", "principles", "criteria")
are correctly routed to SINGLE_HOP, while true cross-company comparisons and sequential round pipelines
are routed to MULTI_HOP.
"""

import pytest
from backend.core.agent.planner import AgentPlanner
from backend.core.agent.tools import DecomposeQueryTool
from backend.core.decision.models import HopMode, RouteType


def test_single_company_with_conjunctions_routes_to_single_hop():
    """Single-company query with conjunction 'and' and 'principles' must NOT trigger multi-hop."""
    planner = AgentPlanner()
    plan = planner.plan(
        "What behavioral interview questions are commonly asked at Amazon, and which Leadership Principles do they test?"
    )
    assert plan["multi_hop_needed"] is False
    assert "MULTI_HOP_RETRIEVAL" not in plan["planned_operations"]


def test_single_company_coding_topics_with_and_routes_to_single_hop():
    """Single-company query asking about DP and graphs must route to single-hop."""
    planner = AgentPlanner()
    plan = planner.plan(
        "What dynamic programming and graph problems frequently appear in Google SDE interviews?"
    )
    assert plan["multi_hop_needed"] is False
    assert "MULTI_HOP_RETRIEVAL" not in plan["planned_operations"]


def test_single_company_behavioral_demonstration_routes_to_single_hop():
    """Standard behavioral preparation question without conflict scenario routes to single-hop."""
    planner = AgentPlanner()
    plan = planner.plan(
        "I'm preparing for an Amazon software engineering interview. Give me behavioral questions I should practice, and explain what a strong answer should demonstrate for each."
    )
    assert plan["multi_hop_needed"] is False
    assert "MULTI_HOP_RETRIEVAL" not in plan["planned_operations"]


def test_cross_company_comparison_routes_to_multi_hop():
    """Queries comparing two companies must route to MULTI_HOP."""
    planner = AgentPlanner()
    plan = planner.plan(
        "Compare Amazon and Google software engineering interviews in terms of their behavioral and technical interview expectations."
    )
    assert plan["multi_hop_needed"] is True
    assert "MULTI_HOP_RETRIEVAL" in plan["planned_operations"]


@pytest.mark.anyio
async def test_heuristic_backend_routes_single_company_to_single_hop():
    """HeuristicDecisionBackend produces HopMode.SINGLE_HOP for single company queries."""
    from backend.core.decision.heuristic import HeuristicDecisionBackend
    backend = HeuristicDecisionBackend()
    dec = await backend.plan_retrieval(
        "What behavioral interview questions are commonly asked at Amazon, and which Leadership Principles do they test?"
    )
    assert dec.hop_mode == HopMode.SINGLE_HOP
    assert dec.route_type == RouteType.SINGLE_HOP


@pytest.mark.anyio
async def test_heuristic_backend_routes_cross_company_to_multi_hop():
    """HeuristicDecisionBackend produces HopMode.MULTI_HOP for multi-company comparisons."""
    from backend.core.decision.heuristic import HeuristicDecisionBackend
    backend = HeuristicDecisionBackend()
    dec = await backend.plan_retrieval(
        "Compare Amazon and Google software engineering interviews in terms of their behavioral and technical interview expectations."
    )
    assert dec.hop_mode == HopMode.MULTI_HOP
    assert dec.route_type == RouteType.MULTI_HOP


def test_cross_company_differ_between_routes_to_multi_hop():
    """Queries asking how interviews differ between two companies route to MULTI_HOP."""
    planner = AgentPlanner()
    plan = planner.plan(
        "How do software engineering interview expectations differ between Amazon and NVIDIA?"
    )
    assert plan["multi_hop_needed"] is True
    assert "MULTI_HOP_RETRIEVAL" in plan["planned_operations"]


def test_multi_stage_sequence_routes_to_multi_hop():
    """Queries inquiring across sequential interview stages route to MULTI_HOP."""
    planner = AgentPlanner()
    plan = planner.plan(
        "What are the main interview rounds for an Amazon candidate, specifically round 1 and round 2?"
    )
    assert plan["multi_hop_needed"] is True
    assert "MULTI_HOP_RETRIEVAL" in plan["planned_operations"]


def test_screening_and_onsite_routes_to_multi_hop():
    """Queries comparing screening and onsite rounds route to MULTI_HOP."""
    planner = AgentPlanner()
    plan = planner.plan(
        "What is the difference between screening and onsite technical expectations at Stripe?"
    )
    assert plan["multi_hop_needed"] is True
    assert "MULTI_HOP_RETRIEVAL" in plan["planned_operations"]


def test_complex_behavioral_disagreement_breakdown_routes_to_multi_hop():
    """Complex strategy question analyzing manager disagreement and leadership principles routes to MULTI_HOP."""
    planner = AgentPlanner()
    plan = planner.plan(
        "How should I answer an Amazon behavioral question about disagreement with a manager, and what leadership principles does it demonstrate?"
    )
    assert plan["multi_hop_needed"] is True
    assert "MULTI_HOP_RETRIEVAL" in plan["planned_operations"]


def test_decompose_query_tool_creates_clean_company_subqueries():
    """DecomposeQueryTool decomposes cross-company comparisons cleanly per company."""
    tool = DecomposeQueryTool()
    res = tool.execute(
        "Compare Amazon and Google software engineering interviews",
        detected_companies=["Amazon", "Google"],
    )
    assert len(res.output) >= 2
    assert any("Amazon" in q for q in res.output)
    assert any("Google" in q for q in res.output)


def test_decompose_query_tool_does_not_inject_manager_disagreement_for_general_queries():
    """DecomposeQueryTool does not inject irrelevant manager conflict into clean multi-company queries."""
    tool = DecomposeQueryTool()
    res = tool.execute(
        "Compare technical rounds at OpenAI vs Anthropic",
        detected_companies=["OpenAI", "Anthropic"],
    )
    assert all("disagreement with manager" not in q.lower() for q in res.output)
