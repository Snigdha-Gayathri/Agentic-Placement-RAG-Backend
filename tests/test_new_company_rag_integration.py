"""Integration and retrieval tests for the 30 newly added company interview datasets."""

from __future__ import annotations

import asyncio
import pytest
from pathlib import Path

from backend.app.models import ChatRequest
from backend.app.service import SecureRAGService, build_request_context
from backend.core.retrieval.router import QueryRouter
from backend.ingestion.metadata_extractor import MetadataExtractor
from backend.ingestion.pipeline import ChunkingConfig, IngestionPipeline


NEW_COMPANIES = [
    "OpenAI", "Anthropic", "Cohere", "Perplexity", "xAI", "Databricks",
    "Snowflake", "Scale AI", "LangChain", "Replit", "Weights & Biases",
    "Harvey", "Runway", "Glean", "Together AI", "Groq", "Salesforce",
    "ServiceNow", "Palantir", "Stripe", "Atlassian", "AMD", "Intel",
    "Qualcomm", "PayPal", "Mastercard", "Spotify", "ByteDance", "Tencent", "Alibaba"
]

EXISTING_COMPANIES = [
    "Amazon", "Google", "Microsoft", "Meta", "NVIDIA", "Oracle", "Adobe"
]


def test_metadata_extractor_recognizes_new_and_existing_companies():
    extractor = MetadataExtractor()

    # Test new companies from markdown filenames and headers
    for comp in NEW_COMPANIES:
        safe = comp.replace(" ", "_").replace("&", "_")
        filename = f"{safe}_interview_intelligence.md"
        text = f"# {comp} — Company-Specific Interview Intelligence\n\n## COMPANY_TECHNICAL_FOCUS\nTechnical domains: LLM, RAG"
        meta = extractor.extract(filename, text=text)
        assert meta.company.lower() == comp.lower(), f"Expected {comp}, got {meta.company}"
        assert meta.source_type == "interview_intelligence"
        assert "rag" in meta.tags or "llm" in meta.tags or "interview_intelligence" in meta.tags

    # Test existing companies from PDF filenames
    meta_nvidia = extractor.extract("NVIDIA.pdf")
    assert meta_nvidia.company == "NVIDIA"

    meta_amazon = extractor.extract("Amazon_DSA.pdf")
    assert meta_amazon.company == "Amazon"


def test_query_router_single_and_multi_company():
    router = QueryRouter()

    # Single new company routing
    dec_openai = router.route("Give me OpenAI LLM agent interview questions.")
    assert dec_openai.metadata_filters.get("company") == "OpenAI"
    assert dec_openai.retriever_type == "hybrid"

    dec_anthropic = router.route("What RAG interview questions should I expect at Anthropic?")
    assert dec_anthropic.metadata_filters.get("company") == "Anthropic"

    # Single existing company routing
    dec_nvidia = router.route("What LLM system design questions could I face at NVIDIA?")
    assert dec_nvidia.metadata_filters.get("company") == "NVIDIA"

    # Multi-company comparison routing
    dec_multi = router.route("Compare the interview focus of OpenAI, Anthropic and Databricks.")
    assert "company" not in dec_multi.metadata_filters  # No restrictive single-company filter
    assert dec_multi.needs_multi_hop is True


@pytest.mark.anyio
async def test_hybrid_retrieval_for_new_and_existing_companies():
    svc = SecureRAGService()

    # Test OpenAI retrieval
    openai_res = await svc.hybrid_retriever.retrieve(
        "OpenAI LLM agent and inference optimization",
        top_k=5,
        metadata_filters={"company": "OpenAI"},
    )
    assert len(openai_res.chunks) >= 1
    assert openai_res.chunks[0].metadata.get("company") == "OpenAI"

    # Test Anthropic retrieval
    anthropic_res = await svc.hybrid_retriever.retrieve(
        "Anthropic RAG and prompt evaluations",
        top_k=5,
        metadata_filters={"company": "Anthropic"},
    )
    assert len(anthropic_res.chunks) >= 1
    assert anthropic_res.chunks[0].metadata.get("company") == "Anthropic"

    # Test NVIDIA (existing company) retrieval
    nvidia_res = await svc.hybrid_retriever.retrieve(
        "NVIDIA GPU architectures and system design",
        top_k=5,
        metadata_filters={"company": "NVIDIA"},
    )
    assert len(nvidia_res.chunks) >= 1
    assert nvidia_res.chunks[0].metadata.get("company") == "NVIDIA"


@pytest.mark.anyio
async def test_end_to_end_chat_pipeline_execution():
    svc = SecureRAGService()

    queries_to_verify = [
        ("Give me OpenAI LLM agent interview questions.", "OpenAI"),
        ("What RAG interview questions should I expect at Anthropic?", "Anthropic"),
        ("Give me agent reliability interview questions for Databricks.", "Databricks"),
        ("What LLM system design questions could I face at NVIDIA?", "NVIDIA"),
        ("What interview questions are relevant for an AI engineer at Cohere?", "Cohere"),
    ]

    for query, expected_company in queries_to_verify:
        ctx = build_request_context()
        req = ChatRequest(query=query)
        resp = await svc.process(req, ctx)
        assert resp is not None
        assert resp.pipeline_data is not None

        ret_info = resp.pipeline_data.retrieval_info
        chunks = ret_info.get("retrieved_chunks", [])
        assert len(chunks) >= 1, f"Expected safe chunks for {query}"
        assert chunks[0]["metadata"]["company"].lower() == expected_company.lower()


def test_chroma_stats_includes_all_companies():
    svc = SecureRAGService()
    stats = svc.chroma_store.get_stats()
    indexed_companies = stats.metadata_distribution.get("companies", {})
    assert stats.total_chunks >= 340

    for comp in NEW_COMPANIES:
        assert comp in indexed_companies, f"New company {comp} missing in Chroma index"

    for comp in EXISTING_COMPANIES:
        assert comp in indexed_companies, f"Existing company {comp} missing in Chroma index"
