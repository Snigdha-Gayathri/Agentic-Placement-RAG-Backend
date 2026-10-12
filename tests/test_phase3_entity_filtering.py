"""Unit tests for Phase 3 Step 1: Entity Extraction & Metadata Pre-Filtering with Fallback."""

import asyncio
from backend.core.retrieval.router import QueryRouter
from backend.core.agent.planner import AgentPlanner
from backend.core.retrieval.bm25 import BM25Retriever
from backend.core.retrieval.hybrid import HybridRetriever
from backend.core.vector_store.base import ScoredChunk


class MockVectorStore:
    """Mock vector store with sample chunks for BM25 and fallback testing."""

    def __init__(self, chunks: list[ScoredChunk]):
        self._chunks = chunks

    def get_all_chunks(self) -> list[ScoredChunk]:
        return self._chunks

    def search(self, query_embedding, top_k=10, metadata_filters=None, fallback_unfiltered=True):
        if metadata_filters:
            matched = [
                c for c in self._chunks
                if all(str(c.metadata.get(k, "")).lower() == str(v).lower() for k, v in metadata_filters.items())
            ]
            if matched:
                return matched[:top_k]
            if fallback_unfiltered:
                return self._chunks[:top_k]
            return []
        return self._chunks[:top_k]


def sample_chunks():
    return [
        ScoredChunk(
            chunk_id="chk_google_sde_01",
            text="Google SDE interview focuses on binary search, trees, and dynamic programming.",
            metadata={"company": "Google", "role": "Software Development Engineer", "topic": "dsa"},
            score=1.0,
        ),
        ScoredChunk(
            chunk_id="chk_anthropic_mle_01",
            text="Anthropic ML Engineer interviews emphasize transformer attention, KV cache, and alignment.",
            metadata={"company": "Anthropic", "role": "Machine Learning Engineer", "topic": "ai_ml"},
            score=1.0,
        ),
        ScoredChunk(
            chunk_id="chk_meta_sys_01",
            text="Meta System Design round covers distributed caching, rate limiting, and feed architecture.",
            metadata={"company": "Meta", "role": "Software Development Engineer", "topic": "system_design"},
            score=1.0,
        ),
    ]


def test_role_entity_extraction_router():
    """Verify QueryRouter extracts canonical roles correctly."""
    router = QueryRouter()

    dec1 = router.route("What topics are emphasized in Anthropic ML Engineer interviews?")
    assert "Anthropic" in dec1.detected_companies
    assert "Machine Learning Engineer" in dec1.detected_roles
    assert dec1.metadata_filters.get("company") == "Anthropic"
    assert dec1.metadata_filters.get("role") == "Machine Learning Engineer"

    dec2 = router.route("Google SDE coding round dynamic programming questions")
    assert "Google" in dec2.detected_companies
    assert "Software Development Engineer" in dec2.detected_roles
    assert dec2.metadata_filters.get("company") == "Google"
    assert dec2.metadata_filters.get("role") == "Software Development Engineer"


def test_router_does_not_hijack_company_topic_questions():
    """Verify company questions mentioning topics are NOT hijacked by METADATA_DIRECT."""
    router = QueryRouter()
    queries = [
        "What technical interview questions and machine learning topics are asked at OpenAI?",
        "What topics and rounds are emphasized in Anthropic ML Engineer interviews?",
        "What GPU architecture, CUDA, or high-performance computing topics are asked at NVIDIA?",
    ]
    for q in queries:
        dec = router.route(q)
        assert dec.route_type != "METADATA_DIRECT", f"Query '{q}' was hijacked as METADATA_DIRECT!"
        assert dec.llm_required is True


def test_planner_role_entity_extraction():
    """Verify AgentPlanner extracts roles and sets metadata filters."""
    planner = AgentPlanner()
    profile = planner.analyze_query_profile("What distributed systems questions are asked for Anthropic ML Engineer?")
    assert "Anthropic" in profile["detected_companies"]
    assert "Machine Learning Engineer" in profile["detected_roles"]

    plan = planner.plan("What distributed systems questions are asked for Anthropic ML Engineer?")
    assert plan["metadata_filters"].get("company") == "Anthropic"
    assert plan["metadata_filters"].get("role") == "Machine Learning Engineer"


def test_bm25_prefilter_fallback():
    """Verify BM25 pre-filtering returns matches when present, and falls back to unfiltered when 0 hits."""
    async def _run():
        chunks = sample_chunks()
        vs = MockVectorStore(chunks)
        bm25 = BM25Retriever(vs)

        # 1. Matching filter: returns only Anthropic chunks
        res_match = await bm25.retrieve(
            "machine learning transformer",
            top_k=5,
            metadata_filters={"company": "Anthropic"},
            fallback_unfiltered=True,
        )
        assert len(res_match.chunks) > 0
        assert all(c.metadata["company"] == "Anthropic" for c in res_match.chunks)

        # 2. Non-matching filter with fallback=True: returns chunks via unfiltered fallback
        res_fallback = await bm25.retrieve(
            "binary search trees dynamic programming",
            top_k=5,
            metadata_filters={"company": "NonExistentCompanyXYZ"},
            fallback_unfiltered=True,
        )
        assert len(res_fallback.chunks) > 0, "BM25 should fall back to unfiltered search when filter yields 0 hits"

        # 3. Non-matching filter with fallback=False: returns 0 chunks
        res_no_fallback = await bm25.retrieve(
            "binary search trees dynamic programming",
            top_k=5,
            metadata_filters={"company": "NonExistentCompanyXYZ"},
            fallback_unfiltered=False,
        )
        assert len(res_no_fallback.chunks) == 0

    asyncio.run(_run())


def test_hybrid_prefilter_fallback():
    """Verify Hybrid retriever falls back to unfiltered fused chunks when post-filter yields 0 hits."""
    async def _run():
        chunks = sample_chunks()
        vs = MockVectorStore(chunks)
        bm25 = BM25Retriever(vs)

        class MockDenseRetriever:
            async def retrieve(self, query, top_k=10, metadata_filters=None, fallback_unfiltered=True):
                from backend.core.retrieval.base import RetrievalResult
                res_chunks = vs.search(None, top_k=top_k, metadata_filters=metadata_filters, fallback_unfiltered=fallback_unfiltered)
                return RetrievalResult(chunks=res_chunks, retriever_type="dense", latency_ms=1.0)

        dense = MockDenseRetriever()
        hybrid = HybridRetriever(dense, bm25)

        # Filter for non-existent company with fallback=True
        res = await hybrid.retrieve(
            "distributed caching and system design",
            top_k=5,
            metadata_filters={"company": "NonExistentCompanyXYZ"},
            fallback_unfiltered=True,
        )
        assert len(res.chunks) > 0, "Hybrid should fall back to unfiltered fused chunks when filter has 0 hits"

    asyncio.run(_run())
