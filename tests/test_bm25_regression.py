import pytest
from unittest.mock import MagicMock
from backend.core.vector_store.base import ScoredChunk
from backend.core.retrieval.bm25 import BM25Retriever


class MockVectorStore:
    def __init__(self, chunks):
        self._chunks = chunks

    def get_all_chunks(self):
        return self._chunks


@pytest.mark.anyio
async def test_bm25_malformed_and_missing_metadata():
    """Verify BM25 handles None, missing fields, or non-dict metadata without throwing errors."""
    chunks = [
        ScoredChunk(chunk_id="c1", text="Google interview algorithms tree graph", metadata=None, score=1.0),
        ScoredChunk(chunk_id="c2", text="Google system design load balancer", metadata={}, score=1.0),
        ScoredChunk(chunk_id="c3", text="Oracle SQL indexing relational queries", metadata={"company": "Oracle"}, score=1.0),
    ]
    vs = MockVectorStore(chunks)
    retriever = BM25Retriever(vector_store=vs)

    # Retrieval with company filter when some chunks have metadata=None or empty dict
    res = await retriever.retrieve("Google tree algorithms", top_k=5, metadata_filters={"company": "Google"})
    assert res.retriever_type == "bm25"
    # Should not raise AttributeError: 'NoneType' object has no attribute 'get'


@pytest.mark.anyio
async def test_bm25_fallback_branch_pollution_fix():
    """Verify that when company filter fallback succeeds, unfiltered keyword matching does NOT pollute candidates."""
    chunks = [
        ScoredChunk(chunk_id="c_target_1", text="Google software engineer dynamic programming", metadata={"company": "Google", "role": "sde"}, score=1.0),
        ScoredChunk(chunk_id="c_target_2", text="Google distributed consensus raft", metadata={"company": "Google", "role": "sde"}, score=1.0),
        ScoredChunk(chunk_id="c_other_1", text="Amazon dynamic programming leetcode practice", metadata={"company": "Amazon", "role": "sde"}, score=1.0),
        ScoredChunk(chunk_id="c_other_2", text="Meta dynamic programming algorithms", metadata={"company": "Meta", "role": "sde"}, score=1.0),
    ]
    vs = MockVectorStore(chunks)
    retriever = BM25Retriever(vector_store=vs)

    # Filter with non-matching role but matching company: compound filter {"company": "Google", "role": "Data Engineer"}
    # Hits 0 on compound filter, then triggers hierarchical fallback to company-only {"company": "Google"}
    res = await retriever.retrieve(
        "dynamic programming",
        top_k=5,
        metadata_filters={"company": "Google", "role": "Data Engineer"},
        fallback_unfiltered=True,
    )

    # Both returned chunks MUST be Google chunks!
    # Under the old bug, the else: branch executed over all chunks, appending Amazon and Meta chunks!
    assert len(res.chunks) > 0
    for chunk in res.chunks:
        assert chunk.metadata.get("company") == "Google"
        assert chunk.chunk_id in ["c_target_1", "c_target_2"]
        assert chunk.chunk_id not in ["c_other_1", "c_other_2"]


@pytest.mark.anyio
async def test_bm25_role_alias_normalization():
    """Verify that synonymous role strings like 'Software Development Engineer' match 'sde' chunks."""
    chunks = [
        ScoredChunk(chunk_id="c1", text="SDE coding round questions", metadata={"company": "Google", "role": "sde"}, score=1.0),
        ScoredChunk(chunk_id="c2", text="Product manager behavioral questions", metadata={"company": "Google", "role": "pm"}, score=1.0),
    ]
    vs = MockVectorStore(chunks)
    retriever = BM25Retriever(vector_store=vs)

    res = await retriever.retrieve(
        "coding round",
        top_k=5,
        metadata_filters={"role": "Software Development Engineer"},
    )
    assert len(res.chunks) == 1
    assert res.chunks[0].chunk_id == "c1"
