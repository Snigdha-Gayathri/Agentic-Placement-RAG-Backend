"""Deterministic unit tests for corrected evaluation metrics.

Covers:
1. Exact NDCG calculation, ensuring score is always in [0.0, 1.0].
2. The specific Phase 1 bug where |GT|=1 but multiple retrieved items matched topic keywords.
3. Precision@k and Recall@k denominators, including fewer than k retrieved results.
4. MRR ranking logic for first genuinely relevant chunk.
5. Deduplication of retrieved chunks so duplicate items cannot inflate scores.
6. Separation of unanswerable queries: ranking metrics return None, abstention is tracked.
7. Aggregate metric calculations with proper answerable/unanswerable denominators.
"""

import math
import pytest
from eval.metrics import (
    compute_dcg,
    compute_idcg,
    compute_ranking_metrics,
    check_abstention,
    aggregate_metrics,
    deduplicate_chunk_ids,
)


def test_ndcg_bounded_between_zero_and_one():
    """Verify NDCG is strictly in [0.0, 1.0] under all rankings."""
    gt = ["c1", "c2", "c3"]

    # Perfect ranking
    res_perfect = compute_ranking_metrics(["c1", "c2", "c3"], gt)
    assert res_perfect["ndcg@1"] == 1.0
    assert res_perfect["ndcg@3"] == 1.0
    assert res_perfect["ndcg@5"] == 1.0

    # Partial ranking: hits at 1 and 3
    # DCG@5 = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG@5 = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1.0 + 0.63092975 + 0.5 = 2.13092975
    # NDCG@5 = 1.5 / 2.13092975 = 0.7039
    res_partial = compute_ranking_metrics(["c1", "other1", "c2"], gt)
    assert 0.7038 <= res_partial["ndcg@5"] <= 0.7040

    # Zero hits
    res_zero = compute_ranking_metrics(["other1", "other2"], gt)
    assert res_zero["ndcg@1"] == 0.0
    assert res_zero["ndcg@5"] == 0.0
    assert res_zero["mrr"] == 0.0


def test_ndcg_phase1_bug_scenario():
    """Replicate the exact Phase 1 bug condition and verify NDCG never exceeds 1.0.
    
    In Phase 1, if ground truth had 1 item, but 3 retrieved items were flagged relevant,
    the broken evaluator produced NDCG > 1.0 (e.g. 1.0384 or 2.13).
    With the corrected chunk ground truth, the relevant set is well-defined.
    """
    gt = ["c1"]
    # Retrieved 5 items where c1 is at rank 1, rest are non-relevant
    res = compute_ranking_metrics(["c1", "c2", "c3", "c4", "c5"], gt)
    assert res["ndcg@5"] == 1.0
    assert res["recall@5"] == 1.0
    assert res["precision@5"] == 0.2

    # Even if ground truth has 1 item and it's at rank 2:
    # DCG@5 = 1/log2(3) = 0.6309, IDCG@5 = 1/log2(2) = 1.0 -> NDCG@5 = 0.6309
    res_rank2 = compute_ranking_metrics(["other", "c1", "other2"], gt)
    assert 0.6308 <= res_rank2["ndcg@5"] <= 0.6310
    assert res_rank2["ndcg@5"] <= 1.0


def test_precision_and_recall_denominators():
    """Verify Precision@k denominator is fixed k, and Recall@k denominator is |GT|."""
    gt = ["c1", "c2", "c3", "c4"]

    # Case A: 3 items retrieved, 2 are relevant
    res_fewer = compute_ranking_metrics(["c1", "other", "c2"], gt, k_values=(1, 3, 5))
    # At k=5: 2 hits out of k=5 -> precision@5 = 2/5 = 0.4
    assert res_fewer["precision@5"] == 0.4
    # At k=5: 2 hits out of |GT|=4 -> recall@5 = 2/4 = 0.5
    assert res_fewer["recall@5"] == 0.5
    # At k=3: 2 hits out of k=3 -> precision@3 = 2/3 = 0.6667
    assert res_fewer["precision@3"] == 0.6667

    # Case B: 5 items retrieved, all 4 relevant are in top 5
    res_all = compute_ranking_metrics(["c1", "c2", "c3", "c4", "other"], gt, k_values=(1, 5))
    assert res_all["precision@5"] == 0.8  # 4 / 5
    assert res_all["recall@5"] == 1.0     # 4 / 4


def test_mrr_calculation():
    """Verify MRR identifies the reciprocal rank of the first relevant chunk."""
    gt = ["target_chunk"]

    # Rank 1 hit
    assert compute_ranking_metrics(["target_chunk", "b", "c"], gt)["mrr"] == 1.0
    # Rank 2 hit
    assert compute_ranking_metrics(["a", "target_chunk", "c"], gt)["mrr"] == 0.5
    # Rank 3 hit
    assert compute_ranking_metrics(["a", "b", "target_chunk"], gt)["mrr"] == 0.3333
    # No hit
    assert compute_ranking_metrics(["a", "b", "c"], gt)["mrr"] == 0.0


def test_deduplication_prevents_score_inflation():
    """Verify that duplicate chunks in retrieved candidate list cannot inflate hits."""
    gt = ["chunk_target"]
    # Retrieved list contains the same target chunk 3 times
    retrieved_with_dups = ["chunk_target", "chunk_target", "chunk_target"]
    res = compute_ranking_metrics(retrieved_with_dups, gt, k_values=(1, 3, 5))

    # Hits must be 1, NOT 3
    assert res["precision@3"] == round(1.0 / 3.0, 4)
    assert res["recall@3"] == 1.0
    assert res["ndcg@3"] == 1.0


def test_unanswerable_queries_not_inflated():
    """Verify unanswerable queries (0 GT items) return None for ranking metrics."""
    gt_empty = []
    retrieved = ["chunk_a", "chunk_b"]

    res = compute_ranking_metrics(retrieved, gt_empty)
    # Must NOT be 1.0! Ranking metrics are not applicable
    assert res["hit_rate@5"] is None
    assert res["precision@5"] is None
    assert res["recall@5"] is None
    assert res["ndcg@5"] is None
    assert res["mrr"] is None
    assert res["total_relevant_ground_truth"] == 0


def test_abstention_check():
    """Verify check_abstention categorizes answers accurately."""
    # Correct abstention
    res_correct_abs = check_abstention(
        "I cannot answer because the material does not contain sufficient evidence.",
        should_abstain=True,
    )
    assert res_correct_abs["correct"] is True
    assert res_correct_abs["status"] == "CORRECT_ABSTENTION"

    # False answer (hallucinated on unanswerable query)
    res_false_ans = check_abstention(
        "Here are 5 common interview questions asked at Figma...",
        should_abstain=True,
    )
    assert res_false_ans["correct"] is False
    assert res_false_ans["status"] == "FALSE_ANSWER"

    # Inappropriate abstention (abstained on answerable query)
    res_inapp = check_abstention(
        "The indexed corpus does not contain enough evidence to answer this.",
        should_abstain=False,
    )
    assert res_inapp["correct"] is False
    assert res_inapp["status"] == "INAPPROPRIATE_ABSTENTION"

    # Correct answer
    res_good_ans = check_abstention(
        "Based on Amazon placement materials, candidates prepare using the STAR format.",
        should_abstain=False,
    )
    assert res_good_ans["correct"] is True
    assert res_good_ans["status"] == "CORRECT_ANSWER"


def test_aggregate_metrics_separation():
    """Verify aggregate_metrics separates answerable and unanswerable denominators."""
    records = [
        # Answerable query 1 (Hit at rank 1)
        {
            "query_id": "q1",
            "should_abstain": False,
            "metrics": compute_ranking_metrics(["c1"], ["c1"]),
            "abstention": check_abstention("Answer about c1", should_abstain=False),
            "ragas_metrics": {"faithfulness": 0.9, "answer_relevancy": 0.8},
            "latency_ms": 100.0,
        },
        # Answerable query 2 (Miss)
        {
            "query_id": "q2",
            "should_abstain": False,
            "metrics": compute_ranking_metrics(["other"], ["c2"]),
            "abstention": check_abstention("Answer about c2", should_abstain=False),
            "ragas_metrics": {"faithfulness": 0.7, "answer_relevancy": 0.8},
            "latency_ms": 200.0,
        },
        # Unanswerable query 3 (Correctly abstained)
        {
            "query_id": "q3",
            "should_abstain": True,
            "metrics": compute_ranking_metrics(["random_chunk"], []),
            "abstention": check_abstention("Insufficient evidence to answer", should_abstain=True),
            "ragas_metrics": {"faithfulness": 1.0, "answer_relevancy": 0.5},
            "latency_ms": 50.0,
        },
    ]

    agg = aggregate_metrics(records)

    # Denominators
    assert agg["total_queries"] == 3
    assert agg["answerable_queries_count"] == 2
    assert agg["unanswerable_queries_count"] == 1

    # Answerable retrieval metrics: macro-averaged over N_answerable=2, NOT 3!
    # q1 MRR=1.0, q2 MRR=0.0 -> mean MRR = 0.50
    assert agg["retrieval_metrics"]["count"] == 2
    assert agg["retrieval_metrics"]["mrr"] == 0.5
    # q1 HitRate@5=1.0, q2 HitRate@5=0.0 -> mean HitRate@5 = 0.50
    assert agg["retrieval_metrics"]["hit_rate@5"] == 0.5

    # Abstention metrics: 1 unanswerable query, 1 correct abstention -> accuracy = 1.0
    assert agg["abstention_metrics"]["abstention_accuracy"] == 1.0
    assert agg["abstention_metrics"]["false_answer_count"] == 0
