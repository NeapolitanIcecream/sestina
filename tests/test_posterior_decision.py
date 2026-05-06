from __future__ import annotations

from sestina.aggregation import AggregationResult, PaperEstimate
from sestina.diagnostics import DiagnosticRecorder
from sestina.models import PairwiseComparison
from sestina.posterior import TopKPosterior
from sestina.posterior_decision import (
    SparsePairwiseShrinkageConfig,
    shrunk_top_k_membership_predictions,
    sparse_pairwise_shrunk_top_k_predictions,
)


def test_shrunk_top_k_membership_blends_low_degree_pairwise_signal_toward_prior() -> None:
    prior = TopKPosterior(
        top_k_probabilities={"prior_favorite": 0.70, "sparse_winner": 0.20},
        mean_sampled_rank={"prior_favorite": 1.2, "sparse_winner": 1.8},
        samples=1000,
        diagnostics={"method": "fixture_prior"},
    )
    pairwise = TopKPosterior(
        top_k_probabilities={"prior_favorite": 0.20, "sparse_winner": 0.80},
        mean_sampled_rank={"prior_favorite": 1.7, "sparse_winner": 1.1},
        samples=1000,
        diagnostics={"method": "fixture_pairwise"},
    )
    aggregation = AggregationResult(
        estimates={
            "prior_favorite": PaperEstimate(
                paper_id="prior_favorite",
                prior_logit=1.0,
                posterior_logit=0.6,
                posterior_good_probability=0.65,
                variance=0.4,
                comparisons_used=0,
                comparisons_won=0.0,
                comparisons_lost=0.0,
            ),
            "sparse_winner": PaperEstimate(
                paper_id="sparse_winner",
                prior_logit=0.2,
                posterior_logit=0.9,
                posterior_good_probability=0.71,
                variance=0.6,
                comparisons_used=1,
                comparisons_won=1.0,
                comparisons_lost=0.0,
            ),
        }
    )

    predictions, diagnostics = shrunk_top_k_membership_predictions(
        prior_posterior=prior,
        pairwise_posterior=pairwise,
        pairwise_aggregation=aggregation,
        prior_degree=2.0,
        k=1,
    )

    assert predictions[0].paper_id == "prior_favorite"
    sparse_row = next(
        row
        for row in diagnostics["decision_outputs"]
        if row["paper_id"] == "sparse_winner"
    )
    assert sparse_row["shrinkage_weight"] == 0.33333333
    assert sparse_row["shrunk_top_k_probability"] == 0.4
    assert diagnostics["coverage"]["selected_zero_degree_count"] == 1
    assert diagnostics["top_k_comparison"]["changed_vs_pairwise_topk_count"] == 1
    assert diagnostics["rule_parameters"]["uses_future_labels_for_decision"] is False


def test_sparse_pairwise_shrinkage_emits_empty_decision_diagnostics() -> None:
    recorder = DiagnosticRecorder()

    result = sparse_pairwise_shrunk_top_k_predictions(
        [],
        [],
        k=1,
        config=SparsePairwiseShrinkageConfig(samples=100),
        diagnostics=recorder,
    )

    assert result.predictions == []
    assert result.diagnostics["coverage"]["paper_count"] == 0
    assert recorder.to_dict()["events"][-1]["code"] == (
        "sparse_pairwise_shrinkage_completed"
    )


def test_sparse_pairwise_shrinkage_uses_pairwise_evidence_after_repeated_comparisons(
    paper_set,
) -> None:
    papers = paper_set(4)
    comparisons = [
        PairwiseComparison(
            left_id="p4",
            right_id="p1",
            winner="left",
            soft_probability=0.95,
            confidence=1.0,
        ),
        PairwiseComparison(
            left_id="p4",
            right_id="p2",
            winner="left",
            soft_probability=0.95,
            confidence=1.0,
        ),
        PairwiseComparison(
            left_id="p4",
            right_id="p3",
            winner="left",
            soft_probability=0.95,
            confidence=1.0,
        ),
    ]

    result = sparse_pairwise_shrunk_top_k_predictions(
        papers,
        comparisons,
        k=1,
        config=SparsePairwiseShrinkageConfig(
            prior_degree=1.0,
            pairwise_strength=5.0,
            samples=600,
            seed=7,
        ),
    )

    selected_ids = {
        row["paper_id"]
        for row in result.diagnostics["decision_outputs"]
        if row["selected_by_shrunk_rule"]
    }
    p4_row = next(
        row for row in result.diagnostics["decision_outputs"] if row["paper_id"] == "p4"
    )
    assert "p4" in selected_ids
    assert p4_row["comparisons_used"] == 3
    assert p4_row["shrinkage_weight"] == 0.75
    assert result.diagnostics["coverage"]["compared_paper_count"] == 4
