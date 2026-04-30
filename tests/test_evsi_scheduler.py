from __future__ import annotations

from sestina.evsi_scheduler import (
    EVSISchedulerConfig,
    posterior_top_k_predictions,
    schedule_evsi_boundary_duels,
)
from sestina.models import PairwiseComparison, PairwiseOrderMetadata
from sestina.scheduler import PairwiseBudget


def test_posterior_top_k_predictions_use_membership_probability(paper_set) -> None:
    papers = paper_set(5)
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
    ]

    predictions, posterior = posterior_top_k_predictions(
        papers,
        comparisons,
        k=2,
        samples=600,
        seed=7,
    )

    ranked_ids = [prediction.paper_id for prediction in predictions[:2]]
    assert "p4" in ranked_ids
    assert posterior.diagnostics["method"] == "independent_laplace_normal_sampling"
    assert abs(sum(posterior.top_k_probabilities.values()) - 2.0) < 0.15


def test_evsi_scheduler_targets_boundary_challengers_and_skips_seen_pairs(
    paper_set,
) -> None:
    papers = paper_set(12)
    prior_comparisons = [
        PairwiseComparison(
            left_id="p1",
            right_id="p2",
            winner="left",
            order=PairwiseOrderMetadata(seed=1),
        )
    ]

    schedule = schedule_evsi_boundary_duels(
        papers,
        prior_comparisons,
        k=3,
        budget=PairwiseBudget(n=len(papers), candidate_size=8, budget=8),
        seed=11,
        config=EVSISchedulerConfig(samples=500, per_item_cap=4),
    )

    keys = {frozenset((pair.left_id, pair.right_id)) for pair in schedule.pairs}
    assert frozenset(("p1", "p2")) not in keys
    assert len(schedule.pairs) == 8
    assert len(keys) == 8
    assert any(pair.purpose == "evsi_boundary_duel" for pair in schedule.pairs)
    assert any(
        pair.diagnostics["pair_role"] == "incumbent_challenger"
        for pair in schedule.pairs
    )
    assert all(pair.order.randomized for pair in schedule.pairs)
    assert schedule.diagnostics["acquisition"]["method"] == "top_k_evsi_approximation"
    assert schedule.diagnostics["coverage"]["incumbent_challenger_pairs"] > 0


def test_evsi_scheduler_emits_empty_diagnostics_for_no_budget(paper_set) -> None:
    schedule = schedule_evsi_boundary_duels(
        paper_set(3),
        [],
        k=1,
        budget=PairwiseBudget(n=3, candidate_size=3, budget=0),
    )

    assert schedule.pairs == []
    assert schedule.diagnostics["scheduled_total"] == 0
    assert schedule.diagnostics["acquisition"]["method"] == "top_k_evsi_approximation"


def test_evsi_scheduler_uses_configured_calibration_fraction(paper_set) -> None:
    schedule = schedule_evsi_boundary_duels(
        paper_set(12),
        [],
        k=3,
        budget=PairwiseBudget(n=12, candidate_size=8, budget=8),
        seed=11,
        config=EVSISchedulerConfig(
            samples=500,
            calibration_fraction=1.0,
            per_item_cap=4,
        ),
    )

    assert schedule.diagnostics["acquisition"]["calibration_fraction"] == 1.0
    assert schedule.diagnostics["purpose_counts"] == {"calibration_discovery": 8}
