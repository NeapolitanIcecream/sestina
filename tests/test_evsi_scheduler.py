from __future__ import annotations

from sestina.evsi_scheduler import (
    CCTDGFSchedulerConfig,
    EVSISchedulerConfig,
    SequentialEVSISchedulerConfig,
    schedule_cache_aware_cctd_gf,
    posterior_top_k_predictions,
    schedule_cache_aware_sequential_evsi,
    schedule_evsi_boundary_duels,
    schedule_exact_pool_random,
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


def test_exact_pool_random_samples_from_same_evsi_feasible_pool(
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
    budget = PairwiseBudget(n=len(papers), candidate_size=8, budget=6)
    config = EVSISchedulerConfig(samples=500, per_item_cap=2)

    evsi_schedule = schedule_evsi_boundary_duels(
        papers,
        prior_comparisons,
        k=3,
        budget=budget,
        seed=11,
        config=config,
    )
    random_schedule = schedule_exact_pool_random(
        papers,
        prior_comparisons,
        k=3,
        budget=budget,
        seed=11,
        config=config,
    )

    seen_key = frozenset(("p1", "p2"))
    random_keys = {
        frozenset((pair.left_id, pair.right_id)) for pair in random_schedule.pairs
    }
    degrees = {
        paper.paper_id: sum(
            paper.paper_id in {pair.left_id, pair.right_id}
            for pair in random_schedule.pairs
        )
        for paper in papers
    }
    assert seen_key not in random_keys
    assert len(random_schedule.pairs) == budget.budget
    assert random_schedule.diagnostics["pairs_considered"] == (
        evsi_schedule.diagnostics["pairs_considered"]
    )
    assert random_schedule.diagnostics["unique_pairs_considered"] == (
        evsi_schedule.diagnostics["unique_pairs_considered"]
    )
    assert max(degrees.values()) <= config.per_item_cap
    assert random_schedule.diagnostics["acquisition"]["method"] == "exact_pool_random"
    assert random_schedule.diagnostics["evsi_score_distribution"][
        "zero_score_rate"
    ] >= 0.0


def test_sequential_evsi_reveals_cached_labels_only_after_batch_selection(
    paper_set,
) -> None:
    papers = paper_set(10)
    revealed_pairs: list[tuple[str, str]] = []

    def reveal_after_selection(pair) -> PairwiseComparison | None:
        revealed_pairs.append(tuple(sorted((pair.left_id, pair.right_id))))
        if len(revealed_pairs) <= 2:
            return PairwiseComparison(
                left_id=pair.left_id,
                right_id=pair.right_id,
                winner="left",
                confidence=1.0,
                order=pair.order,
            )
        return None

    schedule = schedule_cache_aware_sequential_evsi(
        papers,
        [],
        reveal_comparison=reveal_after_selection,
        k=3,
        budget=PairwiseBudget(n=len(papers), candidate_size=8, budget=6),
        seed=31,
        config=SequentialEVSISchedulerConfig(
            evsi=EVSISchedulerConfig(samples=500, per_item_cap=3),
            rounds=3,
            batch_size=2,
            stop_on_novel=True,
        ),
    )

    assert len(schedule.pairs) == 4
    assert len(revealed_pairs) == 4
    assert len(set(revealed_pairs)) == 4
    assert schedule.diagnostics["cached_label_revealed_total"] == 2
    assert schedule.diagnostics["novel_pairs_total"] == 2
    assert schedule.diagnostics["stopped_on_novel"] is True
    assert schedule.diagnostics["batch_history"][0]["comparisons_before_batch"] == 0
    assert schedule.diagnostics["batch_history"][0]["comparisons_after_batch"] == 2
    assert schedule.diagnostics["batch_history"][1]["comparisons_before_batch"] == 2
    assert "top_k_entropy_reduction" in schedule.diagnostics["batch_history"][0]


def test_cctd_gf_uses_round_mix_and_cache_reveals_after_each_batch(
    paper_set,
) -> None:
    papers = paper_set(16)
    revealed_pairs: list[tuple[str, str]] = []

    def reveal_all_selected(pair) -> PairwiseComparison:
        revealed_pairs.append(tuple(sorted((pair.left_id, pair.right_id))))
        return PairwiseComparison(
            left_id=pair.left_id,
            right_id=pair.right_id,
            winner="left",
            confidence=1.0,
            order=pair.order,
        )

    schedule = schedule_cache_aware_cctd_gf(
        papers,
        [],
        reveal_comparison=reveal_all_selected,
        k=4,
        budget=PairwiseBudget(n=len(papers), candidate_size=10, budget=20),
        seed=41,
        config=CCTDGFSchedulerConfig(
            evsi=EVSISchedulerConfig(samples=256, per_item_cap=6),
            stop_on_novel=True,
        ),
    )

    purpose_counts = schedule.diagnostics["purpose_counts"]
    keys = {frozenset((pair.left_id, pair.right_id)) for pair in schedule.pairs}
    assert len(schedule.pairs) == 20
    assert len(keys) == 20
    assert len(revealed_pairs) == 20
    assert purpose_counts == {
        "cctd_gf_disagreement": 12,
        "cctd_gf_graph_floor": 4,
        "cctd_gf_random_floor": 4,
    }
    assert schedule.diagnostics["acquisition"]["method"] == "cctd_gf"
    assert len(schedule.diagnostics["batch_history"]) == 4
    assert all(pair.order.randomized for pair in schedule.pairs)
    assert any(
        pair.diagnostics["top_k_disagreement"] > 0.0
        for pair in schedule.pairs
        if pair.purpose == "cctd_gf_disagreement"
    )
    assert any(
        pair.diagnostics["cross_component"] for pair in schedule.pairs
    )
    assert "cctd_gf_score_distribution" in schedule.diagnostics


def test_cctd_gf_stops_after_batch_with_novel_pair(
    paper_set,
) -> None:
    papers = paper_set(16)

    def reveal_none(pair) -> None:
        return None

    schedule = schedule_cache_aware_cctd_gf(
        papers,
        [],
        reveal_comparison=reveal_none,
        k=4,
        budget=PairwiseBudget(n=len(papers), candidate_size=10, budget=20),
        seed=43,
        config=CCTDGFSchedulerConfig(
            evsi=EVSISchedulerConfig(samples=256, per_item_cap=6),
            stop_on_novel=True,
        ),
    )

    assert len(schedule.pairs) == 5
    assert schedule.diagnostics["purpose_counts"] == {
        "cctd_gf_disagreement": 3,
        "cctd_gf_graph_floor": 1,
        "cctd_gf_random_floor": 1,
    }
    assert schedule.diagnostics["stopped_on_novel"] is True
    assert schedule.diagnostics["batch_history"][0]["novel_pairs_total"] == 5
