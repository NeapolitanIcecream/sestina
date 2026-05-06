from __future__ import annotations

import pytest

from scripts.analyze_random_variance_replication import (
    schedule_cached_exact_pool_random_replay,
    schedule_cached_historical_random,
    summarize_values,
    validate_random_variance_artifact_schema,
)
from sestina.evsi_scheduler import EVSISchedulerConfig, schedule_exact_pool_random
from sestina.models import Paper, PointwiseAssessment
from sestina.scheduler import PairwiseBudget


def test_cached_exact_pool_replay_matches_current_exact_policy_when_all_pairs_cached() -> None:
    """Regression: cached exact replay must not use the CI partition per-item cap."""
    papers = [
        Paper(
            paper_id=f"p{index}",
            title=f"Paper {index}",
            pointwise=PointwiseAssessment(
                good_probability=0.08 + (index * 0.07),
                uncertainty=0.3,
            ),
        )
        for index in range(12)
    ]
    available = {
        tuple(sorted((left.paper_id, right.paper_id)))
        for left in papers
        for right in papers
        if left.paper_id < right.paper_id
    }
    budget = PairwiseBudget(n=len(papers), candidate_size=len(papers), budget=18)
    config = EVSISchedulerConfig(samples=96, pairwise_strength=2.5)

    exact = schedule_exact_pool_random(
        papers,
        [],
        k=4,
        budget=budget,
        seed=211,
        config=config,
    )
    cached = schedule_cached_exact_pool_random_replay(
        papers,
        [],
        k=4,
        budget=budget,
        seed=211,
        config=config,
        available_pair_keys=available,
    )

    assert [
        (
            pair.left_id,
            pair.right_id,
            pair.order.shown_first_id,
            pair.order.shown_second_id,
        )
        for pair in cached.pairs
    ] == [
        (
            pair.left_id,
            pair.right_id,
            pair.order.shown_first_id,
            pair.order.shown_second_id,
        )
        for pair in exact.pairs
    ]
    assert cached.diagnostics["acquisition"]["selection_policy"] == (
        "schedule_exact_pool_random_after_cached_label_filter"
    )


def test_cached_historical_random_schedules_only_available_pairs() -> None:
    candidate_ids = ["p1", "p2", "p3", "p4"]
    available = {("p1", "p2"), ("p1", "p4"), ("p3", "p4")}

    schedule = schedule_cached_historical_random(
        candidate_ids,
        available_pair_keys=available,
        budget=PairwiseBudget(n=4, candidate_size=4, budget=4),
        seed=37,
    )

    scheduled = {tuple(sorted((pair.left_id, pair.right_id))) for pair in schedule}
    assert len(schedule) == 3
    assert scheduled <= available
    assert all(pair.order.randomized for pair in schedule)


def test_summarize_values_reports_seed_unit_uncertainty() -> None:
    summary = summarize_values(
        [0.30, 0.35, 0.40],
        bootstrap_samples=200,
        bootstrap_seed=13,
    )

    assert summary["count"] == 3
    assert summary["mean"] == pytest.approx(0.35)
    assert summary["standard_error"] > 0.0
    assert summary["normal_approx_95_ci"][0] < summary["mean"]
    assert summary["normal_approx_95_ci"][1] > summary["mean"]
    assert summary["bootstrap_percentile_95_ci"][0] <= summary["mean"]
    assert summary["bootstrap_percentile_95_ci"][1] >= summary["mean"]


def test_random_variance_artifact_schema_requires_cached_replay_sections() -> None:
    payload = {
        "artifact_type": "sestina-random-variance-replication",
        "schema_version": 1,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "analysis_parameters": {},
        "full_schedule_cache_probe": {},
        "cached_replay": {
            "aggregate_metrics": {},
            "paired_deltas": {},
            "seed_results": [],
        },
        "uncertainty_summary": {},
        "recommendation": {},
        "limitations": [],
    }

    validate_random_variance_artifact_schema(payload)

    broken = dict(payload)
    broken["cached_replay"] = {"aggregate_metrics": {}}
    with pytest.raises(ValueError, match="paired_deltas"):
        validate_random_variance_artifact_schema(broken)
