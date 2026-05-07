from __future__ import annotations

import pytest

from scripts.run_ci_partition_gate import validate_ci_gate_artifact_schema
from sestina.ci_partition_gate import (
    CIPartitionConfig,
    ReliabilityAwareCIPartitionV2Config,
    confidence_interval_partition,
    replay_ci_partition_gate,
    replay_reliability_aware_ci_partition_v2_gate,
    schedule_ci_partition_pairs,
    schedule_reliability_aware_ci_partition_v2_pairs,
)
from sestina.models import PairwiseComparison
from sestina.scheduler import PairwiseBudget


def test_ci_partition_keeps_noisy_boundary_items_unresolved(paper_set) -> None:
    papers = paper_set(5)
    comparisons = [
        PairwiseComparison(
            left_id="p1",
            right_id="p5",
            winner="left",
            soft_probability=0.7,
            confidence=0.6,
        ),
        PairwiseComparison(
            left_id="p2",
            right_id="p3",
            winner="tie",
            soft_probability=0.52,
            confidence=0.5,
        ),
    ]

    state = confidence_interval_partition(
        papers,
        comparisons,
        k=2,
        config=CIPartitionConfig(prior_strength=1.0, confidence_z=1.64),
    )

    assert state.unresolved_count > 0
    assert state.kth_lower_bound < state.best_outside_upper_bound
    assert "p3" in state.unresolved_ids
    assert state.intervals["p1"].effective_pairwise_n == pytest.approx(0.6)
    assert state.intervals["p2"].effective_pairwise_n == pytest.approx(0.175)
    assert state.intervals["p2"].mean > 0.0
    assert state.intervals["p2"].mean < 1.0


def test_ci_scheduler_preserves_randomized_coverage_floor(paper_set) -> None:
    schedule = schedule_ci_partition_pairs(
        paper_set(12),
        [],
        k=3,
        budget=PairwiseBudget(n=12, candidate_size=8, budget=6),
        seed=23,
        config=CIPartitionConfig(
            posterior_samples=300,
            random_floor_fraction=0.4,
            min_random_floor_pairs=1,
            per_item_cap=4,
        ),
    )

    assert len(schedule.pairs) == 6
    assert schedule.diagnostics["acquisition"]["method"] == (
        "ci_partition_elimination"
    )
    assert schedule.diagnostics["coverage"]["random_floor_pairs"] >= 2
    assert schedule.diagnostics["purpose_counts"]["ci_random_coverage_floor"] >= 2
    assert schedule.diagnostics["ci_partition"]["unresolved_count"] > 0
    assert all(pair.order.randomized for pair in schedule.pairs)


def test_ci_replay_uses_only_cached_pairwise_labels(paper_set) -> None:
    papers = paper_set(10)
    cached = {
        ("p1", "p2"): PairwiseComparison(
            left_id="p1",
            right_id="p2",
            winner="left",
            soft_probability=0.8,
            confidence=0.9,
        ),
        ("p3", "p4"): PairwiseComparison(
            left_id="p3",
            right_id="p4",
            winner="right",
            soft_probability=0.75,
            confidence=0.8,
        ),
        ("p5", "p6"): PairwiseComparison(
            left_id="p5",
            right_id="p6",
            winner="uncertain",
            soft_probability=0.5,
            confidence=0.5,
        ),
    }

    replay = replay_ci_partition_gate(
        papers,
        cached,
        k=3,
        budget=PairwiseBudget(n=10, candidate_size=8, budget=5),
        seed=11,
        config=CIPartitionConfig(
            batch_size=2,
            posterior_samples=300,
            random_floor_fraction=0.5,
            min_random_floor_pairs=1,
        ),
    )

    scheduled_keys = {
        tuple(sorted((pair.left_id, pair.right_id))) for pair in replay.schedule
    }
    assert scheduled_keys <= set(cached)
    assert len(replay.comparisons) == len(replay.schedule)
    assert replay.diagnostics["label_policy"]["missing_pairwise_labels"] == 0
    assert replay.diagnostics["label_policy"]["novel_pairs_scheduled"] == 0
    assert replay.diagnostics["available_label_filter"]["cached_pair_keys_total"] == 3


def test_reliability_aware_ci_v2_falls_back_when_boundary_is_unreliable(
    paper_set,
) -> None:
    schedule = schedule_reliability_aware_ci_partition_v2_pairs(
        paper_set(12),
        [],
        k=3,
        budget=PairwiseBudget(n=12, candidate_size=8, budget=6),
        seed=29,
        config=ReliabilityAwareCIPartitionV2Config(
            posterior_samples=300,
            random_floor_fraction=0.2,
            low_reliability_random_floor_fraction=0.5,
            min_random_floor_pairs=1,
            per_item_cap=4,
        ),
    )

    diagnostics = schedule.diagnostics
    assert diagnostics["acquisition"]["method"] == (
        "reliability_aware_ci_partition_v2"
    )
    assert diagnostics["ci_reliability"]["low_reliability_fallback_active"] is True
    assert diagnostics["ci_reliability"]["unresolved_fraction"] >= 0.85
    assert diagnostics["coverage"]["random_floor_pairs"] >= 3
    assert diagnostics["purpose_counts"]["ci_random_coverage_floor"] >= 3
    assert all(
        "ci_v2_pair_reliability" in pair.diagnostics for pair in schedule.pairs
    )


def test_reliability_aware_ci_v2_replay_uses_only_cached_labels(paper_set) -> None:
    papers = paper_set(10)
    cached = {
        ("p1", "p2"): PairwiseComparison(
            left_id="p1",
            right_id="p2",
            winner="left",
            soft_probability=0.8,
            confidence=0.9,
        ),
        ("p3", "p4"): PairwiseComparison(
            left_id="p3",
            right_id="p4",
            winner="right",
            soft_probability=0.75,
            confidence=0.8,
        ),
        ("p5", "p6"): PairwiseComparison(
            left_id="p5",
            right_id="p6",
            winner="uncertain",
            soft_probability=0.5,
            confidence=0.5,
        ),
    }

    replay = replay_reliability_aware_ci_partition_v2_gate(
        papers,
        cached,
        k=3,
        budget=PairwiseBudget(n=10, candidate_size=8, budget=5),
        seed=11,
        config=ReliabilityAwareCIPartitionV2Config(
            batch_size=2,
            posterior_samples=300,
        ),
    )

    scheduled_keys = {
        tuple(sorted((pair.left_id, pair.right_id))) for pair in replay.schedule
    }
    assert scheduled_keys <= set(cached)
    assert len(replay.comparisons) == len(replay.schedule)
    assert replay.diagnostics["method"] == (
        "reliability_aware_ci_partition_v2_cached_replay"
    )
    assert replay.diagnostics["label_policy"]["missing_pairwise_labels"] == 0
    assert (
        replay.diagnostics["label_policy"]["cached_label_values_used_before_scheduling"]
        is False
    )


def test_ci_gate_artifact_schema_requires_gate_metrics() -> None:
    payload = {
        "artifact_type": "sestina-ci-partition-gate-analysis",
        "schema_version": 1,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "gate_verdict": {"paid_followup_allowed": False},
        "gate_criteria": {},
        "aggregate_metrics": {},
        "paired_deltas_vs_exact_pool_random": {},
        "aggregate_diagnostics": {
            "confidence_bound_unresolved_count": {},
            "graph_connectivity": {},
            "oracle_caps": {},
            "unique_future_positives_touched": {},
            "weak_bucket_deltas": {},
        },
        "bucket_results": [],
        "limitations": [],
    }

    validate_ci_gate_artifact_schema(payload)

    broken = dict(payload)
    broken["aggregate_diagnostics"] = {"oracle_caps": {}}
    with pytest.raises(ValueError, match="confidence_bound_unresolved_count"):
        validate_ci_gate_artifact_schema(broken)
