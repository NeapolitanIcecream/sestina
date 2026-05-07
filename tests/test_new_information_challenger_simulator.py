from __future__ import annotations

import pytest

from scripts.run_new_information_challenger_simulator import (
    validate_new_information_artifact_schema,
)


def test_new_information_artifact_schema_requires_core_diagnostics() -> None:
    payload = {
        "artifact_type": "sestina-new-information-challenger-simulator",
        "schema_version": 1,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 0,
        "active_arm_name": "new_information_challenger_cached_replay",
        "candidate_random_control_baseline": "exact_pool_random_cached_replay",
        "gate_verdict": {"paid_followup_allowed": False},
        "aggregate_metrics": {},
        "paired_deltas_vs_exact_pool_random": {"seed_deltas": {"17": {}}},
        "seed_level_metric_intervals": {},
        "aggregate_diagnostics": {
            "confidence_bound_unresolved_count": {},
            "graph_connectivity": {},
            "oracle_caps": {},
            "unique_future_positives_touched": {},
            "weak_bucket_deltas": {},
            "new_information_challenger": {"row_count": 1},
        },
        "bucket_results": [],
        "cache_and_label_caveats": {},
        "budget_fill": _minimal_budget_fill(),
        "limitations": [],
        "active_arm_gate": {"paid_followup_allowed": False},
    }

    validate_new_information_artifact_schema(payload)

    broken = dict(payload)
    broken["aggregate_diagnostics"] = {
        key: value
        for key, value in payload["aggregate_diagnostics"].items()
        if key != "new_information_challenger"
    }
    with pytest.raises(ValueError, match="new_information_challenger"):
        validate_new_information_artifact_schema(broken)


def test_new_information_artifact_schema_blocks_pointwise_calls() -> None:
    payload = {
        "artifact_type": "sestina-new-information-challenger-simulator",
        "schema_version": 1,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 1,
        "active_arm_name": "new_information_challenger_cached_replay",
        "candidate_random_control_baseline": "exact_pool_random_cached_replay",
        "gate_verdict": {"paid_followup_allowed": False},
        "aggregate_metrics": {},
        "paired_deltas_vs_exact_pool_random": {"seed_deltas": {"17": {}}},
        "seed_level_metric_intervals": {},
        "aggregate_diagnostics": {
            "confidence_bound_unresolved_count": {},
            "graph_connectivity": {},
            "oracle_caps": {},
            "unique_future_positives_touched": {},
            "weak_bucket_deltas": {},
            "new_information_challenger": {"row_count": 1},
        },
        "bucket_results": [],
        "cache_and_label_caveats": {},
        "budget_fill": _minimal_budget_fill(),
        "limitations": [],
        "active_arm_gate": {"paid_followup_allowed": False},
    }

    with pytest.raises(ValueError, match="zero pointwise calls"):
        validate_new_information_artifact_schema(payload)


def test_new_information_artifact_schema_requires_budget_shortfall_caveat() -> None:
    """Regression: under-budget rows were accepted as complete cached replays."""
    payload = _minimal_new_information_payload()
    payload["bucket_results"] = [
        {
            "seed": 101,
            "buckets": [
                {
                    "bucket": "under-budget-bucket",
                    "budget": {"budget": 20},
                    "arms": {
                        "new_information_challenger_cached_replay": {
                            "comparison_source": {
                                "scheduled_pairwise_total": 16,
                                "cached_pairwise_labels_available": 16,
                                "missing_pairwise_labels": 0,
                                "partial": False,
                            }
                        },
                        "exact_pool_random_cached_replay": {
                            "comparison_source": {
                                "scheduled_pairwise_total": 20,
                                "cached_pairwise_labels_available": 20,
                                "missing_pairwise_labels": 0,
                                "partial": False,
                            }
                        },
                    },
                }
            ],
        }
    ]

    with pytest.raises(ValueError, match="budget shortfall"):
        validate_new_information_artifact_schema(payload)

    payload["cache_and_label_caveats"]["budget_completeness"] = {
        "present": True,
        "blocking": True,
        "active_budget_shortfall": 4,
        "random_control_budget_shortfall": 0,
        "active_under_budget_rows": 1,
        "random_control_under_budget_rows": 0,
    }
    payload["gate_verdict"]["paid_followup_allowed"] = False
    payload["active_arm_gate"]["paid_followup_allowed"] = False
    validate_new_information_artifact_schema(payload)


def _minimal_new_information_payload() -> dict:
    return {
        "artifact_type": "sestina-new-information-challenger-simulator",
        "schema_version": 1,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 0,
        "active_arm_name": "new_information_challenger_cached_replay",
        "candidate_random_control_baseline": "exact_pool_random_cached_replay",
        "gate_verdict": {"paid_followup_allowed": False},
        "aggregate_metrics": {},
        "paired_deltas_vs_exact_pool_random": {"seed_deltas": {"17": {}}},
        "seed_level_metric_intervals": {},
        "aggregate_diagnostics": {
            "confidence_bound_unresolved_count": {},
            "graph_connectivity": {},
            "oracle_caps": {},
            "unique_future_positives_touched": {},
            "weak_bucket_deltas": {},
            "new_information_challenger": {"row_count": 1},
        },
        "bucket_results": [],
        "cache_and_label_caveats": {},
        "budget_fill": _minimal_budget_fill(),
        "limitations": [],
        "active_arm_gate": {"paid_followup_allowed": False},
    }


def _minimal_budget_fill() -> dict:
    return {
        "method": {},
        "fallback_policy": {},
        "inputs": {},
        "shortfall_summary": {},
        "prior_incomplete_comparison": {},
        "recommendation": "test fixture",
    }
