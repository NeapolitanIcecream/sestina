from __future__ import annotations

import pytest

from scripts.run_ci_partition_v2_gate_replay import (
    validate_ci_partition_v2_artifact_schema,
)


def test_ci_partition_v2_artifact_schema_requires_reliability_diagnostics() -> None:
    payload = {
        "artifact_type": "sestina-ci-partition-v2-gate-replay",
        "schema_version": 1,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "active_arm_name": "reliability_aware_ci_partition_v2_cached_replay",
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
            "ci_v2_reliability": {"row_count": 1},
        },
        "bucket_results": [],
        "limitations": [],
        "active_arm_gate": {"paid_followup_allowed": False},
    }

    validate_ci_partition_v2_artifact_schema(payload)

    broken = dict(payload)
    broken["aggregate_diagnostics"] = {
        key: value
        for key, value in payload["aggregate_diagnostics"].items()
        if key != "ci_v2_reliability"
    }
    with pytest.raises(ValueError, match="ci_v2_reliability"):
        validate_ci_partition_v2_artifact_schema(broken)
