from __future__ import annotations

import pytest

from sestina.active_arm_gate import (
    build_active_arm_gate,
    validate_active_arm_gate_artifact_schema,
)


def test_gate_allows_paid_followup_when_mean_margin_clears_policy() -> None:
    payload = build_active_arm_gate(
        _active_artifact(
            recall_deltas=[0.6] + ([0.0] * 19),
            ndcg_deltas=[0.0] * 20,
            ap_deltas=[0.0] * 20,
        ),
        _random_reference_artifact(),
        active_artifact_path="artifacts/active.json",
        random_variance_artifact_path="artifacts/random.json",
    )

    assert payload["paid_followup_allowed"] is True
    assert payload["gate_verdict"]["mean_margin_gate_passed"] is True
    assert payload["gate_verdict"]["blocking_reasons"] == []
    validate_active_arm_gate_artifact_schema(payload)


def test_gate_allows_paid_followup_when_recall_delta_is_credibly_positive() -> None:
    payload = build_active_arm_gate(
        _active_artifact(
            recall_deltas=[0.01] * 20,
            ndcg_deltas=[-0.02] * 20,
            ap_deltas=[-0.02] * 20,
        ),
        _random_reference_artifact(),
    )

    assert payload["paid_followup_allowed"] is True
    assert payload["gate_verdict"]["credible_recall_gate_passed"] is True


def test_gate_blocks_when_secondary_metrics_drop_under_mean_margin_path() -> None:
    payload = build_active_arm_gate(
        _active_artifact(
            recall_deltas=[0.6] + ([0.0] * 19),
            ndcg_deltas=[-0.001] * 20,
            ap_deltas=[0.0] * 20,
        ),
        _random_reference_artifact(),
    )

    assert payload["paid_followup_allowed"] is False
    assert "nDCG/AP deltas are not both nonnegative" in (
        payload["gate_verdict"]["blocking_reasons"]
    )


def test_gate_blocks_mean_margin_when_missing_label_caveat_is_present() -> None:
    payload = build_active_arm_gate(
        _active_artifact(
            recall_deltas=[0.6] + ([0.0] * 19),
            ndcg_deltas=[0.0] * 20,
            ap_deltas=[0.0] * 20,
            missing_pairwise_labels=1,
        ),
        _random_reference_artifact(),
    )

    assert payload["paid_followup_allowed"] is False
    assert payload["caveats"]["missing_label_caveat"]["present"] is True
    assert "missing-label caveat is present" in (
        payload["gate_verdict"]["blocking_reasons"]
    )


def test_gate_blocks_when_active_schedule_is_under_resolved_budget() -> None:
    """Regression: under-budget rows passed because cached labels were complete."""
    payload = build_active_arm_gate(
        _active_artifact(
            recall_deltas=[0.6] + ([0.0] * 19),
            ndcg_deltas=[0.0] * 20,
            ap_deltas=[0.0] * 20,
            active_scheduled_pairwise_total=16,
            resolved_pairwise_budget=20,
        ),
        _random_reference_artifact(),
    )

    assert payload["paid_followup_allowed"] is False
    assert payload["caveats"]["budget_completeness_caveat"]["present"] is True
    assert payload["caveats"]["budget_completeness_caveat"][
        "active_budget_shortfall"
    ] == 4
    assert "budget-completeness caveat is present" in (
        payload["gate_verdict"]["blocking_reasons"]
    )


def test_gate_blocks_without_paired_random_control() -> None:
    payload = build_active_arm_gate(
        _active_artifact(
            recall_deltas=[0.6] + ([0.0] * 19),
            ndcg_deltas=[0.0] * 20,
            ap_deltas=[0.0] * 20,
            include_paired_deltas=False,
        ),
        _random_reference_artifact(),
    )

    assert payload["paid_followup_allowed"] is False
    assert "paired random-control deltas are unavailable" in (
        payload["gate_verdict"]["blocking_reasons"]
    )


def test_gate_blocks_non_random_paired_control_baseline() -> None:
    """Regression: any paired payload was accepted as a random control."""
    payload = build_active_arm_gate(
        _active_artifact(
            recall_deltas=[0.01] * 20,
            ndcg_deltas=[0.0] * 20,
            ap_deltas=[0.0] * 20,
            random_arm="pointwise_only_not_random",
        ),
        _random_reference_artifact(),
    )

    assert payload["paid_followup_allowed"] is False
    assert (
        payload["paired_active_minus_random_deltas"][
            "random_control_baseline_is_approved"
        ]
        is False
    )
    assert (
        "paired control baseline is not an approved random/exact-pool random "
        "control"
    ) in payload["gate_verdict"]["blocking_reasons"]


def test_gate_blocks_non_random_paired_control_even_with_random_override() -> None:
    """Regression: a caller override could relabel non-random paired deltas."""
    payload = build_active_arm_gate(
        _active_artifact(
            recall_deltas=[0.01] * 20,
            ndcg_deltas=[0.0] * 20,
            ap_deltas=[0.0] * 20,
            random_arm="pointwise_only_not_random",
        ),
        _random_reference_artifact(),
        candidate_random_control_baseline="exact_pool_random_cached_replay",
    )

    assert payload["paid_followup_allowed"] is False
    assert (
        payload["paired_active_minus_random_deltas"][
            "random_control_baseline_is_approved"
        ]
        is False
    )
    assert (
        payload["paired_active_minus_random_deltas"]["random_control_baseline"]
        == "pointwise_only_not_random"
    )
    assert (
        "paired control baseline is not an approved random/exact-pool random "
        "control"
    ) in payload["gate_verdict"]["blocking_reasons"]


def test_gate_blocks_when_selected_random_reference_intervals_are_incomplete() -> None:
    """Regression: a full-random artifact with empty intervals passed."""
    payload = build_active_arm_gate(
        _active_artifact(
            recall_deltas=[0.01] * 20,
            ndcg_deltas=[0.0] * 20,
            ap_deltas=[0.0] * 20,
        ),
        _random_reference_artifact(
            omit_selected_seed_level_intervals={
                "recall_at_k",
                "ndcg_at_k",
                "average_precision",
            }
        ),
    )

    assert payload["paid_followup_allowed"] is False
    assert (
        payload["random_variance_reference"]["complete_20_seed_reference"] is False
    )
    assert payload["random_variance_reference"][
        "missing_required_seed_level_intervals"
    ] == [
        "average_precision",
        "ndcg_at_k",
        "recall_at_k",
    ]
    assert (
        "completed 20-seed full-random variance reference is unavailable"
    ) in payload["gate_verdict"]["blocking_reasons"]


def test_gate_blocks_when_active_input_lacks_explicit_zero_paid_metadata() -> None:
    """Regression: missing paid metadata was treated as zero paid input."""
    payload = build_active_arm_gate(
        _active_artifact(
            recall_deltas=[0.01] * 20,
            ndcg_deltas=[0.0] * 20,
            ap_deltas=[0.0] * 20,
            include_paid_metadata=False,
        ),
        _random_reference_artifact(),
    )

    assert payload["paid_followup_allowed"] is False
    assert payload["gate_verdict"]["no_paid_active_input"] is False
    assert (
        "active gate input is missing explicit zero-paid metadata: "
        "paid_calls_made, paid_spend_usd"
    ) in payload["gate_verdict"]["blocking_reasons"]


def test_gate_blocks_fractional_paid_calls_as_invalid_zero_paid_evidence() -> None:
    """Regression: fractional paid calls were truncated to zero."""
    payload = build_active_arm_gate(
        _active_artifact(
            recall_deltas=[0.01] * 20,
            ndcg_deltas=[0.0] * 20,
            ap_deltas=[0.0] * 20,
            paid_calls_made=0.9,
            paid_spend_usd=0.0,
        ),
        _random_reference_artifact(),
    )

    assert payload["paid_followup_allowed"] is False
    assert payload["gate_verdict"]["no_paid_active_input"] is False
    assert payload["gate_verdict"]["active_paid_metadata"][
        "explicit_zero_paid_evidence"
    ] is False
    assert "paid_calls_made" in payload["gate_verdict"]["active_paid_metadata"][
        "invalid_fields"
    ]
    assert (
        "active gate input has invalid paid metadata: paid_calls_made"
    ) in payload["gate_verdict"]["blocking_reasons"]


def test_gate_blocks_when_paired_metric_deltas_are_incomplete() -> None:
    """Regression: Recall-only paired deltas could pass the gate."""
    payload = build_active_arm_gate(
        _active_artifact(
            recall_deltas=[0.01] * 20,
            ndcg_deltas=[0.0] * 20,
            ap_deltas=[0.0] * 20,
            omit_paired_metrics={"ndcg_at_k", "average_precision"},
        ),
        _random_reference_artifact(),
    )

    assert payload["paid_followup_allowed"] is False
    assert (
        payload["paired_active_minus_random_deltas"][
            "required_metric_deltas_complete"
        ]
        is False
    )
    assert (
        "paired metric deltas are incomplete for required metrics: "
        "average_precision, ndcg_at_k"
    ) in payload["gate_verdict"]["blocking_reasons"]


def test_gate_blocks_future_label_or_cached_label_leakage_markers() -> None:
    """Future labels and cached label values cannot inform scheduling."""
    payload = build_active_arm_gate(
        _active_artifact(
            recall_deltas=[0.01] * 20,
            ndcg_deltas=[0.0] * 20,
            ap_deltas=[0.0] * 20,
            leakage_policy={
                "future_labels_used_for_scheduling": True,
                "uses_future_labels_for_scheduling": True,
                "cached_label_values_used_before_scheduling": True,
            },
        ),
        _random_reference_artifact(),
    )

    assert payload["paid_followup_allowed"] is False
    assert payload["gate_verdict"]["no_future_label_or_cached_label_leakage"] is False
    assert payload["label_leakage"]["forbidden_true_keys"] == [
        "cached_label_values_used_before_scheduling",
        "future_labels_used_for_scheduling",
        "uses_future_labels_for_scheduling",
    ]
    assert (
        "future-label or cached-label leakage markers are true: "
        "cached_label_values_used_before_scheduling, "
        "future_labels_used_for_scheduling, "
        "uses_future_labels_for_scheduling"
    ) in payload["gate_verdict"]["blocking_reasons"]


@pytest.mark.parametrize(
    "marker",
    [
        "future_labels_used_for_scheduling",
        "uses_future_labels_for_scheduling",
        "future_labels_used_as_model_features",
        "future_labels_used_for_model_visible_selection",
        "future_labels_used_in_model_visible_inputs",
        "future_labels_used_for_prompting",
        "future_labels_used_for_routing",
        "uses_future_labels_for_decision",
        "uses_future_labels_for_calibration",
        "future_citation_labels_used_for_scheduling",
        "citation_labels_used_for_scheduling",
        "citation_outcomes_used_for_scheduling",
        "good_paper_used_for_scheduling",
        "matched_title_used_for_scheduling",
        "matched_work_id_used_for_scheduling",
        "cached_label_values_used_before_scheduling",
    ],
)
def test_gate_blocks_repo_emitted_label_leakage_marker_aliases(marker: str) -> None:
    """Regression: producer-specific leakage marker aliases could pass the gate."""
    payload = build_active_arm_gate(
        _active_artifact(
            recall_deltas=[0.01] * 20,
            ndcg_deltas=[0.0] * 20,
            ap_deltas=[0.0] * 20,
            leakage_policy={marker: True},
        ),
        _random_reference_artifact(),
    )

    assert payload["paid_followup_allowed"] is False
    assert payload["gate_verdict"]["no_future_label_or_cached_label_leakage"] is False
    assert payload["label_leakage"]["forbidden_true_keys"] == [marker]


def test_gate_artifact_schema_requires_diagnostics_and_verdict() -> None:
    payload = build_active_arm_gate(
        _active_artifact(
            recall_deltas=[0.6] + ([0.0] * 19),
            ndcg_deltas=[0.0] * 20,
            ap_deltas=[0.0] * 20,
        ),
        _random_reference_artifact(),
    )
    validate_active_arm_gate_artifact_schema(payload)

    broken = dict(payload)
    broken.pop("diagnostics")
    with pytest.raises(ValueError, match="diagnostics"):
        validate_active_arm_gate_artifact_schema(broken)


def _active_artifact(
    *,
    recall_deltas: list[float],
    ndcg_deltas: list[float],
    ap_deltas: list[float],
    random_arm: str = "exact_pool_random_cached_replay",
    missing_pairwise_labels: int = 0,
    include_paired_deltas: bool = True,
    include_paid_metadata: bool = True,
    omit_paired_metrics: set[str] | None = None,
    paid_calls_made: int | float = 0,
    paid_spend_usd: int | float = 0.0,
    active_scheduled_pairwise_total: int = 20,
    random_scheduled_pairwise_total: int = 20,
    resolved_pairwise_budget: int = 20,
    leakage_policy: dict | None = None,
) -> dict:
    active_arm = "active_candidate"
    seeds = [str(index + 1) for index in range(len(recall_deltas))]
    omitted_metrics = omit_paired_metrics or set()
    paired = {
        "comparison_arm": active_arm,
        "reference_arm": random_arm,
        "seed_deltas": {
            seed: _paired_seed_row(
                recall=recall,
                ndcg=ndcg,
                ap=ap,
                omitted_metrics=omitted_metrics,
            )
            for seed, recall, ndcg, ap in zip(
                seeds,
                recall_deltas,
                ndcg_deltas,
                ap_deltas,
                strict=True,
            )
        },
        "metric_deltas": {},
        "bucket_deltas": [],
    }
    payload = {
        "artifact_type": "sestina-active-candidate-analysis",
        "schema_version": 1,
        "analysis_parameters": {"seeds": [int(seed) for seed in seeds]},
        "arms": [
            {"name": active_arm, "randomized_coverage_floor": True},
            {"name": random_arm, "randomized_coverage_floor": True},
        ],
        "aggregate_metrics": {
            active_arm: {
                "seed_count": len(seeds),
                "seed_metric_rows": {
                    seed: {
                        "recall_at_k": 0.3 + recall_deltas[index],
                        "ndcg_at_k": 0.4 + ndcg_deltas[index],
                        "average_precision": 0.4 + ap_deltas[index],
                    }
                    for index, seed in enumerate(seeds)
                },
            },
            random_arm: {"seed_count": len(seeds), "seed_metric_rows": {}},
        },
        "aggregate_diagnostics": {
            "confidence_bound_unresolved_count": {
                active_arm: {"mean": 1.0},
                random_arm: {"mean": 1.0},
            },
            "graph_connectivity": {
                active_arm: {
                    "mean_component_count": 1.0,
                    "mean_future_positive_degree": 2.0,
                },
                random_arm: {"mean_component_count": 1.0},
            },
            "oracle_caps": {
                active_arm: {
                    "mean_pointwise_plus_touched_recall_cap": 0.5,
                    "mean_positive_negative_pair_recall_cap": 0.5,
                    "mean_observed_positive_winner_recall_cap": 0.5,
                },
                random_arm: {
                    "mean_pointwise_plus_touched_recall_cap": 0.5,
                    "mean_positive_negative_pair_recall_cap": 0.5,
                    "mean_observed_positive_winner_recall_cap": 0.5,
                },
            },
            "randomized_coverage": {
                active_arm: {"random_floor_rate": 0.25, "random_floor_pairs": 4},
                random_arm: {"random_floor_rate": 1.0, "random_floor_pairs": 20},
            },
            "unique_future_positives_touched": {
                active_arm: {"mean_touch_rate": 0.5, "total": 10},
                random_arm: {"mean_touch_rate": 0.5, "total": 10},
            },
            "weak_bucket_deltas": {
                "row_count": 1,
                "mean_pointwise_plus_touched_recall_cap_delta": 0.0,
                "mean_positive_negative_pair_recall_cap_delta": 0.0,
                "rows": [],
            },
        },
        "bucket_results": [
            {
                "seed": int(seeds[0]),
                "buckets": [
                    {
                        "bucket": "bucket-a",
                        "budget": {"budget": resolved_pairwise_budget},
                        "arms": {
                            active_arm: {
                                "comparison_source": {
                                    "scheduled_pairwise_total": active_scheduled_pairwise_total,
                                    "cached_pairwise_labels_available": active_scheduled_pairwise_total
                                    - missing_pairwise_labels,
                                    "missing_pairwise_labels": missing_pairwise_labels,
                                    "partial": missing_pairwise_labels > 0,
                                }
                            },
                            random_arm: {
                                "comparison_source": {
                                    "scheduled_pairwise_total": random_scheduled_pairwise_total,
                                    "cached_pairwise_labels_available": random_scheduled_pairwise_total,
                                    "missing_pairwise_labels": 0,
                                    "partial": False,
                                }
                            },
                        },
                    }
                ],
            }
        ],
    }
    if leakage_policy is not None:
        payload["label_policy"] = leakage_policy
    if include_paid_metadata:
        payload["paid_calls_made"] = paid_calls_made
        payload["paid_spend_usd"] = paid_spend_usd
    if include_paired_deltas:
        payload["paired_deltas_vs_exact_pool_random"] = paired
    return payload


def _paired_seed_row(
    *,
    recall: float,
    ndcg: float,
    ap: float,
    omitted_metrics: set[str],
) -> dict:
    row = {
        "recall_at_k": recall,
        "ndcg_at_k": ndcg,
        "average_precision": ap,
    }
    for metric in omitted_metrics:
        row.pop(metric)
    return row


def _random_reference_artifact(
    *,
    omit_selected_seed_level_intervals: set[str] | None = None,
) -> dict:
    metric_interval = {
        "count": 20,
        "mean": 0.3225,
        "stddev": 0.033,
        "standard_error": 0.007,
        "bootstrap_percentile_95_ci": [0.31, 0.33625],
        "normal_approx_95_ci": [0.308, 0.337],
    }
    selected_seed_level_intervals = {
        "recall_at_k": metric_interval,
        "ndcg_at_k": {**metric_interval, "mean": 0.362799},
        "average_precision": {**metric_interval, "mean": 0.373689},
    }
    for metric in omit_selected_seed_level_intervals or set():
        selected_seed_level_intervals.pop(metric, None)
    return {
        "artifact_type": "sestina-full-random-variance-completion",
        "schema_version": 1,
        "paid_calls_made": 1727,
        "paid_spend_usd": 1.269345,
        "analysis_parameters": {
            "seed_count": 20,
            "seeds": [17, 101],
            "interval_unit": (
                "seed-level means across the 8 buckets; bucket rows are not "
                "treated as independent for headline intervals"
            ),
        },
        "aggregate_metrics": {
            "exact_pool_random_full_schedule": {
                "seed_level_intervals": selected_seed_level_intervals,
                "bucket_seed_row_mean": {
                    "recall_at_k": 0.3225,
                    "ndcg_at_k": 0.362799,
                    "average_precision": 0.373689,
                },
            },
            "historical_random_full_schedule": {
                "seed_level_intervals": {
                    "recall_at_k": {**metric_interval, "mean": 0.3325},
                    "ndcg_at_k": {**metric_interval, "mean": 0.366567},
                    "average_precision": {**metric_interval, "mean": 0.368876},
                },
                "bucket_seed_row_mean": {
                    "recall_at_k": 0.3325,
                    "ndcg_at_k": 0.366567,
                    "average_precision": 0.368876,
                },
            },
        },
        "paired_deltas": {
            "metric_delta_intervals": {
                "recall_at_k": {
                    "count": 20,
                    "mean": 0.01,
                    "bootstrap_percentile_95_ci": [-0.01, 0.02875],
                },
                "ndcg_at_k": {
                    "count": 20,
                    "mean": 0.003768,
                    "bootstrap_percentile_95_ci": [-0.01096, 0.018203],
                },
                "average_precision": {
                    "count": 20,
                    "mean": -0.004813,
                    "bootstrap_percentile_95_ci": [-0.014868, 0.004117],
                },
            }
        },
        "full_schedule_completion_status": {
            "all_seed_bucket_rows_complete": True,
            "arms": {
                "exact_pool_random_full_schedule": {
                    "missing_pairwise_labels": 0,
                    "cache_reuse_rate": 1.0,
                },
                "historical_random_full_schedule": {
                    "missing_pairwise_labels": 0,
                    "cache_reuse_rate": 1.0,
                },
            },
        },
    }
