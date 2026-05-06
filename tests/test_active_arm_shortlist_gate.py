from __future__ import annotations

import pytest

from scripts.run_active_arm_shortlist_gate import (
    build_shortlist_gate_study,
    validate_shortlist_gate_artifact_schema,
)


def test_shortlist_gate_evaluates_only_methodologically_valid_ci_artifact() -> None:
    payload = build_shortlist_gate_study(
        ci_partition_artifact=_ci_partition_artifact(
            recall_deltas=[0.0] * 20,
            ndcg_deltas=[-0.01] * 20,
            ap_deltas=[-0.02] * 20,
        ),
        random_variance_artifact=_random_reference_artifact(),
        random_control_gap_artifact=_random_control_gap_artifact(),
        pairwise_strength_artifact=_posterior_tweak_artifact(
            artifact_type="sestina-pairwise-strength-calibration-analysis",
            tweak_strategy="soft_strength_calibrated_posterior_topk",
        ),
        posterior_decision_artifact=_posterior_tweak_artifact(
            artifact_type="sestina-posterior-decision-shrinkage-analysis",
            tweak_strategy="degree_shrunk_posterior_topk",
        ),
        ci_partition_artifact_path="artifacts/ci.json",
        random_variance_artifact_path="artifacts/random.json",
        created_at="2026-05-02",
    )

    validate_shortlist_gate_artifact_schema(payload)
    assert payload["paid_calls_made"] == 0
    assert payload["paid_spend_usd"] == 0.0
    assert payload["pointwise_calls_made"] == 0
    assert payload["summary"]["any_candidate_paid_followup_allowed"] is False

    candidates = {candidate["id"]: candidate for candidate in payload["candidates"]}
    ci_candidate = candidates["reliability_aware_ci_partition_v2"]
    assert ci_candidate["evaluated_with_active_arm_gate"] is True
    assert ci_candidate["active_arm_gate"]["seed_count"] == 20
    assert ci_candidate["active_arm_gate"]["paid_followup_allowed"] is False
    assert ci_candidate["paid_followup_allowed"] is False

    challenger = candidates["new_information_challenger_construction"]
    assert challenger["evaluated_with_active_arm_gate"] is False
    assert challenger["status"] == "blocked_missing_prerequisite"
    assert "active_arm_gate_reason_not_run" in challenger["gate_inputs"]
    assert challenger["evidence_summary"]["posterior_topk_metrics"][
        "targeted_outsider_random"
    ]["recall_at_k"] == 0.325

    aggregation = candidates["aggregation_cross_check_standard_ranking_models"]
    assert aggregation["evaluated_with_active_arm_gate"] is False
    assert aggregation["evidence_summary"]["pairwise_strength_calibration"][
        "recall_improved_complete_label_arms"
    ] == []

    harness = candidates["active_arm_simulator_harness_gate_integration"]
    assert harness["status"] == "infrastructure_ready_no_paid_arm"
    assert harness["paid_followup_allowed"] is False
    assert harness["evidence_summary"]["reports_spend_estimate"] is True


def test_shortlist_gate_keeps_contextless_candidates_blocked() -> None:
    payload = build_shortlist_gate_study(
        ci_partition_artifact=_ci_partition_artifact(
            recall_deltas=[0.0] * 20,
            ndcg_deltas=[0.0] * 20,
            ap_deltas=[0.0] * 20,
        ),
        random_variance_artifact=_random_reference_artifact(),
        random_control_gap_artifact=None,
        pairwise_strength_artifact=None,
        posterior_decision_artifact=None,
        created_at="2026-05-02",
    )

    candidates = {candidate["id"]: candidate for candidate in payload["candidates"]}
    assert candidates["new_information_challenger_construction"][
        "evidence_summary"
    ] == {"available": False}
    assert candidates["aggregation_cross_check_standard_ranking_models"][
        "evidence_summary"
    ]["pairwise_strength_calibration"] == {"available": False}
    assert payload["summary"]["status_counts"]["blocked_missing_prerequisite"] == 2


def test_shortlist_gate_schema_requires_candidate_contract() -> None:
    payload = build_shortlist_gate_study(
        ci_partition_artifact=_ci_partition_artifact(
            recall_deltas=[0.0] * 20,
            ndcg_deltas=[0.0] * 20,
            ap_deltas=[0.0] * 20,
        ),
        random_variance_artifact=_random_reference_artifact(),
        created_at="2026-05-02",
    )
    broken = dict(payload)
    broken["candidates"] = [dict(payload["candidates"][0])]
    broken["candidates"][0].pop("blocking_reasons")

    with pytest.raises(ValueError, match="blocking_reasons"):
        validate_shortlist_gate_artifact_schema(broken)


def _ci_partition_artifact(
    *,
    recall_deltas: list[float],
    ndcg_deltas: list[float],
    ap_deltas: list[float],
) -> dict:
    active_arm = "ci_partition_elimination"
    random_arm = "exact_pool_random_cached_replay"
    seeds = [str(index + 1) for index in range(len(recall_deltas))]
    return {
        "artifact_type": "sestina-ci-partition-gate-analysis",
        "schema_version": 1,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "aggregate_diagnostics": {
            "confidence_bound_unresolved_count": {
                active_arm: {"mean": 79.25},
                random_arm: {"mean": 79.25},
            },
            "graph_connectivity": {
                active_arm: {"mean_future_positive_degree": 2.0},
                random_arm: {"mean_future_positive_degree": 1.7},
            },
            "oracle_caps": {
                active_arm: {
                    "mean_pointwise_plus_touched_recall_cap": 0.57625,
                    "mean_positive_negative_pair_recall_cap": 0.5275,
                    "mean_observed_positive_winner_recall_cap": 0.49,
                },
                random_arm: {
                    "mean_pointwise_plus_touched_recall_cap": 0.56625,
                    "mean_positive_negative_pair_recall_cap": 0.55875,
                    "mean_observed_positive_winner_recall_cap": 0.535,
                },
            },
            "randomized_coverage": {
                active_arm: {"random_floor_rate": 0.2, "random_floor_pairs": 640},
                random_arm: {"random_floor_rate": 1.0, "random_floor_pairs": 3200},
            },
            "unique_future_positives_touched": {
                active_arm: {"total": 461},
                random_arm: {"total": 447},
            },
            "weak_bucket_deltas": {
                "row_count": 160,
                "mean_pointwise_plus_touched_recall_cap_delta": 0.01,
                "mean_positive_negative_pair_recall_cap_delta": -0.03125,
            },
        },
        "paired_deltas_vs_exact_pool_random": {
            "comparison_arm": active_arm,
            "reference_arm": random_arm,
            "seed_deltas": {
                seed: {
                    "recall_at_k": recall,
                    "ndcg_at_k": ndcg,
                    "average_precision": ap,
                }
                for seed, recall, ndcg, ap in zip(
                    seeds,
                    recall_deltas,
                    ndcg_deltas,
                    ap_deltas,
                    strict=True,
                )
            },
            "bucket_deltas": [],
        },
        "bucket_results": [
            {
                "seed": 1,
                "buckets": [
                    {
                        "bucket": "bucket-a",
                        "arms": {
                            active_arm: {
                                "comparison_source": {
                                    "missing_pairwise_labels": 0,
                                    "partial": False,
                                }
                            },
                            random_arm: {
                                "comparison_source": {
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


def _random_reference_artifact() -> dict:
    interval = {
        "count": 20,
        "mean": 0.3225,
        "normal_approx_95_ci": [0.31, 0.33625],
        "bootstrap_percentile_95_ci": [0.31, 0.33625],
    }
    return {
        "artifact_type": "sestina-full-random-variance-completion",
        "schema_version": 1,
        "paid_calls_made": 1727,
        "paid_spend_usd": 1.269345,
        "analysis_parameters": {"seed_count": 20},
        "aggregate_metrics": {
            "exact_pool_random_full_schedule": {
                "seed_level_intervals": {
                    "recall_at_k": interval,
                    "ndcg_at_k": {**interval, "mean": 0.362799},
                    "average_precision": {**interval, "mean": 0.373689},
                }
            },
            "historical_random_full_schedule": {
                "seed_level_intervals": {
                    "recall_at_k": {**interval, "mean": 0.3325},
                    "ndcg_at_k": {**interval, "mean": 0.366567},
                    "average_precision": {**interval, "mean": 0.368876},
                }
            },
        },
        "full_schedule_completion_status": {
            "all_seed_bucket_rows_complete": True,
        },
    }


def _random_control_gap_artifact() -> dict:
    return {
        "artifact_type": "sestina-random-control-gap-analysis",
        "schema_version": 1,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "aggregate_metrics": {
            "exact_pool_random": {
                "posterior_topk": {
                    "recall_at_k": 0.375,
                    "ndcg_at_k": 0.40468725,
                    "average_precision": 0.38183604,
                }
            },
            "expanded_pool_random": {
                "posterior_topk": {
                    "recall_at_k": 0.325,
                    "ndcg_at_k": 0.37424611,
                    "average_precision": 0.37732137,
                }
            },
            "targeted_outsider_random": {
                "posterior_topk": {
                    "recall_at_k": 0.325,
                    "ndcg_at_k": 0.37188598,
                    "average_precision": 0.38615959,
                }
            },
        },
    }


def _posterior_tweak_artifact(
    *,
    artifact_type: str,
    tweak_strategy: str,
) -> dict:
    return {
        "artifact_type": artifact_type,
        "schema_version": 1,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "aggregate_metrics": {
            "historical_random": {
                "posterior_topk": {
                    "recall_at_k": 0.375,
                    "ndcg_at_k": 0.41209593,
                    "average_precision": 0.40757871,
                },
                tweak_strategy: {
                    "recall_at_k": 0.375,
                    "ndcg_at_k": 0.41023802,
                    "average_precision": 0.40624469,
                },
            },
            "exact_pool_random": {
                "posterior_topk": {
                    "recall_at_k": 0.375,
                    "ndcg_at_k": 0.40468725,
                    "average_precision": 0.38183604,
                },
                tweak_strategy: {
                    "recall_at_k": 0.375,
                    "ndcg_at_k": 0.40468725,
                    "average_precision": 0.38865072,
                },
            },
        },
    }
