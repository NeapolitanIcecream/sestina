from __future__ import annotations

import pytest

from scripts.adjudicate_new_information_caveat import (
    ARTIFACT_TYPE,
    build_row_diagnostics,
    validate_caveat_adjudication_artifact_schema,
)


def test_row_diagnostics_explain_lost_touch_despite_selected_gain() -> None:
    payload = {
        "bucket_results": [
            {
                "seed": 17,
                "buckets": [
                    {
                        "bucket": "bucket_a",
                        "k": 2,
                        "arms": {
                            "new_information_challenger_cached_replay": _arm(
                                selected_ids=["p_good_1", "p_good_2"],
                                touched_ids=["p_good_2"],
                                pointwise_plus_cap=1.0,
                                pair_cap=1.0,
                                observed_cap=0.5,
                                posterior_ids=["p_good_1", "p_good_2"],
                                false_negative_rows=[],
                                fallback_selected_total=1,
                            ),
                            "exact_pool_random_cached_replay": _arm(
                                selected_ids=["p_good_1"],
                                touched_ids=["p_good_1", "p_good_2"],
                                pointwise_plus_cap=1.0,
                                pair_cap=1.0,
                                observed_cap=1.0,
                                posterior_ids=["p_good_1", "p_bad"],
                                false_negative_rows=[
                                    {
                                        "paper_id": "p_good_2",
                                        "pair_degree": 1,
                                        "posterior_top_k_score": 0.2,
                                    }
                                ],
                                fallback_selected_total=0,
                            ),
                        },
                    }
                ],
            }
        ]
    }

    rows = build_row_diagnostics(payload)

    assert len(rows) == 1
    row = rows[0]
    assert row["selected_positive_delta"] == 1
    assert row["touch_deltas"]["unique_future_positives_touched"] == -1
    assert row["lost_future_positive_touch_ids"] == ["p_good_1"]
    assert row["lost_future_positive_details"][0][
        "selected_by_new_information"
    ] is True
    assert row["new_information_scheduler"]["fallback_selected_total"] == 1


def test_caveat_adjudication_schema_requires_zero_paid_calls() -> None:
    payload = _minimal_artifact()
    payload["paid_calls_made"] = 1

    with pytest.raises(ValueError, match="zero-paid"):
        validate_caveat_adjudication_artifact_schema(payload)


def test_caveat_adjudication_schema_rejects_unknown_decision() -> None:
    payload = _minimal_artifact()
    payload["decision"] = "maybe"

    with pytest.raises(ValueError, match="invalid decision"):
        validate_caveat_adjudication_artifact_schema(payload)


def _arm(
    *,
    selected_ids: list[str],
    touched_ids: list[str],
    pointwise_plus_cap: float,
    pair_cap: float,
    observed_cap: float,
    posterior_ids: list[str],
    false_negative_rows: list[dict],
    fallback_selected_total: int,
) -> dict:
    return {
        "metrics": {
            "posterior_topk": {
                "recall_at_k": len(selected_ids) / 2,
                "ndcg_at_k": len(selected_ids) / 2,
                "average_precision": len(selected_ids) / 2,
            }
        },
        "top_k_error_decomposition": {
            "selected_positive_count": len(selected_ids),
            "selected_future_positive_ids": selected_ids,
            "posterior_top_k_ids": posterior_ids,
            "false_negative_rows": false_negative_rows,
        },
        "positive_exposure": {
            "touched_future_positive_ids": touched_ids,
            "unique_future_positives_touched": len(touched_ids),
            "pairs_touching_future_positive": len(touched_ids),
            "positive_negative_pairs": len(touched_ids),
        },
        "oracle_bounds": {
            "pointwise_top_k_positive_ids": ["p_good_1"],
            "pointwise_plus_touched_positive_upper_bound": {
                "recall_cap": pointwise_plus_cap,
                "recoverable_positive_ids": ["p_good_1", "p_good_2"],
            },
            "positive_negative_pair_label_oracle_upper_bound": {
                "recall_cap": pair_cap,
                "recoverable_positive_ids": touched_ids,
            },
            "observed_positive_winner_upper_bound": {
                "recall_cap": observed_cap,
                "recoverable_positive_ids": touched_ids,
            },
        },
        "scheduler_diagnostics": {
            "cached_frontier_fallback": {"selected_total": fallback_selected_total},
            "new_information_challenger": {
                "primary_scheduled_pairwise_shortfall": fallback_selected_total
            },
            "purpose_counts": {},
        },
    }


def _minimal_artifact() -> dict:
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": 1,
        "decision": "caveat_accepted_with_constraints",
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 0,
        "method": {},
        "input_artifacts": {},
        "diagnostics": {
            "active_gate_summary": {},
            "dry_run_summary": {},
            "replay_gate_tension": {},
            "row_classification_summary": {},
            "per_bucket_summary": {},
            "lost_positive_touch_summary": {},
            "selected_positive_source_summary": {},
            "ranking_gain_summary": {},
            "fallback_sensitivity": {},
            "prior_arm_oracle_predictiveness": {},
        },
        "rationale": ["fixture"],
        "constraints": ["fixture"],
        "recommended_next_workflow": "fixture",
        "validation_commands": [],
        "limitations": [],
    }
