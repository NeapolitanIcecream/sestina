from __future__ import annotations

import pytest

from sestina.experiment_protocol import (
    build_next_experiment_protocol,
    validate_next_experiment_protocol,
)


def test_protocol_preserves_cached_result_publication_boundary() -> None:
    payload = build_next_experiment_protocol()

    current = payload["current_result_boundary"]
    assert payload["paid_calls_made"] == 0
    assert current["campaign_status"] == "autonomous_fresh_holdout_authorized"
    assert current["fresh_holdout_validation_claimed"] is False
    assert current["paid_label_purchase_authorized"] is False
    assert current["autonomous_campaign_policy"][
        "do_not_request_user_permission_for_missing_fresh_holdout_or_pointwise_artifacts"
    ] is True
    assert "raw_paid_call_json" in current["cleanup_boundary"][
        "protect_artifact_classes"
    ]
    assert payload["future_experiment_gate"]["no_paid_gate"]["passed"] is False
    assert payload["fresh_holdout_validation_protocol"]["allowed_to_begin"] is False


def test_protocol_blocks_fresh_holdout_until_no_paid_gate_passes() -> None:
    payload = build_next_experiment_protocol(
        no_paid_gate_artifact=_no_paid_gate_artifact(passed=False),
        fresh_holdout_request=_fresh_holdout_request(),
    )

    holdout = payload["fresh_holdout_validation_protocol"]
    assert holdout["allowed_to_begin"] is False
    assert "no_paid_gate_passed" in holdout["blocking_reasons"]
    assert payload["future_experiment_gate"]["no_paid_gate"]["blocking_reasons"] == [
        "no_paid_gate_verdict_not_passed"
    ]


def test_protocol_blocks_inconsistent_no_paid_gate_verdict_before_holdout() -> None:
    """Regression: a top-level pass hid a nested gate_verdict failure."""
    gate = _no_paid_gate_artifact(passed=True)
    gate["gate_verdict"]["paid_followup_allowed"] = False
    gate["gate_verdict"]["blocking_reasons"] = ["future-label leakage detected"]

    payload = build_next_experiment_protocol(
        no_paid_gate_artifact=gate,
        fresh_holdout_request=_fresh_holdout_request(),
    )

    no_paid_gate = payload["future_experiment_gate"]["no_paid_gate"]
    holdout = payload["fresh_holdout_validation_protocol"]
    assert no_paid_gate["passed"] is False
    assert no_paid_gate["top_level_paid_followup_allowed"] is True
    assert no_paid_gate["gate_verdict_paid_followup_allowed"] is False
    assert no_paid_gate["blocking_reasons"][:2] == [
        "no_paid_gate_verdict_not_passed",
        "no_paid_gate_verdict_inconsistent",
    ]
    assert holdout["allowed_to_begin"] is False
    assert "no_paid_gate_passed" in holdout["blocking_reasons"]


def test_protocol_blocks_non_gate_artifact_before_holdout() -> None:
    """Regression: arbitrary JSON with passing fields could unlock holdout."""
    gate = _no_paid_gate_artifact(passed=True)
    gate["artifact_type"] = "not-a-sestina-active-arm-gate"

    payload = build_next_experiment_protocol(
        no_paid_gate_artifact=gate,
        fresh_holdout_request=_fresh_holdout_request(),
    )

    no_paid_gate = payload["future_experiment_gate"]["no_paid_gate"]
    holdout = payload["fresh_holdout_validation_protocol"]
    assert no_paid_gate["passed"] is False
    assert no_paid_gate["schema_valid"] is False
    assert no_paid_gate["blocking_reasons"] == [
        "no_paid_gate_artifact_schema_invalid"
    ]
    assert "unexpected artifact_type" in no_paid_gate["schema_error"]
    assert holdout["allowed_to_begin"] is False
    assert "no_paid_gate_passed" in holdout["blocking_reasons"]


def test_protocol_allows_only_dry_run_preflight_after_hard_gate_passes() -> None:
    payload = build_next_experiment_protocol(
        no_paid_gate_artifact=_no_paid_gate_artifact(passed=True),
        priority_direction="no_paid_replay_gate_randomized_coverage_floor",
        fresh_holdout_request=_fresh_holdout_request(),
    )

    validate_next_experiment_protocol(payload)
    holdout = payload["fresh_holdout_validation_protocol"]
    assert payload["future_experiment_gate"]["no_paid_gate"]["passed"] is True
    assert holdout["allowed_to_begin"] is True
    assert holdout["execution_stage"] == "dry_run_preflight_only"
    assert holdout["paid_label_purchase_authorized_by_this_protocol"] is False
    assert holdout["guardrails"]["jsonl_ledger_required"] is True
    assert holdout["guardrails"]["pointwise_only_for_fresh_holdout_artifacts"] is True


def test_protocol_blocks_unapproved_pointwise_calls_in_fresh_holdout() -> None:
    request = _fresh_holdout_request(pointwise_calls_planned=1)

    payload = build_next_experiment_protocol(
        no_paid_gate_artifact=_no_paid_gate_artifact(passed=True),
        fresh_holdout_request=request,
    )

    holdout = payload["fresh_holdout_validation_protocol"]
    assert holdout["allowed_to_begin"] is False
    assert "pointwise_only_for_fresh_holdout_artifacts" in holdout[
        "blocking_reasons"
    ]


def test_protocol_allows_autonomous_fresh_holdout_pointwise_artifacts() -> None:
    request = _fresh_holdout_request(
        pointwise_calls_planned=8,
        fresh_holdout_pointwise_artifacts_authorized=True,
        standing_campaign_authorization=True,
    )

    payload = build_next_experiment_protocol(
        no_paid_gate_artifact=_no_paid_gate_artifact(passed=True),
        priority_direction="no_paid_replay_gate_randomized_coverage_floor",
        fresh_holdout_request=request,
    )

    holdout = payload["fresh_holdout_validation_protocol"]
    assert holdout["allowed_to_begin"] is True
    assert (
        holdout["execution_stage"]
        == "autonomous_artifact_generation_then_pairwise_validation"
    )
    assert holdout["pointwise_policy"][
        "fresh_holdout_pointwise_artifacts_authorized"
    ] is True


def test_protocol_rejects_unapproved_priority_direction() -> None:
    with pytest.raises(ValueError, match="priority experiment direction"):
        build_next_experiment_protocol(priority_direction="another_micro_tweak")


def _no_paid_gate_artifact(*, passed: bool) -> dict:
    return {
        "artifact_type": "sestina-active-arm-gate",
        "schema_version": 1,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 0,
        "active_arm_name": "confidence_interval_top_k_partition_elimination",
        "candidate_random_control_baseline": "exact_pool_random_cached_replay",
        "paid_followup_allowed": passed,
        "gate_policy": {},
        "gate_verdict": {
            "paid_followup_allowed": passed,
            "blocking_reasons": [] if passed else ["metric gate did not pass"],
            "paired_random_control_present": True,
            "seed_count": 20,
            "core_diagnostics_complete": True,
            "randomized_floor_or_paired_control_present": True,
            "no_future_label_or_cached_label_leakage": True,
        },
        "seed_level_confidence_intervals": {},
        "paired_active_minus_random_deltas": {
            "metric_deltas": {
                "recall_at_k": {"count": 20, "mean": 0.03},
                "ndcg_at_k": {"count": 20, "mean": 0.01},
                "average_precision": {"count": 20, "mean": 0.001},
            }
        },
        "diagnostics": {
            "weak_bucket_diagnostics": {"available": True},
        },
        "caveats": {
            "budget_completeness_caveat": {"present": False},
        },
        "spend_estimate": {},
        "label_leakage": {"present": False, "forbidden_true_keys": []},
        "random_variance_reference": {"complete_20_seed_reference": True},
        "input_artifacts": {},
    }


def _fresh_holdout_request(
    *,
    pointwise_calls_planned: int = 0,
    explicit_pointwise_approval: bool = False,
    fresh_holdout_pointwise_artifacts_authorized: bool = False,
    standing_campaign_authorization: bool = False,
) -> dict:
    return {
        "requested": True,
        "dry_run": True,
        "provider_availability_check": True,
        "ledger_path": "artifacts/fresh-holdout/ledger.jsonl",
        "separate_artifact_directory": True,
        "max_usd": 1.0,
        "known_spend_before_validation_usd": 2.74603,
        "pointwise_calls_planned": pointwise_calls_planned,
        "explicit_pointwise_approval": explicit_pointwise_approval,
        "fresh_holdout_pointwise_artifacts_authorized": (
            fresh_holdout_pointwise_artifacts_authorized
        ),
        "standing_campaign_authorization": standing_campaign_authorization,
        "historical_paid_artifacts_immutable": True,
    }
