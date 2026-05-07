from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from sestina.active_arm_gate import CURRENT_KNOWN_SPEND_USD, DEFAULT_PAID_CAP_USD

ARTIFACT_TYPE = "sestina-next-experiment-protocol"
SCHEMA_VERSION = 1
CURRENT_BEST_RESULT_NAME = "new_information_challenger_cached_replay"
CURRENT_RANDOM_CONTROL_NAME = "exact_pool_random_cached_replay"
PRIORITY_DIRECTIONS = {
    "confidence_interval_top_k_partition_elimination",
    "no_paid_replay_gate_randomized_coverage_floor",
}
REQUIRED_GATE_METRICS = ("recall_at_k", "ndcg_at_k", "average_precision")
REQUIRED_WEAK_BUCKET_DIAGNOSTICS = (
    "pointwise_plus_touched_oracle_cap",
    "positive_negative_pair_oracle_cap",
    "observed_positive_winner_cap",
    "unique_future_positives_touched",
    "graph_connectivity",
    "degree_around_future_positives",
    "degree_around_posterior_top_k",
)
PROTECTED_PUBLICATION_ARTIFACT_CLASSES = (
    "raw_paid_call_json",
    "historical_paid_ledgers",
    "planned_pair_jsonl_manifests",
    "stdout_json",
    "dataset_manifests_with_work_ids",
    "codex_workflow_records",
)


def build_next_experiment_protocol(
    *,
    no_paid_gate_artifact: Mapping[str, Any] | None = None,
    priority_direction: str = "confidence_interval_top_k_partition_elimination",
    fresh_holdout_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an offline protocol artifact for the next Sestina experiment round."""
    no_paid_gate = _no_paid_gate_summary(no_paid_gate_artifact)
    fresh_holdout = _fresh_holdout_protocol(
        no_paid_gate=no_paid_gate,
        request=fresh_holdout_request or {},
    )
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 0,
        "current_result_boundary": _current_result_boundary(),
        "future_experiment_gate": {
            "required_before_any_future_experiment": True,
            "no_paid_gate": no_paid_gate,
            "allowed_next_step": (
                "cleanup_publication_review"
                if not no_paid_gate["passed"]
                else "review_fresh_holdout_dry_run_protocol"
            ),
            "paid_label_purchase_authorized_by_this_protocol": False,
        },
        "priority_experiment_direction": _priority_direction(priority_direction),
        "hard_gate_standards": _hard_gate_standards(),
        "fresh_holdout_validation_protocol": fresh_holdout,
        "reproduction_commands": [
            "uv run python scripts/validate_next_experiment_protocol.py",
            "uv run pytest tests/test_experiment_protocol.py",
            "git diff --check",
        ],
    }
    validate_next_experiment_protocol(payload)
    return payload


def validate_next_experiment_protocol(payload: Mapping[str, Any]) -> None:
    required = {
        "artifact_type",
        "schema_version",
        "paid_calls_made",
        "paid_spend_usd",
        "pointwise_calls_made",
        "current_result_boundary",
        "future_experiment_gate",
        "priority_experiment_direction",
        "hard_gate_standards",
        "fresh_holdout_validation_protocol",
        "reproduction_commands",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            "next experiment protocol missing top-level keys: "
            + ", ".join(missing)
        )
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("next experiment protocol has unexpected artifact_type")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("next experiment protocol has unexpected schema_version")
    if payload.get("paid_calls_made") != 0:
        raise ValueError("next experiment protocol must make zero paid calls")
    if float(payload.get("paid_spend_usd") or 0.0) != 0.0:
        raise ValueError("next experiment protocol must spend zero USD")
    if payload.get("pointwise_calls_made") != 0:
        raise ValueError("next experiment protocol must make zero pointwise calls")

    current = _mapping(payload.get("current_result_boundary"))
    if current.get("campaign_status") != "stopped":
        raise ValueError("current campaign status must remain stopped")
    if current.get("fresh_holdout_validation_claimed") is not False:
        raise ValueError("current cached result cannot be marked fresh validation")
    if current.get("paid_label_purchase_authorized") is not False:
        raise ValueError("current cached result cannot authorize paid labels")

    priority = _mapping(payload.get("priority_experiment_direction"))
    if priority.get("selected") not in PRIORITY_DIRECTIONS:
        raise ValueError("priority experiment direction is not approved")
    if priority.get("randomized_coverage_floor_required") is not True:
        raise ValueError("priority direction must require randomized coverage")

    standards = _mapping(payload.get("hard_gate_standards"))
    if standards.get("primary_metric") != "recall_at_k":
        raise ValueError("Recall@K must remain the primary metric")
    secondary = set(_list(standards.get("secondary_metrics")))
    if secondary != {"ndcg_at_k", "average_precision"}:
        raise ValueError("nDCG@K and AP must remain secondary metrics")
    if int(standards.get("minimum_seed_count") or 0) < 20:
        raise ValueError("hard gate must require at least 20 seeds")
    if standards.get("future_label_leakage_allowed") is not False:
        raise ValueError("future-label leakage must be disallowed")

    gate = _mapping(_mapping(payload.get("future_experiment_gate")).get("no_paid_gate"))
    if gate.get("required_before_any_future_experiment") is not True:
        raise ValueError("future experiments must require a no-paid gate")

    fresh = _mapping(payload.get("fresh_holdout_validation_protocol"))
    if fresh.get("paid_label_purchase_authorized_by_this_protocol") is not False:
        raise ValueError("fresh holdout protocol cannot directly authorize paid labels")
    if fresh.get("allowed_to_begin") is True:
        if gate.get("passed") is not True:
            raise ValueError("fresh holdout cannot begin before no-paid gate passes")
        if fresh.get("guardrails_clear") is not True:
            raise ValueError("fresh holdout cannot begin with guardrail blockers")
        if fresh.get("execution_stage") != "dry_run_preflight_only":
            raise ValueError("fresh holdout starts with dry-run preflight only")


def _current_result_boundary() -> dict[str, Any]:
    return {
        "campaign_status": "stopped",
        "current_best_result": CURRENT_BEST_RESULT_NAME,
        "random_control": CURRENT_RANDOM_CONTROL_NAME,
        "claim_scope": (
            "reviewed internal cached/no-paid result from the budget-filled "
            "new-information replay plus guarded cache-only execution"
        ),
        "fresh_holdout_validation_claimed": False,
        "publication_ready_as_is": False,
        "paid_label_purchase_authorized": False,
        "cleanup_boundary": {
            "public_claim_must_say_cached_no_paid_not_fresh_validation": True,
            "protect_artifact_classes": list(PROTECTED_PUBLICATION_ARTIFACT_CLASSES),
            "publish_sanitized_summaries_before_raw_artifacts": True,
            "do_not_edit_historical_paid_ledgers_or_raw_call_artifacts": True,
            "codex_workflows_are_internal_coordination_records": True,
        },
    }


def _priority_direction(selected: str) -> dict[str, Any]:
    return {
        "selected": selected,
        "approved_values": sorted(PRIORITY_DIRECTIONS),
        "randomized_coverage_floor_required": True,
        "no_future_label_leakage_required": True,
        "description": (
            "Next algorithmic work should be either a confidence-interval "
            "top-K partition/elimination scheduler or a no-paid replay gate "
            "with a randomized coverage floor. It is not a continuation of the "
            "stopped campaign result."
        ),
    }


def _hard_gate_standards() -> dict[str, Any]:
    return {
        "paired_random_or_exact_pool_controls_required": True,
        "minimum_seed_count": 20,
        "primary_metric": "recall_at_k",
        "secondary_metrics": ["ndcg_at_k", "average_precision"],
        "required_metric_deltas": list(REQUIRED_GATE_METRICS),
        "weak_bucket_diagnostics_required": list(REQUIRED_WEAK_BUCKET_DIAGNOSTICS),
        "randomized_coverage_floor_required": True,
        "completed_full_random_variance_reference_required": True,
        "future_label_leakage_allowed": False,
        "cached_label_values_before_scheduling_allowed": False,
        "no_partial_label_aggregate_rows": True,
        "seed_17_standalone_comparator_allowed": False,
    }


def _no_paid_gate_summary(
    artifact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if artifact is None:
        return {
            "required_before_any_future_experiment": True,
            "present": False,
            "passed": False,
            "blocking_reasons": ["no_paid_gate_artifact_missing"],
            "artifact_type": None,
            "paid_calls_made": None,
            "paid_spend_usd": None,
            "pointwise_calls_made": None,
        }

    verdict = _mapping(artifact.get("gate_verdict"))
    deltas = _mapping(artifact.get("paired_active_minus_random_deltas"))
    metric_deltas = _mapping(deltas.get("metric_deltas"))
    diagnostics = _mapping(artifact.get("diagnostics"))
    leakage = _mapping(artifact.get("label_leakage"))
    random_reference = _mapping(artifact.get("random_variance_reference"))
    blocking: list[str] = []

    paid_calls = artifact.get("paid_calls_made")
    paid_spend = artifact.get("paid_spend_usd")
    pointwise_calls = artifact.get("pointwise_calls_made", 0)
    top_level_paid_allowed = artifact.get("paid_followup_allowed")
    verdict_paid_allowed = verdict.get("paid_followup_allowed")
    if paid_calls != 0 or float(paid_spend or 0.0) != 0.0:
        blocking.append("no_paid_gate_made_paid_calls")
    if int(pointwise_calls or 0) != 0:
        blocking.append("no_paid_gate_made_pointwise_calls")
    if top_level_paid_allowed is not True or verdict_paid_allowed is not True:
        blocking.append("no_paid_gate_verdict_not_passed")
    if (
        "paid_followup_allowed" in artifact
        and "paid_followup_allowed" in verdict
        and top_level_paid_allowed != verdict_paid_allowed
    ):
        blocking.append("no_paid_gate_verdict_inconsistent")
    if verdict.get("paired_random_control_present") is not True:
        blocking.append("paired_random_or_exact_pool_control_missing")
    if int(verdict.get("seed_count") or 0) < 20:
        blocking.append("seed_count_below_20")
    if verdict.get("core_diagnostics_complete") is not True:
        blocking.append("weak_bucket_graph_oracle_diagnostics_incomplete")
    if verdict.get("randomized_floor_or_paired_control_present") is not True:
        blocking.append("randomized_floor_or_paired_control_missing")
    if random_reference.get("complete_20_seed_reference") is not True:
        blocking.append("completed_full_random_variance_reference_missing")
    if verdict.get("no_future_label_or_cached_label_leakage") is not True:
        if leakage.get("present") is True:
            blocking.append("future_or_cached_label_leakage_detected")
        else:
            blocking.append("explicit_no_leakage_gate_metadata_missing")

    missing_metrics = sorted(
        metric for metric in REQUIRED_GATE_METRICS if metric not in metric_deltas
    )
    if missing_metrics:
        blocking.append("required_metric_deltas_missing:" + ",".join(missing_metrics))

    return {
        "required_before_any_future_experiment": True,
        "present": True,
        "passed": not blocking,
        "blocking_reasons": blocking,
        "artifact_type": artifact.get("artifact_type"),
        "active_arm_name": artifact.get("active_arm_name"),
        "random_control": artifact.get("candidate_random_control_baseline"),
        "paid_calls_made": paid_calls,
        "paid_spend_usd": paid_spend,
        "pointwise_calls_made": pointwise_calls,
        "top_level_paid_followup_allowed": top_level_paid_allowed,
        "gate_verdict_paid_followup_allowed": verdict_paid_allowed,
        "seed_count": verdict.get("seed_count"),
        "primary_metric": "recall_at_k",
        "secondary_metrics": ["ndcg_at_k", "average_precision"],
        "metric_delta_keys": sorted(metric_deltas),
        "weak_bucket_diagnostics_available": (
            _mapping(diagnostics.get("weak_bucket_diagnostics")).get("available")
            is True
        ),
        "no_future_label_or_cached_label_leakage": verdict.get(
            "no_future_label_or_cached_label_leakage"
        ),
    }


def _fresh_holdout_protocol(
    *,
    no_paid_gate: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    requested = bool(request.get("requested", False))
    pointwise_calls = int(request.get("pointwise_calls_planned", 0) or 0)
    pointwise_approved = bool(request.get("explicit_pointwise_approval", False))
    ledger_path = str(request.get("ledger_path") or "")
    max_usd = float(request.get("max_usd", 0.0) or 0.0)
    guardrails = {
        "no_paid_gate_passed": no_paid_gate.get("passed") is True,
        "dry_run_required": request.get("dry_run") is True,
        "provider_availability_check_required": request.get(
            "provider_availability_check"
        )
        is True,
        "jsonl_ledger_required": bool(ledger_path) and Path(ledger_path).suffix
        == ".jsonl",
        "separate_artifact_directory_required": request.get(
            "separate_artifact_directory"
        )
        is True,
        "max_usd_cap_required": 0.0 < max_usd <= DEFAULT_PAID_CAP_USD,
        "known_spend_preserved": float(
            request.get("known_spend_before_validation_usd", CURRENT_KNOWN_SPEND_USD)
            or 0.0
        )
        == CURRENT_KNOWN_SPEND_USD,
        "zero_pointwise_unless_explicitly_approved": (
            pointwise_calls == 0 or pointwise_approved
        ),
        "historical_paid_artifacts_immutable": request.get(
            "historical_paid_artifacts_immutable"
        )
        is True,
    }
    blockers = sorted(key for key, value in guardrails.items() if not value)
    allowed = requested and not blockers
    return {
        "requested": requested,
        "allowed_to_begin": allowed,
        "execution_stage": "dry_run_preflight_only",
        "guardrails_clear": not blockers,
        "blocking_reasons": blockers,
        "guardrails": guardrails,
        "provider_availability_check": {
            "required_before_any_paid_label_generation": True,
            "allowed_call": "GET /models",
            "label_generation_calls_allowed_in_preflight": 0,
        },
        "ledger": {
            "path": ledger_path or None,
            "format": "jsonl",
            "separate_artifact_directory_required": True,
            "historical_ledgers_rewritten": False,
        },
        "max_usd_cap": {
            "requested_max_usd": max_usd,
            "paid_cap_usd": DEFAULT_PAID_CAP_USD,
            "known_spend_before_validation_usd": CURRENT_KNOWN_SPEND_USD,
        },
        "pointwise_policy": {
            "pointwise_calls_planned": pointwise_calls,
            "explicit_pointwise_approval": pointwise_approved,
            "zero_pointwise_unless_explicitly_approved": True,
        },
        "paid_label_purchase_authorized_by_this_protocol": False,
        "stop_rule": (
            "Stop before any label-generation call if the no-paid gate fails, "
            "dry-run estimate is missing, provider availability is unavailable, "
            "the JSONL ledger is unavailable, the hard max-usd cap would be "
            "exceeded, or an unapproved pointwise call is planned."
        ),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []
