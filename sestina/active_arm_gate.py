from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

ARTIFACT_TYPE = "sestina-active-arm-gate"
SCHEMA_VERSION = 1
CURRENT_KNOWN_SPEND_USD = 2.746030
DEFAULT_PAID_CAP_USD = 100.0
METRICS = ("recall_at_k", "ndcg_at_k", "average_precision")
RANDOM_VARIANCE_ARTIFACT_TYPE = "sestina-full-random-variance-completion"
APPROVED_RANDOM_CONTROL_NAMES = {
    "exact_pool_random",
    "exact_pool_random_cached_replay",
    "exact_pool_random_full_schedule",
    "historical_random",
    "historical_random_cached_replay",
    "historical_random_full_schedule",
    "pairwise_random",
    "random",
    "random_baseline",
    "random_control",
}
APPROVED_RANDOM_CONTROL_PREFIXES = (
    "exact_pool_random_",
    "historical_random_",
    "pairwise_random_",
    "random_baseline_",
    "random_control_",
)
REQUIRED_TOP_LEVEL_KEYS = {
    "artifact_type",
    "schema_version",
    "paid_calls_made",
    "paid_spend_usd",
    "active_arm_name",
    "candidate_random_control_baseline",
    "paid_followup_allowed",
    "gate_policy",
    "gate_verdict",
    "paired_active_minus_random_deltas",
    "seed_level_confidence_intervals",
    "diagnostics",
    "caveats",
    "spend_estimate",
    "random_variance_reference",
    "input_artifacts",
}
CORE_DIAGNOSTIC_KEYS = {
    "graph_connectivity",
    "oracle_caps",
    "unique_future_positives_touched",
    "weak_bucket_deltas",
}
FORBIDDEN_LABEL_LEAKAGE_TRUE_KEYS = {
    "future_labels_used_for_scheduling",
    "uses_future_labels_for_scheduling",
    "future_labels_used_as_model_features",
    "future_labels_used_for_model_visible_selection",
    "future_labels_used_in_model_visible_inputs",
    "future_labels_used_for_prompting",
    "future_labels_used_for_routing",
    "uses_future_labels_for_decision",
    "uses_future_labels_for_calibration",
    "citation_labels_used_for_scheduling",
    "future_citation_labels_used_for_scheduling",
    "citation_outcomes_used_for_scheduling",
    "good_paper_used_for_scheduling",
    "matched_title_used_for_scheduling",
    "matched_work_id_used_for_scheduling",
    "cached_label_values_used_before_scheduling",
}


@dataclass(frozen=True, slots=True)
class ActiveArmGatePolicy:
    recall_mean_margin: float = 0.025
    confidence_level: float = 0.95
    confidence_z: float = 1.96
    minimum_seed_count: int = 20
    require_no_paid_input: bool = True
    require_full_random_reference: bool = True
    require_paired_random_control: bool = True
    require_randomized_floor_or_paired_control: bool = True
    require_core_diagnostics: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_metric": "recall_at_k",
            "secondary_metrics": ["ndcg_at_k", "average_precision"],
            "recall_mean_margin": self.recall_mean_margin,
            "confidence_level": self.confidence_level,
            "confidence_z": self.confidence_z,
            "minimum_seed_count": self.minimum_seed_count,
            "paid_followup_rule": (
                "Block paid follow-up unless paired active-minus-random Recall@K "
                "is credibly positive, or mean Recall@K improves by at least "
                "0.025 with nonnegative nDCG@K/AP deltas and no missing-label "
                "or budget-completeness caveat. Also require a paired random "
                "control, the completed 20-seed full-random variance reference, "
                "no paid active gate input, and core diagnostics where available."
            ),
            "requires_random_or_exact_pool_random_baseline": True,
            "uses_full_random_variance_reference": True,
            "approved_random_control_baselines": sorted(
                APPROVED_RANDOM_CONTROL_NAMES
            ),
            "approved_random_control_prefixes": list(
                APPROVED_RANDOM_CONTROL_PREFIXES
            ),
            "seed_17_policy": (
                "seed-17 is not a stable standalone comparator; paired seed-level "
                "intervals are required"
            ),
        }


def build_active_arm_gate(
    active_artifact: Mapping[str, Any],
    random_variance_artifact: Mapping[str, Any],
    *,
    active_artifact_path: str | None = None,
    random_variance_artifact_path: str | None = None,
    active_arm_name: str | None = None,
    candidate_random_control_baseline: str | None = None,
    paid_followup_estimate_usd: float = 0.0,
    known_spend_usd: float = CURRENT_KNOWN_SPEND_USD,
    paid_cap_usd: float = DEFAULT_PAID_CAP_USD,
    policy: ActiveArmGatePolicy | None = None,
) -> dict[str, Any]:
    """Evaluate a no-paid active-arm artifact against the consolidation gate."""
    cfg = policy or ActiveArmGatePolicy()
    paired_payload, paired_source = _find_paired_deltas(active_artifact)
    paired_reference_arm = _string_value(paired_payload, "reference_arm")
    active_arm = active_arm_name or _string_value(
        paired_payload,
        "comparison_arm",
    )
    random_control = paired_reference_arm or candidate_random_control_baseline
    if active_arm is None or random_control is None:
        inferred_active, inferred_random = _infer_arm_names(active_artifact)
        active_arm = active_arm or inferred_active
        random_control = random_control or inferred_random
    active_arm = active_arm or "unknown_active_arm"
    random_control = random_control or "unknown_random_control"
    paid_input = _paid_input_summary(active_artifact)

    paired_deltas = _paired_delta_summary(
        paired_payload,
        source=paired_source,
        active_arm=active_arm,
        random_control=random_control,
        paired_reference_arm=paired_reference_arm,
        caller_random_control_baseline=candidate_random_control_baseline,
        policy=cfg,
    )
    seed_intervals = {
        metric: {
            "normal_approx_95_ci": paired_deltas["metric_deltas"][metric][
                "normal_approx_95_ci"
            ],
            "count": paired_deltas["metric_deltas"][metric]["count"],
            "mean": paired_deltas["metric_deltas"][metric]["mean"],
            "source": "paired active-minus-random seed deltas",
        }
        for metric in METRICS
    }
    caveats = _caveats(
        active_artifact,
        active_arm=active_arm,
        random_control=random_control,
    )
    diagnostics = _diagnostics(
        active_artifact,
        active_arm=active_arm,
        random_control=random_control,
    )
    label_leakage = _label_leakage_summary(active_artifact)
    random_reference = _random_variance_reference(
        random_variance_artifact,
        random_control=random_control,
        path=random_variance_artifact_path,
        policy=cfg,
    )
    spend_estimate = _spend_estimate(
        paid_followup_estimate_usd=paid_followup_estimate_usd,
        known_spend_usd=known_spend_usd,
        paid_cap_usd=paid_cap_usd,
    )
    gate_verdict = _gate_verdict(
        paired_deltas=paired_deltas,
        caveats=caveats,
        diagnostics=diagnostics,
        label_leakage=label_leakage,
        random_reference=random_reference,
        spend_estimate=spend_estimate,
        paid_input=paid_input,
        policy=cfg,
    )
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "active_arm_name": active_arm,
        "candidate_random_control_baseline": random_control,
        "paid_followup_allowed": gate_verdict["paid_followup_allowed"],
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 0,
        "spend_policy": (
            "This gate is offline-only and makes zero Sestina paid LLM calls; "
            "pointwise calls and paid labeling are outside this harness."
        ),
        "gate_policy": cfg.to_dict(),
        "gate_verdict": gate_verdict,
        "paired_active_minus_random_deltas": paired_deltas,
        "seed_level_confidence_intervals": seed_intervals,
        "diagnostics": diagnostics,
        "label_leakage": label_leakage,
        "caveats": caveats,
        "spend_estimate": spend_estimate,
        "random_variance_reference": random_reference,
        "input_artifacts": {
            "active_artifact_path": active_artifact_path,
            "active_artifact_type": active_artifact.get("artifact_type"),
            "active_artifact_paid_calls_made": paid_input["paid_calls_made"],
            "active_artifact_paid_spend_usd": paid_input["paid_spend_usd"],
            "active_artifact_paid_metadata": paid_input,
            "random_variance_artifact_path": random_variance_artifact_path,
            "random_variance_artifact_type": random_variance_artifact.get(
                "artifact_type"
            ),
        },
        "recommended_next_action": gate_verdict["recommended_next_action"],
    }
    validate_active_arm_gate_artifact_schema(payload)
    return payload


def validate_active_arm_gate_artifact_schema(payload: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(payload))
    if missing:
        raise ValueError(
            "active arm gate artifact missing top-level keys: "
            + ", ".join(missing)
        )
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("active arm gate artifact has unexpected artifact_type")
    verdict = payload.get("gate_verdict")
    if not isinstance(verdict, Mapping):
        raise ValueError("active arm gate artifact gate_verdict must be an object")
    if "paid_followup_allowed" not in verdict:
        raise ValueError("active arm gate verdict missing paid_followup_allowed")
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("active arm gate artifact diagnostics must be an object")
    deltas = payload.get("paired_active_minus_random_deltas")
    if not isinstance(deltas, Mapping):
        raise ValueError(
            "active arm gate artifact paired_active_minus_random_deltas "
            "must be an object"
        )
    metric_deltas = deltas.get("metric_deltas")
    if not isinstance(metric_deltas, Mapping):
        raise ValueError("active arm gate artifact missing metric_deltas")
    missing_metrics = sorted(set(METRICS) - set(metric_deltas))
    if missing_metrics:
        raise ValueError(
            "active arm gate artifact missing metric deltas: "
            + ", ".join(missing_metrics)
        )
    caveats = payload.get("caveats")
    if not isinstance(caveats, Mapping):
        raise ValueError("active arm gate artifact caveats must be an object")
    if "budget_completeness_caveat" not in caveats:
        raise ValueError(
            "active arm gate artifact missing budget_completeness_caveat"
        )


def _gate_verdict(
    *,
    paired_deltas: Mapping[str, Any],
    caveats: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    label_leakage: Mapping[str, Any],
    random_reference: Mapping[str, Any],
    spend_estimate: Mapping[str, Any],
    paid_input: Mapping[str, Any],
    policy: ActiveArmGatePolicy,
) -> dict[str, Any]:
    metric_deltas = paired_deltas["metric_deltas"]
    recall = metric_deltas["recall_at_k"]
    ndcg = metric_deltas["ndcg_at_k"]
    ap = metric_deltas["average_precision"]
    recall_mean = float(recall["mean"])
    ndcg_mean = float(ndcg["mean"])
    ap_mean = float(ap["mean"])
    recall_ci = recall["normal_approx_95_ci"]
    recall_ci_lower = recall_ci[0] if recall_ci[0] is not None else None
    seed_count = int(recall["count"])
    paired_payload_present = bool(
        paired_deltas["paired_control_present"] and seed_count > 0
    )
    paired_baseline_approved = bool(
        paired_deltas["random_control_baseline_is_approved"]
    )
    random_control_override_consistent = bool(
        paired_deltas["random_control_override_matches_paired_reference"]
    )
    paired_random_control_present = bool(
        paired_payload_present
        and paired_baseline_approved
        and random_control_override_consistent
    )
    paired_metric_deltas_complete = bool(
        paired_deltas["required_metric_deltas_complete"]
    )
    seed_count_ok = seed_count >= policy.minimum_seed_count
    missing_label_present = bool(caveats["missing_label_caveat"]["present"])
    no_paid_input = bool(paid_input["explicit_zero_paid_evidence"])
    random_reference_ok = bool(random_reference["complete_20_seed_reference"])
    diagnostics_complete = bool(diagnostics["core_diagnostics_complete"])
    randomized_or_paired = bool(
        diagnostics["randomized_floor_or_paired_control"]["present"]
    )
    no_label_leakage = not bool(label_leakage["present"])
    budget_ok = bool(spend_estimate["within_paid_cap"])
    budget_shortfall_present = bool(
        caveats["budget_completeness_caveat"]["present"]
    )
    credible_recall = bool(
        recall_ci_lower is not None
        and seed_count_ok
        and paired_random_control_present
        and paired_metric_deltas_complete
        and recall_ci_lower > 0.0
    )
    secondary_nonnegative = ndcg_mean >= 0.0 and ap_mean >= 0.0
    mean_margin = bool(
        paired_random_control_present
        and paired_metric_deltas_complete
        and seed_count_ok
        and recall_mean >= policy.recall_mean_margin
        and secondary_nonnegative
        and not missing_label_present
        and not budget_shortfall_present
    )
    metric_gate_passed = credible_recall or mean_margin

    blocking_reasons = []
    if policy.require_no_paid_input and paid_input["missing_fields"]:
        blocking_reasons.append(
            "active gate input is missing explicit zero-paid metadata: "
            + ", ".join(paid_input["missing_fields"])
        )
    if policy.require_no_paid_input and paid_input["invalid_fields"]:
        blocking_reasons.append(
            "active gate input has invalid paid metadata: "
            + ", ".join(paid_input["invalid_fields"])
        )
    if (
        policy.require_no_paid_input
        and not paid_input["missing_fields"]
        and not paid_input["invalid_fields"]
        and not no_paid_input
    ):
        blocking_reasons.append("active gate input already made paid calls or spend")
    if policy.require_paired_random_control and not paired_payload_present:
        blocking_reasons.append("paired random-control deltas are unavailable")
    if (
        policy.require_paired_random_control
        and paired_payload_present
        and not paired_baseline_approved
    ):
        blocking_reasons.append(
            "paired control baseline is not an approved random/exact-pool random "
            "control"
        )
    if (
        policy.require_paired_random_control
        and paired_payload_present
        and not random_control_override_consistent
    ):
        blocking_reasons.append(
            "random-control override does not match paired payload reference_arm"
        )
    if paired_payload_present and not paired_metric_deltas_complete:
        blocking_reasons.append(
            "paired metric deltas are incomplete for required metrics: "
            + ", ".join(paired_deltas["incomplete_required_metric_deltas"])
        )
    if seed_count < policy.minimum_seed_count:
        blocking_reasons.append(
            f"paired seed count {seed_count} is below required "
            f"{policy.minimum_seed_count}"
        )
    if policy.require_full_random_reference and not random_reference_ok:
        blocking_reasons.append(
            "completed 20-seed full-random variance reference is unavailable"
        )
    if policy.require_randomized_floor_or_paired_control and not randomized_or_paired:
        blocking_reasons.append(
            "randomized floor or paired random-control presence is unavailable"
        )
    if policy.require_core_diagnostics and not diagnostics_complete:
        blocking_reasons.append(
            "core weak-bucket/graph/oracle diagnostics are incomplete"
        )
    if not no_label_leakage:
        blocking_reasons.append(
            "future-label or cached-label leakage markers are true: "
            + ", ".join(label_leakage["forbidden_true_keys"])
        )
    if missing_label_present:
        blocking_reasons.append("missing-label caveat is present")
    if budget_shortfall_present:
        blocking_reasons.append("budget-completeness caveat is present")
    if not budget_ok:
        blocking_reasons.append("paid follow-up estimate would exceed paid cap")
    if (
        paired_random_control_present
        and paired_metric_deltas_complete
        and seed_count_ok
        and not metric_gate_passed
    ):
        if recall_mean < policy.recall_mean_margin and not credible_recall:
            blocking_reasons.append(
                "mean Recall@K delta is below +0.025 and Recall@K CI is not "
                "credibly positive"
            )
        if recall_mean >= policy.recall_mean_margin and not secondary_nonnegative:
            blocking_reasons.append("nDCG/AP deltas are not both nonnegative")
    paid_allowed = bool(
        metric_gate_passed
        and no_paid_input
        and paired_random_control_present
        and paired_metric_deltas_complete
        and seed_count_ok
        and random_reference_ok
        and randomized_or_paired
        and diagnostics_complete
        and no_label_leakage
        and not missing_label_present
        and not budget_shortfall_present
        and budget_ok
    )
    if paid_allowed:
        recommendation = (
            "Paid follow-up is allowed by this no-paid gate only as a "
            "guarded pairwise-only validation run after a separate dry-run "
            "estimate, provider model availability check, JSONL ledger, new "
            "artifact directory, and hard --max-usd cap. Fresh-holdout "
            "pointwise artifact generation is governed separately by the "
            "standing autonomous campaign policy and must stay scoped to that "
            "fresh holdout."
        )
    else:
        recommendation = (
            "Do not spend on this active arm. Revise the no-paid simulator or "
            "candidate policy and rerun the gate against paired random/exact-pool "
            "controls and the 20-seed full-random variance reference."
        )
    return {
        "paid_followup_allowed": paid_allowed,
        "credible_recall_gate_passed": credible_recall,
        "mean_margin_gate_passed": mean_margin,
        "metric_gate_passed": metric_gate_passed,
        "paired_random_control_present": paired_random_control_present,
        "paired_control_baseline_approved": paired_baseline_approved,
        "random_control_override_consistent": random_control_override_consistent,
        "paired_metric_deltas_complete": paired_metric_deltas_complete,
        "paired_metric_delta_counts": paired_deltas["metric_delta_counts"],
        "incomplete_paired_metric_deltas": paired_deltas[
            "incomplete_required_metric_deltas"
        ],
        "seed_count": seed_count,
        "minimum_seed_count": policy.minimum_seed_count,
        "randomized_floor_or_paired_control_present": randomized_or_paired,
        "core_diagnostics_complete": diagnostics_complete,
        "no_future_label_or_cached_label_leakage": no_label_leakage,
        "label_leakage": label_leakage,
        "full_random_variance_reference_complete": random_reference_ok,
        "no_paid_active_input": no_paid_input,
        "active_paid_metadata": paid_input,
        "missing_label_caveat_present": missing_label_present,
        "budget_completeness_caveat_present": budget_shortfall_present,
        "budget_within_paid_cap": budget_ok,
        "mean_recall_delta": recall_mean,
        "recall_delta_ci": recall_ci,
        "mean_ndcg_delta": ndcg_mean,
        "mean_average_precision_delta": ap_mean,
        "blocking_reasons": blocking_reasons,
        "recommended_next_action": recommendation,
    }


def _find_paired_deltas(
    artifact: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str | None]:
    for key in (
        "paired_active_minus_random_deltas",
        "paired_deltas_vs_exact_pool_random",
        "paired_deltas",
    ):
        payload = artifact.get(key)
        if isinstance(payload, Mapping):
            return payload, key
    return None, None


def _paired_delta_summary(
    paired_payload: Mapping[str, Any] | None,
    *,
    source: str | None,
    active_arm: str,
    random_control: str,
    paired_reference_arm: str | None,
    caller_random_control_baseline: str | None,
    policy: ActiveArmGatePolicy,
) -> dict[str, Any]:
    seed_deltas = _seed_deltas(paired_payload)
    metric_deltas = {
        metric: _summary(
            [
                float(row[metric])
                for row in seed_deltas.values()
                if isinstance(row, Mapping) and metric in row
            ],
            z=policy.confidence_z,
        )
        for metric in METRICS
    }
    metric_delta_counts = {
        metric: int(metric_deltas[metric]["count"]) for metric in METRICS
    }
    incomplete_metrics = sorted(
        metric
        for metric, count in metric_delta_counts.items()
        if count < policy.minimum_seed_count
    )
    baseline_for_validation = paired_reference_arm or random_control
    baseline_is_approved = _is_approved_random_control_baseline(
        baseline_for_validation
    )
    override_matches_paired_reference = _override_matches_paired_reference(
        caller_random_control_baseline,
        paired_reference_arm,
    )
    return {
        "active_arm": active_arm,
        "random_control_baseline": baseline_for_validation,
        "paired_payload_reference_arm": paired_reference_arm,
        "caller_random_control_baseline": caller_random_control_baseline,
        "random_control_override_matches_paired_reference": (
            override_matches_paired_reference
        ),
        "random_control_baseline_is_approved": baseline_is_approved,
        "source": source,
        "paired_control_present": bool(seed_deltas),
        "required_metric_deltas_complete": not incomplete_metrics,
        "metric_delta_counts": metric_delta_counts,
        "incomplete_required_metric_deltas": incomplete_metrics,
        "metric_deltas": metric_deltas,
        "seed_deltas": seed_deltas,
        "bucket_deltas": list(_list_value(paired_payload, "bucket_deltas")),
        "selected_positive_total_delta": (
            paired_payload.get("selected_positive_total_delta")
            if isinstance(paired_payload, Mapping)
            else None
        ),
        "confidence_interval_method": (
            "normal approximation over paired seed-level active-minus-random "
            "deltas"
        ),
    }


def _seed_deltas(
    paired_payload: Mapping[str, Any] | None,
) -> dict[str, dict[str, float]]:
    if not isinstance(paired_payload, Mapping):
        return {}
    raw = paired_payload.get("seed_deltas")
    if not isinstance(raw, Mapping):
        return {}
    rows: dict[str, dict[str, float]] = {}
    for seed, row in raw.items():
        if not isinstance(row, Mapping):
            continue
        metric_row = {}
        for metric in METRICS:
            if metric in row:
                metric_row[metric] = round(float(row[metric]), 8)
        if metric_row:
            rows[str(seed)] = metric_row
    return dict(sorted(rows.items(), key=lambda item: _seed_sort_key(item[0])))


def _summary(values: Sequence[float], *, z: float) -> dict[str, Any]:
    items = [float(value) for value in values]
    if not items:
        return {
            "count": 0,
            "mean": 0.0,
            "stddev": 0.0,
            "standard_error": 0.0,
            "min": 0.0,
            "max": 0.0,
            "normal_approx_95_ci": [None, None],
        }
    value_mean = mean(items)
    value_stddev = stdev(items) if len(items) > 1 else 0.0
    standard_error = value_stddev / math.sqrt(len(items)) if len(items) > 1 else 0.0
    return {
        "count": len(items),
        "mean": _round(value_mean),
        "stddev": _round(value_stddev),
        "standard_error": _round(standard_error),
        "min": _round(min(items)),
        "max": _round(max(items)),
        "normal_approx_95_ci": [
            _round(value_mean - (z * standard_error)),
            _round(value_mean + (z * standard_error)),
        ],
    }


def _caveats(
    artifact: Mapping[str, Any],
    *,
    active_arm: str,
    random_control: str,
) -> dict[str, Any]:
    missing = _comparison_source_counts(
        artifact,
        active_arm=active_arm,
        random_control=random_control,
        key="missing_pairwise_labels",
    )
    partial = _comparison_source_counts(
        artifact,
        active_arm=active_arm,
        random_control=random_control,
        key="partial",
    )
    cache_reuse = _cache_reuse_summary(artifact)
    budget_completeness = build_budget_completeness_caveat(
        artifact,
        active_arm=active_arm,
        random_control=random_control,
    )
    missing_present = bool(
        missing["active_total"] > 0
        or missing["random_control_total"] > 0
        or partial["active_total"] > 0
        or partial["random_control_total"] > 0
    )
    return {
        "missing_label_caveat": {
            "present": missing_present,
            "active_missing_pairwise_labels": missing["active_total"],
            "random_control_missing_pairwise_labels": missing[
                "random_control_total"
            ],
            "active_partial_rows": partial["active_total"],
            "random_control_partial_rows": partial["random_control_total"],
        },
        "budget_completeness_caveat": budget_completeness,
        "cache_reuse_caveat": cache_reuse,
        "offline_replay_caveat": {
            "present": True,
            "message": (
                "Gate inputs are no-paid cached or simulated artifacts; passing "
                "the gate is permission to consider a paid dry-run, not proof "
                "from fresh paid labels."
            ),
        },
    }


def build_budget_completeness_caveat(
    artifact: Mapping[str, Any],
    *,
    active_arm: str,
    random_control: str,
) -> dict[str, Any]:
    rows = _budget_shortfall_rows(
        artifact,
        active_arm=active_arm,
        random_control=random_control,
    )
    active_rows = [row for row in rows if row["arm_role"] == "active"]
    random_rows = [row for row in rows if row["arm_role"] == "random_control"]
    active_shortfall = sum(int(row["budget_shortfall"]) for row in active_rows)
    random_shortfall = sum(int(row["budget_shortfall"]) for row in random_rows)
    present = bool(active_shortfall or random_shortfall)
    return {
        "present": present,
        "blocking": present,
        "active_budget_shortfall": active_shortfall,
        "random_control_budget_shortfall": random_shortfall,
        "active_under_budget_rows": len(active_rows),
        "random_control_under_budget_rows": len(random_rows),
        "under_budget_row_count": len(rows),
        "message": (
            "One or more arms scheduled fewer pairs than the resolved per-row "
            "budget; treat the cached replay as incomplete for paid follow-up "
            "unless the shortfall is filled by a predeclared no-future-label "
            "fallback."
        )
        if present
        else (
            "All rows with resolved budget metadata scheduled at least the "
            "resolved per-row pairwise budget."
        ),
        "rows": rows,
    }


def _budget_shortfall_rows(
    artifact: Mapping[str, Any],
    *,
    active_arm: str,
    random_control: str,
) -> list[dict[str, Any]]:
    rows = []
    for seed_payload in _list_value(artifact, "bucket_results"):
        if not isinstance(seed_payload, Mapping):
            continue
        seed = seed_payload.get("seed")
        for bucket in _list_value(seed_payload, "buckets"):
            if not isinstance(bucket, Mapping):
                continue
            arms = bucket.get("arms")
            if not isinstance(arms, Mapping):
                continue
            for arm_role, arm_name in (
                ("active", active_arm),
                ("random_control", random_control),
            ):
                arm_payload = arms.get(arm_name)
                if not isinstance(arm_payload, Mapping):
                    continue
                source = arm_payload.get("comparison_source")
                if not isinstance(source, Mapping):
                    continue
                resolved_budget = _resolved_pairwise_budget(bucket, source)
                scheduled_total = _int_value(source.get("scheduled_pairwise_total"))
                if resolved_budget is None or scheduled_total is None:
                    continue
                shortfall = max(0, resolved_budget - scheduled_total)
                if shortfall <= 0:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "bucket": bucket.get("bucket"),
                        "arm": arm_name,
                        "arm_role": arm_role,
                        "resolved_pairwise_budget": resolved_budget,
                        "scheduled_pairwise_total": scheduled_total,
                        "budget_shortfall": shortfall,
                    }
                )
    return rows


def _resolved_pairwise_budget(
    bucket: Mapping[str, Any],
    source: Mapping[str, Any],
) -> int | None:
    for value in (
        source.get("resolved_pairwise_budget"),
        source.get("pairwise_budget"),
        source.get("budget"),
        bucket.get("budget"),
    ):
        resolved = _budget_value(value)
        if resolved is not None:
            return resolved
    return None


def _budget_value(value: Any) -> int | None:
    if isinstance(value, Mapping):
        return _int_value(value.get("budget"))
    return _int_value(value)


def _comparison_source_counts(
    artifact: Mapping[str, Any],
    *,
    active_arm: str,
    random_control: str,
    key: str,
) -> dict[str, int]:
    active_total = 0
    random_total = 0
    inspected_selected_sources = False
    for seed_payload in _list_value(artifact, "bucket_results"):
        if not isinstance(seed_payload, Mapping):
            continue
        for bucket in _list_value(seed_payload, "buckets"):
            if not isinstance(bucket, Mapping):
                continue
            arms = bucket.get("arms")
            if not isinstance(arms, Mapping):
                continue
            active_payload = arms.get(active_arm)
            random_payload = arms.get(random_control)
            if _has_comparison_source(active_payload):
                inspected_selected_sources = True
                active_total += _comparison_value(active_payload, key)
            if _has_comparison_source(random_payload):
                inspected_selected_sources = True
                random_total += _comparison_value(random_payload, key)
    if inspected_selected_sources:
        return {"active_total": active_total, "random_control_total": random_total}
    fallback = _recursive_nonzero_count(artifact, key)
    return {"active_total": fallback, "random_control_total": 0}


def _has_comparison_source(arm_payload: Any) -> bool:
    return (
        isinstance(arm_payload, Mapping)
        and isinstance(arm_payload.get("comparison_source"), Mapping)
    )


def _comparison_value(arm_payload: Any, key: str) -> int:
    if not isinstance(arm_payload, Mapping):
        return 0
    source = arm_payload.get("comparison_source")
    if not isinstance(source, Mapping):
        return 0
    value = source.get(key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    return 0


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _recursive_nonzero_count(value: Any, key: str) -> int:
    if isinstance(value, Mapping):
        total = 0
        for current_key, current_value in value.items():
            if current_key == key:
                if isinstance(current_value, bool):
                    total += int(current_value)
                elif isinstance(current_value, int | float):
                    total += int(current_value)
            total += _recursive_nonzero_count(current_value, key)
        return total
    if isinstance(value, list):
        return sum(_recursive_nonzero_count(item, key) for item in value)
    return 0


def _cache_reuse_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    text_markers = _recursive_string_matches(artifact, "cached")
    paid_calls = artifact.get("paid_calls_made")
    explicit_zero_paid_calls = (
        not isinstance(paid_calls, bool)
        and isinstance(paid_calls, int | float)
        and paid_calls == 0
    )
    return {
        "present": bool(text_markers or explicit_zero_paid_calls),
        "message": (
            "Artifact appears to reuse cached labels or run offline; cache reuse "
            "is acceptable for the no-paid gate but must be reported before any "
            "paid follow-up."
        ),
        "cached_marker_count": text_markers,
    }


def _diagnostics(
    artifact: Mapping[str, Any],
    *,
    active_arm: str,
    random_control: str,
) -> dict[str, Any]:
    aggregate = artifact.get("aggregate_diagnostics")
    aggregate = aggregate if isinstance(aggregate, Mapping) else {}
    available = sorted(key for key in CORE_DIAGNOSTIC_KEYS if key in aggregate)
    missing = sorted(CORE_DIAGNOSTIC_KEYS - set(available))
    randomized_floor = _randomized_floor_diagnostics(
        artifact,
        aggregate=aggregate,
        active_arm=active_arm,
    )
    unavailable_fields = _diagnostic_unavailable_fields(aggregate)
    weak_bucket = aggregate.get("weak_bucket_deltas")
    paired_payload = _find_paired_deltas(artifact)[0]
    paired_present = bool(
        paired_payload
        and _seed_deltas(paired_payload)
        and _is_approved_random_control_baseline(random_control)
    )
    return {
        "core_diagnostics_complete": not missing,
        "core_diagnostic_keys_available": available,
        "core_diagnostic_keys_missing": missing,
        "randomized_floor_or_paired_control": {
            "present": bool(randomized_floor["present"] or paired_present),
            "randomized_floor_present": randomized_floor["present"],
            "paired_random_control_present": paired_present,
            "active_random_floor_rate": randomized_floor["active_random_floor_rate"],
        },
        "weak_bucket_diagnostics": {
            "available": isinstance(weak_bucket, Mapping),
            "value": weak_bucket if isinstance(weak_bucket, Mapping) else None,
            "unavailable_reason": None
            if isinstance(weak_bucket, Mapping)
            else "aggregate_diagnostics.weak_bucket_deltas not present",
        },
        "explicit_unavailable_fields": unavailable_fields,
        "aggregate_diagnostics": aggregate,
    }


def _label_leakage_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    markers = _recursive_true_keys(
        artifact,
        keys=FORBIDDEN_LABEL_LEAKAGE_TRUE_KEYS,
    )
    return {
        "present": bool(markers),
        "forbidden_true_keys": sorted(markers),
        "checked_keys": sorted(FORBIDDEN_LABEL_LEAKAGE_TRUE_KEYS),
        "policy": (
            "Future labels, citation outcomes, good_paper labels, matched work "
            "identifiers, and cached label values must not be used for "
            "scheduling, routing, prompts, or model-visible inputs. Future "
            "labels are allowed only for retrospective evaluation and "
            "diagnostics."
        ),
    }


def _randomized_floor_diagnostics(
    artifact: Mapping[str, Any],
    *,
    aggregate: Mapping[str, Any],
    active_arm: str,
) -> dict[str, Any]:
    randomized = aggregate.get("randomized_coverage")
    active_rate = 0.0
    present = False
    if isinstance(randomized, Mapping):
        active_payload = randomized.get(active_arm)
        if isinstance(active_payload, Mapping):
            active_rate = float(active_payload.get("random_floor_rate", 0.0) or 0.0)
            present = active_rate > 0.0 or int(
                active_payload.get("random_floor_pairs", 0) or 0
            ) > 0
    for arm in _list_value(artifact, "arms"):
        if isinstance(arm, Mapping) and arm.get("name") == active_arm:
            present = present or bool(arm.get("randomized_coverage_floor"))
    return {"present": present, "active_random_floor_rate": _round(active_rate)}


def _diagnostic_unavailable_fields(
    aggregate: Mapping[str, Any],
) -> dict[str, dict[str, str | bool]]:
    graph = aggregate.get("graph_connectivity")
    oracle = aggregate.get("oracle_caps")
    weak = aggregate.get("weak_bucket_deltas")
    return {
        "posterior_top_k_degree": {
            "available": _contains_key(graph, "posterior_top_k_degree"),
            "unavailable_reason": (
                None
                if _contains_key(graph, "posterior_top_k_degree")
                else "not aggregated in the input artifact"
            ),
        },
        "pointwise_plus_touched_oracle_cap": {
            "available": _contains_key(
                oracle,
                "mean_pointwise_plus_touched_recall_cap",
            )
            or _contains_key(weak, "pointwise_plus_touched_recall_cap_delta"),
            "unavailable_reason": None
            if (
                _contains_key(oracle, "mean_pointwise_plus_touched_recall_cap")
                or _contains_key(weak, "pointwise_plus_touched_recall_cap_delta")
            )
            else "not aggregated in the input artifact",
        },
        "positive_negative_pair_oracle_cap": {
            "available": _contains_key(
                oracle,
                "mean_positive_negative_pair_recall_cap",
            )
            or _contains_key(weak, "positive_negative_pair_recall_cap_delta"),
            "unavailable_reason": None
            if (
                _contains_key(oracle, "mean_positive_negative_pair_recall_cap")
                or _contains_key(weak, "positive_negative_pair_recall_cap_delta")
            )
            else "not aggregated in the input artifact",
        },
        "observed_positive_winner_cap": {
            "available": _contains_key(
                oracle,
                "mean_observed_positive_winner_recall_cap",
            ),
            "unavailable_reason": None
            if _contains_key(oracle, "mean_observed_positive_winner_recall_cap")
            else "not aggregated in the input artifact",
        },
    }


def _random_variance_reference(
    artifact: Mapping[str, Any],
    *,
    random_control: str,
    path: str | None,
    policy: ActiveArmGatePolicy,
) -> dict[str, Any]:
    aggregate = artifact.get("aggregate_metrics")
    aggregate = aggregate if isinstance(aggregate, Mapping) else {}
    baseline_key = _random_reference_key(random_control)
    baseline = aggregate.get(baseline_key) if baseline_key is not None else None
    baseline = baseline if isinstance(baseline, Mapping) else {}
    intervals = baseline.get("seed_level_intervals")
    intervals = intervals if isinstance(intervals, Mapping) else {}
    paired = artifact.get("paired_deltas")
    paired = paired if isinstance(paired, Mapping) else {}
    completion = artifact.get("full_schedule_completion_status")
    completion = completion if isinstance(completion, Mapping) else {}
    seed_count = int(
        (artifact.get("analysis_parameters") or {}).get("seed_count", 0)
        if isinstance(artifact.get("analysis_parameters"), Mapping)
        else 0
    )
    interval_status = {
        metric: _seed_interval_status(
            intervals.get(metric),
            minimum_seed_count=policy.minimum_seed_count,
        )
        for metric in METRICS
    }
    missing_required_intervals = sorted(
        metric for metric, status in interval_status.items() if not status["complete"]
    )
    complete = bool(
        artifact.get("artifact_type") == RANDOM_VARIANCE_ARTIFACT_TYPE
        and baseline_key is not None
        and seed_count >= policy.minimum_seed_count
        and completion.get("all_seed_bucket_rows_complete") is True
        and not missing_required_intervals
    )
    return {
        "artifact_path": path,
        "artifact_type": artifact.get("artifact_type"),
        "selected_reference_arm": baseline_key,
        "candidate_random_control_baseline": random_control,
        "candidate_random_control_baseline_is_approved": baseline_key is not None,
        "complete_20_seed_reference": complete,
        "seed_count": seed_count,
        "seed_level_intervals": {
            metric: intervals.get(metric) for metric in METRICS
        },
        "required_seed_level_interval_status": interval_status,
        "missing_required_seed_level_intervals": missing_required_intervals,
        "historical_minus_exact_pool_random_delta_intervals": (
            paired.get("metric_delta_intervals")
            if isinstance(paired.get("metric_delta_intervals"), Mapping)
            else {}
        ),
        "paid_calls_made_in_historical_reference": int(
            artifact.get("paid_calls_made", 0) or 0
        ),
        "paid_spend_usd_in_historical_reference": float(
            artifact.get("paid_spend_usd", 0.0) or 0.0
        ),
    }


def _paid_input_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    required_fields = ("paid_calls_made", "paid_spend_usd")
    missing_fields = [field for field in required_fields if field not in artifact]
    invalid_fields = []
    if "paid_calls_made" in artifact and not _is_integer(
        artifact.get("paid_calls_made")
    ):
        invalid_fields.append("paid_calls_made")
    if "paid_spend_usd" in artifact and not _is_finite_number(
        artifact.get("paid_spend_usd")
    ):
        invalid_fields.append("paid_spend_usd")
    paid_calls = (
        artifact["paid_calls_made"]
        if "paid_calls_made" in artifact
        and "paid_calls_made" not in invalid_fields
        else None
    )
    paid_spend = (
        float(artifact["paid_spend_usd"])
        if "paid_spend_usd" in artifact
        and "paid_spend_usd" not in invalid_fields
        else None
    )
    return {
        "missing_fields": missing_fields,
        "invalid_fields": invalid_fields,
        "paid_calls_made": paid_calls,
        "paid_spend_usd": paid_spend,
        "explicit_zero_paid_evidence": (
            not missing_fields
            and not invalid_fields
            and paid_calls == 0
            and paid_spend == 0.0
        ),
    }


def _seed_interval_status(
    value: Any,
    *,
    minimum_seed_count: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {
            "present": False,
            "count": 0,
            "has_confidence_interval": False,
            "complete": False,
        }
    count_value = value.get("count")
    count = int(count_value) if _is_number(count_value) else 0
    has_interval = _has_confidence_interval(value)
    return {
        "present": True,
        "count": count,
        "has_confidence_interval": has_interval,
        "complete": count >= minimum_seed_count and has_interval,
    }


def _has_confidence_interval(value: Mapping[str, Any]) -> bool:
    for key in ("normal_approx_95_ci", "bootstrap_percentile_95_ci"):
        bounds = value.get(key)
        if (
            isinstance(bounds, list)
            and len(bounds) == 2
            and all(_is_number(bound) for bound in bounds)
        ):
            return True
    return False


def _spend_estimate(
    *,
    paid_followup_estimate_usd: float,
    known_spend_usd: float,
    paid_cap_usd: float,
) -> dict[str, Any]:
    projected = known_spend_usd + paid_followup_estimate_usd
    return {
        "gate_paid_calls_made": 0,
        "gate_paid_spend_usd": 0.0,
        "paid_followup_estimate_usd": _round(paid_followup_estimate_usd, 6),
        "known_spend_usd_before_followup": _round(known_spend_usd, 6),
        "paid_cap_usd": _round(paid_cap_usd, 6),
        "projected_spend_usd_after_followup": _round(projected, 6),
        "remaining_cap_usd_before_followup": _round(
            paid_cap_usd - known_spend_usd,
            6,
        ),
        "within_paid_cap": projected <= paid_cap_usd,
    }


def _infer_arm_names(artifact: Mapping[str, Any]) -> tuple[str | None, str | None]:
    aggregate = artifact.get("aggregate_metrics")
    if not isinstance(aggregate, Mapping):
        return None, None
    names = [str(name) for name in aggregate]
    random_names = [
        name
        for name in names
        if "random" in name or "control" in name or "baseline" in name
    ]
    active_names = [name for name in names if name not in random_names]
    return (
        active_names[0] if active_names else None,
        random_names[0] if random_names else None,
    )


def _random_reference_key(random_control: str) -> str | None:
    normalized = _normalize_control_name(random_control)
    if not _is_approved_normalized_random_control(normalized):
        return None
    if normalized == "exact_pool_random" or normalized.startswith(
        "exact_pool_random_"
    ):
        return "exact_pool_random_full_schedule"
    return "historical_random_full_schedule"


def _is_approved_random_control_baseline(random_control: str) -> bool:
    return _is_approved_normalized_random_control(
        _normalize_control_name(random_control)
    )


def _override_matches_paired_reference(
    override: str | None,
    paired_reference_arm: str | None,
) -> bool:
    if override is None or paired_reference_arm is None:
        return True
    return _normalize_control_name(override) == _normalize_control_name(
        paired_reference_arm
    )


def _is_approved_normalized_random_control(normalized: str) -> bool:
    return normalized in APPROVED_RANDOM_CONTROL_NAMES or normalized.startswith(
        APPROVED_RANDOM_CONTROL_PREFIXES
    )


def _normalize_control_name(value: str) -> str:
    normalized = str(value).strip().lower()
    for character in ("-", " ", "."):
        normalized = normalized.replace(character, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _contains_key(value: Any, target: str) -> bool:
    if isinstance(value, Mapping):
        return any(
            key == target or _contains_key(child, target)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_key(item, target) for item in value)
    return False


def _recursive_string_matches(value: Any, needle: str) -> int:
    if isinstance(value, str):
        return int(needle in value.lower())
    if isinstance(value, Mapping):
        return sum(_recursive_string_matches(child, needle) for child in value.values())
    if isinstance(value, list):
        return sum(_recursive_string_matches(item, needle) for item in value)
    return 0


def _recursive_true_keys(value: Any, *, keys: set[str]) -> set[str]:
    if isinstance(value, Mapping):
        matches: set[str] = set()
        for current_key, current_value in value.items():
            if current_key in keys and current_value is True:
                matches.add(current_key)
            matches.update(_recursive_true_keys(current_value, keys=keys))
        return matches
    if isinstance(value, list):
        matches: set[str] = set()
        for item in value:
            matches.update(_recursive_true_keys(item, keys=keys))
        return matches
    return set()


def _string_value(payload: Mapping[str, Any] | None, key: str) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(key)
    return str(value) if value is not None else None


def _list_value(payload: Mapping[str, Any] | None, key: str) -> list[Any]:
    if not isinstance(payload, Mapping):
        return []
    value = payload.get(key)
    return list(value) if isinstance(value, list) else []


def _seed_sort_key(seed: str) -> tuple[int, int | str]:
    try:
        return (0, int(seed))
    except ValueError:
        return (1, seed)


def _round(value: float, digits: int = 8) -> float:
    return round(float(value), digits)


def _is_integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int)


def _is_finite_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float)
