#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sestina.active_arm_gate import (  # noqa: E402
    CURRENT_KNOWN_SPEND_USD,
    DEFAULT_PAID_CAP_USD,
    METRICS,
    build_active_arm_gate,
    validate_active_arm_gate_artifact_schema,
)
from sestina.diagnostics import write_json_artifact  # noqa: E402


ARTIFACT_TYPE = "sestina-active-arm-shortlist-gate"
SCHEMA_VERSION = 1
WORKFLOW = "sestina-active-arm-shortlist-gate"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-active-arm-shortlist-gate"
    / "shortlist-gate-study.json"
)
DEFAULT_CI_PARTITION_ARTIFACT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-ci-partition-gate"
    / "ci-partition-gate-analysis.json"
)
DEFAULT_RANDOM_VARIANCE_ARTIFACT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-full-random-variance-completion"
    / "full-random-variance-completion.json"
)
DEFAULT_RANDOM_CONTROL_GAP_ARTIFACT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-random-control-diagnosis"
    / "random-control-gap-analysis.json"
)
DEFAULT_PAIRWISE_STRENGTH_ARTIFACT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-pairwise-strength-calibration"
    / "strength-calibration-analysis.json"
)
DEFAULT_POSTERIOR_DECISION_ARTIFACT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-posterior-decision-shrinkage"
    / "decision-shrinkage-analysis.json"
)
REQUIRED_TOP_LEVEL_KEYS = {
    "artifact_type",
    "schema_version",
    "workflow",
    "created_at",
    "paid_calls_made",
    "paid_spend_usd",
    "pointwise_calls_made",
    "known_paid_spend_usd",
    "paid_cap_usd",
    "spend_policy",
    "method",
    "random_baseline_policy",
    "input_artifacts",
    "candidates",
    "summary",
    "recommended_next_action",
}
REQUIRED_CANDIDATE_KEYS = {
    "id",
    "name",
    "memo_no_paid_gate",
    "status",
    "evaluated_with_active_arm_gate",
    "gate_inputs",
    "paid_followup_allowed",
    "blocking_reasons",
    "missing_prerequisites",
    "evidence_summary",
    "recommended_next_action",
}
ALLOWED_STATUSES = {
    "evaluated_blocked",
    "blocked_missing_prerequisite",
    "infrastructure_ready_no_paid_arm",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the no-paid active-arm shortlist gate study using reviewed "
            "cached artifacts and the active-arm gate harness."
        )
    )
    parser.add_argument(
        "--ci-partition-artifact",
        type=Path,
        default=DEFAULT_CI_PARTITION_ARTIFACT,
    )
    parser.add_argument(
        "--random-variance-artifact",
        type=Path,
        default=DEFAULT_RANDOM_VARIANCE_ARTIFACT,
    )
    parser.add_argument(
        "--random-control-gap-artifact",
        type=Path,
        default=DEFAULT_RANDOM_CONTROL_GAP_ARTIFACT,
    )
    parser.add_argument(
        "--pairwise-strength-artifact",
        type=Path,
        default=DEFAULT_PAIRWISE_STRENGTH_ARTIFACT,
    )
    parser.add_argument(
        "--posterior-decision-artifact",
        type=Path,
        default=DEFAULT_POSTERIOR_DECISION_ARTIFACT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--known-spend-usd",
        type=float,
        default=CURRENT_KNOWN_SPEND_USD,
    )
    parser.add_argument("--paid-cap-usd", type=float, default=DEFAULT_PAID_CAP_USD)
    args = parser.parse_args(argv)

    ci_partition_artifact = _read_json(args.ci_partition_artifact)
    random_variance_artifact = _read_json(args.random_variance_artifact)
    random_control_gap_artifact = _read_optional_json(
        args.random_control_gap_artifact
    )
    pairwise_strength_artifact = _read_optional_json(
        args.pairwise_strength_artifact
    )
    posterior_decision_artifact = _read_optional_json(
        args.posterior_decision_artifact
    )

    payload = build_shortlist_gate_study(
        ci_partition_artifact=ci_partition_artifact,
        random_variance_artifact=random_variance_artifact,
        random_control_gap_artifact=random_control_gap_artifact,
        pairwise_strength_artifact=pairwise_strength_artifact,
        posterior_decision_artifact=posterior_decision_artifact,
        ci_partition_artifact_path=str(args.ci_partition_artifact),
        random_variance_artifact_path=str(args.random_variance_artifact),
        random_control_gap_artifact_path=str(args.random_control_gap_artifact),
        pairwise_strength_artifact_path=str(args.pairwise_strength_artifact),
        posterior_decision_artifact_path=str(args.posterior_decision_artifact),
        output_path=str(args.output),
        known_spend_usd=args.known_spend_usd,
        paid_cap_usd=args.paid_cap_usd,
    )
    validate_shortlist_gate_artifact_schema(payload)
    write_json_artifact(args.output, payload)
    sys.stdout.write(json.dumps(_stdout_summary(payload), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def build_shortlist_gate_study(
    *,
    ci_partition_artifact: Mapping[str, Any],
    random_variance_artifact: Mapping[str, Any],
    random_control_gap_artifact: Mapping[str, Any] | None = None,
    pairwise_strength_artifact: Mapping[str, Any] | None = None,
    posterior_decision_artifact: Mapping[str, Any] | None = None,
    ci_partition_artifact_path: str | None = None,
    random_variance_artifact_path: str | None = None,
    random_control_gap_artifact_path: str | None = None,
    pairwise_strength_artifact_path: str | None = None,
    posterior_decision_artifact_path: str | None = None,
    output_path: str | None = None,
    known_spend_usd: float = CURRENT_KNOWN_SPEND_USD,
    paid_cap_usd: float = DEFAULT_PAID_CAP_USD,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the shortlist study without making paid calls or relabeling data."""
    ci_gate = build_active_arm_gate(
        ci_partition_artifact,
        random_variance_artifact,
        active_artifact_path=ci_partition_artifact_path,
        random_variance_artifact_path=random_variance_artifact_path,
        paid_followup_estimate_usd=0.0,
        known_spend_usd=known_spend_usd,
        paid_cap_usd=paid_cap_usd,
    )
    validate_active_arm_gate_artifact_schema(ci_gate)

    candidates = [
        _reliability_ci_candidate(
            ci_partition_artifact=ci_partition_artifact,
            ci_gate=ci_gate,
            ci_partition_artifact_path=ci_partition_artifact_path,
            random_variance_artifact_path=random_variance_artifact_path,
        ),
        _new_information_candidate(
            random_control_gap_artifact=random_control_gap_artifact,
            random_control_gap_artifact_path=random_control_gap_artifact_path,
            random_variance_artifact_path=random_variance_artifact_path,
        ),
        _aggregation_cross_check_candidate(
            pairwise_strength_artifact=pairwise_strength_artifact,
            posterior_decision_artifact=posterior_decision_artifact,
            pairwise_strength_artifact_path=pairwise_strength_artifact_path,
            posterior_decision_artifact_path=posterior_decision_artifact_path,
            random_variance_artifact_path=random_variance_artifact_path,
        ),
        _simulator_harness_candidate(
            ci_gate=ci_gate,
            ci_partition_artifact_path=ci_partition_artifact_path,
            random_variance_artifact_path=random_variance_artifact_path,
        ),
    ]
    status_counts = Counter(candidate["status"] for candidate in candidates)
    any_paid_followup_allowed = any(
        bool(candidate["paid_followup_allowed"]) for candidate in candidates
    )
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "created_at": created_at or date.today().isoformat(),
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 0,
        "known_paid_spend_usd": _round(known_spend_usd),
        "paid_cap_usd": _round(paid_cap_usd),
        "spend_policy": (
            "No paid Sestina LLM calls, pointwise calls, paid labeling, ledger "
            "rewrites, or paid-call artifact rewrites were performed. Cached "
            "and reviewed artifacts are used only as gate evidence."
        ),
        "method": {
            "decision_memo": "docs/internal/sestina-experiment-decision-memo.md",
            "related_work_audit": "docs/internal/related-work-audit.md",
            "active_arm_gate_harness": "sestina/active_arm_gate.py",
            "criteria": [
                "Only run the active-arm gate where the input has explicit "
                "zero-paid metadata, a paired random/exact-pool control, 20 "
                "seed-level metric deltas, the completed full-random variance "
                "reference, and core diagnostics.",
                "Record candidates without a complete no-paid replay/simulation "
                "artifact as blocked instead of inferring a pass from partial "
                "or paid single-seed context.",
                "Preserve random/exact-pool random as mandatory baselines and "
                "do not use seed-17 as a standalone comparator.",
            ],
            "output_path": output_path,
        },
        "random_baseline_policy": _random_baseline_policy(
            random_variance_artifact,
            random_variance_artifact_path=random_variance_artifact_path,
        ),
        "input_artifacts": {
            "ci_partition_gate": _artifact_ref(
                ci_partition_artifact,
                ci_partition_artifact_path,
            ),
            "random_variance_reference": _artifact_ref(
                random_variance_artifact,
                random_variance_artifact_path,
            ),
            "random_control_gap": _artifact_ref(
                random_control_gap_artifact,
                random_control_gap_artifact_path,
            ),
            "pairwise_strength_calibration": _artifact_ref(
                pairwise_strength_artifact,
                pairwise_strength_artifact_path,
            ),
            "posterior_decision_shrinkage": _artifact_ref(
                posterior_decision_artifact,
                posterior_decision_artifact_path,
            ),
        },
        "candidates": candidates,
        "summary": {
            "candidate_count": len(candidates),
            "status_counts": dict(sorted(status_counts.items())),
            "evaluated_with_active_arm_gate_count": sum(
                1
                for candidate in candidates
                if candidate["evaluated_with_active_arm_gate"]
            ),
            "any_candidate_paid_followup_allowed": any_paid_followup_allowed,
            "paid_followup_allowed_candidate_ids": [
                candidate["id"]
                for candidate in candidates
                if candidate["paid_followup_allowed"]
            ],
            "zero_paid_call_handoff": True,
        },
        "recommended_next_action": (
            "Do not start a paid active-arm workflow. Build a complete no-paid "
            "replay/simulation artifact for one blocked candidate, then rerun "
            "this shortlist gate and the active-arm gate before any paid labels."
        ),
    }
    validate_shortlist_gate_artifact_schema(payload)
    return payload


def validate_shortlist_gate_artifact_schema(payload: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(payload))
    if missing:
        raise ValueError(
            "shortlist gate artifact missing top-level keys: " + ", ".join(missing)
        )
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("shortlist gate artifact has unexpected artifact_type")
    if (
        payload.get("paid_calls_made") != 0
        or payload.get("paid_spend_usd") != 0.0
    ):
        raise ValueError("shortlist gate artifact must be zero-paid")
    if payload.get("pointwise_calls_made") != 0:
        raise ValueError("shortlist gate artifact must make zero pointwise calls")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(
            "shortlist gate artifact candidates must be a non-empty list"
        )
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"candidate {index} must be an object")
        missing_candidate = sorted(REQUIRED_CANDIDATE_KEYS - set(candidate))
        if missing_candidate:
            raise ValueError(
                f"candidate {index} missing keys: " + ", ".join(missing_candidate)
            )
        if candidate["status"] not in ALLOWED_STATUSES:
            raise ValueError(f"candidate {candidate['id']} has invalid status")
        if candidate["paid_followup_allowed"] is not False:
            raise ValueError(
                f"candidate {candidate['id']} unexpectedly allows paid follow-up"
            )
        if not isinstance(candidate["blocking_reasons"], list):
            raise ValueError(f"candidate {candidate['id']} blocking_reasons invalid")
        if not isinstance(candidate["missing_prerequisites"], list):
            raise ValueError(
                f"candidate {candidate['id']} missing_prerequisites invalid"
            )
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("shortlist gate artifact summary must be an object")
    if summary.get("any_candidate_paid_followup_allowed") is not False:
        raise ValueError("shortlist gate summary must block paid follow-up")


def _reliability_ci_candidate(
    *,
    ci_partition_artifact: Mapping[str, Any],
    ci_gate: Mapping[str, Any],
    ci_partition_artifact_path: str | None,
    random_variance_artifact_path: str | None,
) -> dict[str, Any]:
    gate_verdict = ci_gate["gate_verdict"]
    active_arm = ci_gate["active_arm_name"]
    random_control = ci_gate["candidate_random_control_baseline"]
    return {
        "id": "reliability_aware_ci_partition_v2",
        "name": "Reliability-aware CI partition v2",
        "memo_no_paid_gate": (
            "Cached replay must reduce unresolved boundary count, improve or "
            "preserve positive-negative oracle cap, and beat exact-pool random "
            "on paired Recall@K/nDCG without losing the randomized floor."
        ),
        "status": "evaluated_blocked",
        "evaluated_with_active_arm_gate": True,
        "gate_inputs": {
            "active_artifact_path": ci_partition_artifact_path,
            "active_artifact_type": ci_partition_artifact.get("artifact_type"),
            "active_arm_name": active_arm,
            "random_control_baseline": random_control,
            "random_variance_artifact_path": random_variance_artifact_path,
        },
        "paid_followup_allowed": False,
        "active_arm_gate": _compact_gate_result(ci_gate),
        "blocking_reasons": [
            *list(gate_verdict["blocking_reasons"]),
            (
                "The available CI replay is the current CI partition arm, not a "
                "new reliability-aware v2 artifact; it cannot justify paying "
                "for v2 labels."
            ),
        ],
        "missing_prerequisites": [
            (
                "Implement or produce a no-paid reliability-aware v2 replay "
                "artifact with explicit zero-paid metadata."
            ),
            (
                "Show lower unresolved boundary count and nonnegative "
                "positive-negative oracle-cap delta versus exact-pool random "
                "across the required seed set."
            ),
        ],
        "evidence_summary": _ci_evidence_summary(
            ci_partition_artifact,
            ci_gate=ci_gate,
        ),
        "recommended_next_action": (
            "Do not spend on CI partition v2. First change the reliability model "
            "offline, rerun cached replay, and require the active-arm gate to "
            "clear with nonnegative secondary metrics and no missing-label caveat."
        ),
    }


def _new_information_candidate(
    *,
    random_control_gap_artifact: Mapping[str, Any] | None,
    random_control_gap_artifact_path: str | None,
    random_variance_artifact_path: str | None,
) -> dict[str, Any]:
    return {
        "id": "new_information_challenger_construction",
        "name": "New-information challenger construction",
        "memo_no_paid_gate": (
            "Show better weak-bucket false-negative exposure and "
            "pointwise-plus-touched cap versus exact-pool random across seeds, "
            "without lowering Recall@K/nDCG in cached replay."
        ),
        "status": "blocked_missing_prerequisite",
        "evaluated_with_active_arm_gate": False,
        "gate_inputs": {
            "context_artifact_path": random_control_gap_artifact_path,
            "context_artifact_type": _mapping_get(
                random_control_gap_artifact,
                "artifact_type",
            ),
            "random_variance_artifact_path": random_variance_artifact_path,
            "active_arm_gate_reason_not_run": (
                "No complete zero-paid challenger-construction artifact with "
                "paired seed-level random-control deltas is available."
            ),
        },
        "paid_followup_allowed": False,
        "blocking_reasons": [
            (
                "No predeclared no-paid replay/simulation artifact exists for a "
                "new model-visible challenger construction policy."
            ),
            (
                "Existing expanded-pool and targeted-outsider rows are paid "
                "single-seed context, not a valid no-paid active-arm gate input."
            ),
            (
                "Available zero-paid diagnosis remains negative or inconclusive: "
                "expanded-pool and targeted-outsider posterior top-K Recall@K "
                "and nDCG@K trail exact-pool random."
            ),
        ],
        "missing_prerequisites": [
            (
                "Define the genuinely new model-visible challenger signal and "
                "feasible proposal pool before seeing future labels."
            ),
            (
                "Produce 20-seed cached/simulated replay rows with paired "
                "exact-pool or historical random controls."
            ),
            (
                "Aggregate weak-bucket false-negative exposure, "
                "pointwise-plus-touched oracle cap, Recall@K, nDCG@K, AP, and "
                "missing-label diagnostics."
            ),
        ],
        "evidence_summary": _challenger_context(random_control_gap_artifact),
        "recommended_next_action": (
            "Design a no-paid challenger simulator first. Do not buy labels for "
            "another pool change until the simulator shows decision-relevant "
            "false-negative exposure without a Recall@K/nDCG drop."
        ),
    }


def _aggregation_cross_check_candidate(
    *,
    pairwise_strength_artifact: Mapping[str, Any] | None,
    posterior_decision_artifact: Mapping[str, Any] | None,
    pairwise_strength_artifact_path: str | None,
    posterior_decision_artifact_path: str | None,
    random_variance_artifact_path: str | None,
) -> dict[str, Any]:
    return {
        "id": "aggregation_cross_check_standard_ranking_models",
        "name": "Aggregation cross-check against standard ranking models",
        "memo_no_paid_gate": (
            "Must improve Recall@K for at least one strong random-control "
            "baseline without hurting historical/exact random nDCG/AP or "
            "active-arm comparability."
        ),
        "status": "blocked_missing_prerequisite",
        "evaluated_with_active_arm_gate": False,
        "gate_inputs": {
            "pairwise_strength_context_artifact_path": (
                pairwise_strength_artifact_path
            ),
            "posterior_decision_context_artifact_path": (
                posterior_decision_artifact_path
            ),
            "random_variance_artifact_path": random_variance_artifact_path,
            "active_arm_gate_reason_not_run": (
                "Existing artifacts test internal posterior tweaks, not a "
                "standard BT/PL/Rank Centrality cross-check with paired seed "
                "deltas and active-arm comparability diagnostics."
            ),
        },
        "paid_followup_allowed": False,
        "blocking_reasons": [
            (
                "No artifact currently evaluates standard BT/PL/Rank Centrality "
                "implementations across the full random-control seed reference."
            ),
            (
                "Degree shrinkage and soft-strength calibration are zero-paid "
                "context but negative/inconclusive; they did not improve "
                "Recall@K on complete-label arms."
            ),
            (
                "The active-arm gate is not methodologically valid for these "
                "artifacts because they are aggregation diagnostics, not paired "
                "active acquisition arms with random-control deltas."
            ),
        ],
        "missing_prerequisites": [
            (
                "Implement a no-paid standard-ranking cross-check over cached "
                "pairwise labels without new paid labels."
            ),
            (
                "Report seed-level Recall@K/nDCG/AP against historical and "
                "exact-pool random controls with active-arm comparability caveats."
            ),
            (
                "Demonstrate a Recall@K gain without hurting historical/exact "
                "random nDCG/AP before considering any downstream paid arm."
            ),
        ],
        "evidence_summary": _aggregation_context(
            pairwise_strength_artifact=pairwise_strength_artifact,
            posterior_decision_artifact=posterior_decision_artifact,
        ),
        "recommended_next_action": (
            "Run the standard-model aggregation cross-check offline only. It "
            "should remain a diagnostic until it improves a strong random-control "
            "baseline without secondary-metric regressions."
        ),
    }


def _simulator_harness_candidate(
    *,
    ci_gate: Mapping[str, Any],
    ci_partition_artifact_path: str | None,
    random_variance_artifact_path: str | None,
) -> dict[str, Any]:
    return {
        "id": "active_arm_simulator_harness_gate_integration",
        "name": (
            "Active-arm simulator harness / gate integration for future paid runs"
        ),
        "memo_no_paid_gate": (
            "Harness must produce seed/bucket rows, paired deltas, CIs, "
            "missing-label caveats, and spend estimates from cached or simulated "
            "labels before any new runner is allowed to pay."
        ),
        "status": "infrastructure_ready_no_paid_arm",
        "evaluated_with_active_arm_gate": True,
        "gate_inputs": {
            "smoke_active_artifact_path": ci_partition_artifact_path,
            "random_variance_artifact_path": random_variance_artifact_path,
            "active_arm_name": ci_gate["active_arm_name"],
            "random_control_baseline": ci_gate[
                "candidate_random_control_baseline"
            ],
        },
        "paid_followup_allowed": False,
        "active_arm_gate": _compact_gate_result(ci_gate),
        "blocking_reasons": [
            (
                "The harness is infrastructure, not an active policy; it cannot "
                "by itself justify paid labels."
            ),
            (
                "The concrete smoke input remains blocked, so no future paid "
                "active arm is currently allowed."
            ),
        ],
        "missing_prerequisites": [
            (
                "Integrate future candidate runners so they emit the active-arm "
                "gate input shape before paid execution."
            ),
            (
                "Require a concrete candidate artifact to pass the gate before "
                "any pairwise-only paid dry run."
            ),
        ],
        "evidence_summary": {
            "harness_schema_validated": True,
            "produces_paid_followup_allowed": True,
            "produces_seed_level_confidence_intervals": bool(
                ci_gate.get("seed_level_confidence_intervals")
            ),
            "reports_missing_label_caveat": "missing_label_caveat"
            in ci_gate.get("caveats", {}),
            "reports_spend_estimate": bool(ci_gate.get("spend_estimate")),
            "reports_random_variance_reference": bool(
                ci_gate.get("random_variance_reference")
            ),
            "smoke_paid_followup_allowed": ci_gate["paid_followup_allowed"],
        },
        "recommended_next_action": (
            "Keep this harness as a mandatory pre-paid integration point. The "
            "next useful work is adapting one blocked candidate to emit complete "
            "no-paid gate evidence."
        ),
    }


def _compact_gate_result(gate: Mapping[str, Any]) -> dict[str, Any]:
    deltas = gate["paired_active_minus_random_deltas"]
    metric_deltas = deltas["metric_deltas"]
    return {
        "artifact_type": gate["artifact_type"],
        "paid_calls_made": gate["paid_calls_made"],
        "paid_spend_usd": gate["paid_spend_usd"],
        "active_arm_name": gate["active_arm_name"],
        "candidate_random_control_baseline": gate[
            "candidate_random_control_baseline"
        ],
        "paid_followup_allowed": gate["paid_followup_allowed"],
        "seed_count": gate["gate_verdict"]["seed_count"],
        "blocking_reasons": list(gate["gate_verdict"]["blocking_reasons"]),
        "metric_deltas": {
            metric: {
                "count": metric_deltas[metric]["count"],
                "mean": metric_deltas[metric]["mean"],
                "normal_approx_95_ci": metric_deltas[metric][
                    "normal_approx_95_ci"
                ],
            }
            for metric in METRICS
        },
        "missing_label_caveat_present": gate["gate_verdict"][
            "missing_label_caveat_present"
        ],
        "core_diagnostics_complete": gate["gate_verdict"][
            "core_diagnostics_complete"
        ],
        "full_random_variance_reference_complete": gate["gate_verdict"][
            "full_random_variance_reference_complete"
        ],
        "spend_estimate": gate["spend_estimate"],
    }


def _ci_evidence_summary(
    artifact: Mapping[str, Any],
    *,
    ci_gate: Mapping[str, Any],
) -> dict[str, Any]:
    active = ci_gate["active_arm_name"]
    random_control = ci_gate["candidate_random_control_baseline"]
    aggregate = _mapping(artifact.get("aggregate_diagnostics"))
    confidence = _mapping(aggregate.get("confidence_bound_unresolved_count"))
    oracle = _mapping(aggregate.get("oracle_caps"))
    randomized = _mapping(aggregate.get("randomized_coverage"))
    weak = _mapping(aggregate.get("weak_bucket_deltas"))
    active_confidence = _mapping(confidence.get(active))
    random_confidence = _mapping(confidence.get(random_control))
    active_oracle = _mapping(oracle.get(active))
    random_oracle = _mapping(oracle.get(random_control))
    active_floor = _mapping(randomized.get(active))
    return {
        "closest_available_artifact": artifact.get("artifact_type"),
        "paired_seed_count": ci_gate["gate_verdict"]["seed_count"],
        "mean_recall_delta": ci_gate["gate_verdict"]["mean_recall_delta"],
        "recall_delta_ci": ci_gate["gate_verdict"]["recall_delta_ci"],
        "mean_ndcg_delta": ci_gate["gate_verdict"]["mean_ndcg_delta"],
        "mean_average_precision_delta": ci_gate["gate_verdict"][
            "mean_average_precision_delta"
        ],
        "random_floor_rate": active_floor.get("random_floor_rate"),
        "unresolved_boundary_count_delta": _delta(
            active_confidence.get("mean"),
            random_confidence.get("mean"),
        ),
        "pointwise_plus_touched_oracle_cap_delta": _delta(
            active_oracle.get("mean_pointwise_plus_touched_recall_cap"),
            random_oracle.get("mean_pointwise_plus_touched_recall_cap"),
        ),
        "positive_negative_pair_oracle_cap_delta": _delta(
            active_oracle.get("mean_positive_negative_pair_recall_cap"),
            random_oracle.get("mean_positive_negative_pair_recall_cap"),
        ),
        "weak_bucket_pointwise_plus_touched_recall_cap_delta": weak.get(
            "mean_pointwise_plus_touched_recall_cap_delta"
        ),
        "weak_bucket_positive_negative_pair_recall_cap_delta": weak.get(
            "mean_positive_negative_pair_recall_cap_delta"
        ),
    }


def _challenger_context(artifact: Mapping[str, Any] | None) -> dict[str, Any]:
    if artifact is None:
        return {"available": False}
    arms = (
        "exact_pool_random",
        "expanded_pool_random",
        "targeted_outsider_random",
        "historical_random",
    )
    metrics = {}
    for arm in arms:
        row = _posterior_metrics(artifact, arm)
        if row:
            metrics[arm] = row
    exact = metrics.get("exact_pool_random", {})
    return {
        "available": True,
        "artifact_type": artifact.get("artifact_type"),
        "paid_calls_made": artifact.get("paid_calls_made"),
        "paid_spend_usd": artifact.get("paid_spend_usd"),
        "posterior_topk_metrics": metrics,
        "expanded_pool_minus_exact_pool_random": _metric_delta(
            metrics.get("expanded_pool_random", {}),
            exact,
        ),
        "targeted_outsider_minus_exact_pool_random": _metric_delta(
            metrics.get("targeted_outsider_random", {}),
            exact,
        ),
        "methodological_caveat": (
            "This diagnosis summarizes existing paid single-seed arm context "
            "under zero additional spend; it is not a complete no-paid active "
            "gate for a new challenger policy."
        ),
    }


def _aggregation_context(
    *,
    pairwise_strength_artifact: Mapping[str, Any] | None,
    posterior_decision_artifact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "pairwise_strength_calibration": _posterior_tweak_context(
            pairwise_strength_artifact,
            tweak_strategy="soft_strength_calibrated_posterior_topk",
        ),
        "posterior_decision_shrinkage": _posterior_tweak_context(
            posterior_decision_artifact,
            tweak_strategy="degree_shrunk_posterior_topk",
        ),
        "methodological_caveat": (
            "These are internal posterior-layer diagnostics, not standard "
            "BT/PL/Rank Centrality implementation checks."
        ),
    }


def _posterior_tweak_context(
    artifact: Mapping[str, Any] | None,
    *,
    tweak_strategy: str,
) -> dict[str, Any]:
    if artifact is None:
        return {"available": False}
    arms = (
        "historical_random",
        "exact_pool_random",
        "historical_active",
        "targeted_outsider_random",
    )
    deltas = {}
    improved_recall_arms = []
    for arm in arms:
        baseline = _strategy_metrics(artifact, arm, "posterior_topk")
        tweak = _strategy_metrics(artifact, arm, tweak_strategy)
        if baseline and tweak:
            delta = _metric_delta(tweak, baseline)
            deltas[arm] = delta
            if (delta.get("recall_at_k") or 0.0) > 0.0:
                improved_recall_arms.append(arm)
    return {
        "available": True,
        "artifact_type": artifact.get("artifact_type"),
        "paid_calls_made": artifact.get("paid_calls_made"),
        "paid_spend_usd": artifact.get("paid_spend_usd"),
        "tweak_strategy": tweak_strategy,
        "deltas_vs_posterior_topk": deltas,
        "recall_improved_complete_label_arms": improved_recall_arms,
    }


def _random_baseline_policy(
    artifact: Mapping[str, Any],
    *,
    random_variance_artifact_path: str | None,
) -> dict[str, Any]:
    aggregate = _mapping(artifact.get("aggregate_metrics"))
    return {
        "random_or_exact_pool_random_required": True,
        "seed_17_standalone_comparator_allowed": False,
        "stop_random_baseline_spending": True,
        "variance_reference_artifact_path": random_variance_artifact_path,
        "variance_reference_artifact_type": artifact.get("artifact_type"),
        "variance_reference_paid_calls_made": artifact.get("paid_calls_made"),
        "variance_reference_paid_spend_usd": artifact.get("paid_spend_usd"),
        "historical_random_full_schedule": _random_reference_metric_summary(
            aggregate,
            "historical_random_full_schedule",
        ),
        "exact_pool_random_full_schedule": _random_reference_metric_summary(
            aggregate,
            "exact_pool_random_full_schedule",
        ),
    }


def _random_reference_metric_summary(
    aggregate: Mapping[str, Any],
    arm: str,
) -> dict[str, Any]:
    intervals = _mapping(_mapping(aggregate.get(arm)).get("seed_level_intervals"))
    summary = {}
    for metric in METRICS:
        row = _mapping(intervals.get(metric))
        summary[metric] = {
            "count": row.get("count"),
            "mean": row.get("mean"),
            "normal_approx_95_ci": row.get("normal_approx_95_ci"),
            "bootstrap_percentile_95_ci": row.get("bootstrap_percentile_95_ci"),
        }
    return summary


def _artifact_ref(
    artifact: Mapping[str, Any] | None,
    path: str | None,
) -> dict[str, Any]:
    return {
        "path": path,
        "present": artifact is not None,
        "artifact_type": artifact.get("artifact_type") if artifact else None,
        "schema_version": artifact.get("schema_version") if artifact else None,
        "paid_calls_made": artifact.get("paid_calls_made") if artifact else None,
        "paid_spend_usd": artifact.get("paid_spend_usd") if artifact else None,
    }


def _posterior_metrics(artifact: Mapping[str, Any], arm: str) -> dict[str, Any]:
    return _strategy_metrics(artifact, arm, "posterior_topk")


def _strategy_metrics(
    artifact: Mapping[str, Any],
    arm: str,
    strategy: str,
) -> dict[str, Any]:
    metrics = _mapping(_mapping(artifact.get("aggregate_metrics")).get(arm))
    row = _mapping(metrics.get(strategy))
    return {
        metric: row[metric]
        for metric in METRICS
        if isinstance(row.get(metric), int | float)
    }


def _metric_delta(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, float]:
    return {
        metric: _round(float(left[metric]) - float(right[metric]))
        for metric in METRICS
        if isinstance(left.get(metric), int | float)
        and isinstance(right.get(metric), int | float)
    }


def _delta(left: Any, right: Any) -> float | None:
    if not isinstance(left, int | float) or not isinstance(right, int | float):
        return None
    return _round(float(left) - float(right))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_get(artifact: Mapping[str, Any] | None, key: str) -> Any:
    return artifact.get(key) if isinstance(artifact, Mapping) else None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _stdout_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_path": payload["method"]["output_path"],
        "artifact_type": payload["artifact_type"],
        "paid_calls_made": payload["paid_calls_made"],
        "paid_spend_usd": payload["paid_spend_usd"],
        "pointwise_calls_made": payload["pointwise_calls_made"],
        "known_paid_spend_usd": payload["known_paid_spend_usd"],
        "any_candidate_paid_followup_allowed": payload["summary"][
            "any_candidate_paid_followup_allowed"
        ],
        "candidate_statuses": {
            candidate["id"]: candidate["status"]
            for candidate in payload["candidates"]
        },
        "recommended_next_action": payload["recommended_next_action"],
    }


def _round(value: float) -> float:
    return round(float(value), 6)


if __name__ == "__main__":
    raise SystemExit(main())
