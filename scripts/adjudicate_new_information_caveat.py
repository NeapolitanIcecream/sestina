#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sestina.diagnostics import write_json_artifact  # noqa: E402


ARTIFACT_TYPE = "sestina-new-information-caveat-adjudication"
SCHEMA_VERSION = 1
ARM_NEW_INFO = "new_information_challenger_cached_replay"
ARM_EXACT = "exact_pool_random_cached_replay"
POSTERIOR_STRATEGY = "posterior_topk"
ALLOWED_DECISIONS = {
    "caveat_remains_blocking",
    "caveat_accepted_with_constraints",
    "requires_policy_revision",
}
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-new-information-caveat-adjudication"
    / "caveat-adjudication.json"
)
DEFAULT_BUDGET_FILL_ARTIFACT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-new-information-budget-fill-gate"
    / "new-information-budget-fill-gate.json"
)
DEFAULT_ACTIVE_GATE_ARTIFACT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-new-information-budget-fill-gate"
    / "active-arm-gate.json"
)
DEFAULT_DRY_RUN_ARTIFACT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-new-information-paid-dry-run"
    / "paid-dry-run-go-no-go.json"
)
DEFAULT_PLANNED_PAIRS = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-new-information-paid-dry-run"
    / "planned-pair-occurrences.jsonl"
)
DEFAULT_RANDOM_CONTROL_ARTIFACT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-random-control-diagnosis"
    / "random-control-gap-analysis.json"
)
DEFAULT_HISTORICAL_RESULTS_DOC = (
    REPO_ROOT / "docs" / "internal" / "historical-arxiv-pilot-results.md"
)
DEFAULT_DECISION_MEMO = (
    REPO_ROOT / "docs" / "internal" / "sestina-experiment-decision-memo.md"
)
REQUIRED_TOP_LEVEL_KEYS = {
    "artifact_type",
    "schema_version",
    "decision",
    "paid_calls_made",
    "paid_spend_usd",
    "pointwise_calls_made",
    "method",
    "input_artifacts",
    "diagnostics",
    "rationale",
    "constraints",
    "recommended_next_workflow",
    "validation_commands",
    "limitations",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Adjudicate the replay-local weak-bucket oracle-headroom caveat for "
            "the budget-filled new-information challenger without paid calls."
        )
    )
    parser.add_argument(
        "--budget-fill-artifact",
        type=Path,
        default=DEFAULT_BUDGET_FILL_ARTIFACT,
    )
    parser.add_argument(
        "--active-gate-artifact",
        type=Path,
        default=DEFAULT_ACTIVE_GATE_ARTIFACT,
    )
    parser.add_argument("--dry-run-artifact", type=Path, default=DEFAULT_DRY_RUN_ARTIFACT)
    parser.add_argument("--planned-pairs", type=Path, default=DEFAULT_PLANNED_PAIRS)
    parser.add_argument(
        "--random-control-artifact",
        type=Path,
        default=DEFAULT_RANDOM_CONTROL_ARTIFACT,
    )
    parser.add_argument(
        "--historical-results-doc",
        type=Path,
        default=DEFAULT_HISTORICAL_RESULTS_DOC,
    )
    parser.add_argument("--decision-memo", type=Path, default=DEFAULT_DECISION_MEMO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    payload = build_caveat_adjudication(
        budget_fill_artifact_path=args.budget_fill_artifact,
        active_gate_artifact_path=args.active_gate_artifact,
        dry_run_artifact_path=args.dry_run_artifact,
        planned_pairs_path=args.planned_pairs,
        random_control_artifact_path=args.random_control_artifact,
        historical_results_doc_path=args.historical_results_doc,
        decision_memo_path=args.decision_memo,
        output_path=args.output,
    )
    sys.stdout.write(json.dumps(_stdout_summary(payload), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def build_caveat_adjudication(
    *,
    budget_fill_artifact_path: Path,
    active_gate_artifact_path: Path,
    dry_run_artifact_path: Path,
    planned_pairs_path: Path,
    random_control_artifact_path: Path,
    historical_results_doc_path: Path,
    decision_memo_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    budget_fill = _read_json(budget_fill_artifact_path)
    active_gate = _read_json(active_gate_artifact_path)
    dry_run = _read_json(dry_run_artifact_path)
    random_control = _read_json(random_control_artifact_path)
    planned_pairs = _read_jsonl(planned_pairs_path)

    rows = build_row_diagnostics(budget_fill)
    planned_pair_summary = _planned_pair_summary(planned_pairs)
    diagnostics = {
        "active_gate_summary": _active_gate_summary(active_gate),
        "dry_run_summary": _dry_run_summary(dry_run),
        "replay_gate_tension": _replay_gate_tension(budget_fill),
        "row_classification_summary": _row_classification_summary(rows),
        "per_bucket_summary": _per_bucket_summary(rows),
        "lost_positive_touch_summary": _lost_positive_touch_summary(rows),
        "selected_positive_source_summary": _selected_positive_source_summary(rows),
        "ranking_gain_summary": _ranking_gain_summary(rows),
        "fallback_sensitivity": _fallback_sensitivity(rows, planned_pair_summary),
        "leave_one_bucket_out_sensitivity": _leave_one_bucket_out_sensitivity(rows),
        "prior_arm_oracle_predictiveness": _prior_arm_oracle_predictiveness(
            random_control
        ),
        "planned_pair_manifest_summary": planned_pair_summary,
    }
    decision = _decision(diagnostics)
    rationale = _rationale(decision=decision, diagnostics=diagnostics)
    constraints = _constraints(decision)
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 0,
        "known_paid_spend_before_workflow_usd": dry_run.get(
            "known_paid_spend_before_workflow_usd"
        ),
        "paid_cap_usd": dry_run.get("paid_cap_usd"),
        "method": {
            "summary": (
                "No-paid retrospective adjudication over existing reviewed "
                "budget-fill replay, active-gate, paid-dry-run, planned-pair, "
                "and random-control diagnostic artifacts."
            ),
            "future_labels_used_for_scheduling": False,
            "future_labels_used_for_model_visible_selection": False,
            "future_labels_used_for_retrospective_diagnostics_only": True,
            "paid_runner_invoked": False,
            "paid_labeling_invoked": False,
            "pointwise_runner_invoked": False,
        },
        "input_artifacts": {
            "budget_fill_artifact_path": str(budget_fill_artifact_path),
            "budget_fill_artifact_sha256": _sha256(budget_fill_artifact_path),
            "active_gate_artifact_path": str(active_gate_artifact_path),
            "active_gate_artifact_sha256": _sha256(active_gate_artifact_path),
            "dry_run_artifact_path": str(dry_run_artifact_path),
            "dry_run_artifact_sha256": _sha256(dry_run_artifact_path),
            "planned_pairs_path": str(planned_pairs_path),
            "planned_pairs_sha256": _sha256(planned_pairs_path),
            "random_control_artifact_path": str(random_control_artifact_path),
            "random_control_artifact_sha256": _sha256(random_control_artifact_path),
            "historical_results_doc_path": str(historical_results_doc_path),
            "decision_memo_path": str(decision_memo_path),
        },
        "diagnostics": diagnostics,
        "rationale": rationale,
        "constraints": constraints,
        "recommended_next_workflow": _recommended_next_workflow(decision),
        "limitations": [
            "This adjudication is retrospective and cannot prove fresh paid acquisition behavior.",
            "Future citation labels are used only after the frozen schedules for diagnostics.",
            "Fallback sensitivity is row-level and manifest-level; it does not recompute posterior top-K after deleting fallback labels.",
            "The random-control predictiveness check uses the prior one-seed complete-arm diagnosis, so it is supporting evidence rather than a gate replacement.",
            "The missing guarded 20-seed pairwise-only paid runner is documented as a separate guardrail gap and is not solved here.",
        ],
        "validation_commands": [
            "uv run python scripts/adjudicate_new_information_caveat.py",
            "uv run pytest tests/test_new_information_caveat_adjudication.py",
            "uv run python -m json.tool artifacts/backtest-arxiv-new-information-caveat-adjudication/caveat-adjudication.json >/dev/null",
            "git diff --check",
            "uv run pytest -p no:cacheprovider",
        ],
        "output_path": str(output_path),
    }
    validate_caveat_adjudication_artifact_schema(payload)
    write_json_artifact(output_path, payload)
    return payload


def validate_caveat_adjudication_artifact_schema(payload: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(payload))
    if missing:
        raise ValueError(
            "caveat adjudication artifact missing top-level keys: "
            + ", ".join(missing)
        )
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("caveat adjudication artifact has unexpected artifact_type")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("caveat adjudication artifact has unexpected schema_version")
    if payload.get("decision") not in ALLOWED_DECISIONS:
        raise ValueError("caveat adjudication artifact has invalid decision")
    if payload.get("paid_calls_made") != 0 or payload.get("paid_spend_usd") != 0.0:
        raise ValueError("caveat adjudication artifact must be zero-paid")
    if payload.get("pointwise_calls_made") != 0:
        raise ValueError("caveat adjudication artifact must make zero pointwise calls")
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("caveat adjudication diagnostics must be an object")
    required_diagnostics = {
        "active_gate_summary",
        "dry_run_summary",
        "replay_gate_tension",
        "row_classification_summary",
        "per_bucket_summary",
        "lost_positive_touch_summary",
        "selected_positive_source_summary",
        "ranking_gain_summary",
        "fallback_sensitivity",
        "prior_arm_oracle_predictiveness",
    }
    missing_diagnostics = sorted(required_diagnostics - set(diagnostics))
    if missing_diagnostics:
        raise ValueError(
            "caveat adjudication artifact missing diagnostics: "
            + ", ".join(missing_diagnostics)
        )
    constraints = payload.get("constraints")
    if not isinstance(constraints, list) or not constraints:
        raise ValueError("accepted/rejected caveat decision must include constraints")


def build_row_diagnostics(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_payload in payload["bucket_results"]:
        seed = int(seed_payload["seed"])
        for bucket_payload in seed_payload["buckets"]:
            bucket = str(bucket_payload["bucket"])
            k = int(bucket_payload["k"])
            active = bucket_payload["arms"][ARM_NEW_INFO]
            exact = bucket_payload["arms"][ARM_EXACT]
            active_metrics = active["metrics"][POSTERIOR_STRATEGY]
            exact_metrics = exact["metrics"][POSTERIOR_STRATEGY]
            active_error = active["top_k_error_decomposition"]
            exact_error = exact["top_k_error_decomposition"]
            active_exposure = active["positive_exposure"]
            exact_exposure = exact["positive_exposure"]
            active_oracle = active["oracle_bounds"]
            exact_oracle = exact["oracle_bounds"]
            active_touch_ids = set(active_exposure["touched_future_positive_ids"])
            exact_touch_ids = set(exact_exposure["touched_future_positive_ids"])
            active_selected_ids = set(active_error["selected_future_positive_ids"])
            exact_selected_ids = set(exact_error["selected_future_positive_ids"])
            active_false_negative_rows = {
                row["paper_id"]: row for row in active_error["false_negative_rows"]
            }
            lost_touch_ids = sorted(exact_touch_ids - active_touch_ids)
            gained_touch_ids = sorted(active_touch_ids - exact_touch_ids)
            lost_details = []
            active_pointwise_positive_ids = set(
                active_oracle.get("pointwise_top_k_positive_ids", [])
            )
            active_pointwise_plus_ids = _recoverable_ids(
                active_oracle,
                "pointwise_plus_touched_positive_upper_bound",
            )
            active_pair_oracle_ids = _recoverable_ids(
                active_oracle,
                "positive_negative_pair_label_oracle_upper_bound",
            )
            for paper_id in lost_touch_ids:
                false_negative = active_false_negative_rows.get(paper_id)
                lost_details.append(
                    {
                        "paper_id": paper_id,
                        "selected_by_new_information": paper_id
                        in active_selected_ids,
                        "selected_by_exact_pool_random": paper_id
                        in exact_selected_ids,
                        "pointwise_top_k_positive_under_new_information": paper_id
                        in active_pointwise_positive_ids,
                        "recoverable_under_new_information_pointwise_plus_touched_cap": (
                            paper_id in active_pointwise_plus_ids
                        ),
                        "recoverable_under_new_information_positive_negative_pair_cap": (
                            paper_id in active_pair_oracle_ids
                        ),
                        "new_information_false_negative": false_negative is not None,
                        "new_information_pair_degree": (
                            false_negative.get("pair_degree")
                            if false_negative is not None
                            else None
                        ),
                        "new_information_posterior_top_k_score": (
                            false_negative.get("posterior_top_k_score")
                            if false_negative is not None
                            else None
                        ),
                    }
                )
            selected_delta = int(
                active_error["selected_positive_count"]
                - exact_error["selected_positive_count"]
            )
            rows.append(
                {
                    "seed": seed,
                    "bucket": bucket,
                    "k": k,
                    "metric_deltas": {
                        metric: _round(float(active_metrics[metric]) - float(exact_metrics[metric]))
                        for metric in (
                            "recall_at_k",
                            "ndcg_at_k",
                            "average_precision",
                        )
                    },
                    "selected_positive_delta": selected_delta,
                    "active_selected_positive_count": int(
                        active_error["selected_positive_count"]
                    ),
                    "exact_selected_positive_count": int(
                        exact_error["selected_positive_count"]
                    ),
                    "oracle_cap_deltas": {
                        "pointwise_plus_touched": _round(
                            _recall_cap(
                                active_oracle,
                                "pointwise_plus_touched_positive_upper_bound",
                            )
                            - _recall_cap(
                                exact_oracle,
                                "pointwise_plus_touched_positive_upper_bound",
                            )
                        ),
                        "positive_negative_pair": _round(
                            _recall_cap(
                                active_oracle,
                                "positive_negative_pair_label_oracle_upper_bound",
                            )
                            - _recall_cap(
                                exact_oracle,
                                "positive_negative_pair_label_oracle_upper_bound",
                            )
                        ),
                        "observed_positive_winner": _round(
                            _recall_cap(
                                active_oracle,
                                "observed_positive_winner_upper_bound",
                            )
                            - _recall_cap(
                                exact_oracle,
                                "observed_positive_winner_upper_bound",
                            )
                        ),
                    },
                    "touch_deltas": {
                        "unique_future_positives_touched": int(
                            active_exposure["unique_future_positives_touched"]
                        )
                        - int(exact_exposure["unique_future_positives_touched"]),
                        "pairs_touching_future_positive": int(
                            active_exposure["pairs_touching_future_positive"]
                        )
                        - int(exact_exposure["pairs_touching_future_positive"]),
                        "positive_negative_pairs": int(
                            active_exposure["positive_negative_pairs"]
                        )
                        - int(exact_exposure["positive_negative_pairs"]),
                    },
                    "selected_positive_sources": {
                        "active_selected_touched_count": len(
                            active_selected_ids & active_touch_ids
                        ),
                        "active_selected_untouched_count": len(
                            active_selected_ids - active_touch_ids
                        ),
                        "exact_selected_touched_count": len(
                            exact_selected_ids & exact_touch_ids
                        ),
                        "exact_selected_untouched_count": len(
                            exact_selected_ids - exact_touch_ids
                        ),
                    },
                    "active_posterior_positive_ranks": _positive_ranks(
                        active_error["posterior_top_k_ids"],
                        active_selected_ids,
                    ),
                    "exact_posterior_positive_ranks": _positive_ranks(
                        exact_error["posterior_top_k_ids"],
                        exact_selected_ids,
                    ),
                    "new_information_scheduler": {
                        "fallback_selected_total": int(
                            active["scheduler_diagnostics"]
                            .get("cached_frontier_fallback", {})
                            .get("selected_total", 0)
                        ),
                        "primary_scheduled_pairwise_shortfall": int(
                            active["scheduler_diagnostics"]
                            .get("new_information_challenger", {})
                            .get("primary_scheduled_pairwise_shortfall", 0)
                        ),
                        "purpose_counts": active["scheduler_diagnostics"].get(
                            "purpose_counts", {}
                        ),
                    },
                    "lost_future_positive_touch_ids": lost_touch_ids,
                    "gained_future_positive_touch_ids": gained_touch_ids,
                    "lost_future_positive_details": lost_details,
                }
            )
    return rows


def _active_gate_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    gate = payload["gate_verdict"]
    intervals = payload["seed_level_confidence_intervals"]
    return {
        "paid_followup_allowed": bool(payload["paid_followup_allowed"]),
        "blocking_reasons": gate.get("blocking_reasons", []),
        "seed_count": gate.get("seed_count"),
        "mean_recall_delta": gate.get("mean_recall_delta"),
        "recall_delta_ci": gate.get("recall_delta_ci"),
        "mean_ndcg_delta": gate.get("mean_ndcg_delta"),
        "ndcg_delta_ci": intervals.get("ndcg_at_k", {}).get("normal_approx_95_ci"),
        "mean_average_precision_delta": gate.get("mean_average_precision_delta"),
        "average_precision_delta_ci": intervals.get("average_precision", {}).get(
            "normal_approx_95_ci"
        ),
        "budget_completeness_caveat_present": gate.get(
            "budget_completeness_caveat_present"
        ),
        "missing_label_caveat_present": gate.get("missing_label_caveat_present"),
    }


def _dry_run_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    totals = payload["totals"]
    guardrails = payload["guardrails"]
    go_no_go = payload["go_no_go"]
    return {
        "decision": go_no_go["decision"],
        "caveat_blocking_reasons": go_no_go.get("caveat_blocking_reasons", []),
        "guardrail_blocking_reasons": go_no_go.get("guardrail_blocking_reasons", []),
        "planned_pair_occurrence_count": payload["planned_pair_occurrence_count"],
        "unique_planned_pair_labels": totals["unique_planned_pair_labels"],
        "unique_missing_pairwise_labels": totals["unique_missing_pairwise_labels"],
        "estimated_additional_spend_usd": totals["estimated_additional_spend_usd"],
        "pointwise_calls": totals["pointwise_calls"],
        "active_budget_shortfall": totals["active_budget_shortfall"],
        "random_control_budget_shortfall": totals["random_control_budget_shortfall"],
        "guarded_runner_ready": guardrails["checks"][
            "guarded_pairwise_runner_ready_for_new_information"
        ],
        "guarded_runner_note": guardrails["guarded_runner_note"],
    }


def _replay_gate_tension(payload: Mapping[str, Any]) -> dict[str, Any]:
    replay_gate = payload["new_information_replay_gate_verdict"]
    weak = payload["aggregate_diagnostics"]["weak_bucket_deltas"]
    active_gate = payload["gate_verdict"]
    return {
        "reviewed_active_gate_paid_followup_allowed": active_gate[
            "paid_followup_allowed"
        ],
        "local_replay_gate_paid_followup_allowed": replay_gate[
            "paid_followup_allowed"
        ],
        "local_replay_gate_blocking_reasons": replay_gate["blocking_reasons"],
        "credible_metric_improvement": replay_gate["credible_metric_improvement"],
        "weak_oracle_headroom_preserved": replay_gate[
            "weak_oracle_headroom_preserved"
        ],
        "weak_bucket_row_count": weak["row_count"],
        "selected_positive_delta_total": weak["selected_positive_delta_total"],
        "unique_future_positives_touched_delta_total": weak[
            "unique_future_positives_touched_delta_total"
        ],
        "mean_pointwise_plus_touched_recall_cap_delta": weak[
            "mean_pointwise_plus_touched_recall_cap_delta"
        ],
        "mean_positive_negative_pair_recall_cap_delta": weak[
            "mean_positive_negative_pair_recall_cap_delta"
        ],
    }


def _row_classification_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    for row in rows:
        selected_delta = int(row["selected_positive_delta"])
        touch_delta = int(row["touch_deltas"]["unique_future_positives_touched"])
        pointwise_cap_delta = float(row["oracle_cap_deltas"]["pointwise_plus_touched"])
        pair_cap_delta = float(row["oracle_cap_deltas"]["positive_negative_pair"])
        ndcg_delta = float(row["metric_deltas"]["ndcg_at_k"])
        if selected_delta > 0:
            counters["rows_with_selected_positive_gain"] += 1
        elif selected_delta < 0:
            counters["rows_with_selected_positive_loss"] += 1
        else:
            counters["rows_with_selected_positive_tie"] += 1
        if touch_delta < 0:
            counters["rows_with_touch_loss"] += 1
        elif touch_delta > 0:
            counters["rows_with_touch_gain"] += 1
        else:
            counters["rows_with_touch_tie"] += 1
        if pointwise_cap_delta < 0 or pair_cap_delta < 0:
            counters["rows_with_any_oracle_cap_loss"] += 1
        if selected_delta > 0 and touch_delta <= 0:
            counters["rows_with_selected_gain_despite_no_touch_gain"] += 1
        if selected_delta > 0 and (pointwise_cap_delta < 0 or pair_cap_delta < 0):
            counters["rows_with_selected_gain_despite_oracle_cap_loss"] += 1
        if selected_delta == 0 and ndcg_delta > 0:
            counters["rows_with_same_recall_but_ndcg_gain"] += 1
    return {
        "row_count": len(rows),
        **dict(sorted(counters.items())),
        "selected_positive_delta_total": sum(
            int(row["selected_positive_delta"]) for row in rows
        ),
        "touch_delta_total": sum(
            int(row["touch_deltas"]["unique_future_positives_touched"])
            for row in rows
        ),
        "pointwise_plus_touched_cap_delta_total_positive_count_equivalent": _round(
            sum(
                float(row["oracle_cap_deltas"]["pointwise_plus_touched"])
                * int(row["k"])
                for row in rows
            )
        ),
        "positive_negative_pair_cap_delta_total_positive_count_equivalent": _round(
            sum(
                float(row["oracle_cap_deltas"]["positive_negative_pair"])
                * int(row["k"])
                for row in rows
            )
        ),
    }


def _per_bucket_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["bucket"])].append(row)
    output = []
    for bucket, bucket_rows in sorted(grouped.items()):
        output.append(
            {
                "bucket": bucket,
                "row_count": len(bucket_rows),
                "mean_recall_delta": _mean_metric(bucket_rows, "recall_at_k"),
                "mean_ndcg_delta": _mean_metric(bucket_rows, "ndcg_at_k"),
                "mean_average_precision_delta": _mean_metric(
                    bucket_rows,
                    "average_precision",
                ),
                "selected_positive_delta_total": sum(
                    int(row["selected_positive_delta"]) for row in bucket_rows
                ),
                "touch_delta_total": sum(
                    int(row["touch_deltas"]["unique_future_positives_touched"])
                    for row in bucket_rows
                ),
                "mean_pointwise_plus_touched_cap_delta": _mean_oracle(
                    bucket_rows,
                    "pointwise_plus_touched",
                ),
                "mean_positive_negative_pair_cap_delta": _mean_oracle(
                    bucket_rows,
                    "positive_negative_pair",
                ),
                "rows_with_selected_positive_gain": sum(
                    1 for row in bucket_rows if int(row["selected_positive_delta"]) > 0
                ),
                "rows_with_touch_loss": sum(
                    1
                    for row in bucket_rows
                    if int(row["touch_deltas"]["unique_future_positives_touched"]) < 0
                ),
                "fallback_selected_total": sum(
                    int(row["new_information_scheduler"]["fallback_selected_total"])
                    for row in bucket_rows
                ),
            }
        )
    return output


def _lost_positive_touch_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lost_occurrences = []
    gained_occurrences = []
    for row in rows:
        for detail in row["lost_future_positive_details"]:
            lost_occurrences.append({**detail, "bucket": row["bucket"], "seed": row["seed"]})
        for paper_id in row["gained_future_positive_touch_ids"]:
            gained_occurrences.append(
                {"paper_id": paper_id, "bucket": row["bucket"], "seed": row["seed"]}
            )
    lost_by_bucket = Counter(str(row["bucket"]) for row in lost_occurrences)
    gained_by_bucket = Counter(str(row["bucket"]) for row in gained_occurrences)
    lost_selected = [
        row for row in lost_occurrences if row["selected_by_new_information"]
    ]
    lost_false_negative = [
        row for row in lost_occurrences if row["new_information_false_negative"]
    ]
    return {
        "lost_touch_occurrences": len(lost_occurrences),
        "gained_touch_occurrences": len(gained_occurrences),
        "net_touch_occurrence_delta": len(gained_occurrences) - len(lost_occurrences),
        "unique_lost_future_positive_ids": len(
            {str(row["paper_id"]) for row in lost_occurrences}
        ),
        "unique_gained_future_positive_ids": len(
            {str(row["paper_id"]) for row in gained_occurrences}
        ),
        "lost_occurrences_still_selected_by_new_information": len(lost_selected),
        "lost_occurrences_selected_by_exact_pool_random": sum(
            1 for row in lost_occurrences if row["selected_by_exact_pool_random"]
        ),
        "lost_occurrences_pointwise_top_k_under_new_information": sum(
            1
            for row in lost_occurrences
            if row["pointwise_top_k_positive_under_new_information"]
        ),
        "lost_occurrences_recoverable_under_new_pointwise_plus_touched_cap": sum(
            1
            for row in lost_occurrences
            if row["recoverable_under_new_information_pointwise_plus_touched_cap"]
        ),
        "lost_occurrences_recoverable_under_new_positive_negative_pair_cap": sum(
            1
            for row in lost_occurrences
            if row["recoverable_under_new_information_positive_negative_pair_cap"]
        ),
        "lost_occurrences_new_information_false_negative": len(lost_false_negative),
        "lost_occurrences_new_information_false_negative_zero_degree": sum(
            1
            for row in lost_false_negative
            if int(row["new_information_pair_degree"] or 0) == 0
        ),
        "lost_touch_by_bucket": dict(sorted(lost_by_bucket.items())),
        "gained_touch_by_bucket": dict(sorted(gained_by_bucket.items())),
        "top_lost_positive_occurrences": _top_occurrences(lost_occurrences),
        "top_gained_positive_occurrences": _top_occurrences(gained_occurrences),
    }


def _selected_positive_source_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    active_touched = sum(
        int(row["selected_positive_sources"]["active_selected_touched_count"])
        for row in rows
    )
    active_untouched = sum(
        int(row["selected_positive_sources"]["active_selected_untouched_count"])
        for row in rows
    )
    exact_touched = sum(
        int(row["selected_positive_sources"]["exact_selected_touched_count"])
        for row in rows
    )
    exact_untouched = sum(
        int(row["selected_positive_sources"]["exact_selected_untouched_count"])
        for row in rows
    )
    return {
        "active_selected_positive_occurrences": active_touched + active_untouched,
        "exact_selected_positive_occurrences": exact_touched + exact_untouched,
        "active_selected_touched_positive_occurrences": active_touched,
        "active_selected_untouched_positive_occurrences": active_untouched,
        "exact_selected_touched_positive_occurrences": exact_touched,
        "exact_selected_untouched_positive_occurrences": exact_untouched,
        "selected_positive_occurrence_delta": (
            active_touched + active_untouched - exact_touched - exact_untouched
        ),
        "selected_touched_positive_occurrence_delta": active_touched - exact_touched,
        "selected_untouched_positive_occurrence_delta": (
            active_untouched - exact_untouched
        ),
    }


def _ranking_gain_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    same_recall_ndcg_gain = [
        row
        for row in rows
        if int(row["selected_positive_delta"]) == 0
        and float(row["metric_deltas"]["ndcg_at_k"]) > 0.0
    ]
    recall_gain_oracle_loss = [
        row
        for row in rows
        if int(row["selected_positive_delta"]) > 0
        and (
            float(row["oracle_cap_deltas"]["pointwise_plus_touched"]) < 0.0
            or float(row["oracle_cap_deltas"]["positive_negative_pair"]) < 0.0
        )
    ]
    return {
        "rows_with_same_recall_but_ndcg_gain": len(same_recall_ndcg_gain),
        "mean_ndcg_delta_when_recall_tied": _mean(
            [
                float(row["metric_deltas"]["ndcg_at_k"])
                for row in rows
                if int(row["selected_positive_delta"]) == 0
            ]
        ),
        "mean_ap_delta_when_recall_tied": _mean(
            [
                float(row["metric_deltas"]["average_precision"])
                for row in rows
                if int(row["selected_positive_delta"]) == 0
            ]
        ),
        "rows_with_recall_gain_and_oracle_cap_loss": len(recall_gain_oracle_loss),
        "mean_positive_rank_sum_delta": _round(
            _mean(
                [
                    sum(row["exact_posterior_positive_ranks"])
                    - sum(row["active_posterior_positive_ranks"])
                    for row in rows
                    if int(row["selected_positive_delta"]) == 0
                    and row["active_posterior_positive_ranks"]
                    and row["exact_posterior_positive_ranks"]
                ]
            )
        ),
        "interpretation": (
            "Positive rank-sum delta means the new-information arm ranks the same "
            "number of selected positives earlier inside top-K."
        ),
    }


def _fallback_sensitivity(
    rows: Sequence[Mapping[str, Any]],
    planned_pair_summary: Mapping[str, Any],
) -> dict[str, Any]:
    fallback_rows = [
        row
        for row in rows
        if int(row["new_information_scheduler"]["fallback_selected_total"]) > 0
    ]
    nonfallback_rows = [
        row
        for row in rows
        if int(row["new_information_scheduler"]["fallback_selected_total"]) == 0
    ]
    return {
        "fallback_rows": _row_subset_summary(fallback_rows),
        "nonfallback_rows": _row_subset_summary(nonfallback_rows),
        "planned_pair_purpose_counts": planned_pair_summary["purpose_counts"],
        "fallback_manifest_occurrences": planned_pair_summary["purpose_counts"].get(
            "new_information_cached_frontier_fallback",
            0,
        ),
        "exact_ablation_feasible_without_recomputing_posterior": False,
        "reason_exact_ablation_not_claimed": (
            "Removing fallback comparisons changes the posterior graph and would "
            "also reintroduce an active budget shortfall; this artifact therefore "
            "reports row-level sensitivity instead of a counterfactual gate."
        ),
    }


def _leave_one_bucket_out_sensitivity(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    buckets = sorted({str(row["bucket"]) for row in rows})
    output = []
    for bucket in buckets:
        kept = [row for row in rows if row["bucket"] != bucket]
        seed_metric_rows: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in kept:
            seed_metric_rows[int(row["seed"])].append(row)
        seed_recall_deltas = [
            _mean([float(row["metric_deltas"]["recall_at_k"]) for row in seed_rows])
            for seed_rows in seed_metric_rows.values()
        ]
        seed_ndcg_deltas = [
            _mean([float(row["metric_deltas"]["ndcg_at_k"]) for row in seed_rows])
            for seed_rows in seed_metric_rows.values()
        ]
        seed_ap_deltas = [
            _mean(
                [
                    float(row["metric_deltas"]["average_precision"])
                    for row in seed_rows
                ]
            )
            for seed_rows in seed_metric_rows.values()
        ]
        output.append(
            {
                "excluded_bucket": bucket,
                "remaining_row_count": len(kept),
                "recall_delta": _summary_with_ci(seed_recall_deltas),
                "ndcg_delta": _summary_with_ci(seed_ndcg_deltas),
                "average_precision_delta": _summary_with_ci(seed_ap_deltas),
                "selected_positive_delta_total": sum(
                    int(row["selected_positive_delta"]) for row in kept
                ),
                "touch_delta_total": sum(
                    int(row["touch_deltas"]["unique_future_positives_touched"])
                    for row in kept
                ),
                "mean_pointwise_plus_touched_cap_delta": _mean_oracle(
                    kept,
                    "pointwise_plus_touched",
                ),
                "mean_positive_negative_pair_cap_delta": _mean_oracle(
                    kept,
                    "positive_negative_pair",
                ),
            }
        )
    return output


def _prior_arm_oracle_predictiveness(payload: Mapping[str, Any]) -> dict[str, Any]:
    posterior = payload["random_control_gap_diagnosis"]["posterior_topk_metric_table"]
    oracle = payload["aggregate_diagnostics"]["oracle_bounds"]
    rows = []
    for arm, metrics in posterior.items():
        if arm not in oracle:
            continue
        rows.append(
            {
                "arm": arm,
                "recall_at_k": float(metrics["recall_at_k"]),
                "ndcg_at_k": float(metrics["ndcg_at_k"]),
                "average_precision": float(metrics["average_precision"]),
                "pointwise_plus_touched_cap": float(
                    oracle[arm]["mean_pointwise_plus_touched_positive_recall_cap"]
                ),
                "positive_negative_pair_cap": float(
                    oracle[arm][
                        "mean_positive_negative_pair_label_oracle_recall_cap"
                    ]
                ),
                "observed_positive_winner_cap": float(
                    oracle[arm]["mean_observed_positive_winner_recall_cap"]
                ),
            }
        )
    return {
        "source": (
            "prior one-seed random-control diagnosis; included only to test "
            "whether oracle caps were monotonic enough to be a hard blocker"
        ),
        "arm_count": len(rows),
        "spearman_recall_vs_pointwise_plus_touched_cap": _spearman(
            [row["pointwise_plus_touched_cap"] for row in rows],
            [row["recall_at_k"] for row in rows],
        ),
        "spearman_recall_vs_positive_negative_pair_cap": _spearman(
            [row["positive_negative_pair_cap"] for row in rows],
            [row["recall_at_k"] for row in rows],
        ),
        "rows": rows,
        "counterexamples": [
            row
            for row in rows
            if row["pointwise_plus_touched_cap"] >= 0.55 and row["recall_at_k"] < 0.35
        ],
        "interpretation": (
            "Prior complete-arm rows show oracle-cap headroom is useful as a "
            "safety diagnostic but not monotonic enough to override a reviewed "
            "paired top-K metric gate by itself."
        ),
    }


def _planned_pair_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    purpose_counts = Counter(str(row.get("purpose")) for row in rows)
    source_purpose_counts = Counter(
        str(row.get("source_new_information_purpose")) for row in rows
    )
    cache_status_counts = Counter(str(row.get("cache_status")) for row in rows)
    bucket_counts = Counter(str(row.get("bucket")) for row in rows)
    unique_pairs = {
        (str(row.get("bucket")), tuple(row.get("pair_key", []))) for row in rows
    }
    return {
        "occurrence_count": len(rows),
        "unique_same_bucket_pair_labels": len(unique_pairs),
        "purpose_counts": dict(sorted(purpose_counts.items())),
        "source_new_information_purpose_counts": dict(
            sorted(source_purpose_counts.items())
        ),
        "cache_status_counts": dict(sorted(cache_status_counts.items())),
        "bucket_occurrence_counts": dict(sorted(bucket_counts.items())),
    }


def _decision(diagnostics: Mapping[str, Any]) -> str:
    active = diagnostics["active_gate_summary"]
    dry_run = diagnostics["dry_run_summary"]
    tension = diagnostics["replay_gate_tension"]
    selected_sources = diagnostics["selected_positive_source_summary"]
    prior = diagnostics["prior_arm_oracle_predictiveness"]
    if active["paid_followup_allowed"] is not True:
        return "caveat_remains_blocking"
    if dry_run["unique_missing_pairwise_labels"] != 0:
        return "requires_policy_revision"
    if tension["selected_positive_delta_total"] <= 0:
        return "caveat_remains_blocking"
    if selected_sources["selected_positive_occurrence_delta"] <= 0:
        return "caveat_remains_blocking"
    if prior["spearman_recall_vs_pointwise_plus_touched_cap"] >= 0.8:
        return "requires_policy_revision"
    return "caveat_accepted_with_constraints"


def _rationale(*, decision: str, diagnostics: Mapping[str, Any]) -> list[str]:
    active = diagnostics["active_gate_summary"]
    dry_run = diagnostics["dry_run_summary"]
    tension = diagnostics["replay_gate_tension"]
    row_summary = diagnostics["row_classification_summary"]
    lost = diagnostics["lost_positive_touch_summary"]
    selected = diagnostics["selected_positive_source_summary"]
    prior = diagnostics["prior_arm_oracle_predictiveness"]
    if decision != "caveat_accepted_with_constraints":
        return [
            "The reviewed active-arm gate or selected-positive evidence did not clear the adjudication criteria.",
            "Keep the replay-local weak-bucket oracle-headroom caveat blocking until the simulator/policy is revised.",
        ]
    return [
        (
            "The reviewed active-arm gate already cleared the primary paid-followup "
            "criteria: Recall@K delta "
            f"{active['mean_recall_delta']:+.6f} with CI "
            f"{active['recall_delta_ci']}, nDCG@K delta "
            f"{active['mean_ndcg_delta']:+.6f}, AP delta "
            f"{active['mean_average_precision_delta']:+.6f}, 20 seeds, no "
            "missing-label caveat, and no budget-completeness caveat."
        ),
        (
            "The weak-bucket caveat is real but explains broad retrospective "
            "headroom, not the observed posterior top-K outcome: selected future "
            f"positive occurrences rose by {tension['selected_positive_delta_total']} "
            "even as unique future-positive touch occurrences fell by "
            f"{tension['unique_future_positives_touched_delta_total']}."
        ),
        (
            "The lost-touch positives are real misses, not a metric artifact: "
            f"{lost['lost_occurrences_still_selected_by_new_information']} of "
            f"{lost['lost_touch_occurrences']} lost-touch occurrences are still "
            "selected by the new-information posterior top-K and "
            f"{lost['lost_occurrences_new_information_false_negative_zero_degree']} "
            "lost-touch occurrences become zero-degree active false negatives. "
            "The caveat is accepted only because these misses are offset by a net "
            "selected-positive occurrence gain of "
            f"{selected['selected_positive_occurrence_delta']}."
        ),
        (
            "Row-level diagnostics show the gate improvement is not merely extra "
            "broad exposure: "
            f"{row_summary['rows_with_selected_gain_despite_no_touch_gain']} rows "
            "gain selected positives despite no unique-touch gain, and "
            f"{row_summary['rows_with_same_recall_but_ndcg_gain']} rows improve "
            "nDCG with the same positive count."
        ),
        (
            "Historical complete-arm diagnostics make oracle cap headroom too "
            "non-monotonic to use as a hard override by itself; the one-seed "
            "Spearman correlation between recall and pointwise-plus-touched cap "
            f"is {prior['spearman_recall_vs_pointwise_plus_touched_cap']:+.6f}."
        ),
        (
            "The dry-run found no labels to buy for the frozen manifest "
            f"({dry_run['unique_missing_pairwise_labels']} unique missing labels, "
            f"USD {dry_run['estimated_additional_spend_usd']:.6f} estimated "
            "additional spend), so accepting the caveat does not authorize paid "
            "label purchase by this artifact."
        ),
    ]


def _constraints(decision: str) -> list[str]:
    common = [
        "Do not weaken or bypass the reviewed active-arm gate.",
        "Use future labels only for retrospective diagnostics, never for scheduling or model-visible selection.",
        "Make zero pointwise calls in any follow-up tied to this manifest.",
        "Do not rewrite historical paid ledgers or paid-call artifacts.",
        "Keep the missing guarded 20-seed pairwise-only paid runner as a separate unresolved guardrail until reviewed implementation exists.",
    ]
    if decision != "caveat_accepted_with_constraints":
        return common + [
            "Do not proceed to any paid active execution until the weak-bucket oracle-headroom caveat is resolved by policy/simulator change or a new reviewed adjudication."
        ]
    return common + [
        "Acceptance is scoped only to the frozen budget-filled new-information manifest and its current reviewed artifacts.",
        "Acceptance permits reviewer consideration of a later workflow only after a separate go/no-go artifact clears all non-caveat guardrails.",
        "Because the dry-run found zero unique missing labels, this adjudication authorizes no paid label purchase by itself.",
        "Any later paid workflow must use a reviewed guarded pairwise-only runner, provider model availability checks, JSONL ledger, hard max-usd cap, separate artifact directory, and abort on any pointwise-call attempt.",
        "If a later manifest has missing labels, rerun the dry-run and this caveat adjudication before buying them; do not transfer this acceptance to a changed schedule.",
        "Report weak-bucket oracle caps, unique-positive touch deltas, selected-positive deltas, fallback usage, parse/retry counts, and paired seed-level Recall/nDCG/AP intervals in the follow-up handoff.",
    ]


def _recommended_next_workflow(decision: str) -> str:
    if decision == "caveat_accepted_with_constraints":
        return (
            "Treat the weak-bucket oracle-headroom caveat as an accepted, "
            "reviewer-auditable risk for this frozen zero-missing-label manifest. "
            "Do not run paid calls now; next work is a separate guarded "
            "20-seed pairwise-only runner/go-no-go workflow, which remains "
            "out of scope here."
        )
    if decision == "requires_policy_revision":
        return (
            "Revise the policy/simulator so the replay-local weak-bucket caveat "
            "and reviewed active-arm gate have a single predeclared decision role "
            "before any paid workflow."
        )
    return (
        "Keep the caveat blocking and revise the no-paid active policy before any paid workflow."
    )


def _row_subset_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "mean_recall_delta": _mean_metric(rows, "recall_at_k"),
        "mean_ndcg_delta": _mean_metric(rows, "ndcg_at_k"),
        "mean_average_precision_delta": _mean_metric(rows, "average_precision"),
        "selected_positive_delta_total": sum(
            int(row["selected_positive_delta"]) for row in rows
        ),
        "touch_delta_total": sum(
            int(row["touch_deltas"]["unique_future_positives_touched"])
            for row in rows
        ),
        "mean_pointwise_plus_touched_cap_delta": _mean_oracle(
            rows,
            "pointwise_plus_touched",
        ),
        "mean_positive_negative_pair_cap_delta": _mean_oracle(
            rows,
            "positive_negative_pair",
        ),
    }


def _top_occurrences(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row["paper_id"]) for row in rows)
    buckets_by_paper: dict[str, str] = {}
    for row in rows:
        buckets_by_paper.setdefault(str(row["paper_id"]), str(row["bucket"]))
    return [
        {
            "paper_id": paper_id,
            "bucket": buckets_by_paper[paper_id],
            "occurrences": count,
        }
        for paper_id, count in counts.most_common(12)
    ]


def _mean_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    return _mean([float(row["metric_deltas"][metric]) for row in rows])


def _mean_oracle(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    return _mean([float(row["oracle_cap_deltas"][metric]) for row in rows])


def _recoverable_ids(oracle: Mapping[str, Any], key: str) -> set[str]:
    return set(oracle[key].get("recoverable_positive_ids", []))


def _recall_cap(oracle: Mapping[str, Any], key: str) -> float:
    return float(oracle[key]["recall_cap"])


def _positive_ranks(ids: Sequence[str], positive_ids: set[str]) -> list[int]:
    return [index for index, paper_id in enumerate(ids, start=1) if paper_id in positive_ids]


def _summary_with_ci(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    if len(values) == 1:
        value = _round(values[0])
        return {
            "count": 1,
            "mean": value,
            "min": value,
            "max": value,
            "standard_error": 0.0,
            "normal_approx_95_ci": [value, value],
        }
    se = stdev(values) / math.sqrt(len(values))
    avg = mean(values)
    return {
        "count": len(values),
        "mean": _round(avg),
        "min": _round(min(values)),
        "max": _round(max(values)),
        "standard_error": _round(se),
        "normal_approx_95_ci": [_round(avg - 1.96 * se), _round(avg + 1.96 * se)],
    }


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_ranks = _average_ranks(xs)
    y_ranks = _average_ranks(ys)
    x_mean = mean(x_ranks)
    y_mean = mean(y_ranks)
    numerator = sum(
        (x_rank - x_mean) * (y_rank - y_mean)
        for x_rank, y_rank in zip(x_ranks, y_ranks)
    )
    x_den = math.sqrt(sum((x_rank - x_mean) ** 2 for x_rank in x_ranks))
    y_den = math.sqrt(sum((y_rank - y_mean) ** 2 for y_rank in y_ranks))
    if x_den == 0.0 or y_den == 0.0:
        return 0.0
    return _round(numerator / (x_den * y_den))


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = avg_rank
        index = end
    return ranks


def _mean(values: Iterable[float]) -> float:
    concrete = list(values)
    if not concrete:
        return 0.0
    return _round(mean(concrete))


def _round(value: float) -> float:
    return round(float(value), 8)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stdout_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = payload["diagnostics"]
    return {
        "artifact_type": payload["artifact_type"],
        "decision": payload["decision"],
        "output_path": payload["output_path"],
        "paid_calls_made": payload["paid_calls_made"],
        "pointwise_calls_made": payload["pointwise_calls_made"],
        "active_gate_paid_followup_allowed": diagnostics["active_gate_summary"][
            "paid_followup_allowed"
        ],
        "dry_run_unique_missing_pairwise_labels": diagnostics["dry_run_summary"][
            "unique_missing_pairwise_labels"
        ],
        "selected_positive_delta_total": diagnostics["replay_gate_tension"][
            "selected_positive_delta_total"
        ],
        "touch_delta_total": diagnostics["replay_gate_tension"][
            "unique_future_positives_touched_delta_total"
        ],
        "recommended_next_workflow": payload["recommended_next_workflow"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
