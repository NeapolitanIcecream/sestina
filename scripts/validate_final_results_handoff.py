#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_HANDOFF_SUMMARY = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-final-results-handoff"
    / "final-results-handoff-summary.json"
)
DEFAULT_GUARDED_EXECUTION = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-new-information-guarded-execution"
    / "guarded-execution-go-no-go.json"
)
DEFAULT_PLANNED_PAIRS = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-new-information-paid-dry-run"
    / "planned-pair-occurrences.jsonl"
)
DEFAULT_GUARDED_LEDGER = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-new-information-guarded-execution"
    / "guarded-execution-ledger.jsonl"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run an offline consistency check over the final Sestina handoff, "
            "guarded cache-only execution artifact, frozen planned-pair "
            "manifest, and empty guarded ledger."
        )
    )
    parser.add_argument("--handoff-summary", type=Path, default=DEFAULT_HANDOFF_SUMMARY)
    parser.add_argument(
        "--guarded-execution-artifact",
        type=Path,
        default=DEFAULT_GUARDED_EXECUTION,
    )
    parser.add_argument("--planned-pairs", type=Path, default=DEFAULT_PLANNED_PAIRS)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_GUARDED_LEDGER)
    args = parser.parse_args(argv)

    payload = validate_final_results_handoff(
        handoff_summary_path=args.handoff_summary,
        guarded_execution_path=args.guarded_execution_artifact,
        planned_pairs_path=args.planned_pairs,
        ledger_path=args.ledger,
    )
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def validate_final_results_handoff(
    *,
    handoff_summary_path: Path,
    guarded_execution_path: Path,
    planned_pairs_path: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    handoff = _read_json(handoff_summary_path)
    guarded = _read_json(guarded_execution_path)
    planned_pairs = _read_jsonl(planned_pairs_path)
    ledger_entries = _read_jsonl(ledger_path)

    planned_summary = _summarize_planned_pairs(planned_pairs)
    ledger_summary = {"line_count": len(ledger_entries)}

    errors: list[str] = []
    _check_handoff_summary(handoff, errors)
    _check_guarded_execution(guarded, errors)
    _check_cross_artifact_totals(
        handoff,
        guarded,
        planned_summary=planned_summary,
        ledger_summary=ledger_summary,
        errors=errors,
    )

    if planned_summary["forbidden_pointwise_like_rows"]:
        errors.append("planned-pair manifest contains pointwise-like rows")
    if planned_summary["future_label_scheduling_rows"]:
        errors.append("planned-pair manifest uses future labels for scheduling")
    if planned_summary["cached_label_value_scheduling_rows"]:
        errors.append("planned-pair manifest uses cached label values for scheduling")
    if planned_summary["invalid_cache_status_rows"]:
        errors.append("planned-pair manifest contains non-cached rows")
    if ledger_entries:
        errors.append("guarded execution ledger is not empty")

    return {
        "artifact_type": "sestina-final-results-handoff-validation",
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "inputs": {
            "handoff_summary": _display_path(handoff_summary_path),
            "guarded_execution_artifact": _display_path(guarded_execution_path),
            "planned_pairs": _display_path(planned_pairs_path),
            "ledger": _display_path(ledger_path),
        },
        "decision": {
            "campaign_status": handoff.get("decision", {}).get("campaign_status"),
            "best_result": handoff.get("decision", {}).get("best_result"),
            "ready_for_pr_publication_cleanup": handoff.get("decision", {}).get(
                "ready_for_pr_publication_cleanup"
            ),
            "ready_to_publish_without_cleanup": handoff.get("decision", {}).get(
                "ready_to_publish_without_cleanup"
            ),
            "paid_label_purchase_authorized": handoff.get("caveat_scope", {}).get(
                "paid_label_purchase_authorized"
            ),
        },
        "guarded_execution": {
            "status": guarded.get("execution_summary", {}).get("status"),
            "decision": guarded.get("go_no_go", {}).get("decision"),
            "expected_execution_mode": guarded.get("planned_execution", {}).get(
                "expected_execution_mode"
            ),
            "paid_calls_made": guarded.get("paid_calls_made"),
            "pointwise_calls_made": guarded.get("pointwise_calls_made"),
            "paid_spend_usd": guarded.get("paid_spend_usd"),
        },
        "planned_pairs": planned_summary,
        "ledger": ledger_summary,
        "paid_work_requires_explicit_voyager_approval": True,
    }


def _check_handoff_summary(payload: dict[str, Any], errors: list[str]) -> None:
    _expect_equal(
        payload.get("artifact_type"),
        "sestina_final_results_handoff_summary",
        "handoff artifact type",
        errors,
    )
    _expect_equal(
        payload.get("paid_calls_made_by_this_handoff"),
        0,
        "handoff paid call count",
        errors,
    )
    _expect_equal(
        payload.get("pointwise_calls_made_by_this_handoff"),
        0,
        "handoff pointwise call count",
        errors,
    )
    _expect_equal(
        payload.get("paid_spend_usd_by_this_handoff"),
        0.0,
        "handoff paid spend",
        errors,
    )
    _expect_equal(
        payload.get("decision", {}).get("campaign_status"),
        "stop_experiments_now",
        "campaign status",
        errors,
    )
    _expect_equal(
        payload.get("decision", {}).get("ready_for_pr_publication_cleanup"),
        True,
        "PR/publication cleanup readiness",
        errors,
    )
    _expect_equal(
        payload.get("decision", {}).get("ready_to_publish_without_cleanup"),
        False,
        "publish-without-cleanup readiness",
        errors,
    )
    _expect_equal(
        payload.get("caveat_scope", {}).get("paid_label_purchase_authorized"),
        False,
        "paid-label purchase authorization",
        errors,
    )

    current = payload.get("current_best_result", {})
    _expect_equal(
        current.get("active_arm"),
        "new_information_challenger_cached_replay",
        "current active arm",
        errors,
    )
    _expect_equal(
        current.get("random_control_arm"),
        "exact_pool_random_cached_replay",
        "current random control arm",
        errors,
    )
    _expect_equal(current.get("seed_count"), 20, "current seed count", errors)
    _expect_equal(current.get("bucket_count"), 8, "current bucket count", errors)


def _check_guarded_execution(payload: dict[str, Any], errors: list[str]) -> None:
    _expect_equal(
        payload.get("artifact_type"),
        "sestina-new-information-guarded-runner-go-no-go",
        "guarded execution artifact type",
        errors,
    )
    _expect_equal(payload.get("mode"), "execute", "guarded execution mode", errors)
    _expect_equal(payload.get("paid_calls_made"), 0, "guarded paid calls", errors)
    _expect_equal(
        payload.get("pointwise_calls_made"),
        0,
        "guarded pointwise calls",
        errors,
    )
    _expect_equal(payload.get("paid_spend_usd"), 0.0, "guarded paid spend", errors)
    _expect_equal(
        payload.get("execution_summary", {}).get("status"),
        "cache_only_zero_missing_labels",
        "guarded execution status",
        errors,
    )
    _expect_equal(
        payload.get("go_no_go", {}).get("decision"),
        "go",
        "guarded go/no-go decision",
        errors,
    )
    _expect_equal(
        payload.get("go_no_go", {}).get(
            "paid_label_purchase_authorized_by_this_artifact"
        ),
        False,
        "guarded paid-label authorization",
        errors,
    )
    _expect_equal(
        payload.get("planned_execution", {}).get("expected_execution_mode"),
        "cache_only_zero_spend",
        "guarded expected execution mode",
        errors,
    )


def _check_cross_artifact_totals(
    handoff: dict[str, Any],
    guarded: dict[str, Any],
    *,
    planned_summary: dict[str, Any],
    ledger_summary: dict[str, Any],
    errors: list[str],
) -> None:
    handoff_guarded = handoff.get("guarded_execution", {})
    guarded_totals = guarded.get("totals", {})

    _expect_equal(
        handoff_guarded.get("planned_pair_occurrences"),
        planned_summary["line_count"],
        "handoff planned-pair occurrence count",
        errors,
    )
    _expect_equal(
        guarded_totals.get("pairwise_scheduled_occurrences"),
        planned_summary["line_count"],
        "guarded planned-pair occurrence count",
        errors,
    )
    _expect_equal(
        handoff_guarded.get("unique_planned_pair_labels"),
        planned_summary["unique_pair_labels"],
        "handoff unique planned pair labels",
        errors,
    )
    _expect_equal(
        guarded_totals.get("unique_planned_pair_labels"),
        planned_summary["unique_pair_labels"],
        "guarded unique planned pair labels",
        errors,
    )
    _expect_equal(
        handoff_guarded.get("ledger_line_count"),
        ledger_summary["line_count"],
        "handoff ledger line count",
        errors,
    )
    _expect_equal(
        guarded.get("ledger", {}).get("new_entries_this_invocation"),
        ledger_summary["line_count"],
        "guarded ledger entry count",
        errors,
    )
    zero_fields = (
        (
            "handoff missing pairwise labels",
            handoff_guarded.get("unique_missing_pairwise_labels"),
        ),
        (
            "guarded missing pairwise labels",
            guarded_totals.get("unique_missing_pairwise_labels"),
        ),
        (
            "handoff paid pairwise attempts",
            handoff_guarded.get("paid_pairwise_calls_attempted"),
        ),
        (
            "guarded paid pairwise attempts",
            guarded.get("execution_summary", {}).get("paid_pairwise_calls_attempted"),
        ),
        ("handoff pointwise calls", handoff_guarded.get("pointwise_calls")),
        ("guarded pointwise calls", guarded_totals.get("pointwise_calls")),
    )
    for label, actual in zero_fields:
        _expect_equal(actual, 0, label, errors)


def _summarize_planned_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair_keys: set[tuple[str, str]] = set()
    bucket_counts: Counter[str] = Counter()
    cache_status_counts: Counter[str] = Counter()
    purpose_counts: Counter[str] = Counter()
    invalid_pair_key_rows = 0
    invalid_cache_status_rows = 0
    forbidden_pointwise_like_rows = 0
    future_label_scheduling_rows = 0
    cached_label_value_scheduling_rows = 0

    for row in rows:
        pair_key = row.get("pair_key")
        if (
            isinstance(pair_key, list)
            and len(pair_key) == 2
            and all(isinstance(value, str) for value in pair_key)
        ):
            pair_keys.add(tuple(sorted(pair_key)))
        else:
            invalid_pair_key_rows += 1

        bucket = row.get("bucket")
        if isinstance(bucket, str):
            bucket_counts[bucket] += 1

        cache_status = row.get("cache_status")
        if isinstance(cache_status, str):
            cache_status_counts[cache_status] += 1
        if cache_status != "cached_reuse":
            invalid_cache_status_rows += 1

        purpose = row.get("purpose")
        if isinstance(purpose, str):
            purpose_counts[purpose] += 1

        call_kind = row.get("planned_call_kind") or row.get("cached_artifact_kind")
        if isinstance(call_kind, str) and "pointwise" in call_kind:
            forbidden_pointwise_like_rows += 1
        if row.get("future_labels_used_for_scheduling") is not False:
            future_label_scheduling_rows += 1
        if row.get("cached_label_values_used_before_scheduling") is not False:
            cached_label_value_scheduling_rows += 1

    return {
        "line_count": len(rows),
        "unique_pair_labels": len(pair_keys),
        "bucket_count": len(bucket_counts),
        "bucket_occurrence_counts": dict(sorted(bucket_counts.items())),
        "cache_status_counts": dict(sorted(cache_status_counts.items())),
        "purpose_counts": dict(sorted(purpose_counts.items())),
        "invalid_pair_key_rows": invalid_pair_key_rows,
        "invalid_cache_status_rows": invalid_cache_status_rows,
        "forbidden_pointwise_like_rows": forbidden_pointwise_like_rows,
        "future_label_scheduling_rows": future_label_scheduling_rows,
        "cached_label_value_scheduling_rows": cached_label_value_scheduling_rows,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _expect_equal(
    actual: object,
    expected: object,
    label: str,
    errors: list[str],
) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, got {actual!r}")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


if __name__ == "__main__":
    raise SystemExit(main())
