from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_final_results_handoff import validate_final_results_handoff


def test_final_results_handoff_validation_passes_cache_only_fixture(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)

    payload = validate_final_results_handoff(**fixture)

    assert payload["status"] == "passed"
    assert payload["errors"] == []
    assert payload["decision"]["campaign_status"] == "stop_experiments_now"
    assert payload["decision"]["paid_label_purchase_authorized"] is False
    assert payload["guarded_execution"]["status"] == "cache_only_zero_missing_labels"
    assert payload["planned_pairs"]["line_count"] == 2
    assert payload["planned_pairs"]["unique_pair_labels"] == 2
    assert payload["ledger"]["line_count"] == 0
    assert payload["paid_work_requires_explicit_voyager_approval"] is True


def test_final_results_handoff_validation_fails_pointwise_and_nonempty_ledger(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    _write_jsonl(
        fixture["planned_pairs_path"],
        [
            _planned_pair(
                "p1",
                "p2",
                cached_artifact_kind="pointwise",
                future_labels_used_for_scheduling=True,
            ),
            _planned_pair("p3", "p4", cache_status="missing"),
        ],
    )
    _write_jsonl(fixture["ledger_path"], [{"kind": "pairwise_active"}])

    payload = validate_final_results_handoff(**fixture)

    assert payload["status"] == "failed"
    assert "planned-pair manifest contains pointwise-like rows" in payload["errors"]
    assert (
        "planned-pair manifest uses future labels for scheduling"
        in payload["errors"]
    )
    assert "planned-pair manifest contains non-cached rows" in payload["errors"]
    assert "guarded execution ledger is not empty" in payload["errors"]
    assert "handoff ledger line count: expected 1, got 0" in payload["errors"]


def test_final_results_handoff_validation_fails_paid_authorization_drift(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    handoff = json.loads(fixture["handoff_summary_path"].read_text())
    handoff["caveat_scope"]["paid_label_purchase_authorized"] = True
    fixture["handoff_summary_path"].write_text(json.dumps(handoff))

    payload = validate_final_results_handoff(**fixture)

    assert payload["status"] == "failed"
    assert (
        "paid-label purchase authorization: expected False, got True"
        in payload["errors"]
    )


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    handoff_path = tmp_path / "final-results-handoff-summary.json"
    guarded_path = tmp_path / "guarded-execution-go-no-go.json"
    planned_pairs_path = tmp_path / "planned-pair-occurrences.jsonl"
    ledger_path = tmp_path / "guarded-execution-ledger.jsonl"

    handoff_path.write_text(json.dumps(_handoff_summary()))
    guarded_path.write_text(json.dumps(_guarded_execution()))
    _write_jsonl(
        planned_pairs_path,
        [_planned_pair("p1", "p2"), _planned_pair("p3", "p4")],
    )
    ledger_path.write_text("")

    return {
        "handoff_summary_path": handoff_path,
        "guarded_execution_path": guarded_path,
        "planned_pairs_path": planned_pairs_path,
        "ledger_path": ledger_path,
    }


def _handoff_summary() -> dict[str, object]:
    return {
        "artifact_type": "sestina_final_results_handoff_summary",
        "paid_calls_made_by_this_handoff": 0,
        "pointwise_calls_made_by_this_handoff": 0,
        "paid_spend_usd_by_this_handoff": 0.0,
        "decision": {
            "campaign_status": "stop_experiments_now",
            "ready_for_pr_publication_cleanup": True,
            "ready_to_publish_without_cleanup": False,
            "best_result": (
                "budget_filled_new_information_challenger_cached_replay_with_"
                "guarded_cache_only_execution"
            ),
        },
        "current_best_result": {
            "active_arm": "new_information_challenger_cached_replay",
            "random_control_arm": "exact_pool_random_cached_replay",
            "seed_count": 20,
            "bucket_count": 8,
        },
        "guarded_execution": {
            "planned_pair_occurrences": 2,
            "unique_planned_pair_labels": 2,
            "unique_missing_pairwise_labels": 0,
            "paid_pairwise_calls_attempted": 0,
            "pointwise_calls": 0,
            "ledger_line_count": 0,
        },
        "caveat_scope": {"paid_label_purchase_authorized": False},
    }


def _guarded_execution() -> dict[str, object]:
    return {
        "artifact_type": "sestina-new-information-guarded-runner-go-no-go",
        "mode": "execute",
        "paid_calls_made": 0,
        "pointwise_calls_made": 0,
        "paid_spend_usd": 0.0,
        "execution_summary": {
            "status": "cache_only_zero_missing_labels",
            "paid_pairwise_calls_attempted": 0,
        },
        "go_no_go": {
            "decision": "go",
            "paid_label_purchase_authorized_by_this_artifact": False,
        },
        "planned_execution": {"expected_execution_mode": "cache_only_zero_spend"},
        "ledger": {"new_entries_this_invocation": 0},
        "totals": {
            "pairwise_scheduled_occurrences": 2,
            "unique_planned_pair_labels": 2,
            "unique_missing_pairwise_labels": 0,
            "pointwise_calls": 0,
        },
    }


def _planned_pair(
    left_id: str,
    right_id: str,
    *,
    cache_status: str = "cached_reuse",
    cached_artifact_kind: str = "pairwise_active",
    future_labels_used_for_scheduling: bool = False,
) -> dict[str, object]:
    return {
        "bucket": "bucket",
        "cache_status": cache_status,
        "cached_artifact_kind": cached_artifact_kind,
        "cached_label_values_used_before_scheduling": False,
        "future_labels_used_for_scheduling": future_labels_used_for_scheduling,
        "left_id": left_id,
        "right_id": right_id,
        "pair_key": sorted([left_id, right_id]),
        "purpose": "new_information_false_negative_challenge",
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
