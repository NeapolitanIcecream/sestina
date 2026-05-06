from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.run_new_information_guarded_runner import (
    PointwiseCallForbiddenError,
    _assert_pairwise_only_call_kind,
    build_new_information_guarded_runner_go_no_go,
)


def test_guarded_runner_planning_marks_zero_missing_manifest_ready(
    tmp_path: Path,
) -> None:
    fixture = _write_frozen_fixture(tmp_path)

    payload = build_new_information_guarded_runner_go_no_go(
        config_path=fixture["config_path"],
        manifest_path=fixture["manifest_path"],
        source_artifact_dir=fixture["source_dir"],
        budget_fill_artifact_path=fixture["budget_fill_path"],
        active_gate_artifact_path=fixture["active_gate_path"],
        dry_run_artifact_path=fixture["dry_run_path"],
        planned_pairs_path=fixture["planned_pairs_path"],
        caveat_adjudication_path=fixture["caveat_path"],
        artifact_dir=fixture["artifact_dir"],
        ledger_path=fixture["artifact_dir"] / "guarded-runner-ledger.jsonl",
        output_path=fixture["artifact_dir"] / "guarded-runner-go-no-go.json",
        mode="planning",
        max_usd=0.01,
    )

    assert payload["paid_calls_made"] == 0
    assert payload["pointwise_calls_made"] == 0
    assert payload["go_no_go"]["decision"] == "go"
    assert payload["go_no_go"]["runner_ready_for_later_execution"] is True
    assert payload["totals"]["unique_missing_pairwise_labels"] == 0
    assert payload["totals"]["pairwise_calls_to_buy"] == 0
    assert payload["planned_execution"]["expected_execution_mode"] == (
        "cache_only_zero_spend"
    )
    assert payload["model_availability"]["status"] == (
        "required_later_not_checked_planning"
    )
    assert payload["ledger"]["existing_spend_usd_before_workflow"] == 0.0
    assert (fixture["artifact_dir"] / "guarded-runner-ledger.jsonl").exists()
    assert (
        fixture["artifact_dir"] / "guarded-runner-ledger.jsonl"
    ).read_text() == ""


def test_guarded_runner_planning_reports_pointwise_like_row_as_no_go(
    tmp_path: Path,
) -> None:
    fixture = _write_frozen_fixture(tmp_path)
    rows = _read_jsonl(fixture["planned_pairs_path"])
    rows[0]["planned_call_kind"] = "pointwise"
    _write_jsonl(fixture["planned_pairs_path"], rows)
    _refresh_caveat_planned_pairs_sha(fixture)

    payload = build_new_information_guarded_runner_go_no_go(
        config_path=fixture["config_path"],
        manifest_path=fixture["manifest_path"],
        source_artifact_dir=fixture["source_dir"],
        budget_fill_artifact_path=fixture["budget_fill_path"],
        active_gate_artifact_path=fixture["active_gate_path"],
        dry_run_artifact_path=fixture["dry_run_path"],
        planned_pairs_path=fixture["planned_pairs_path"],
        caveat_adjudication_path=fixture["caveat_path"],
        artifact_dir=fixture["artifact_dir"],
        ledger_path=fixture["artifact_dir"] / "guarded-runner-ledger.jsonl",
        output_path=fixture["artifact_dir"] / "guarded-runner-go-no-go.json",
        mode="planning",
        max_usd=0.01,
    )

    assert payload["go_no_go"]["decision"] == "no_go"
    assert payload["go_no_go"]["runner_ready_for_later_execution"] is False
    assert "planned_rows_pairwise_only" in payload["frozen_manifest_validation"][
        "blocking_reasons"
    ]
    assert payload["totals"]["pointwise_like_planned_rows"] == 1


def test_guarded_runner_requires_accepted_caveat_scope(tmp_path: Path) -> None:
    fixture = _write_frozen_fixture(tmp_path)
    caveat = json.loads(fixture["caveat_path"].read_text())
    caveat["decision"] = "caveat_remains_blocking"
    fixture["caveat_path"].write_text(json.dumps(caveat))

    payload = build_new_information_guarded_runner_go_no_go(
        config_path=fixture["config_path"],
        manifest_path=fixture["manifest_path"],
        source_artifact_dir=fixture["source_dir"],
        budget_fill_artifact_path=fixture["budget_fill_path"],
        active_gate_artifact_path=fixture["active_gate_path"],
        dry_run_artifact_path=fixture["dry_run_path"],
        planned_pairs_path=fixture["planned_pairs_path"],
        caveat_adjudication_path=fixture["caveat_path"],
        artifact_dir=fixture["artifact_dir"],
        ledger_path=fixture["artifact_dir"] / "guarded-runner-ledger.jsonl",
        output_path=fixture["artifact_dir"] / "guarded-runner-go-no-go.json",
        mode="planning",
        max_usd=0.01,
    )

    assert payload["go_no_go"]["decision"] == "no_go"
    assert "caveat_adjudication_accepted_with_constraints" in payload[
        "caveat_scope"
    ]["blocking_reasons"]


def test_pairwise_only_call_guard_aborts_pointwise_and_random_control() -> None:
    with pytest.raises(PointwiseCallForbiddenError, match="pointwise"):
        _assert_pairwise_only_call_kind("pointwise")

    with pytest.raises(PointwiseCallForbiddenError, match="pairwise_active"):
        _assert_pairwise_only_call_kind("pairwise_random")

    _assert_pairwise_only_call_kind("pairwise_active")


def _write_frozen_fixture(tmp_path: Path) -> dict[str, Path]:
    source_dir = tmp_path / "source"
    artifact_dir = tmp_path / "guarded-runner"
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "manifest.json"
    budget_fill_path = tmp_path / "budget-fill.json"
    active_gate_path = tmp_path / "active-gate.json"
    dry_run_path = tmp_path / "paid-dry-run.json"
    planned_pairs_path = tmp_path / "planned-pair-occurrences.jsonl"
    caveat_path = tmp_path / "caveat.json"

    config_path.write_text(
        json.dumps(
            {
                "rate_card": {
                    "openai/mini": {
                        "input_usd_per_1m_tokens": 1.0,
                        "output_usd_per_1m_tokens": 1.0,
                    }
                },
                "token_assumptions": {
                    "pairwise": {
                        "input_tokens_per_call": 100,
                        "output_tokens_per_call": 10,
                    }
                },
                "phases": [
                    {
                        "name": "pilot",
                        "pointwise_model": "openai/mini",
                        "pairwise_model": "openai/mini",
                        "audit_model": "openai/mini",
                        "buckets": [{"name": "tiny_bucket", "n": 3, "k": 1}],
                    }
                ],
            }
        )
    )
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "sestina-backtest-dataset-manifest",
                "buckets": [
                    {
                        "name": "tiny_bucket",
                        "phase": "pilot",
                        "k": 1,
                        "papers": [
                            {
                                "paper_id": "p1",
                                "title": "First",
                                "abstract": "First abstract",
                                "labels": {"good_paper": True},
                            },
                            {
                                "paper_id": "p2",
                                "title": "Second",
                                "abstract": "Second abstract",
                                "labels": {"good_paper": False},
                            },
                        ],
                    }
                ],
            }
        )
    )
    budget_fill_path.write_text(
        json.dumps(
            {
                "artifact_type": "sestina-new-information-challenger-simulator",
                "paid_calls_made": 0,
                "paid_spend_usd": 0.0,
                "pointwise_calls_made": 0,
                "budget_fill": {
                    "inputs": {"pairwise_cache_artifact_dirs": [str(source_dir)]},
                    "fallback_policy": {
                        "name": "predeclared_cached_frontier_challenger_fallback",
                        "future_labels_used_for_scheduling": False,
                        "cached_label_values_used_before_scheduling": False,
                    },
                },
            }
        )
    )
    active_gate_path.write_text(
        json.dumps(
            {
                "artifact_type": "sestina-active-arm-gate",
                "schema_version": 1,
                "paid_calls_made": 0,
                "paid_spend_usd": 0.0,
                "active_arm_name": "new_information_challenger_cached_replay",
                "candidate_random_control_baseline": "exact_pool_random_cached_replay",
                "paid_followup_allowed": True,
                "gate_verdict": {
                    "blocking_reasons": [],
                    "paid_followup_allowed": True,
                },
                "caveats": {
                    "budget_completeness_caveat": {"present": False},
                    "missing_label_caveat": {"present": False},
                },
            }
        )
    )
    _write_jsonl(
        planned_pairs_path,
        [
            {
                "row_id": "17:tiny_bucket",
                "seed": 17,
                "bucket": "tiny_bucket",
                "k": 1,
                "pair_index": 1,
                "pair_key": ["p1", "p2"],
                "left_id": "p1",
                "right_id": "p2",
                "purpose": "new_information_false_negative_challenge",
                "cache_status": "cached_reuse",
                "cached_artifact_path": str(
                    source_dir
                    / "pilot"
                    / "tiny_bucket"
                    / "calls"
                    / "0001-pairwise_active-fixture.json"
                ),
                "cached_artifact_kind": "pairwise_active",
                "future_labels_used_for_scheduling": False,
                "cached_label_values_used_before_scheduling": False,
            },
            {
                "row_id": "17:tiny_bucket",
                "seed": 17,
                "bucket": "tiny_bucket",
                "k": 1,
                "pair_index": 2,
                "pair_key": ["p1", "p2"],
                "left_id": "p1",
                "right_id": "p2",
                "purpose": "new_information_random_floor",
                "cache_status": "cached_reuse",
                "cached_artifact_path": str(
                    source_dir
                    / "pilot"
                    / "tiny_bucket"
                    / "calls"
                    / "0001-pairwise_active-fixture.json"
                ),
                "cached_artifact_kind": "pairwise_active",
                "future_labels_used_for_scheduling": False,
                "cached_label_values_used_before_scheduling": False,
            },
        ],
    )
    dry_run = {
        "artifact_type": "sestina-new-information-paid-dry-run",
        "schema_version": 1,
        "dry_run": True,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 0,
        "known_paid_spend_before_workflow_usd": 2.74603,
        "paid_cap_usd": 100.0,
        "input_artifacts": {
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "source_artifact_dir": str(source_dir),
            "budget_fill_artifact_path": str(budget_fill_path),
            "budget_fill_artifact_sha256": _sha256(budget_fill_path),
            "active_gate_artifact_path": str(active_gate_path),
            "active_gate_artifact_sha256": _sha256(active_gate_path),
        },
        "frozen_inputs": {
            "active_arm_name": "new_information_challenger_cached_replay",
            "random_control_comparator": "exact_pool_random_cached_replay",
            "phase": "pilot",
            "seeds": [17],
            "seed_count": 1,
            "candidate_construction_policy": {
                "method": "new_information_challenger_cached_replay",
                "future_labels_used_for_scheduling": False,
                "cached_label_values_used_before_scheduling": False,
            },
            "pairwise_cache_artifact_dirs": [str(source_dir)],
            "model_provider": "openai",
            "pairwise_model": "openai/mini",
            "artifact_dir": str(tmp_path / "paid-dry-run"),
            "ledger_path": str(tmp_path / "paid-dry-run" / "ledger.jsonl"),
        },
        "planned_execution": {
            "planned_pairwise_label_kind": "pairwise_active",
            "pointwise_calls_planned": 0,
            "random_control_paid_labels_planned": 0,
            "model_availability": {
                "status": "not_checked_dry_run",
                "required_before_paid_calls": True,
                "models_requiring_check": ["openai/mini"],
            },
        },
        "totals": {
            "pointwise_calls": 0,
            "pairwise_scheduled_occurrences": 2,
            "pairwise_cached_occurrences": 2,
            "pairwise_missing_occurrences": 0,
            "unique_planned_pair_labels": 1,
            "unique_missing_pairwise_labels": 0,
            "pairwise_calls_to_buy": 0,
            "estimated_additional_spend_usd": 0.0,
            "known_paid_spend_before_workflow_usd": 2.74603,
            "projected_known_paid_spend_after_workflow_usd": 2.74603,
            "active_budget_shortfall": 0,
            "random_control_budget_shortfall": 0,
        },
        "guardrails": {},
        "caveats": {
            "unresolved_blocking_caveats": [
                "replay_local_weak_bucket_oracle_headroom_fell"
            ]
        },
        "go_no_go": {"decision": "no_go", "requested_max_usd": 2.0},
        "planned_pair_occurrences_path": str(planned_pairs_path),
        "planned_pair_occurrence_count": 2,
        "planned_unique_pair_labels_by_bucket": {},
        "validation_commands": [],
    }
    dry_run_path.write_text(json.dumps(dry_run))
    caveat_path.write_text(
        json.dumps(
            {
                "artifact_type": "sestina-new-information-caveat-adjudication",
                "schema_version": 1,
                "decision": "caveat_accepted_with_constraints",
                "paid_calls_made": 0,
                "paid_spend_usd": 0.0,
                "pointwise_calls_made": 0,
                "input_artifacts": {
                    "budget_fill_artifact_path": str(budget_fill_path),
                    "budget_fill_artifact_sha256": _sha256(budget_fill_path),
                    "active_gate_artifact_path": str(active_gate_path),
                    "active_gate_artifact_sha256": _sha256(active_gate_path),
                    "dry_run_artifact_path": str(dry_run_path),
                    "dry_run_artifact_sha256": _sha256(dry_run_path),
                    "planned_pairs_path": str(planned_pairs_path),
                    "planned_pairs_sha256": _sha256(planned_pairs_path),
                },
                "constraints": [
                    "Do not weaken or bypass the reviewed active-arm gate.",
                    "Use future labels only for retrospective diagnostics, never for scheduling or model-visible selection.",
                    "Make zero pointwise calls in any follow-up tied to this manifest.",
                    "Do not rewrite historical paid ledgers or paid-call artifacts.",
                    "Acceptance is scoped only to the frozen budget-filled new-information manifest and its current reviewed artifacts.",
                    "Because the dry-run found zero unique missing labels, this adjudication authorizes no paid label purchase by itself.",
                    "Any later paid workflow must use a reviewed guarded pairwise-only runner, provider model availability checks, JSONL ledger, hard max-usd cap, separate artifact directory, and abort on any pointwise-call attempt.",
                ],
            }
        )
    )
    return {
        "source_dir": source_dir,
        "artifact_dir": artifact_dir,
        "config_path": config_path,
        "manifest_path": manifest_path,
        "budget_fill_path": budget_fill_path,
        "active_gate_path": active_gate_path,
        "dry_run_path": dry_run_path,
        "planned_pairs_path": planned_pairs_path,
        "caveat_path": caveat_path,
    }


def _refresh_caveat_planned_pairs_sha(fixture: dict[str, Path]) -> None:
    caveat = json.loads(fixture["caveat_path"].read_text())
    caveat["input_artifacts"]["planned_pairs_sha256"] = _sha256(
        fixture["planned_pairs_path"]
    )
    fixture["caveat_path"].write_text(json.dumps(caveat))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
