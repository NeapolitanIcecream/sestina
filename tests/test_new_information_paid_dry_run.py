from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.run_new_information_paid_dry_run import (
    build_new_information_paid_dry_run,
    validate_paid_dry_run_artifact_schema,
)
from sestina.diagnostics import fingerprint
from sestina.models import PointwiseAssessment


def test_paid_dry_run_freezes_pairs_and_blocks_unresolved_caveat(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    source_dir = tmp_path / "source"
    _write_pointwise_artifacts(source_dir)
    _write_pairwise_cache_artifact(source_dir)
    budget_fill_path = _write_budget_fill_artifact(tmp_path, source_dir=source_dir)
    active_gate_path = _write_active_gate_artifact(tmp_path)
    random_variance_path = _write_random_variance_artifact(tmp_path)
    artifact_dir = tmp_path / "dry-run"

    payload = build_new_information_paid_dry_run(
        config_path=config_path,
        manifest_path=manifest_path,
        source_artifact_dir=source_dir,
        budget_fill_artifact_path=budget_fill_path,
        active_gate_artifact_path=active_gate_path,
        random_variance_artifact_path=random_variance_path,
        artifact_dir=artifact_dir,
        ledger_path=artifact_dir / "ledger.jsonl",
        output_path=artifact_dir / "paid-dry-run-go-no-go.json",
        planned_pairs_output_path=artifact_dir / "planned-pair-occurrences.jsonl",
        phase="pilot",
        max_usd=2.0,
        seeds=[17],
        scheduler_samples=16,
        posterior_samples=16,
        pairwise_strength=2.5,
        random_floor_fraction=0.2,
        anchor_multiplier=2,
        challenger_multiplier=3,
        min_challengers=1,
        minimum_rubric_residual=0.02,
        per_item_cap=6,
        pairwise_cache_artifact_dirs=[source_dir],
    )

    assert payload["paid_calls_made"] == 0
    assert payload["pointwise_calls_made"] == 0
    assert payload["totals"]["pairwise_scheduled_occurrences"] == 1
    assert payload["totals"]["pairwise_cached_occurrences"] == 1
    assert payload["totals"]["unique_missing_pairwise_labels"] == 0
    assert payload["go_no_go"]["decision"] == "no_go"
    assert "replay_local_weak_bucket_oracle_headroom_fell" in payload["go_no_go"][
        "caveat_blocking_reasons"
    ]
    assert "guarded_pairwise_runner_ready_for_new_information" in payload[
        "go_no_go"
    ]["guardrail_blocking_reasons"]
    assert payload["planned_execution"]["model_availability"]["status"] == (
        "not_checked_dry_run"
    )
    assert not (artifact_dir / "ledger.jsonl").exists()
    rows = [
        json.loads(line)
        for line in (artifact_dir / "planned-pair-occurrences.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["cache_status"] == "cached_reuse"


def test_paid_dry_run_schema_requires_zero_pointwise_calls() -> None:
    payload: dict[str, Any] = {
        "artifact_type": "sestina-new-information-paid-dry-run",
        "schema_version": 1,
        "dry_run": True,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 1,
        "input_artifacts": {},
        "frozen_inputs": {},
        "planned_execution": {},
        "totals": {"pointwise_calls": 0},
        "guardrails": {},
        "caveats": {},
        "go_no_go": {"decision": "no_go"},
        "planned_pair_occurrences_path": "planned.jsonl",
        "planned_unique_pair_labels_by_bucket": {},
        "validation_commands": [],
    }

    with pytest.raises(ValueError, match="zero pointwise calls"):
        validate_paid_dry_run_artifact_schema(payload)


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "budget_cap_usd": 100.0,
                "rate_card": {
                    "openai/mini": {
                        "input_usd_per_1m_tokens": 1.0,
                        "output_usd_per_1m_tokens": 1.0,
                    }
                },
                "token_assumptions": {
                    "pointwise": {
                        "input_tokens_per_call": 100,
                        "output_tokens_per_call": 10,
                    },
                    "pairwise": {
                        "input_tokens_per_call": 10,
                        "output_tokens_per_call": 5,
                    },
                },
                "phases": [
                    {
                        "name": "pilot",
                        "allocation_fraction": 1.0,
                        "pointwise_model": "openai/mini",
                        "pairwise_model": "openai/mini",
                        "audit_model": "openai/mini",
                        "strategies": [
                            "pointwise_only",
                            "sestina_active_pairwise",
                        ],
                        "buckets": [{"name": "tiny_bucket", "n": 3, "k": 1}],
                    }
                ],
            }
        )
    )
    return path


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
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
                                "title": "First paper",
                                "abstract": "A strong abstract.",
                                "baseline_score": 0.9,
                                "labels": {"good_paper": True},
                                "metadata": {"primary_category": "cs.LG"},
                            },
                            {
                                "paper_id": "p2",
                                "title": "Second paper",
                                "abstract": "A challenger abstract.",
                                "baseline_score": 0.4,
                                "labels": {"good_paper": False},
                                "metadata": {"primary_category": "cs.CL"},
                            },
                            {
                                "paper_id": "p3",
                                "title": "Third paper",
                                "abstract": "Another challenger abstract.",
                                "baseline_score": 0.3,
                                "labels": {"good_paper": False},
                                "metadata": {"primary_category": "cs.AI"},
                            },
                        ],
                    }
                ],
            }
        )
    )
    return path


def _write_pointwise_artifacts(source_dir: Path) -> None:
    calls_dir = source_dir / "pilot" / "tiny_bucket" / "calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    responses = {
        "p1": PointwiseAssessment(
            good_probability=0.9,
            uncertainty=0.2,
            rubric_scores={"novelty": 0.7, "technical_depth": 0.7},
            summary="strong",
        ).to_dict(),
        "p2": PointwiseAssessment(
            good_probability=0.4,
            uncertainty=0.7,
            rubric_scores={"novelty": 0.8, "technical_depth": 0.8},
            summary="challenger",
        ).to_dict(),
        "p3": PointwiseAssessment(
            good_probability=0.3,
            uncertainty=0.8,
            rubric_scores={"novelty": 0.85, "technical_depth": 0.85},
            summary="second challenger",
        ).to_dict(),
    }
    for index, paper_id in enumerate(["p1", "p2", "p3"], start=1):
        path = calls_dir / f"{index:04d}-pointwise-{fingerprint(paper_id)}.json"
        path.write_text(
            json.dumps(
                {
                    "artifact_type": "sestina-backtest-call",
                    "phase": "pilot",
                    "bucket": "tiny_bucket",
                    "model": "openai/mini",
                    "kind": "pointwise",
                    "status": "ok",
                    "response": responses[paper_id],
                    "subject": {"paper_id": paper_id},
                }
            )
        )


def _write_pairwise_cache_artifact(source_dir: Path) -> None:
    calls_dir = source_dir / "pilot" / "tiny_bucket" / "calls"
    for index, (left_id, right_id) in enumerate(
        [("p1", "p3"), ("p2", "p3")],
        start=1,
    ):
        path = calls_dir / (
            f"{index:04d}-pairwise_active-{fingerprint(left_id + ':' + right_id)}.json"
        )
        path.write_text(
            json.dumps(
                {
                    "artifact_type": "sestina-backtest-call",
                    "phase": "pilot",
                    "bucket": "tiny_bucket",
                    "model": "openai/mini",
                    "kind": "pairwise_active",
                    "status": "ok",
                    "response": {
                        "winner": "left",
                        "soft_probability": 0.8,
                        "confidence": 0.9,
                        "reasons": ["fixture"],
                    },
                    "subject": {"left_id": left_id, "right_id": right_id},
                }
            )
        )


def _write_budget_fill_artifact(tmp_path: Path, *, source_dir: Path) -> Path:
    path = tmp_path / "budget-fill.json"
    path.write_text(
        json.dumps(
            {
                "paid_calls_made": 0,
                "paid_spend_usd": 0.0,
                "pointwise_calls_made": 0,
                "budget_fill": {
                    "inputs": {
                        "pairwise_cache_artifact_dirs": [str(source_dir)],
                    },
                    "fallback_policy": {
                        "name": "predeclared_cached_frontier_challenger_fallback",
                        "enabled": True,
                        "future_labels_used_for_scheduling": False,
                        "cached_label_values_used_before_scheduling": False,
                    },
                },
                "new_information_replay_gate_verdict": {
                    "paid_followup_allowed": False,
                    "blocking_reasons": [
                        "weak-bucket oracle headroom fell versus exact-pool random"
                    ],
                    "weak_oracle_headroom_preserved": False,
                },
                "aggregate_diagnostics": {
                    "weak_bucket_deltas": {
                        "mean_pointwise_plus_touched_recall_cap_delta": -0.1,
                        "mean_positive_negative_pair_recall_cap_delta": -0.1,
                        "unique_future_positives_touched_delta_total": -1,
                    }
                },
            }
        )
    )
    return path


def _write_active_gate_artifact(tmp_path: Path) -> Path:
    path = tmp_path / "active-gate.json"
    path.write_text(
        json.dumps(
            {
                "paid_followup_allowed": True,
                "gate_verdict": {"blocking_reasons": []},
                "caveats": {
                    "budget_completeness_caveat": {"present": False},
                    "missing_label_caveat": {"present": False},
                },
            }
        )
    )
    return path


def _write_random_variance_artifact(tmp_path: Path) -> Path:
    path = tmp_path / "random-variance.json"
    path.write_text(
        json.dumps({"artifact_type": "sestina-full-random-variance-completion"})
    )
    return path
