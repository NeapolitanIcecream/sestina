from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_coverage_floor_fresh_validation import (
    ARTIFACT_TYPE,
    analyze_coverage_floor_fresh_validation,
)
from sestina.diagnostics import fingerprint
from sestina.models import PointwiseAssessment


def test_fresh_validation_analysis_claims_complete_only_with_all_pairwise_labels(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path, omit_pair=None)

    payload = analyze_coverage_floor_fresh_validation(
        config_path=fixture["config"],
        no_paid_sweep_artifact_path=fixture["sweep"],
        active_gate_artifact_path=fixture["gate"],
        manifest_path=fixture["manifest"],
        source_artifact_dir=fixture["pointwise_dir"],
        pairwise_artifact_dir=fixture["pairwise_dir"],
        output_path=tmp_path / "analysis.json",
    )

    assert payload["artifact_type"] == ARTIFACT_TYPE
    assert payload["fresh_validation_claim"]["complete"] is True
    assert payload["fresh_validation_claim"]["can_claim_fresh_paid_validation"] is True
    assert payload["paired_deltas_vs_exact_pool_random"]["metric_deltas"][
        "recall_at_k"
    ]["count"] == 2
    assert (tmp_path / "analysis.json").exists()


def test_fresh_validation_analysis_blocks_claim_when_pairwise_labels_missing(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path, omit_pair=("fresh-1", "fresh-2"))

    payload = analyze_coverage_floor_fresh_validation(
        config_path=fixture["config"],
        no_paid_sweep_artifact_path=fixture["sweep"],
        active_gate_artifact_path=fixture["gate"],
        manifest_path=fixture["manifest"],
        source_artifact_dir=fixture["pointwise_dir"],
        pairwise_artifact_dir=fixture["pairwise_dir"],
        output_path=tmp_path / "analysis.json",
    )

    assert payload["fresh_validation_claim"]["complete"] is False
    assert payload["fresh_validation_claim"]["do_not_claim_success_if_incomplete"] is True
    assert payload["completeness"]["missing_pairwise_occurrences"] > 0


def _write_fixture(
    tmp_path: Path,
    *,
    omit_pair: tuple[str, str] | None,
) -> dict[str, Path]:
    config = _write_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    pointwise_dir = tmp_path / "pointwise"
    pairwise_dir = tmp_path / "pairwise"
    _write_pointwise(pointwise_dir)
    _write_pairwise(pairwise_dir, omit_pair=omit_pair)
    sweep, gate = _write_no_paid(tmp_path, manifest)
    return {
        "config": config,
        "manifest": manifest,
        "pointwise_dir": pointwise_dir,
        "pairwise_dir": pairwise_dir,
        "sweep": sweep,
        "gate": gate,
    }


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
                "phases": [
                    {
                        "name": "pilot",
                        "allocation_fraction": 1.0,
                        "pointwise_model": "openai/mini",
                        "pairwise_model": "openai/mini",
                        "audit_model": "openai/mini",
                        "strategies": ["pointwise_only"],
                        "buckets": [{"name": "fixture", "n": 3, "k": 1}],
                    }
                ],
            }
        )
    )
    return path


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    papers = []
    for index, paper_id in enumerate(("fresh-1", "fresh-2", "fresh-3"), start=1):
        papers.append(
            {
                "paper_id": paper_id,
                "title": f"Fresh paper {index}",
                "abstract": f"Fresh abstract {index}",
                "baseline_score": 0.5,
                "labels": {"good_paper": index == 1},
                "metadata": {"primary_category": "cs.LG"},
            }
        )
    path.write_text(
        json.dumps(
            {
                "artifact_type": "sestina-backtest-dataset-manifest",
                "buckets": [
                    {
                        "name": "fresh-bucket",
                        "phase": "pilot",
                        "k": 1,
                        "papers": papers,
                    }
                ],
            }
        )
    )
    return path


def _write_pointwise(pointwise_dir: Path) -> None:
    calls = pointwise_dir / "pilot" / "fresh-bucket" / "calls"
    calls.mkdir(parents=True, exist_ok=True)
    for index, paper_id in enumerate(("fresh-1", "fresh-2", "fresh-3"), start=1):
        assessment = PointwiseAssessment(
            good_probability=0.8 - (0.1 * index),
            uncertainty=0.2,
            summary=f"summary {index}",
        )
        path = calls / f"{index:04d}-pointwise-{fingerprint(paper_id)}.json"
        path.write_text(
            json.dumps(
                {
                    "artifact_type": "sestina-backtest-call",
                    "phase": "pilot",
                    "bucket": "fresh-bucket",
                    "model": "openai/mini",
                    "kind": "pointwise",
                    "status": "ok",
                    "response": assessment.to_dict(),
                    "subject": {"paper_id": paper_id},
                }
            )
        )


def _write_pairwise(
    pairwise_dir: Path,
    *,
    omit_pair: tuple[str, str] | None,
) -> None:
    calls = pairwise_dir / "pilot" / "fresh-bucket" / "calls"
    calls.mkdir(parents=True, exist_ok=True)
    pairs = [("fresh-1", "fresh-2"), ("fresh-1", "fresh-3"), ("fresh-2", "fresh-3")]
    for index, (left_id, right_id) in enumerate(pairs, start=1):
        if omit_pair is not None and tuple(sorted(omit_pair)) == tuple(
            sorted((left_id, right_id))
        ):
            continue
        path = calls / f"{index:04d}-pairwise_active-{fingerprint(left_id + ':' + right_id)}.json"
        path.write_text(
            json.dumps(
                {
                    "artifact_type": "sestina-backtest-call",
                    "phase": "pilot",
                    "bucket": "fresh-bucket",
                    "model": "openai/mini",
                    "kind": "pairwise_active",
                    "status": "ok",
                    "subject": {"left_id": left_id, "right_id": right_id},
                    "response": {
                        "winner": "left",
                        "soft_probability": 0.8,
                        "confidence": 0.9,
                        "reasons": ["fixture"],
                    },
                }
            )
        )


def _write_no_paid(tmp_path: Path, manifest: Path) -> tuple[Path, Path]:
    sweep = tmp_path / "sweep.json"
    gate = tmp_path / "gate.json"
    sweep.write_text(
        json.dumps(
            {
                "artifact_type": "sestina-no-paid-algorithm-sweep",
                "schema_version": 1,
                "phase": "pilot",
                "paid_calls_made": 0,
                "paid_spend_usd": 0.0,
                "pointwise_calls_made": 0,
                "manifest_path": str(manifest),
                "analysis_parameters": {
                    "seeds": [17, 101],
                    "scheduler_samples": 32,
                    "posterior_samples": 32,
                    "pairwise_strength": 2.5,
                    "confidence_z": 1.96,
                },
                "aggregate_metrics": {
                    "randomized_coverage_floor_hybrid_cached_replay": {
                        "recall_at_k": 0.5,
                        "ndcg_at_k": 0.5,
                        "average_precision": 0.5,
                    }
                },
                "paired_deltas_vs_exact_pool_random": {
                    "metric_deltas": {
                        "recall_at_k": {"count": 2, "mean": 0.1},
                        "ndcg_at_k": {"count": 2, "mean": 0.1},
                        "average_precision": {"count": 2, "mean": 0.1},
                    }
                },
                "label_policy": {"cache_availability_used_for_scheduling": True},
                "candidate_arms_tried": [],
                "control_arms": [],
                "bucket_results": [
                    {"seed": 17, "buckets": [{"bucket": "development-bucket"}]}
                ],
            }
        )
    )
    gate.write_text(
        json.dumps(
            {
                "artifact_type": "sestina-active-arm-gate",
                "schema_version": 1,
                "paid_calls_made": 0,
                "paid_spend_usd": 0.0,
                "pointwise_calls_made": 0,
                "active_arm_name": "randomized_coverage_floor_hybrid_cached_replay",
                "candidate_random_control_baseline": "exact_pool_random_cached_replay",
                "paid_followup_allowed": True,
                "gate_policy": {},
                "gate_verdict": {
                    "paid_followup_allowed": True,
                    "blocking_reasons": [],
                    "paired_random_control_present": True,
                    "seed_count": 2,
                    "core_diagnostics_complete": True,
                    "randomized_floor_or_paired_control_present": True,
                    "no_future_label_or_cached_label_leakage": True,
                },
                "seed_level_confidence_intervals": {},
                "paired_active_minus_random_deltas": {"metric_deltas": {}},
                "diagnostics": {"weak_bucket_diagnostics": {"available": True}},
                "caveats": {},
                "spend_estimate": {},
                "label_leakage": {"present": False, "forbidden_true_keys": []},
                "random_variance_reference": {"complete_20_seed_reference": True},
                "input_artifacts": {},
            }
        )
    )
    return sweep, gate
