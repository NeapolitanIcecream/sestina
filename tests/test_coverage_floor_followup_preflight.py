from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.run_coverage_floor_followup_preflight import (
    DEFAULT_ACTIVE_GATE_ARTIFACT,
    DEFAULT_CONFIG,
    DEFAULT_NO_PAID_SWEEP_ARTIFACT,
    PointwiseCallForbiddenError,
    _assert_pairwise_only_call_kind,
    build_coverage_floor_followup_preflight,
    validate_coverage_floor_preflight_artifact_schema,
)
from sestina.diagnostics import fingerprint
from sestina.models import PointwiseAssessment


class _FakeResponse:
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {"data": [{"id": "openai/gpt-5.4-mini"}]},
        ).encode("utf-8")


def _fake_urlopen(*args: object, **kwargs: object) -> _FakeResponse:
    return _FakeResponse()


def test_preflight_blocks_missing_fresh_holdout_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SESTINA_LLM_API_KEY", "test-key")
    monkeypatch.setenv("SESTINA_LLM_BASE_URL", "https://llm.test")
    artifact_dir = tmp_path / "coverage-floor-preflight"

    payload = build_coverage_floor_followup_preflight(
        config_path=DEFAULT_CONFIG,
        no_paid_sweep_artifact_path=DEFAULT_NO_PAID_SWEEP_ARTIFACT,
        active_gate_artifact_path=DEFAULT_ACTIVE_GATE_ARTIFACT,
        fresh_holdout_manifest_path=tmp_path / "missing-manifest.json",
        source_artifact_dir=tmp_path / "missing-pointwise",
        artifact_dir=artifact_dir,
        ledger_path=artifact_dir / "ledger.jsonl",
        output_path=artifact_dir / "preflight.json",
        planned_pairs_output_path=artifact_dir / "planned-pair-occurrences.jsonl",
        max_usd=2.0,
        urlopen=_fake_urlopen,
    )

    assert payload["paid_calls_made"] == 0
    assert payload["paid_spend_usd"] == 0.0
    assert payload["pointwise_calls_made"] == 0
    assert payload["provider_model_availability"]["status"] == "available"
    assert payload["fresh_holdout"]["status"] == (
        "blocked_missing_fresh_holdout_manifest"
    )
    assert payload["final_go_no_go"]["decision"] == "no_go"
    assert "fresh_holdout_manifest_present" in payload["final_go_no_go"][
        "blocking_reasons"
    ]
    assert payload["input_artifacts"]["planned_pairs_path"] is None
    assert not (artifact_dir / "planned-pair-occurrences.jsonl").exists()


def test_preflight_blocks_development_replay_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SESTINA_LLM_API_KEY", "test-key")
    monkeypatch.setenv("SESTINA_LLM_BASE_URL", "https://llm.test")
    artifact_dir = tmp_path / "coverage-floor-preflight"
    development_manifest = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "backtest-datasets"
        / "arxiv-historical-pilot-manifest.json"
    )

    payload = build_coverage_floor_followup_preflight(
        config_path=DEFAULT_CONFIG,
        no_paid_sweep_artifact_path=DEFAULT_NO_PAID_SWEEP_ARTIFACT,
        active_gate_artifact_path=DEFAULT_ACTIVE_GATE_ARTIFACT,
        fresh_holdout_manifest_path=development_manifest,
        source_artifact_dir=tmp_path / "missing-pointwise",
        artifact_dir=artifact_dir,
        ledger_path=artifact_dir / "ledger.jsonl",
        output_path=artifact_dir / "preflight.json",
        planned_pairs_output_path=artifact_dir / "planned-pair-occurrences.jsonl",
        max_usd=2.0,
        urlopen=_fake_urlopen,
    )

    assert payload["fresh_holdout"]["status"] == "blocked_manifest_not_fresh"
    assert "fresh_holdout_manifest_is_development_replay" in payload[
        "fresh_holdout"
    ]["blocking_reasons"]
    assert payload["final_go_no_go"]["decision"] == "no_go"
    assert payload["pointwise_calls_made"] == 0


def test_preflight_plans_pairwise_only_rows_when_fresh_inputs_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SESTINA_LLM_API_KEY", "test-key")
    monkeypatch.setenv("SESTINA_LLM_BASE_URL", "https://llm.test")
    manifest_path = _write_fresh_manifest(tmp_path)
    source_dir = tmp_path / "fresh-pointwise"
    _write_pointwise_artifacts(source_dir)
    artifact_dir = tmp_path / "coverage-floor-preflight"

    payload = build_coverage_floor_followup_preflight(
        config_path=DEFAULT_CONFIG,
        no_paid_sweep_artifact_path=DEFAULT_NO_PAID_SWEEP_ARTIFACT,
        active_gate_artifact_path=DEFAULT_ACTIVE_GATE_ARTIFACT,
        fresh_holdout_manifest_path=manifest_path,
        source_artifact_dir=source_dir,
        artifact_dir=artifact_dir,
        ledger_path=artifact_dir / "ledger.jsonl",
        output_path=artifact_dir / "preflight.json",
        planned_pairs_output_path=artifact_dir / "planned-pair-occurrences.jsonl",
        max_usd=2.0,
        urlopen=_fake_urlopen,
    )

    assert payload["final_go_no_go"]["decision"] == "go"
    assert payload["final_go_no_go"]["paid_validation_may_run_now"] is False
    assert payload["paid_calls_made"] == 0
    assert payload["pointwise_calls_made"] == 0
    assert payload["fresh_holdout"]["pointwise_artifacts"]["status"] == "available"
    assert payload["totals"]["pairwise_scheduled_occurrences"] > 0
    assert payload["totals"]["pointwise_like_planned_rows"] == 0
    assert payload["totals"]["non_pairwise_call_rows"] == 0
    assert payload["totals"]["estimated_additional_spend_usd"] <= 2.0
    rows = [
        json.loads(line)
        for line in (artifact_dir / "planned-pair-occurrences.jsonl")
        .read_text()
        .splitlines()
    ]
    assert rows
    assert {row["planned_call_kind"] for row in rows} == {"pairwise_active"}
    assert {row["row_role"] for row in rows} == {
        "coverage_floor_active",
        "exact_pool_random_control",
    }
    assert all(row["future_labels_used_for_scheduling"] is False for row in rows)
    assert all(
        row["cached_label_values_used_before_scheduling"] is False for row in rows
    )


def test_preflight_schema_rejects_pointwise_calls() -> None:
    payload: dict[str, Any] = {
        "artifact_type": "sestina-coverage-floor-followup-preflight",
        "schema_version": 1,
        "mode": "planning",
        "dry_run": True,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 1,
        "method": {},
        "input_artifacts": {},
        "frozen_no_paid_sweep": {},
        "fresh_holdout": {},
        "provider_model_availability": {},
        "planned_execution": {},
        "ledger": {},
        "max_usd_cap": {"requested_max_usd": 2.0},
        "totals": {"pointwise_calls": 0},
        "guardrails": {},
        "final_go_no_go": {"decision": "no_go"},
        "validation_commands": [],
    }

    with pytest.raises(ValueError, match="zero pointwise calls"):
        validate_coverage_floor_preflight_artifact_schema(payload)


def test_pairwise_only_guard_rejects_pointwise_call_kind() -> None:
    with pytest.raises(PointwiseCallForbiddenError, match="pointwise"):
        _assert_pairwise_only_call_kind("pointwise")

    _assert_pairwise_only_call_kind("pairwise_active")


def _write_fresh_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "fresh-manifest.json"
    papers = []
    for index, paper_id in enumerate(["fresh_p1", "fresh_p2", "fresh_p3", "fresh_p4"]):
        papers.append(
            {
                "paper_id": paper_id,
                "title": f"Fresh holdout paper {index}",
                "abstract": f"Fresh abstract {index}",
                "baseline_score": 0.5,
                "labels": {"good_paper": index == 0},
                "metadata": {"primary_category": "cs.LG"},
            }
        )
    path.write_text(
        json.dumps(
            {
                "artifact_type": "sestina-backtest-dataset-manifest",
                "label_definition": {
                    "good_paper": "fresh fixture labels for evaluation only"
                },
                "buckets": [
                    {
                        "name": "arxiv_cs_LG_2024_01_coverage_floor_holdout",
                        "phase": "pilot",
                        "k": 1,
                        "papers": papers,
                    }
                ],
            }
        )
    )
    return path


def _write_pointwise_artifacts(source_dir: Path) -> None:
    bucket = "arxiv_cs_LG_2024_01_coverage_floor_holdout"
    calls_dir = source_dir / "pilot" / bucket / "calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    for index, paper_id in enumerate(["fresh_p1", "fresh_p2", "fresh_p3", "fresh_p4"], start=1):
        assessment = PointwiseAssessment(
            good_probability=0.8 - (index * 0.1),
            uncertainty=0.2 + (index * 0.1),
            rubric_scores={"novelty": 0.5 + (index * 0.05)},
            summary=f"fixture pointwise {index}",
        )
        path = calls_dir / f"{index:04d}-pointwise-{fingerprint(paper_id)}.json"
        path.write_text(
            json.dumps(
                {
                    "artifact_type": "sestina-backtest-call",
                    "phase": "pilot",
                    "bucket": bucket,
                    "model": "openai/gpt-5.4-mini",
                    "kind": "pointwise",
                    "status": "ok",
                    "response": assessment.to_dict(),
                    "subject": {
                        "paper_id": paper_id,
                        "title": f"Fresh holdout paper {index}",
                        "abstract": f"Fresh abstract {index}",
                        "metadata": {"primary_category": "cs.LG"},
                    },
                }
            )
        )
