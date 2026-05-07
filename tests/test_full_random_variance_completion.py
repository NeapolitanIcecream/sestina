from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.run_full_random_variance_completion import (
    PAIRWISE_COMPLETION_KIND,
    _next_pairwise_attempt_path,
    build_missing_label_plan,
    run_guarded_pairwise_label_completion,
)
from sestina.diagnostics import fingerprint
from sestina.models import PointwiseAssessment


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_missing_label_plan_dedupes_full_random_schedules_pairwise_only(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    source_dir = tmp_path / "source"
    _write_pointwise_artifacts(source_dir)
    artifact_dir = tmp_path / "completion"

    plan = build_missing_label_plan(
        config_path=config_path,
        manifest_path=manifest_path,
        source_artifact_dir=source_dir,
        artifact_dir=artifact_dir,
        ledger_path=artifact_dir / "ledger.jsonl",
        output_path=artifact_dir / "missing-label-plan.json",
        phase="pilot",
        max_usd=5.0,
        seeds=[17],
        scheduler_samples=16,
        posterior_samples=16,
        pairwise_strength=2.5,
        bootstrap_samples=20,
        pairwise_cache_artifact_dirs=None,
    )

    assert plan["totals"]["pointwise_calls"] == 0
    assert plan["totals"]["pairwise_scheduled_occurrences"] == 2
    assert plan["totals"]["unique_missing_pairwise_labels"] == 1
    assert plan["guardrails"]["paid_run_allowed_after_plan"] is True
    missing = plan["missing_pairs_by_bucket"]["tiny_bucket"]
    assert len(missing) == 1
    assert {row["arm"] for row in missing[0]["required_by"]} == {
        "historical_random_full_schedule",
        "exact_pool_random_full_schedule",
    }


def test_guarded_completion_makes_pairwise_calls_without_pointwise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    source_dir = tmp_path / "source"
    _write_pointwise_artifacts(source_dir)
    artifact_dir = tmp_path / "completion"
    plan = build_missing_label_plan(
        config_path=config_path,
        manifest_path=manifest_path,
        source_artifact_dir=source_dir,
        artifact_dir=artifact_dir,
        ledger_path=artifact_dir / "ledger.jsonl",
        output_path=artifact_dir / "missing-label-plan.json",
        phase="pilot",
        max_usd=5.0,
        seeds=[17],
        scheduler_samples=16,
        posterior_samples=16,
        pairwise_strength=2.5,
        bootstrap_samples=20,
        pairwise_cache_artifact_dirs=None,
    )
    chat_payloads: list[dict[str, Any]] = []

    def fake_urlopen(request: Any, **kwargs: object) -> _FakeResponse:
        if request.full_url.endswith("/models"):
            return _FakeResponse({"data": [{"id": "openai/mini"}]})
        payload = json.loads(request.data.decode("utf-8"))
        chat_payloads.append(payload)
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "winner": "left",
                                    "soft_probability": 0.8,
                                    "confidence": 0.9,
                                    "reasons": ["fixture pairwise label"],
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setenv("SESTINA_LLM_API_KEY", "secret")
    monkeypatch.setenv("SESTINA_LLM_BASE_URL", "https://llm.example/v1")

    summary = run_guarded_pairwise_label_completion(
        plan,
        config_path=config_path,
        manifest_path=manifest_path,
        source_artifact_dir=source_dir,
        artifact_dir=artifact_dir,
        ledger_path=artifact_dir / "ledger.jsonl",
        summary_path=artifact_dir / "labeling-summary-pilot.json",
        phase="pilot",
        max_usd=5.0,
        urlopen=fake_urlopen,
    )

    assert summary["pointwise_calls"] == 0
    assert summary["new_ok_call_artifacts_this_invocation"] == 1
    assert len(chat_payloads) == 1
    assert "Compare two candidate papers" in chat_payloads[0]["messages"][0]["content"]
    assert "Assess whether a paper" not in chat_payloads[0]["messages"][0]["content"]
    ledger_rows = [
        json.loads(line)
        for line in (artifact_dir / "ledger.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [row["kind"] for row in ledger_rows] == [PAIRWISE_COMPLETION_KIND]


def test_retry_path_preserves_existing_failed_paid_artifact(tmp_path: Path) -> None:
    """Regression: paid retries must not overwrite a failed call artifact."""
    path = tmp_path / "0001-pairwise_full_random_variance-deadbeef.json"
    path.write_text(json.dumps({"status": "parse_error"}))

    retry = _next_pairwise_attempt_path(path)

    assert retry.name == "0001-pairwise_full_random_variance-deadbeef-attempt-2.json"
    assert path.read_text() == json.dumps({"status": "parse_error"})


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
                        "strategies": ["pointwise_only"],
                        "buckets": [{"name": "tiny_bucket", "n": 2, "k": 1}],
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
                                "baseline_score": 0.8,
                                "labels": {"good_paper": True},
                                "metadata": {
                                    "primary_category": "cs.LG",
                                    "source": "arxiv",
                                },
                            },
                            {
                                "paper_id": "p2",
                                "title": "Second paper",
                                "abstract": "A weaker abstract.",
                                "baseline_score": 0.4,
                                "labels": {"good_paper": False},
                                "metadata": {
                                    "primary_category": "cs.CL",
                                    "source": "arxiv",
                                },
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
            good_probability=0.8,
            uncertainty=0.2,
            summary="strong",
        ).to_dict(),
        "p2": PointwiseAssessment(
            good_probability=0.4,
            uncertainty=0.6,
            summary="weak",
        ).to_dict(),
    }
    for index, paper_id in enumerate(["p1", "p2"], start=1):
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
