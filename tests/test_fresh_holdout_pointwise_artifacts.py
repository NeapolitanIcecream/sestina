from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.run_fresh_holdout_pointwise_artifacts import (
    ARTIFACT_TYPE,
    build_fresh_holdout_pointwise_artifacts,
)
from sestina.diagnostics import fingerprint


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_pointwise_planning_reports_missing_artifacts_without_calls(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)

    payload = build_fresh_holdout_pointwise_artifacts(
        config_path=_write_config(tmp_path),
        manifest_path=manifest,
        artifact_dir=tmp_path / "pointwise",
        ledger_path=tmp_path / "pointwise" / "ledger.jsonl",
        output_path=tmp_path / "pointwise" / "review.json",
        mode="planning",
    )

    assert payload["artifact_type"] == ARTIFACT_TYPE
    assert payload["paid_calls_made"] == 0
    assert payload["pointwise_calls_made"] == 0
    assert payload["fresh_holdout"]["missing_pointwise_artifacts"] == 2
    assert payload["final_status"]["status"] == "incomplete"
    assert "pointwise_artifacts_missing" in payload["final_status"][
        "blocking_reasons"
    ]
    assert not (tmp_path / "pointwise" / "ledger.jsonl").exists()


def test_pointwise_execute_writes_reviewed_artifacts_and_usage_ledger(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SESTINA_LLM_API_KEY", "test-key")
    monkeypatch.setenv("SESTINA_LLM_BASE_URL", "https://llm.test")
    manifest = _write_manifest(tmp_path)
    calls = {"chat": 0}

    def fake_urlopen(request: Any, **kwargs: Any) -> _FakeResponse:
        url = getattr(request, "full_url", "")
        if url.endswith("/models"):
            return _FakeResponse({"data": [{"id": "openai/mini"}]})
        calls["chat"] += 1
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "good_probability": 0.7,
                                    "uncertainty": 0.2,
                                    "summary": "promising",
                                    "reasons": ["clear contribution"],
                                    "rubric_scores": {"novelty": 0.8},
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
            }
        )

    payload = build_fresh_holdout_pointwise_artifacts(
        config_path=_write_config(tmp_path),
        manifest_path=manifest,
        artifact_dir=tmp_path / "pointwise",
        ledger_path=tmp_path / "pointwise" / "ledger.jsonl",
        output_path=tmp_path / "pointwise" / "review.json",
        mode="execute",
        confirm_fresh_holdout_pointwise_generation=True,
        urlopen=fake_urlopen,
    )

    assert calls["chat"] == 2
    assert payload["final_status"]["status"] == "complete"
    assert payload["paid_calls_made"] == 2
    assert payload["fresh_holdout"]["available_pointwise_artifacts"] == 2
    ledger_rows = [
        json.loads(line)
        for line in (tmp_path / "pointwise" / "ledger.jsonl").read_text().splitlines()
    ]
    assert {row["cost_source"] for row in ledger_rows} == {
        "upstream_returned_usage_configured_rates"
    }
    first_artifact = (
        tmp_path
        / "pointwise"
        / "pilot"
        / "fresh-bucket"
        / "calls"
        / f"0001-pointwise-{fingerprint('fresh-1')}.json"
    )
    artifact = json.loads(first_artifact.read_text())
    assert "labels" not in json.dumps(artifact["subject"])
    assert artifact["subject"]["title"] == "Fresh paper 1"


def test_pointwise_leakage_review_blocks_model_visible_citation_metadata(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path, metadata={"citation_count": 99})

    payload = build_fresh_holdout_pointwise_artifacts(
        config_path=_write_config(tmp_path),
        manifest_path=manifest,
        artifact_dir=tmp_path / "pointwise",
        ledger_path=tmp_path / "pointwise" / "ledger.jsonl",
        output_path=tmp_path / "pointwise" / "review.json",
        mode="planning",
    )

    assert payload["model_visible_leakage_review"]["present"] is True
    assert "model_visible_leakage_present" in payload["final_status"][
        "blocking_reasons"
    ]


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "budget_cap_usd": 100.0,
                "rate_card": {
                    "openai/mini": {
                        "input_usd_per_1m_tokens": 1.0,
                        "output_usd_per_1m_tokens": 2.0,
                    }
                },
                "token_assumptions": {
                    "pointwise": {
                        "input_tokens_per_call": 900,
                        "output_tokens_per_call": 220,
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
                        "buckets": [{"name": "fixture", "n": 2, "k": 1}],
                    }
                ],
            }
        )
    )
    return path


def _write_manifest(
    tmp_path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = tmp_path / "manifest.json"
    paper_metadata = {"primary_category": "cs.LG"} | (metadata or {})
    path.write_text(
        json.dumps(
            {
                "artifact_type": "sestina-backtest-dataset-manifest",
                "buckets": [
                    {
                        "name": "fresh-bucket",
                        "phase": "pilot",
                        "k": 1,
                        "papers": [
                            {
                                "paper_id": "fresh-1",
                                "title": "Fresh paper 1",
                                "abstract": "Fresh abstract 1",
                                "baseline_score": 0.5,
                                "labels": {
                                    "good_paper": True,
                                    "citation_count": 20,
                                },
                                "metadata": paper_metadata,
                            },
                            {
                                "paper_id": "fresh-2",
                                "title": "Fresh paper 2",
                                "abstract": "Fresh abstract 2",
                                "baseline_score": 0.5,
                                "labels": {
                                    "good_paper": False,
                                    "citation_count": 1,
                                },
                                "metadata": paper_metadata,
                            },
                        ],
                    }
                ],
            }
        )
    )
    return path
