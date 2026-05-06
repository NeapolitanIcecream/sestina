from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.run_new_information_execution_preflight import (
    build_new_information_execution_preflight,
    validate_execution_preflight_artifact_schema,
)
from scripts.run_new_information_guarded_runner import (
    build_new_information_guarded_runner_go_no_go,
)
from tests.test_new_information_guarded_runner import _write_frozen_fixture


def test_execution_preflight_closes_model_gap_without_label_calls(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    fixture = _write_reviewed_guarded_fixture(tmp_path)
    monkeypatch.setenv("SESTINA_LLM_API_KEY", "secret-for-test")
    monkeypatch.setenv("SESTINA_LLM_BASE_URL", "https://llm.example/v1")
    requests: list[Any] = []

    def fake_urlopen(request: Any, *args: Any, **kwargs: Any) -> _FakeResponse:
        requests.append(request)
        assert request.get_method() == "GET"
        assert request.full_url.endswith("/models")
        assert "chat/completions" not in request.full_url
        return _FakeResponse({"data": [{"id": "openai/mini"}]})

    payload = build_new_information_execution_preflight(
        config_path=fixture["config_path"],
        manifest_path=fixture["manifest_path"],
        source_artifact_dir=fixture["source_dir"],
        budget_fill_artifact_path=fixture["budget_fill_path"],
        active_gate_artifact_path=fixture["active_gate_path"],
        dry_run_artifact_path=fixture["dry_run_path"],
        planned_pairs_path=fixture["planned_pairs_path"],
        caveat_adjudication_path=fixture["caveat_path"],
        guarded_runner_artifact_path=fixture["guarded_path"],
        output_path=tmp_path / "preflight" / "execution-preflight-go-no-go.json",
        urlopen=fake_urlopen,
    )

    validate_execution_preflight_artifact_schema(payload)
    assert len(requests) == 1
    assert payload["provider_model_availability"]["status"] == "available"
    assert payload["final_go_no_go"]["decision"] == "go"
    assert payload["final_go_no_go"]["recommended_later_execution_cap_usd"] == 0.01
    assert payload["final_go_no_go"]["expected_execution_mode"] == (
        "cache_only_zero_spend"
    )
    assert payload["paid_calls_made"] == 0
    assert payload["pointwise_calls_made"] == 0
    assert payload["method"]["chat_completions_calls_made"] == 0
    assert payload["totals"]["unique_missing_pairwise_labels"] == 0
    assert payload["totals"]["pairwise_calls_to_buy"] == 0
    assert "secret-for-test" not in json.dumps(payload)


def test_execution_preflight_reports_unavailable_model_as_no_go(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    fixture = _write_reviewed_guarded_fixture(tmp_path)
    monkeypatch.setenv("SESTINA_LLM_API_KEY", "secret-for-test")
    monkeypatch.setenv("SESTINA_LLM_BASE_URL", "https://llm.example/v1")

    payload = build_new_information_execution_preflight(
        config_path=fixture["config_path"],
        manifest_path=fixture["manifest_path"],
        source_artifact_dir=fixture["source_dir"],
        budget_fill_artifact_path=fixture["budget_fill_path"],
        active_gate_artifact_path=fixture["active_gate_path"],
        dry_run_artifact_path=fixture["dry_run_path"],
        planned_pairs_path=fixture["planned_pairs_path"],
        caveat_adjudication_path=fixture["caveat_path"],
        guarded_runner_artifact_path=fixture["guarded_path"],
        output_path=tmp_path / "preflight" / "execution-preflight-go-no-go.json",
        urlopen=lambda *args, **kwargs: _FakeResponse(
            {"data": [{"id": "openai/other"}]}
        ),
    )

    validate_execution_preflight_artifact_schema(payload)
    assert payload["provider_model_availability"]["status"] == "unavailable"
    assert payload["final_go_no_go"]["decision"] == "no_go"
    assert "provider_model_availability" in payload["final_go_no_go"][
        "blocking_reasons"
    ]
    assert "guardrail:model_availability_available" in payload["final_go_no_go"][
        "blocking_reasons"
    ]


def test_execution_preflight_detects_planned_pair_manifest_drift(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    fixture = _write_reviewed_guarded_fixture(tmp_path)
    monkeypatch.setenv("SESTINA_LLM_API_KEY", "secret-for-test")
    monkeypatch.setenv("SESTINA_LLM_BASE_URL", "https://llm.example/v1")
    fixture["planned_pairs_path"].write_text(
        fixture["planned_pairs_path"].read_text() + "\n"
    )

    payload = build_new_information_execution_preflight(
        config_path=fixture["config_path"],
        manifest_path=fixture["manifest_path"],
        source_artifact_dir=fixture["source_dir"],
        budget_fill_artifact_path=fixture["budget_fill_path"],
        active_gate_artifact_path=fixture["active_gate_path"],
        dry_run_artifact_path=fixture["dry_run_path"],
        planned_pairs_path=fixture["planned_pairs_path"],
        caveat_adjudication_path=fixture["caveat_path"],
        guarded_runner_artifact_path=fixture["guarded_path"],
        output_path=tmp_path / "preflight" / "execution-preflight-go-no-go.json",
        urlopen=lambda *args, **kwargs: _FakeResponse(
            {"data": [{"id": "openai/mini"}]}
        ),
    )

    validate_execution_preflight_artifact_schema(payload)
    assert payload["final_go_no_go"]["decision"] == "no_go"
    assert "planned_pairs_sha_matches_guarded" in payload["final_go_no_go"][
        "blocking_reasons"
    ]
    assert "planned_pairs_sha_matches_caveat" in payload["final_go_no_go"][
        "blocking_reasons"
    ]


def _write_reviewed_guarded_fixture(tmp_path: Path) -> dict[str, Path]:
    fixture = _write_frozen_fixture(tmp_path)
    _add_model_policy_to_fixture(fixture)
    guarded_path = fixture["artifact_dir"] / "guarded-runner-go-no-go.json"
    build_new_information_guarded_runner_go_no_go(
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
        output_path=guarded_path,
        mode="planning",
        max_usd=0.01,
    )
    return {**fixture, "guarded_path": guarded_path}


def _add_model_policy_to_fixture(fixture: dict[str, Path]) -> None:
    config = json.loads(fixture["config_path"].read_text())
    config["model_name_policy"] = {
        "require_provider_prefix": True,
        "availability_check_required_before_paid_run": True,
    }
    fixture["config_path"].write_text(json.dumps(config))

    dry_run = json.loads(fixture["dry_run_path"].read_text())
    dry_run["input_artifacts"]["config_sha256"] = _sha256(fixture["config_path"])
    fixture["dry_run_path"].write_text(json.dumps(dry_run))

    caveat = json.loads(fixture["caveat_path"].read_text())
    caveat["input_artifacts"]["dry_run_artifact_sha256"] = _sha256(
        fixture["dry_run_path"]
    )
    fixture["caveat_path"].write_text(json.dumps(caveat))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")
