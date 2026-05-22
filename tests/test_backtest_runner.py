from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sestina.backtest_runner import (
    BacktestRunner,
    BudgetLimitExceeded,
    CallEstimate,
    ChatJsonResponse,
    JsonlLedger,
    LedgerEntry,
    ModelAvailabilityError,
    check_model_availability,
    usage_cost_payload,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_runner_defaults_to_dry_run_without_model_check_or_ledger_calls(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
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
                        "name": "smoke",
                        "allocation_fraction": 0.05,
                        "pointwise_model": "openai/mini",
                        "pairwise_model": "openai/mini",
                        "audit_model": "openai/mini",
                        "strategies": ["pointwise_only"],
                        "buckets": [{"name": "tiny", "n": 2, "k": 1}],
                    }
                ],
            }
        )
    )

    summary = BacktestRunner(
        config_path=config_path,
        phase="smoke",
        max_usd=100.0,
        artifact_dir=tmp_path / "artifacts",
        ledger_path=tmp_path / "ledger.jsonl",
    ).run()

    assert summary["dry_run"] is True
    assert summary["model_availability"]["status"] == "not_checked_dry_run"
    assert summary["paid_calls_made"] == 0
    assert not (tmp_path / "ledger.jsonl").exists()
    assert (tmp_path / "artifacts" / "estimate-smoke.json").exists()


def test_model_availability_allows_unprefixed_model_names() -> None:
    result = check_model_availability(
        base_url="https://llm.example/v1",
        api_key="secret",
        models=["gpt-5.4-mini"],
        urlopen=lambda *args, **kwargs: _FakeResponse(
            {"data": [{"id": "gpt-5.4-mini"}]}
        ),
    )

    assert result == {
        "status": "available",
        "requested_models": ["gpt-5.4-mini"],
        "missing_models": [],
        "available_requested_models": ["gpt-5.4-mini"],
    }


def test_model_availability_rejects_missing_endpoint_model() -> None:
    def fake_urlopen(*args: object, **kwargs: object) -> _FakeResponse:
        return _FakeResponse({"data": [{"id": "openai/other"}]})

    with pytest.raises(ModelAvailabilityError, match="not available"):
        check_model_availability(
            base_url="https://llm.example/v1",
            api_key="secret",
            models=["openai/mini"],
            urlopen=fake_urlopen,
        )


def test_model_availability_returns_checked_models_without_secrets() -> None:
    def fake_urlopen(*args: object, **kwargs: object) -> _FakeResponse:
        return _FakeResponse({"data": [{"id": "openai/mini"}]})

    result = check_model_availability(
        base_url="https://llm.example/v1",
        api_key="secret",
        models=["openai/mini"],
        urlopen=fake_urlopen,
    )

    assert result == {
        "status": "available",
        "requested_models": ["openai/mini"],
        "missing_models": [],
        "available_requested_models": ["openai/mini"],
    }
    assert "secret" not in json.dumps(result)


def test_jsonl_ledger_writes_required_call_fields_and_totals_spend(
    tmp_path: Path,
) -> None:
    ledger = JsonlLedger(tmp_path / "ledger.jsonl")

    ledger.append(
        LedgerEntry(
            phase="smoke",
            bucket="bucket-a",
            model="openai/mini",
            kind="pointwise",
            estimated_input_tokens=900,
            estimated_output_tokens=220,
            estimated_cost_usd=0.123456,
            status="ok",
            artifact_path=str(tmp_path / "call.json"),
            created_at_unix=1.0,
        )
    )

    entry = json.loads((tmp_path / "ledger.jsonl").read_text())
    assert entry["phase"] == "smoke"
    assert entry["bucket"] == "bucket-a"
    assert entry["model"] == "openai/mini"
    assert entry["kind"] == "pointwise"
    assert entry["estimated_tokens"] == {"input": 900, "output": 220}
    assert entry["estimated_cost_usd"] == 0.123456
    assert entry["billable_cost_usd"] == 0.123456
    assert entry["cost_source"] == "configured_token_estimate"
    assert entry["status"] == "ok"
    assert entry["artifact_path"].endswith("call.json")
    assert ledger.existing_spend_usd() == pytest.approx(0.123456)


def test_ledger_prefers_billable_cost_when_upstream_usage_is_recorded(
    tmp_path: Path,
) -> None:
    ledger = JsonlLedger(tmp_path / "ledger.jsonl")

    ledger.append(
        LedgerEntry(
            phase="smoke",
            bucket="bucket-a",
            model="openai/mini",
            kind="pointwise",
            estimated_input_tokens=900,
            estimated_output_tokens=220,
            estimated_cost_usd=0.50,
            billable_cost_usd=0.12,
            cost_source="upstream_returned_usage_configured_rates",
            upstream_usage={"prompt_tokens": 100, "completion_tokens": 10},
            status="ok",
            artifact_path=str(tmp_path / "call.json"),
            created_at_unix=1.0,
        )
    )

    entry = json.loads((tmp_path / "ledger.jsonl").read_text())
    assert entry["billable_cost_usd"] == 0.12
    assert entry["upstream_usage"] == {"prompt_tokens": 100, "completion_tokens": 10}
    assert ledger.existing_spend_usd() == pytest.approx(0.12)


def test_usage_cost_payload_estimates_from_returned_usage_and_configured_rates() -> None:
    payload = usage_cost_payload(
        model="openai/mini",
        estimate=CallEstimate(input_tokens=900, output_tokens=220, cost_usd=0.50),
        rates={
            "openai/mini": {
                "input_usd_per_1m_tokens": 1.0,
                "output_usd_per_1m_tokens": 2.0,
                "discount_multiplier": 1.0,
            }
        },
        response=ChatJsonResponse(
            content={"ok": True},
            upstream_usage={"prompt_tokens": 1000, "completion_tokens": 200},
            upstream_cost_usd=None,
        ),
    )

    assert payload["cost_source"] == "upstream_returned_usage_configured_rates"
    assert payload["billable_cost_usd"] == pytest.approx(0.0014)


def test_budget_guard_stops_before_projected_spend_exceeds_cap(
    tmp_path: Path,
) -> None:
    ledger = JsonlLedger(tmp_path / "ledger.jsonl")
    ledger.append(
        LedgerEntry(
            phase="smoke",
            bucket="bucket-a",
            model="openai/mini",
            kind="pointwise",
            estimated_input_tokens=1,
            estimated_output_tokens=1,
            estimated_cost_usd=0.09,
            status="ok",
            artifact_path=str(tmp_path / "call.json"),
            created_at_unix=1.0,
        )
    )

    with pytest.raises(BudgetLimitExceeded, match="exceeds budget cap"):
        ledger.guard_projected_spend(cap_usd=0.10, next_cost_usd=0.02)
