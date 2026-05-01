from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sestina.backtest_runner import _call_estimate, _normalize_rates_from_config
from sestina.backtest_runner import _normalize_token_assumptions_from_config
from sestina.diagnostics import fingerprint
from sestina.models import Paper, PointwiseAssessment
from sestina.scheduler import resolve_pairwise_budget
from sestina.scheduler_followup import (
    PointwiseArtifactError,
    SchedulerOnlyRunner,
    legacy_schedule_pairs,
    legacy_select_candidates,
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


def test_scheduler_only_paid_run_never_makes_pointwise_llm_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduler-only paid runs may call pairwise, but never pointwise."""
    config_path = _write_config(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    source_dir = tmp_path / "source"
    _write_pointwise_artifacts(source_dir)
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
                                    "soft_probability": 0.82,
                                    "confidence": 0.9,
                                    "reasons": ["fixture pairwise judgment"],
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setenv("SESTINA_LLM_API_KEY", "secret")
    monkeypatch.setenv("SESTINA_LLM_BASE_URL", "https://llm.example/v1")

    summary = SchedulerOnlyRunner(
        config_path=config_path,
        manifest_path=manifest_path,
        source_artifact_dir=source_dir,
        artifact_dir=tmp_path / "followup",
        ledger_path=tmp_path / "followup" / "ledger.jsonl",
        max_usd=0.50,
        confirm_paid=True,
        urlopen=fake_urlopen,
    ).run()

    assert summary["pointwise_calls"] == 0
    assert summary["pairwise_novel_total"] == 1
    assert summary["aggregate_metrics"]["sestina_active_pairwise"]["bucket_count"] == 1
    assert len(chat_payloads) == 1
    system_prompt = chat_payloads[0]["messages"][0]["content"]
    assert "Compare two candidate papers" in system_prompt
    assert "Assess whether a paper" not in system_prompt
    ledger_rows = _read_ledger(tmp_path / "followup" / "ledger.jsonl")
    assert [row["kind"] for row in ledger_rows] == ["pairwise_active"]


def test_missing_pointwise_artifact_fails_before_paid_pairwise_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    source_dir = tmp_path / "source"
    _write_pointwise_artifacts(source_dir, missing_ids={"p2"})
    endpoint_calls = 0

    def fake_urlopen(*args: object, **kwargs: object) -> _FakeResponse:
        nonlocal endpoint_calls
        endpoint_calls += 1
        return _FakeResponse({"data": [{"id": "openai/mini"}]})

    monkeypatch.setenv("SESTINA_LLM_API_KEY", "secret")
    monkeypatch.setenv("SESTINA_LLM_BASE_URL", "https://llm.example/v1")

    with pytest.raises(PointwiseArtifactError, match="missing successful pointwise"):
        SchedulerOnlyRunner(
            config_path=config_path,
            manifest_path=manifest_path,
            source_artifact_dir=source_dir,
            artifact_dir=tmp_path / "followup",
            ledger_path=tmp_path / "followup" / "ledger.jsonl",
            max_usd=0.50,
            confirm_paid=True,
            urlopen=fake_urlopen,
        ).run()

    assert endpoint_calls == 0
    assert not (tmp_path / "followup" / "ledger.jsonl").exists()


def test_scheduler_only_dry_run_estimates_only_novel_pairwise_calls(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    source_dir = tmp_path / "source"
    _write_pointwise_artifacts(source_dir)

    summary = SchedulerOnlyRunner(
        config_path=config_path,
        manifest_path=manifest_path,
        source_artifact_dir=source_dir,
        artifact_dir=tmp_path / "dry",
        ledger_path=tmp_path / "dry" / "ledger.jsonl",
        max_usd=0.50,
    ).run()

    assert summary["dry_run"] is True
    assert summary["pointwise_calls"] == 0
    assert summary["pairwise_calls"] == 1
    assert summary["pairwise_novel_total"] == 1
    assert summary["input_tokens"] == 10
    assert summary["output_tokens"] == 5
    assert not (tmp_path / "dry" / "ledger.jsonl").exists()


def test_scheduler_only_reuses_duplicate_successful_pairwise_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    source_dir = tmp_path / "source"
    _write_pointwise_artifacts(source_dir)
    _write_legacy_pairwise_artifact(source_dir)
    chat_calls = 0

    def fake_urlopen(request: Any, **kwargs: object) -> _FakeResponse:
        nonlocal chat_calls
        if request.full_url.endswith("/models"):
            return _FakeResponse({"data": [{"id": "openai/mini"}]})
        chat_calls += 1
        return _FakeResponse(
            {"choices": [{"message": {"content": json.dumps({"winner": "left"})}}]}
        )

    monkeypatch.setenv("SESTINA_LLM_API_KEY", "secret")
    monkeypatch.setenv("SESTINA_LLM_BASE_URL", "https://llm.example/v1")

    summary = SchedulerOnlyRunner(
        config_path=config_path,
        manifest_path=manifest_path,
        source_artifact_dir=source_dir,
        artifact_dir=tmp_path / "followup",
        ledger_path=tmp_path / "followup" / "ledger.jsonl",
        max_usd=0.50,
        confirm_paid=True,
        urlopen=fake_urlopen,
    ).run()

    assert summary["pairwise_reused_total"] == 1
    assert summary["pairwise_novel_total"] == 0
    assert chat_calls == 0
    assert not (tmp_path / "followup" / "ledger.jsonl").exists()
    bucket_result = json.loads(
        (
            tmp_path
            / "followup"
            / "pilot"
            / "tiny_bucket"
            / "bucket-result.json"
        ).read_text()
    )
    assert bucket_result["calls"]["pairwise_active_reused"] == 1
    assert bucket_result["strategies"]["sestina_active_pairwise"]["k"] == 1


def test_scheduler_only_evsi_mode_can_report_posterior_topk_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    source_dir = tmp_path / "source"
    _write_pointwise_artifacts(source_dir)

    def fake_urlopen(request: Any, **kwargs: object) -> _FakeResponse:
        if request.full_url.endswith("/models"):
            return _FakeResponse({"data": [{"id": "openai/mini"}]})
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "winner": "left",
                                    "soft_probability": 0.82,
                                    "confidence": 0.9,
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setenv("SESTINA_LLM_API_KEY", "secret")
    monkeypatch.setenv("SESTINA_LLM_BASE_URL", "https://llm.example/v1")

    summary = SchedulerOnlyRunner(
        config_path=config_path,
        manifest_path=manifest_path,
        source_artifact_dir=source_dir,
        artifact_dir=tmp_path / "followup",
        ledger_path=tmp_path / "followup" / "ledger.jsonl",
        max_usd=0.50,
        confirm_paid=True,
        scheduler_kind="evsi",
        aggregation_mode="both",
        urlopen=fake_urlopen,
    ).run()

    bucket_result = summary["bucket_results"][0]
    assert bucket_result["scheduler_diagnostics"]["acquisition"]["method"] == (
        "top_k_evsi_approximation"
    )
    assert "sestina_active_posterior_topk" in bucket_result["strategies"]
    assert bucket_result["posterior_topk_diagnostics"]["method"] == (
        "independent_laplace_normal_sampling"
    )


def test_scheduler_only_exact_pool_random_mode_reports_pool_diagnostics(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    source_dir = tmp_path / "source"
    _write_pointwise_artifacts(source_dir)

    summary = SchedulerOnlyRunner(
        config_path=config_path,
        manifest_path=manifest_path,
        source_artifact_dir=source_dir,
        artifact_dir=tmp_path / "dry",
        ledger_path=tmp_path / "dry" / "ledger.jsonl",
        max_usd=0.50,
        scheduler_kind="exact_pool_random",
        aggregation_mode="posterior_topk",
    ).run()

    estimate = json.loads(Path(summary["estimate_path"]).read_text())
    diagnostics = estimate["buckets"][0]["scheduler_diagnostics"]
    assert summary["scheduler_kind"] == "exact_pool_random"
    assert diagnostics["acquisition"]["method"] == "exact_pool_random"
    assert diagnostics["acquisition"]["source_method"] == "top_k_evsi_approximation"
    assert "isolation" in diagnostics
    assert "offline_bucket_results_path" in summary


def test_scheduler_only_sequential_evsi_paid_resume_reveals_followup_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sequential EVSI resumes by revealing labels paid in prior invocations."""
    config_path = _write_config(tmp_path)
    papers = _many_manifest_papers(n=32, k=5)
    manifest_path = _write_manifest_for_papers(tmp_path, papers=papers, k=5)
    source_dir = tmp_path / "source"
    _write_pointwise_artifacts_for_papers(source_dir, papers=papers)
    chat_calls = 0

    def fake_urlopen(request: Any, **kwargs: object) -> _FakeResponse:
        nonlocal chat_calls
        if request.full_url.endswith("/models"):
            return _FakeResponse({"data": [{"id": "openai/mini"}]})
        chat_calls += 1
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "winner": "left",
                                    "soft_probability": 0.82,
                                    "confidence": 0.9,
                                    "reasons": ["fixture pairwise judgment"],
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setenv("SESTINA_LLM_API_KEY", "secret")
    monkeypatch.setenv("SESTINA_LLM_BASE_URL", "https://llm.example/v1")

    artifact_dir = tmp_path / "followup"
    ledger_path = artifact_dir / "ledger.jsonl"
    first = SchedulerOnlyRunner(
        config_path=config_path,
        manifest_path=manifest_path,
        source_artifact_dir=source_dir,
        artifact_dir=artifact_dir,
        ledger_path=ledger_path,
        max_usd=0.50,
        confirm_paid=True,
        scheduler_kind="sequential_evsi",
        aggregation_mode="posterior_topk",
        urlopen=fake_urlopen,
    ).run()

    assert first["pairwise_scheduled_total"] == 4
    assert first["pairwise_novel_total"] == 4
    assert first["new_ledger_entries_this_invocation"] == 4
    assert first["sequential_evsi_completion"]["status"] == (
        "needs_guarded_paid_resume"
    )
    assert chat_calls == 4

    second = SchedulerOnlyRunner(
        config_path=config_path,
        manifest_path=manifest_path,
        source_artifact_dir=source_dir,
        artifact_dir=artifact_dir,
        ledger_path=ledger_path,
        max_usd=0.50,
        scheduler_kind="sequential_evsi",
        aggregation_mode="posterior_topk",
        urlopen=fake_urlopen,
    ).run()

    assert second["pairwise_scheduled_total"] == 8
    assert second["pairwise_reused_total"] == 4
    assert second["pairwise_novel_total"] == 4
    assert second["sequential_evsi_completion"]["status"] == "complete"
    diagnostics = second["offline_bucket_results"][0]["scheduler_diagnostics"]
    assert diagnostics["batch_history"][0]["cached_label_revealed_total"] == 4
    assert diagnostics["batch_history"][1]["novel_pairs_total"] == 4
    assert chat_calls == 4


def test_scheduler_only_cctd_gf_mode_reports_mixed_batch_diagnostics(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    papers = _many_manifest_papers(n=32, k=5)
    manifest_path = _write_manifest_for_papers(tmp_path, papers=papers, k=5)
    source_dir = tmp_path / "source"
    _write_pointwise_artifacts_for_papers(source_dir, papers=papers)

    summary = SchedulerOnlyRunner(
        config_path=config_path,
        manifest_path=manifest_path,
        source_artifact_dir=source_dir,
        artifact_dir=tmp_path / "dry",
        ledger_path=tmp_path / "dry" / "ledger.jsonl",
        max_usd=2.00,
        scheduler_kind="cctd_gf",
        aggregation_mode="posterior_topk",
    ).run()

    estimate = json.loads(Path(summary["estimate_path"]).read_text())
    diagnostics = estimate["buckets"][0]["scheduler_diagnostics"]
    assert summary["scheduler_kind"] == "cctd_gf"
    assert summary["pairwise_scheduled_total"] == 5
    assert diagnostics["acquisition"]["method"] == "cctd_gf"
    assert diagnostics["purpose_counts"] == {
        "cctd_gf_disagreement": 3,
        "cctd_gf_graph_floor": 1,
        "cctd_gf_random_floor": 1,
    }
    assert diagnostics["batch_history"][0]["selected_total"] == 5
    assert diagnostics["cctd_gf_score_distribution"]["proposal_count"] > 0
    assert summary["cctd_gf_completion"]["status"] == "needs_guarded_paid_resume"


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
                        "buckets": [{"name": "tiny_bucket", "n": 2, "k": 1}],
                    }
                ],
            }
        )
    )
    return path


def _many_manifest_papers(*, n: int, k: int) -> list[dict[str, Any]]:
    return [
        {
            "paper_id": f"p{index:02d}",
            "title": f"Paper {index}",
            "abstract": f"Abstract for paper {index}.",
            "baseline_score": max(0.05, 0.96 - (index * 0.02)),
            "uncertainty": min(0.9, 0.2 + (index * 0.01)),
            "labels": {"good_paper": index <= k},
            "metadata": {
                "primary_category": "cs.LG" if index % 2 else "cs.CL",
                "source": "arxiv",
            },
        }
        for index in range(1, n + 1)
    ]


def _write_manifest_for_papers(
    tmp_path: Path,
    *,
    papers: list[dict[str, Any]],
    k: int,
) -> Path:
    path = tmp_path / "manifest-many.json"
    path.write_text(
        json.dumps(
            {
                "artifact_type": "sestina-backtest-dataset-manifest",
                "buckets": [
                    {
                        "name": "tiny_bucket",
                        "phase": "pilot",
                        "k": k,
                        "papers": papers,
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
                                "abstract": "A good abstract.",
                                "baseline_score": 0.5,
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
                                "baseline_score": 0.5,
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


def _write_pointwise_artifacts(
    source_dir: Path,
    *,
    missing_ids: set[str] | None = None,
) -> None:
    missing = missing_ids or set()
    calls_dir = source_dir / "pilot" / "tiny_bucket" / "calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    responses = {
        "p1": PointwiseAssessment(
            good_probability=0.9,
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
        if paper_id in missing:
            continue
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


def _write_pointwise_artifacts_for_papers(
    source_dir: Path,
    *,
    papers: list[dict[str, Any]],
) -> None:
    calls_dir = source_dir / "pilot" / "tiny_bucket" / "calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    for index, paper in enumerate(papers, start=1):
        paper_id = str(paper["paper_id"])
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
                    "response": PointwiseAssessment(
                        good_probability=float(paper["baseline_score"]),
                        uncertainty=float(paper["uncertainty"]),
                        summary="fixture",
                    ).to_dict(),
                    "subject": {"paper_id": paper_id},
                }
            )
        )


def _write_legacy_pairwise_artifact(source_dir: Path) -> None:
    papers = [
        Paper(
            paper_id="p1",
            title="First paper",
            pointwise=PointwiseAssessment(good_probability=0.9, uncertainty=0.2),
            metadata={"primary_category": "cs.LG", "source": "arxiv"},
        ),
        Paper(
            paper_id="p2",
            title="Second paper",
            pointwise=PointwiseAssessment(good_probability=0.4, uncertainty=0.6),
            metadata={"primary_category": "cs.CL", "source": "arxiv"},
        ),
    ]
    selection = legacy_select_candidates(papers, k=1)
    budget = resolve_pairwise_budget(
        n=len(papers),
        candidate_size=len(selection.candidate_ids),
    )
    pair = legacy_schedule_pairs(
        papers,
        candidate_selection=selection,
        k=1,
        budget=budget,
        seed=17,
    ).pairs[0]
    config = json.loads(
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
                        "input_tokens_per_call": 10,
                        "output_tokens_per_call": 5,
                    }
                },
            }
        )
    )
    estimate = _call_estimate(
        "pairwise",
        "openai/mini",
        _normalize_token_assumptions_from_config(config),
        _normalize_rates_from_config(config),
    )
    calls_dir = source_dir / "pilot" / "tiny_bucket" / "calls"
    path = calls_dir / (
        "0001-pairwise_active-"
        f"{fingerprint(pair.left_id + ':' + pair.right_id)}.json"
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
                "estimated_cost_usd": estimate.cost_usd,
                "estimated_tokens": {
                    "input": estimate.input_tokens,
                    "output": estimate.output_tokens,
                },
                "response": {
                    "winner": "left",
                    "soft_probability": 0.8,
                    "confidence": 0.9,
                    "reasons": ["historical fixture"],
                },
                "subject": {
                    "left_id": pair.left_id,
                    "right_id": pair.right_id,
                },
            }
        )
    )


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
