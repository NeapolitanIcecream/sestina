from __future__ import annotations

import json
import subprocess
import sys

from sestina.models import PairwiseComparison, TargetSpec
from sestina.pipeline import run_pipeline


def test_pipeline_emits_diagnostics_for_each_runtime_stage(paper_set) -> None:
    papers = paper_set(12)

    result = run_pipeline(
        papers,
        TargetSpec(top_k=3),
        comparisons=[
            PairwiseComparison(
                left_id="p4",
                right_id="p2",
                winner="left",
                soft_probability=0.8,
                confidence=0.7,
            )
        ],
        config={"posterior_samples": 300, "seed": 7},
    )

    payload = result.to_dict()
    steps = {event["step"] for event in payload["diagnostics"]["events"]}
    assert {
        "input_validation",
        "target",
        "candidate_selection",
        "pairwise_budget",
        "pair_scheduling",
        "comparison_ingestion",
        "aggregation",
        "uncertainty",
        "output",
    }.issubset(steps)
    assert len(payload["recommendations"]["recommended_good_papers"]) == 3


def test_pipeline_candidate_size_config_matches_diagnostics(paper_set) -> None:
    papers = paper_set(12)

    result = run_pipeline(
        papers,
        TargetSpec(top_k=3),
        config={"candidate_size": 6, "posterior_samples": 300, "seed": 7},
    )

    payload = result.to_dict()
    candidate_diagnostics = payload["candidate_selection"]["diagnostics"]
    assert payload["config"]["candidate_size"] == 6
    assert candidate_diagnostics["candidate_size_source"] == "override"
    assert candidate_diagnostics["candidate_size_requested"] == 6
    assert candidate_diagnostics["candidate_size"] == 6
    assert payload["pairwise_schedule"]["budget"]["candidate_size"] == 6


def test_pipeline_default_candidate_size_is_explicitly_diagnosed(paper_set) -> None:
    papers = paper_set(12)

    result = run_pipeline(
        papers,
        TargetSpec(top_k=3),
        config={"posterior_samples": 300, "seed": 7},
    )

    payload = result.to_dict()
    candidate_diagnostics = payload["candidate_selection"]["diagnostics"]
    assert payload["config"]["candidate_size"] is None
    assert candidate_diagnostics["candidate_size_source"] == "default"
    assert candidate_diagnostics["candidate_size_requested"] is None
    assert (
        candidate_diagnostics["candidate_size"]
        == candidate_diagnostics["candidate_size_default"]
    )


def test_cli_runs_offline_and_writes_json_markdown_and_debug_artifact(tmp_path) -> None:
    output_path = tmp_path / "out.json"
    report_path = tmp_path / "report.md"
    debug_dir = tmp_path / "debug"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sestina.cli",
            "run",
            "examples/sample_papers.json",
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--debug-dir",
            str(debug_dir),
            "--posterior-samples",
            "300",
            "--seed",
            "11",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text())
    assert payload["target"]["k"] == 3
    assert len(payload["recommendations"]["recommended_good_papers"]) == 3
    assert "Recommended Good Papers" in report_path.read_text()
    assert (debug_dir / "sestina-run-diagnostics.json").exists()


def test_cli_accepts_jsonl_with_cli_target(tmp_path, paper_set) -> None:
    jsonl_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "jsonl-out.json"
    jsonl_path.write_text(
        "\n".join(
            json.dumps(paper.to_dict(include_text=True)) for paper in paper_set(5)
        )
        + "\n"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sestina.cli",
            "run",
            str(jsonl_path),
            "--top-alpha",
            "0.4",
            "--output",
            str(output_path),
            "--posterior-samples",
            "300",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text())
    assert payload["target"]["k"] == 2
    assert len(payload["recommendations"]["recommended_good_papers"]) == 2


def test_cli_failure_writes_error_debug_artifact(tmp_path) -> None:
    bad_input = tmp_path / "bad.json"
    bad_input.write_text(json.dumps({"papers": [], "target": {}}))
    debug_dir = tmp_path / "debug"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sestina.cli",
            "run",
            str(bad_input),
            "--debug-dir",
            str(debug_dir),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    artifact = debug_dir / "sestina-error-debug.json"
    assert artifact.exists()
    payload = json.loads(artifact.read_text())
    assert payload["diagnostics"]["events"][-1]["code"] == "cli_run_failed"


def test_cli_candidate_size_zero_retains_structured_diagnostic(tmp_path) -> None:
    debug_dir = tmp_path / "debug"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sestina.cli",
            "run",
            "examples/sample_papers.json",
            "--candidate-size",
            "0",
            "--debug-dir",
            str(debug_dir),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert "candidate_size must be at least resolved top K" in completed.stderr
    artifact = debug_dir / "sestina-error-debug.json"
    payload = json.loads(artifact.read_text())
    codes = [event["code"] for event in payload["diagnostics"]["events"]]
    assert "candidate_size_invalid" in codes
    assert codes[-1] == "cli_run_failed"
