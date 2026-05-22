from __future__ import annotations

import json
from pathlib import Path

from scripts.design_coverage_floor_fresh_holdout import (
    ARTIFACT_TYPE,
    DEFAULT_BUCKETS,
    build_fresh_holdout_design,
    validate_fresh_holdout_design,
)


def test_fresh_holdout_design_skips_old_development_buckets(tmp_path: Path) -> None:
    development_manifest = _write_development_manifest(tmp_path)

    payload = build_fresh_holdout_design(
        development_manifest_path=development_manifest,
        output_path=tmp_path / "design.json",
        manifest_output_path=tmp_path / "fresh-manifest.json",
        part_dir=tmp_path / "parts",
        bucket_values=[
            "cs.LG:2023-01",
            "cs.CL:2023-01",
            "cs.LG:2023-03",
            "cs.CL:2023-03",
        ],
        target_bucket_count=2,
    )

    validate_fresh_holdout_design(payload)
    assert payload["artifact_type"] == ARTIFACT_TYPE
    assert payload["predeclared_before_results_analysis"] is True
    assert [row["bucket"] for row in payload["fresh_bucket_specs"]] == [
        "cs.LG:2023-03",
        "cs.CL:2023-03",
    ]
    assert {row["reason"] for row in payload["rejected_bucket_specs"]} == {
        "overlaps_development_category_date_bucket"
    }
    assert "--target-bucket-count" in payload["builder_command"]
    assert "--arxiv-metadata-source" in payload["builder_command"]
    assert "huldra" in payload["builder_command"]
    assert "--huldra-base-url" in payload["builder_command"]
    assert "http://127.0.0.1:8765" in payload["builder_command"]
    assert "--huldra-wait-timeout-seconds" in payload["builder_command"]
    assert "600" in payload["builder_command"]
    assert "--arxiv-page-size" in payload["builder_command"]
    assert "--arxiv-pacing-delays-seconds" in payload["builder_command"]
    assert "15" in payload["builder_command"]
    assert payload["selection_policy"]["arxiv_page_size"] == 5
    assert payload["selection_policy"]["arxiv_pacing_delays_seconds"] == "15"
    assert payload["selection_policy"]["arxiv_metadata_source"] == "huldra"
    assert payload["selection_policy"]["huldra_base_url"] == "http://127.0.0.1:8765"
    assert payload["selection_policy"]["huldra_wait_timeout_seconds"] == 600
    assert (tmp_path / "design.json").exists()


def test_fresh_holdout_design_validation_rejects_development_overlap() -> None:
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": 1,
        "predeclared_before_results_analysis": True,
        "fresh_bucket_specs": [
            {
                "category": "cs.LG",
                "date_bucket": "2023-01",
                "expected_bucket_name": "arxiv_cs_LG_2023_01_historical_citation_pilot",
            }
        ],
        "development_manifest": {
            "bucket_names": ["arxiv_cs_LG_2023_01_historical_citation_pilot"],
            "category_date_buckets": [
                {"category": "cs.LG", "date_bucket": "2023-01"}
            ],
        },
    }

    try:
        validate_fresh_holdout_design(payload)
    except ValueError as exc:
        assert "overlaps development bucket names" in str(exc)
    else:
        raise AssertionError("development overlap was accepted")


def test_default_fresh_holdout_buckets_remain_frozen() -> None:
    assert DEFAULT_BUCKETS == (
        "cs.LG:2023-03",
        "cs.LG:2023-04",
        "cs.CL:2023-03",
        "cs.CL:2023-04",
        "cs.AI:2023-03",
        "cs.AI:2023-04",
        "cs.CV:2023-03",
        "cs.CV:2023-04",
    )


def _write_development_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "development.json"
    path.write_text(
        json.dumps(
            {
                "artifact_type": "sestina-backtest-dataset-manifest",
                "buckets": [
                    _bucket("arxiv_cs_LG_2023_01_historical_citation_pilot", "cs.LG"),
                    _bucket("arxiv_cs_CL_2023_01_historical_citation_pilot", "cs.CL"),
                ],
            }
        )
    )
    return path


def _bucket(name: str, category: str) -> dict:
    return {
        "name": name,
        "phase": "pilot",
        "k": 1,
        "source": {
            "category": category,
            "date_bucket": "2023-01",
        },
        "papers": [
            {
                "paper_id": f"{name}:1",
                "title": "Development paper",
                "abstract": "Development abstract",
                "baseline_score": 0.5,
                "labels": {"good_paper": True, "arxiv_id": f"{name}-arxiv"},
                "metadata": {"primary_category": category},
            }
        ],
    }
