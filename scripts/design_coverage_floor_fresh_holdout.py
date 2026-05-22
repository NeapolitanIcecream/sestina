#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_arxiv_historical_manifest import parse_bucket_spec  # noqa: E402
from sestina.backtest_runner import load_dataset_manifest  # noqa: E402
from sestina.diagnostics import write_json_artifact  # noqa: E402


ARTIFACT_TYPE = "sestina-coverage-floor-fresh-holdout-design"
SCHEMA_VERSION = 1
DEFAULT_DEVELOPMENT_MANIFEST = (
    REPO_ROOT / "artifacts" / "backtest-datasets" / "arxiv-historical-pilot-manifest.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-autonomous-holdout-campaign"
    / "fresh-holdout-design.json"
)
DEFAULT_MANIFEST_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-datasets"
    / "arxiv-historical-coverage-floor-fresh-holdout-manifest.json"
)
DEFAULT_PART_DIR = (
    REPO_ROOT
    / "artifacts"
    / "backtest-datasets"
    / "arxiv-historical-coverage-floor-fresh-holdout-parts"
)
DEFAULT_BUCKETS = (
    "cs.LG:2023-03",
    "cs.LG:2023-04",
    "cs.CL:2023-03",
    "cs.CL:2023-04",
    "cs.AI:2023-03",
    "cs.AI:2023-04",
    "cs.CV:2023-03",
    "cs.CV:2023-04",
)
DEFAULT_ARXIV_PAGE_SIZE = 5
DEFAULT_ARXIV_PACING_DELAYS_SECONDS = "15"
DEFAULT_HULDRA_BASE_URL = "http://127.0.0.1:8765"
DEFAULT_HULDRA_WAIT_TIMEOUT_SECONDS = 600


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the autonomous fresh holdout bucket design before any "
            "fresh-holdout pointwise, pairwise, or result analysis runs."
        )
    )
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=DEFAULT_DEVELOPMENT_MANIFEST,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument("--part-dir", type=Path, default=DEFAULT_PART_DIR)
    parser.add_argument("--bucket", action="append", default=[])
    parser.add_argument("--phase", default="pilot")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--metadata-provider", default="auto")
    parser.add_argument("--unmatched-policy", default="drop")
    parser.add_argument("--target-bucket-count", type=int, default=8)
    parser.add_argument("--arxiv-page-size", type=int, default=DEFAULT_ARXIV_PAGE_SIZE)
    parser.add_argument("--huldra-base-url", default=DEFAULT_HULDRA_BASE_URL)
    parser.add_argument(
        "--huldra-wait-timeout-seconds",
        type=int,
        default=DEFAULT_HULDRA_WAIT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--arxiv-pacing-delays-seconds",
        default=DEFAULT_ARXIV_PACING_DELAYS_SECONDS,
        help=(
            "comma-separated request spacing tiers for arXiv manifest builds; "
            "the campaign default keeps every arXiv request at least 15 seconds apart"
        ),
    )
    args = parser.parse_args(argv)

    payload = build_fresh_holdout_design(
        development_manifest_path=args.development_manifest,
        output_path=args.output,
        manifest_output_path=args.manifest_output,
        part_dir=args.part_dir,
        bucket_values=args.bucket or DEFAULT_BUCKETS,
        phase=args.phase,
        limit=args.limit,
        k=args.k,
        metadata_provider=args.metadata_provider,
        unmatched_policy=args.unmatched_policy,
        target_bucket_count=args.target_bucket_count,
        arxiv_page_size=args.arxiv_page_size,
        arxiv_pacing_delays_seconds=args.arxiv_pacing_delays_seconds,
        huldra_base_url=args.huldra_base_url,
        huldra_wait_timeout_seconds=args.huldra_wait_timeout_seconds,
    )
    sys.stdout.write(json.dumps(_stdout_summary(payload), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def build_fresh_holdout_design(
    *,
    development_manifest_path: Path,
    output_path: Path,
    manifest_output_path: Path,
    part_dir: Path,
    bucket_values: Sequence[str],
    phase: str = "pilot",
    limit: int = 80,
    k: int = 5,
    metadata_provider: str = "auto",
    unmatched_policy: str = "drop",
    target_bucket_count: int = 8,
    arxiv_page_size: int = DEFAULT_ARXIV_PAGE_SIZE,
    arxiv_pacing_delays_seconds: str = DEFAULT_ARXIV_PACING_DELAYS_SECONDS,
    huldra_base_url: str = DEFAULT_HULDRA_BASE_URL,
    huldra_wait_timeout_seconds: int = DEFAULT_HULDRA_WAIT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if target_bucket_count <= 0:
        raise ValueError("target_bucket_count must be greater than zero")
    if arxiv_page_size <= 0:
        raise ValueError("arxiv_page_size must be greater than zero")
    if huldra_wait_timeout_seconds <= 0:
        raise ValueError("huldra_wait_timeout_seconds must be greater than zero")
    if len(bucket_values) < target_bucket_count:
        raise ValueError("not enough bucket specs to satisfy target_bucket_count")

    development = load_dataset_manifest(development_manifest_path)
    development_sources = _development_sources(development.payload)
    development_bucket_names = {bucket.name for bucket in development.buckets}
    development_arxiv_ids = _manifest_arxiv_ids(development.payload)

    selected_specs = []
    selected_identity = set()
    rejected = []
    for raw_value in bucket_values:
        spec = parse_bucket_spec(raw_value)
        identity = (spec.category, spec.date_bucket.label)
        bucket_name = _bucket_name(
            category=spec.category,
            label=spec.date_bucket.label,
            phase=phase,
        )
        if identity in selected_identity:
            rejected.append(
                {
                    "bucket": raw_value,
                    "reason": "duplicate_requested_bucket_identity",
                }
            )
            continue
        if identity in development_sources:
            rejected.append(
                {
                    "bucket": raw_value,
                    "reason": "overlaps_development_category_date_bucket",
                }
            )
            continue
        if bucket_name in development_bucket_names:
            rejected.append(
                {
                    "bucket": raw_value,
                    "reason": "overlaps_development_bucket_name",
                }
            )
            continue
        selected_specs.append(
            {
                "bucket": raw_value,
                "category": spec.category,
                "date_bucket": spec.date_bucket.label,
                "start_date": spec.date_bucket.start.isoformat(),
                "end_date": spec.date_bucket.end.isoformat(),
                "expected_bucket_name": bucket_name,
            }
        )
        selected_identity.add(identity)
        if len(selected_specs) >= target_bucket_count:
            break

    if len(selected_specs) != target_bucket_count:
        raise RuntimeError(
            f"selected {len(selected_specs)} fresh buckets; "
            f"target is {target_bucket_count}"
        )

    command = [
        "uv",
        "run",
        "python",
        "scripts/build_arxiv_historical_manifest.py",
    ]
    for row in selected_specs:
        command.extend(["--bucket", str(row["bucket"])])
    command.extend(
        [
            "--limit",
            str(limit),
            "--arxiv-metadata-source",
            "huldra",
            "--huldra-base-url",
            huldra_base_url,
            "--huldra-wait-timeout-seconds",
            str(huldra_wait_timeout_seconds),
            "--arxiv-page-size",
            str(arxiv_page_size),
            "--arxiv-pacing-delays-seconds",
            arxiv_pacing_delays_seconds,
            "--k",
            str(k),
            "--phase",
            phase,
            "--metadata-provider",
            metadata_provider,
            "--unmatched-policy",
            unmatched_policy,
            "--part-dir",
            str(part_dir),
            "--reuse-parts",
            "--write-parts",
            "--target-bucket-count",
            str(target_bucket_count),
            "--output",
            str(manifest_output_path),
        ]
    )

    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_predeclared",
        "predeclared_before_results_analysis": True,
        "selection_policy": {
            "summary": (
                "Use the same four development categories for comparability, "
                "but the next two chronological months after the old 2023-01 "
                "and 2023-02 development buckets."
            ),
            "uses_fresh_result_metrics_or_labels": False,
            "uses_future_citation_labels_for_selection": False,
            "old_development_buckets_excluded": True,
            "target_bucket_count": target_bucket_count,
            "limit_per_bucket": limit,
            "arxiv_page_size": arxiv_page_size,
            "arxiv_pacing_delays_seconds": arxiv_pacing_delays_seconds,
            "arxiv_metadata_source": "huldra",
            "huldra_base_url": huldra_base_url,
            "huldra_wait_timeout_seconds": huldra_wait_timeout_seconds,
            "k_per_bucket": k,
        },
        "development_manifest": {
            "path": str(development_manifest_path),
            "sha256": _sha256(development_manifest_path),
            "bucket_names": sorted(development_bucket_names),
            "category_date_buckets": [
                {"category": category, "date_bucket": date_bucket}
                for category, date_bucket in sorted(development_sources)
            ],
            "arxiv_id_count": len(development_arxiv_ids),
        },
        "fresh_bucket_specs": selected_specs,
        "rejected_bucket_specs": rejected,
        "expected_manifest_output": str(manifest_output_path),
        "part_dir": str(part_dir),
        "builder_command": command,
        "leakage_policy": {
            "model_visible_inputs_later": [
                "title",
                "abstract",
                "primary_category",
                "categories",
                "publication and bucket dates",
            ],
            "forbidden_model_visible_inputs": [
                "good_paper",
                "citation_count",
                "citation_rank",
                "citation_percentile",
                "matched_title",
                "matched_work_id",
                "evaluation_labels",
            ],
        },
        "validation_commands": [
            "uv run python scripts/design_coverage_floor_fresh_holdout.py",
            "uv run pytest tests/test_coverage_floor_fresh_holdout_design.py",
            "git diff --check",
        ],
        "output_path": str(output_path),
    }
    validate_fresh_holdout_design(payload)
    write_json_artifact(output_path, payload)
    return {**payload, "artifact_path": str(output_path)}


def validate_fresh_holdout_design(payload: Mapping[str, Any]) -> None:
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("fresh holdout design has unexpected artifact_type")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("fresh holdout design has unexpected schema_version")
    if payload.get("predeclared_before_results_analysis") is not True:
        raise ValueError("fresh holdout design must be predeclared")
    specs = payload.get("fresh_bucket_specs")
    if not isinstance(specs, list) or not specs:
        raise ValueError("fresh holdout design must include bucket specs")
    expected_names = [str(row.get("expected_bucket_name") or "") for row in specs]
    if len(set(expected_names)) != len(expected_names):
        raise ValueError("fresh holdout design has duplicate bucket names")
    development = payload.get("development_manifest")
    if not isinstance(development, Mapping):
        raise ValueError("fresh holdout design missing development manifest")
    development_names = set(development.get("bucket_names") or [])
    if set(expected_names) & development_names:
        raise ValueError("fresh holdout design overlaps development bucket names")
    development_pairs = {
        (str(row.get("category")), str(row.get("date_bucket")))
        for row in development.get("category_date_buckets") or []
        if isinstance(row, Mapping)
    }
    fresh_pairs = {
        (str(row.get("category")), str(row.get("date_bucket")))
        for row in specs
        if isinstance(row, Mapping)
    }
    if fresh_pairs & development_pairs:
        raise ValueError("fresh holdout design overlaps development category/date")


def _development_sources(payload: Mapping[str, Any]) -> set[tuple[str, str]]:
    output = set()
    for bucket in payload.get("buckets") or []:
        if not isinstance(bucket, Mapping):
            continue
        source = bucket.get("source") or {}
        if not isinstance(source, Mapping):
            continue
        category = source.get("category")
        date_bucket = source.get("date_bucket")
        if category and date_bucket:
            output.add((str(category), str(date_bucket)))
    return output


def _manifest_arxiv_ids(payload: Mapping[str, Any]) -> set[str]:
    output = set()
    for bucket in payload.get("buckets") or []:
        if not isinstance(bucket, Mapping):
            continue
        for paper in bucket.get("papers") or []:
            if not isinstance(paper, Mapping):
                continue
            labels = paper.get("labels") or {}
            if isinstance(labels, Mapping) and labels.get("arxiv_id"):
                output.add(str(labels["arxiv_id"]))
    return output


def _bucket_name(*, category: str, label: str, phase: str) -> str:
    safe_category = re.sub(r"[^A-Za-z0-9]+", "_", category).strip("_")
    safe_label = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    safe_phase = re.sub(r"[^A-Za-z0-9]+", "_", phase).strip("_")
    return f"arxiv_{safe_category}_{safe_label}_historical_citation_{safe_phase}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stdout_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_path": payload.get("artifact_path") or payload.get("output_path"),
        "artifact_type": payload["artifact_type"],
        "status": payload["status"],
        "predeclared_before_results_analysis": payload[
            "predeclared_before_results_analysis"
        ],
        "bucket_count": len(payload["fresh_bucket_specs"]),
        "expected_manifest_output": payload["expected_manifest_output"],
        "builder_command": payload["builder_command"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
