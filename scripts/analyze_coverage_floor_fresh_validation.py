#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ci_partition_gate import (  # noqa: E402
    DEFAULT_SEEDS,
    _orient_cached_comparison,
    _parse_seeds,
    load_cached_pairwise_labels,
)
from scripts.run_coverage_floor_followup_preflight import (  # noqa: E402
    ARM_COVERAGE,
    ARM_EXACT,
    DEFAULT_ACTIVE_GATE_ARTIFACT,
    DEFAULT_CONFIG,
    DEFAULT_FRESH_HOLDOUT_MANIFEST,
    DEFAULT_NO_PAID_SWEEP_ARTIFACT,
    DEFAULT_SOURCE_ARTIFACT_DIR,
    HybridScheduleConfig,
    _all_unordered_pair_keys,
    _freeze_no_paid_winner,
)
from sestina.backtest import compare_strategies  # noqa: E402
from sestina.backtest_budget import load_config  # noqa: E402
from sestina.backtest_runner import _config_for_phase, load_dataset_manifest  # noqa: E402
from sestina.candidates import select_candidates  # noqa: E402
from sestina.ci_partition_gate import CIPartitionConfig, schedule_cached_exact_pool_random  # noqa: E402
from sestina.diagnostics import write_json_artifact  # noqa: E402
from sestina.evsi_scheduler import posterior_top_k_predictions  # noqa: E402
from sestina.no_paid_algorithm_sweep import (  # noqa: E402
    canonical_pair_key,
    paired_seed_metric_deltas,
    schedule_model_visible_hybrid_pairs,
    summarize_values,
)
from sestina.scheduler import resolve_pairwise_budget  # noqa: E402
from sestina.scheduler_followup import (  # noqa: E402
    PointwiseArtifactError,
    load_pointwise_papers_from_artifacts,
)


ARTIFACT_TYPE = "sestina-coverage-floor-fresh-validation-analysis"
SCHEMA_VERSION = 1
POSTERIOR_STRATEGY = "posterior_topk"
METRICS = ("recall_at_k", "ndcg_at_k", "average_precision")
DEFAULT_PAIRWISE_ARTIFACT_DIR = (
    REPO_ROOT / "artifacts" / "backtest-arxiv-coverage-floor-followup-preflight"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-autonomous-holdout-campaign"
    / "fresh-validation-analysis.json"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the autonomous fresh holdout coverage-floor validation "
            "against exact-pool random after pairwise-only artifacts exist."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--no-paid-sweep-artifact",
        type=Path,
        default=DEFAULT_NO_PAID_SWEEP_ARTIFACT,
    )
    parser.add_argument(
        "--active-gate-artifact",
        type=Path,
        default=DEFAULT_ACTIVE_GATE_ARTIFACT,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_FRESH_HOLDOUT_MANIFEST)
    parser.add_argument(
        "--source-artifact-dir",
        type=Path,
        default=DEFAULT_SOURCE_ARTIFACT_DIR,
    )
    parser.add_argument(
        "--pairwise-artifact-dir",
        type=Path,
        default=DEFAULT_PAIRWISE_ARTIFACT_DIR,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", default=None)
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args(argv)

    payload = analyze_coverage_floor_fresh_validation(
        config_path=args.config,
        no_paid_sweep_artifact_path=args.no_paid_sweep_artifact,
        active_gate_artifact_path=args.active_gate_artifact,
        manifest_path=args.manifest,
        source_artifact_dir=args.source_artifact_dir,
        pairwise_artifact_dir=args.pairwise_artifact_dir,
        output_path=args.output,
        phase=args.phase,
        seeds=_parse_seeds(args.seeds) if args.seeds else None,
    )
    sys.stdout.write(json.dumps(_stdout_summary(payload), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0 if payload["fresh_validation_claim"]["complete"] else 2


def analyze_coverage_floor_fresh_validation(
    *,
    config_path: Path,
    no_paid_sweep_artifact_path: Path,
    active_gate_artifact_path: Path,
    manifest_path: Path,
    source_artifact_dir: Path,
    pairwise_artifact_dir: Path,
    output_path: Path,
    phase: str | None = None,
    seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    no_paid_sweep = _read_json(no_paid_sweep_artifact_path)
    active_gate = _read_json(active_gate_artifact_path)
    frozen = _freeze_no_paid_winner(
        no_paid_sweep=no_paid_sweep,
        active_gate=active_gate,
        no_paid_sweep_artifact_path=no_paid_sweep_artifact_path,
        active_gate_artifact_path=active_gate_artifact_path,
        requested_phase=phase,
        requested_seeds=seeds,
    )
    phase_name = str(frozen["phase"])
    seed_set = [int(seed) for seed in frozen["seeds"]]
    raw_config = load_config(config_path)
    phase_config = _config_for_phase(raw_config, phase=phase_name)["phases"][0]
    manifest = load_dataset_manifest(manifest_path)
    policy = frozen["active_policy"]
    active_config = HybridScheduleConfig(
        name=ARM_COVERAGE,
        random_floor_fraction=float(policy.get("random_floor_fraction") or 0.35),
        min_random_floor_pairs=int(policy.get("min_random_floor_pairs") or 1),
        per_item_cap=int(policy.get("per_item_cap") or 6),
        anchor_multiplier=int(policy.get("anchor_multiplier") or 2),
        challenger_multiplier=int(policy.get("challenger_multiplier") or 5),
    )
    random_config = CIPartitionConfig(
        pairwise_strength=float(policy.get("pairwise_strength") or 2.5),
        posterior_samples=int(policy.get("scheduler_samples") or 800),
        confidence_z=float(policy.get("confidence_z") or 1.96),
        random_floor_fraction=float(policy.get("random_floor_fraction") or 0.35),
    )
    posterior_samples = int(policy.get("posterior_samples") or 900)
    pairwise_strength = float(policy.get("pairwise_strength") or 2.5)

    bucket_results = []
    completeness = Counter()
    cache_stats_by_bucket: dict[str, dict[str, Any]] = {}
    pointwise_error: str | None = None
    for seed in seed_set:
        seed_payload = {"seed": int(seed), "buckets": []}
        for bucket in manifest.buckets_for_phase(phase_name):
            try:
                papers = load_pointwise_papers_from_artifacts(
                    bucket,
                    source_artifact_dir=source_artifact_dir,
                    phase=phase_name,
                )
            except PointwiseArtifactError as exc:
                pointwise_error = str(exc)
                completeness["missing_pointwise_buckets"] += 1
                continue
            selection = select_candidates(papers, k=bucket.k)
            budget = resolve_pairwise_budget(
                n=len(papers),
                candidate_size=len(selection.candidate_ids),
            )
            cached, cache_stats = load_cached_pairwise_labels(
                bucket.name,
                artifact_dirs=[pairwise_artifact_dir],
                phase=phase_name,
            )
            cache_stats_by_bucket.setdefault(bucket.name, cache_stats)
            all_pair_keys = _all_unordered_pair_keys(papers)
            active_schedule, active_diagnostics = schedule_model_visible_hybrid_pairs(
                papers,
                k=bucket.k,
                budget=budget,
                seed=int(seed),
                available_pair_keys=all_pair_keys,
                config=active_config,
            )
            random_schedule = schedule_cached_exact_pool_random(
                papers,
                [],
                k=bucket.k,
                budget=budget,
                seed=int(seed),
                config=random_config,
                available_pair_keys=None,
            )
            active_comparisons, active_missing = _comparisons_for_schedule(
                active_schedule,
                cached,
            )
            random_comparisons, random_missing = _comparisons_for_schedule(
                random_schedule.pairs,
                cached,
            )
            completeness["scheduled_pairwise_occurrences"] += (
                len(active_schedule) + len(random_schedule.pairs)
            )
            completeness["missing_pairwise_occurrences"] += (
                active_missing + random_missing
            )
            seed_payload["buckets"].append(
                {
                    "bucket": bucket.name,
                    "seed": int(seed),
                    "k": bucket.k,
                    "papers_total": len(papers),
                    "positive_labels_total": len(bucket.relevant_ids),
                    "budget": budget.to_dict(),
                    "arms": {
                        ARM_COVERAGE: _arm_metrics(
                            papers,
                            relevant_ids=bucket.relevant_ids,
                            k=bucket.k,
                            schedule=active_schedule,
                            comparisons=active_comparisons,
                            missing_pairwise_labels=active_missing,
                            scheduler_diagnostics=active_diagnostics,
                            seed=int(seed),
                            posterior_samples=posterior_samples,
                            pairwise_strength=pairwise_strength,
                        ),
                        ARM_EXACT: _arm_metrics(
                            papers,
                            relevant_ids=bucket.relevant_ids,
                            k=bucket.k,
                            schedule=random_schedule.pairs,
                            comparisons=random_comparisons,
                            missing_pairwise_labels=random_missing,
                            scheduler_diagnostics=random_schedule.diagnostics,
                            seed=int(seed),
                            posterior_samples=posterior_samples,
                            pairwise_strength=pairwise_strength,
                        ),
                    },
                }
            )
        bucket_results.append(seed_payload)

    aggregate_metrics, seed_metric_rows = _aggregate_metrics(bucket_results)
    paired_deltas = paired_seed_metric_deltas(
        seed_metric_rows,
        comparison_arm=ARM_COVERAGE,
        reference_arm=ARM_EXACT,
    )
    complete = (
        completeness["missing_pointwise_buckets"] == 0
        and completeness["missing_pairwise_occurrences"] == 0
        and bool(bucket_results)
        and all(seed_payload["buckets"] for seed_payload in bucket_results)
    )
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 0,
        "phase": phase_name,
        "input_artifacts": {
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "no_paid_sweep_artifact_path": str(no_paid_sweep_artifact_path),
            "no_paid_sweep_artifact_sha256": _sha256(no_paid_sweep_artifact_path),
            "active_gate_artifact_path": str(active_gate_artifact_path),
            "active_gate_artifact_sha256": _sha256(active_gate_artifact_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "source_artifact_dir": str(source_artifact_dir),
            "pairwise_artifact_dir": str(pairwise_artifact_dir),
        },
        "frozen_policy": frozen,
        "analysis_parameters": {
            "seeds": seed_set,
            "seed_count": len(seed_set),
            "posterior_samples": posterior_samples,
            "pairwise_strength": pairwise_strength,
            "pairwise_model": str(phase_config["pairwise_model"]),
            "primary_metric": "recall_at_k",
            "secondary_metrics": ["ndcg_at_k", "average_precision"],
        },
        "label_policy": {
            "future_labels_used_for_scheduling": False,
            "future_labels_used_as_model_features": False,
            "future_labels_used_for_retrospective_evaluation_only": True,
            "cached_label_values_used_before_scheduling": False,
        },
        "pairwise_cache_stats_by_bucket": cache_stats_by_bucket,
        "completeness": {
            **dict(completeness),
            "pointwise_error": pointwise_error,
        },
        "aggregate_metrics": aggregate_metrics,
        "paired_deltas_vs_exact_pool_random": paired_deltas,
        "bucket_results": bucket_results,
        "fresh_validation_claim": {
            "complete": complete,
            "can_claim_fresh_paid_validation": complete,
            "primary_metric": "recall_at_k",
            "do_not_claim_success_if_incomplete": not complete,
        },
        "limitations": _limitations(complete=complete),
        "validation_commands": [
            "uv run python scripts/analyze_coverage_floor_fresh_validation.py",
            "uv run pytest tests/test_coverage_floor_fresh_validation_analysis.py",
            "git diff --check",
        ],
        "output_path": str(output_path),
    }
    validate_fresh_validation_analysis(payload)
    write_json_artifact(output_path, payload)
    return {**payload, "artifact_path": str(output_path)}


def validate_fresh_validation_analysis(payload: Mapping[str, Any]) -> None:
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("fresh validation analysis has unexpected artifact_type")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("fresh validation analysis has unexpected schema_version")
    if payload.get("paid_calls_made") != 0 or payload.get("pointwise_calls_made") != 0:
        raise ValueError("fresh validation analysis must not make paid calls")
    claim = payload.get("fresh_validation_claim")
    if not isinstance(claim, Mapping):
        raise ValueError("fresh validation analysis missing claim")
    completeness = payload.get("completeness")
    if not isinstance(completeness, Mapping):
        raise ValueError("fresh validation analysis missing completeness")
    if claim.get("can_claim_fresh_paid_validation") is True and int(
        completeness.get("missing_pairwise_occurrences") or 0
    ) != 0:
        raise ValueError("fresh validation cannot be claimed with missing pairwise")


def _comparisons_for_schedule(
    schedule: Sequence[Any],
    cached: Mapping[tuple[str, str], Any],
) -> tuple[list[Any], int]:
    comparisons = []
    missing = 0
    for pair in schedule:
        key = canonical_pair_key(pair.left_id, pair.right_id)
        if key not in cached:
            missing += 1
            continue
        comparisons.append(_orient_cached_comparison(cached[key], pair))
    return comparisons, missing


def _arm_metrics(
    papers: list[Any],
    *,
    relevant_ids: set[str],
    k: int,
    schedule: Sequence[Any],
    comparisons: Sequence[Any],
    missing_pairwise_labels: int,
    scheduler_diagnostics: Mapping[str, Any],
    seed: int,
    posterior_samples: int,
    pairwise_strength: float,
) -> dict[str, Any]:
    predictions, posterior = posterior_top_k_predictions(
        papers,
        comparisons,
        k=k,
        pairwise_strength=pairwise_strength,
        samples=posterior_samples,
        seed=seed,
    )
    metrics = compare_strategies(
        {POSTERIOR_STRATEGY: predictions},
        relevant_ids=relevant_ids,
        k=k,
    )
    return {
        "selected_strategy": POSTERIOR_STRATEGY,
        "metrics": {
            POSTERIOR_STRATEGY: metrics[POSTERIOR_STRATEGY].to_dict(),
        },
        "comparison_source": {
            "scheduled_pairwise_total": len(schedule),
            "pairwise_labels_available": len(comparisons),
            "missing_pairwise_labels": missing_pairwise_labels,
            "partial": missing_pairwise_labels != 0,
        },
        "posterior_topk_diagnostics": posterior.diagnostics,
        "scheduler_diagnostics": dict(scheduler_diagnostics),
    }


def _aggregate_metrics(
    bucket_results: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, float | int]]]]:
    rows_by_arm: dict[str, list[dict[str, float | int]]] = {
        ARM_COVERAGE: [],
        ARM_EXACT: [],
    }
    seed_rows: dict[str, dict[str, list[dict[str, float | int]]]] = {}
    seed_metric_rows: dict[str, dict[str, dict[str, float | int]]] = {}
    for seed_payload in bucket_results:
        seed = str(seed_payload["seed"])
        seed_rows.setdefault(seed, {ARM_COVERAGE: [], ARM_EXACT: []})
        for bucket in seed_payload.get("buckets") or []:
            for arm in (ARM_COVERAGE, ARM_EXACT):
                selected = bucket["arms"][arm]["selected_strategy"]
                row = bucket["arms"][arm]["metrics"][selected]
                rows_by_arm[arm].append(row)
                seed_rows[seed][arm].append(row)
        seed_metric_rows[seed] = {
            arm: _mean_metric_rows(seed_rows[seed][arm])
            for arm in (ARM_COVERAGE, ARM_EXACT)
        }
    aggregate = {}
    for arm in (ARM_COVERAGE, ARM_EXACT):
        arm_seed_rows = {
            seed: seed_metric_rows[seed][arm]
            for seed in sorted(seed_metric_rows, key=int)
        }
        aggregate[arm] = {
            **_mean_metric_rows(rows_by_arm[arm]),
            "seed_count": len(arm_seed_rows),
            "selected_strategy": POSTERIOR_STRATEGY,
            "seed_metric_rows": arm_seed_rows,
            "seed_level_intervals": {
                metric: summarize_values(
                    [
                        float(row[metric])
                        for row in arm_seed_rows.values()
                        if metric in row
                    ]
                )
                for metric in METRICS
            },
        }
    return aggregate, seed_metric_rows


def _mean_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {metric: 0.0 for metric in METRICS} | {"bucket_count": 0}
    return {
        metric: round(mean(float(row.get(metric, 0.0)) for row in rows), 8)
        for metric in METRICS
    } | {"bucket_count": len(rows)}


def _limitations(*, complete: bool) -> list[str]:
    base = [
        "Fresh holdout citation labels are used only for retrospective metrics.",
        "Pointwise prompts and pairwise prompts use only model-visible paper text, sanitized metadata, and pointwise assessments.",
        "The analysis makes zero paid calls and reads existing reviewed artifacts only.",
    ]
    if not complete:
        base.append(
            "Do not claim success: pointwise artifacts or pairwise labels are incomplete."
        )
    return base


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stdout_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    deltas = payload["paired_deltas_vs_exact_pool_random"]["metric_deltas"]
    return {
        "artifact_path": payload.get("artifact_path") or payload.get("output_path"),
        "artifact_type": payload["artifact_type"],
        "complete": payload["fresh_validation_claim"]["complete"],
        "can_claim_fresh_paid_validation": payload["fresh_validation_claim"][
            "can_claim_fresh_paid_validation"
        ],
        "mean_recall_delta": deltas["recall_at_k"]["mean"],
        "mean_ndcg_delta": deltas["ndcg_at_k"]["mean"],
        "mean_average_precision_delta": deltas["average_precision"]["mean"],
        "missing_pairwise_occurrences": payload["completeness"].get(
            "missing_pairwise_occurrences",
            0,
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
