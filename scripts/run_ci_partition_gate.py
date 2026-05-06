#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_random_control_gap import (  # noqa: E402
    oracle_bounds,
    pair_graph_diagnostics,
    positive_exposure_diagnostics,
    top_k_error_decomposition,
)
from sestina.backtest import Prediction, compare_strategies  # noqa: E402
from sestina.backtest_budget import load_config  # noqa: E402
from sestina.backtest_runner import (  # noqa: E402
    _config_for_phase,
    load_dataset_manifest,
    validate_model_names,
)
from sestina.candidates import select_candidates  # noqa: E402
from sestina.ci_partition_gate import (  # noqa: E402
    CIPartitionConfig,
    confidence_interval_partition,
    replay_ci_partition_gate,
    schedule_cached_exact_pool_random,
)
from sestina.diagnostics import write_json_artifact  # noqa: E402
from sestina.evsi_scheduler import posterior_top_k_predictions  # noqa: E402
from sestina.models import PairwiseComparison, Paper, ScheduledPair  # noqa: E402
from sestina.scheduler import resolve_pairwise_budget  # noqa: E402
from sestina.scheduler_followup import load_pointwise_papers_from_artifacts  # noqa: E402


DEFAULT_SEEDS = (
    17,
    101,
    211,
    307,
    401,
    503,
    607,
    709,
    811,
    907,
    1009,
    1103,
    1201,
    1301,
    1409,
    1511,
    1601,
    1709,
    1801,
    1901,
)
ARM_CI = "ci_partition_elimination"
ARM_EXACT = "exact_pool_random_cached_replay"
POSTERIOR_STRATEGY = "posterior_topk"
REQUIRED_ARTIFACT_KEYS = {
    "artifact_type",
    "schema_version",
    "paid_calls_made",
    "paid_spend_usd",
    "gate_verdict",
    "gate_criteria",
    "aggregate_metrics",
    "paired_deltas_vs_exact_pool_random",
    "aggregate_diagnostics",
    "bucket_results",
    "limitations",
}
REQUIRED_AGGREGATE_DIAGNOSTICS = {
    "confidence_bound_unresolved_count",
    "graph_connectivity",
    "oracle_caps",
    "unique_future_positives_touched",
    "weak_bucket_deltas",
}


@dataclass(frozen=True, slots=True)
class CachedPairwiseLabel:
    comparison: PairwiseComparison
    artifact_path: Path
    artifact_dir: Path
    kind: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a no-paid confidence-interval top-K partition/elimination replay "
            "gate over cached historical arXiv pairwise labels."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "experiments" / "arxiv_historical_pilot_budget_config.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-datasets"
            / "arxiv-historical-pilot-manifest.json"
        ),
    )
    parser.add_argument(
        "--source-artifact-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "backtest-arxiv-pilot-live",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-ci-partition-gate"
            / "ci-partition-gate-analysis.json"
        ),
    )
    parser.add_argument("--phase", default="pilot")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="comma-separated replay seeds",
    )
    parser.add_argument("--scheduler-samples", type=int, default=800)
    parser.add_argument("--posterior-samples", type=int, default=1200)
    parser.add_argument("--pairwise-strength", type=float, default=2.5)
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument("--random-floor-fraction", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument(
        "--pairwise-cache-artifact-dir",
        action="append",
        type=Path,
        default=None,
        help=(
            "Additional artifact directories to scan for cached pairwise labels. "
            "Defaults to all artifacts/backtest-arxiv-*-live directories."
        ),
    )
    args = parser.parse_args(argv)

    payload = run_ci_partition_gate(
        config_path=args.config,
        manifest_path=args.manifest,
        source_artifact_dir=args.source_artifact_dir,
        output_path=args.output,
        phase=args.phase,
        seeds=_parse_seeds(args.seeds),
        scheduler_samples=args.scheduler_samples,
        posterior_samples=args.posterior_samples,
        pairwise_strength=args.pairwise_strength,
        confidence_z=args.confidence_z,
        random_floor_fraction=args.random_floor_fraction,
        batch_size=args.batch_size,
        pairwise_cache_artifact_dirs=args.pairwise_cache_artifact_dir,
    )
    sys.stdout.write(json.dumps(_stdout_summary(payload), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def run_ci_partition_gate(
    *,
    config_path: Path,
    manifest_path: Path,
    source_artifact_dir: Path,
    output_path: Path,
    phase: str = "pilot",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    scheduler_samples: int = 800,
    posterior_samples: int = 1200,
    pairwise_strength: float = 2.5,
    confidence_z: float = 1.96,
    random_floor_fraction: float = 0.25,
    batch_size: int = 5,
    pairwise_cache_artifact_dirs: Sequence[Path] | None = None,
) -> dict[str, Any]:
    raw_config = load_config(config_path)
    phase_config = _config_for_phase(raw_config, phase=phase)["phases"][0]
    pairwise_model = str(phase_config["pairwise_model"])
    validate_model_names([pairwise_model])
    manifest = load_dataset_manifest(manifest_path)
    buckets = manifest.buckets_for_phase(phase)
    labels_by_bucket = _manifest_label_lookup(manifest.payload)
    cache_dirs = _pairwise_cache_dirs(
        source_artifact_dir,
        phase=phase,
        explicit_dirs=pairwise_cache_artifact_dirs,
    )
    ci_config = CIPartitionConfig(
        confidence_z=confidence_z,
        pairwise_strength=pairwise_strength,
        posterior_samples=scheduler_samples,
        random_floor_fraction=random_floor_fraction,
        batch_size=batch_size,
    )

    bucket_results = []
    cache_stats_by_bucket: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        seed_payload = {"seed": seed, "buckets": []}
        for bucket in buckets:
            papers = load_pointwise_papers_from_artifacts(
                bucket,
                source_artifact_dir=source_artifact_dir,
                phase=phase,
            )
            selection = select_candidates(papers, k=bucket.k)
            budget = resolve_pairwise_budget(
                n=len(papers),
                candidate_size=len(selection.candidate_ids),
            )
            cached, cache_stats = load_cached_pairwise_labels(
                bucket.name,
                artifact_dirs=cache_dirs,
                phase=phase,
            )
            cache_stats_by_bucket.setdefault(bucket.name, cache_stats)
            available = set(cached)
            comparison_map = {
                key: row.comparison for key, row in cached.items()
            }
            pointwise_predictions = [
                Prediction(paper.paper_id, paper.pointwise.good_probability)
                for paper in papers
            ]
            pointwise_top_k_ids = _top_k_ids(pointwise_predictions, k=bucket.k)
            ci_replay = replay_ci_partition_gate(
                papers,
                comparison_map,
                k=bucket.k,
                budget=budget,
                seed=seed,
                config=ci_config,
            )
            exact_schedule = schedule_cached_exact_pool_random(
                papers,
                [],
                k=bucket.k,
                budget=budget,
                seed=seed,
                config=ci_config,
                available_pair_keys=available,
            )
            exact_comparisons = [
                _orient_cached_comparison(cached[_pair_key(pair.left_id, pair.right_id)], pair)
                for pair in exact_schedule.pairs
                if _pair_key(pair.left_id, pair.right_id) in cached
            ]
            bucket_payload = {
                "bucket": bucket.name,
                "seed": seed,
                "k": bucket.k,
                "papers_total": len(papers),
                "positive_labels_total": len(bucket.relevant_ids),
                "budget": budget.to_dict(),
                "pointwise_metrics": compare_strategies(
                    {"pointwise_only": pointwise_predictions},
                    relevant_ids=bucket.relevant_ids,
                    k=bucket.k,
                )["pointwise_only"].to_dict(),
                "pointwise_top_k_ids": pointwise_top_k_ids,
                "pairwise_cache": cache_stats,
                "arms": {
                    ARM_CI: _arm_result_payload(
                        papers,
                        relevant_ids=bucket.relevant_ids,
                        k=bucket.k,
                        schedule=ci_replay.schedule,
                        comparisons=ci_replay.comparisons,
                        pointwise_predictions=pointwise_predictions,
                        pointwise_top_k_ids=pointwise_top_k_ids,
                        labels_by_id=labels_by_bucket.get(bucket.name, {}),
                        posterior_samples=posterior_samples,
                        pairwise_strength=pairwise_strength,
                        interval_config=ci_config,
                        seed=seed,
                        comparison_source={
                            "source": "cached_pairwise_replay",
                            "scheduled_pairwise_total": len(ci_replay.schedule),
                            "cached_pairwise_labels_available": len(
                                ci_replay.comparisons
                            ),
                            "missing_pairwise_labels": (
                                len(ci_replay.schedule) - len(ci_replay.comparisons)
                            ),
                            "partial": len(ci_replay.schedule)
                            != len(ci_replay.comparisons),
                        },
                        scheduler_diagnostics=ci_replay.diagnostics,
                    ),
                    ARM_EXACT: _arm_result_payload(
                        papers,
                        relevant_ids=bucket.relevant_ids,
                        k=bucket.k,
                        schedule=exact_schedule.pairs,
                        comparisons=exact_comparisons,
                        pointwise_predictions=pointwise_predictions,
                        pointwise_top_k_ids=pointwise_top_k_ids,
                        labels_by_id=labels_by_bucket.get(bucket.name, {}),
                        posterior_samples=posterior_samples,
                        pairwise_strength=pairwise_strength,
                        interval_config=ci_config,
                        seed=seed,
                        comparison_source={
                            "source": "cached_exact_pool_random_replay",
                            "scheduled_pairwise_total": len(exact_schedule.pairs),
                            "cached_pairwise_labels_available": len(
                                exact_comparisons
                            ),
                            "missing_pairwise_labels": (
                                len(exact_schedule.pairs) - len(exact_comparisons)
                            ),
                            "partial": len(exact_schedule.pairs)
                            != len(exact_comparisons),
                        },
                        scheduler_diagnostics={
                            **exact_schedule.diagnostics,
                            "final_ci_partition": confidence_interval_partition(
                                papers,
                                exact_comparisons,
                                k=bucket.k,
                                config=ci_config,
                            ).to_dict(),
                        },
                    ),
                },
            }
            seed_payload["buckets"].append(bucket_payload)
        bucket_results.append(seed_payload)

    aggregate_metrics = _aggregate_metrics(bucket_results)
    paired_deltas = _paired_deltas(bucket_results)
    aggregate_diagnostics = _aggregate_diagnostics(bucket_results)
    gate_criteria = _gate_criteria(random_floor_fraction=random_floor_fraction)
    gate_verdict = _gate_verdict(
        paired_deltas=paired_deltas,
        aggregate_diagnostics=aggregate_diagnostics,
        criteria=gate_criteria,
    )
    payload = {
        "artifact_type": "sestina-ci-partition-gate-analysis",
        "schema_version": 1,
        "phase": phase,
        "manifest_path": str(manifest_path),
        "source_artifact_dir": str(source_artifact_dir),
        "output_path": str(output_path),
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "spend_policy": (
            "no-paid offline replay over existing pointwise and cached pairwise "
            "artifacts; no LLM calls are made"
        ),
        "known_paid_spend_before_workflow_usd": 1.476685,
        "pairwise_model_validated_from_config": pairwise_model,
        "label_policy": {
            "future_labels_used_as_model_features": False,
            "future_labels_used_for_scheduling": False,
            "future_labels_used_for_retrospective_diagnostics_only": True,
            "pairwise_labels_used_for_replay": (
                "cached historical/follow-up pairwise labels only"
            ),
            "pointwise_paid_calls_made": 0,
        },
        "analysis_parameters": {
            "seeds": list(seeds),
            "seed_count": len(seeds),
            "scheduler_samples": scheduler_samples,
            "posterior_samples": posterior_samples,
            "pairwise_strength": pairwise_strength,
            "confidence_z": confidence_z,
            "random_floor_fraction": random_floor_fraction,
            "batch_size": batch_size,
            "comparison_pool_note": (
                "Both arms are restricted to cached labels from the exact EVSI "
                "feasible proposal pool. CI replay updates that pool after cached "
                "labels are revealed; exact-pool random is the one-shot cached-pool "
                "control."
            ),
        },
        "pairwise_cache_artifact_dirs": [str(path) for path in cache_dirs],
        "pairwise_cache_stats_by_bucket": cache_stats_by_bucket,
        "arms": [
            {
                "name": ARM_CI,
                "method": "confidence_interval_top_k_partition_elimination",
                "randomized_coverage_floor": True,
            },
            {
                "name": ARM_EXACT,
                "method": "cached_exact_pool_random",
                "randomized_coverage_floor": True,
            },
        ],
        "gate_criteria": gate_criteria,
        "gate_verdict": gate_verdict,
        "aggregate_metrics": aggregate_metrics,
        "paired_deltas_vs_exact_pool_random": paired_deltas,
        "aggregate_diagnostics": aggregate_diagnostics,
        "bucket_results": bucket_results,
        "limitations": [
            "This is an offline cached-label replay, not a fresh paid acquisition run.",
            "The exact comparison pool is approximated by cached labels intersected with the exact EVSI feasible proposal pool.",
            "CI replay is adaptive within cached labels; exact-pool random is a one-shot cached-pool control.",
            "Retrospective citation labels are used only for evaluation, weak-bucket deltas, and oracle caps.",
            "The pilot remains 8 historical arXiv buckets; paired seed variance is still material.",
        ],
        "recommended_next_action": gate_verdict["recommended_next_action"],
    }
    validate_ci_gate_artifact_schema(payload)
    write_json_artifact(output_path, payload)
    return {**payload, "artifact_path": str(output_path)}


def validate_ci_gate_artifact_schema(payload: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_ARTIFACT_KEYS - set(payload))
    if missing:
        raise ValueError(f"CI gate artifact missing top-level keys: {missing}")
    if payload.get("artifact_type") != "sestina-ci-partition-gate-analysis":
        raise ValueError("CI gate artifact has unexpected artifact_type")
    diagnostics = payload.get("aggregate_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("CI gate artifact aggregate_diagnostics must be an object")
    missing_diagnostics = sorted(REQUIRED_AGGREGATE_DIAGNOSTICS - set(diagnostics))
    if missing_diagnostics:
        raise ValueError(
            "CI gate artifact missing aggregate diagnostics: "
            + ", ".join(missing_diagnostics)
        )


def load_cached_pairwise_labels(
    bucket_name: str,
    *,
    artifact_dirs: Sequence[Path],
    phase: str,
) -> tuple[dict[tuple[str, str], CachedPairwiseLabel], dict[str, Any]]:
    cached: dict[tuple[str, str], CachedPairwiseLabel] = {}
    stats = Counter(
        {
            "artifact_dirs_scanned": 0,
            "call_artifacts_seen": 0,
            "successful_pairwise_artifacts": 0,
            "duplicate_pair_keys": 0,
            "malformed_pairwise_artifacts": 0,
        }
    )
    by_kind: Counter[str] = Counter()
    by_dir: Counter[str] = Counter()
    for artifact_dir in artifact_dirs:
        calls_dir = artifact_dir / phase / bucket_name / "calls"
        if not calls_dir.exists():
            continue
        stats["artifact_dirs_scanned"] += 1
        for path in sorted(calls_dir.glob("*pairwise*.json")):
            stats["call_artifacts_seen"] += 1
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                stats["malformed_pairwise_artifacts"] += 1
                continue
            if payload.get("status") != "ok":
                continue
            comparison = _comparison_from_artifact(payload)
            if comparison is None:
                stats["malformed_pairwise_artifacts"] += 1
                continue
            key = _pair_key(comparison.left_id, comparison.right_id)
            stats["successful_pairwise_artifacts"] += 1
            kind = str(payload.get("kind") or "pairwise_unknown")
            by_kind[kind] += 1
            by_dir[str(artifact_dir)] += 1
            if key in cached:
                stats["duplicate_pair_keys"] += 1
                continue
            cached[key] = CachedPairwiseLabel(
                comparison=comparison,
                artifact_path=path,
                artifact_dir=artifact_dir,
                kind=kind,
            )
    return cached, {
        **dict(stats),
        "successful_pairwise_by_kind": dict(sorted(by_kind.items())),
        "successful_pairwise_by_artifact_dir": dict(sorted(by_dir.items())),
        "unique_cached_pair_keys": len(cached),
    }


def _arm_result_payload(
    papers: list[Paper],
    *,
    relevant_ids: set[str],
    k: int,
    schedule: list[ScheduledPair],
    comparisons: list[PairwiseComparison],
    pointwise_predictions: list[Prediction],
    pointwise_top_k_ids: list[str],
    labels_by_id: dict[str, dict[str, Any]],
    posterior_samples: int,
    pairwise_strength: float,
    interval_config: CIPartitionConfig,
    seed: int,
    comparison_source: dict[str, Any],
    scheduler_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    posterior_predictions, posterior = posterior_top_k_predictions(
        papers,
        comparisons,
        k=k,
        pairwise_strength=pairwise_strength,
        samples=posterior_samples,
        seed=seed,
    )
    metrics = compare_strategies(
        {
            "pointwise_only": pointwise_predictions,
            POSTERIOR_STRATEGY: posterior_predictions,
        },
        relevant_ids=relevant_ids,
        k=k,
    )
    posterior_top_k_ids = _top_k_ids(posterior_predictions, k=k)
    ci_state = confidence_interval_partition(
        papers,
        comparisons,
        k=k,
        config=interval_config,
    )
    return {
        "comparison_source": comparison_source,
        "metrics": {name: metric.to_dict() for name, metric in metrics.items()},
        "positive_exposure": positive_exposure_diagnostics(
            schedule,
            relevant_ids=relevant_ids,
            paper_count=len(papers),
        ),
        "pair_graph": pair_graph_diagnostics(
            schedule,
            relevant_ids=relevant_ids,
            posterior_top_k_ids=posterior_top_k_ids,
            pointwise_top_k_ids=pointwise_top_k_ids,
        ),
        "oracle_bounds": oracle_bounds(
            k=k,
            relevant_ids=relevant_ids,
            pointwise_top_k_ids=pointwise_top_k_ids,
            schedule=schedule,
            comparisons=comparisons,
        ),
        "top_k_error_decomposition": top_k_error_decomposition(
            predictions=posterior_predictions,
            relevant_ids=relevant_ids,
            k=k,
            schedule=schedule,
            pointwise_top_k_ids=pointwise_top_k_ids,
            labels_by_id=labels_by_id,
        ),
        "confidence_partition": ci_state.to_dict(),
        "confidence_bound_unresolved_count": ci_state.unresolved_count,
        "posterior_topk_diagnostics": posterior.diagnostics,
        "scheduler_diagnostics": scheduler_diagnostics,
    }


def _aggregate_metrics(bucket_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows_by_arm: dict[str, list[dict[str, float | int]]] = {
        ARM_CI: [],
        ARM_EXACT: [],
    }
    seed_rows: dict[str, dict[int, list[dict[str, float | int]]]] = {
        ARM_CI: {},
        ARM_EXACT: {},
    }
    for seed_payload in bucket_results:
        seed = int(seed_payload["seed"])
        for bucket in seed_payload["buckets"]:
            for arm in (ARM_CI, ARM_EXACT):
                row = bucket["arms"][arm]["metrics"][POSTERIOR_STRATEGY]
                rows_by_arm[arm].append(row)
                seed_rows[arm].setdefault(seed, []).append(row)
    return {
        arm: {
            **_mean_metric_rows(rows),
            "seed_count": len(seed_rows[arm]),
            "seed_metric_rows": {
                str(seed): _mean_metric_rows(rows)
                for seed, rows in sorted(seed_rows[arm].items())
            },
        }
        for arm, rows in rows_by_arm.items()
    }


def _paired_deltas(bucket_results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("recall_at_k", "ndcg_at_k", "average_precision")
    seed_deltas: dict[int, dict[str, float]] = {}
    bucket_deltas = []
    for seed_payload in bucket_results:
        seed = int(seed_payload["seed"])
        per_metric_totals = Counter()
        bucket_count = 0
        for bucket in seed_payload["buckets"]:
            ci = bucket["arms"][ARM_CI]["metrics"][POSTERIOR_STRATEGY]
            exact = bucket["arms"][ARM_EXACT]["metrics"][POSTERIOR_STRATEGY]
            bucket_count += 1
            row = {"bucket": bucket["bucket"], "seed": seed}
            for metric in metrics:
                delta = round(float(ci[metric]) - float(exact[metric]), 8)
                per_metric_totals[metric] += delta
                row[f"{metric}_delta"] = delta
            ci_hits = int(
                bucket["arms"][ARM_CI]["top_k_error_decomposition"][
                    "selected_positive_count"
                ]
            )
            exact_hits = int(
                bucket["arms"][ARM_EXACT]["top_k_error_decomposition"][
                    "selected_positive_count"
                ]
            )
            row["selected_positive_delta"] = ci_hits - exact_hits
            bucket_deltas.append(row)
        seed_deltas[seed] = {
            metric: round(per_metric_totals[metric] / bucket_count, 8)
            if bucket_count
            else 0.0
            for metric in metrics
        }
    return {
        "reference_arm": ARM_EXACT,
        "comparison_arm": ARM_CI,
        "metric_deltas": {
            metric: _summary(
                [rows[metric] for rows in seed_deltas.values()]
            )
            for metric in metrics
        },
        "seed_deltas": {str(seed): rows for seed, rows in sorted(seed_deltas.items())},
        "bucket_deltas": bucket_deltas,
        "selected_positive_total_delta": sum(
            int(row["selected_positive_delta"]) for row in bucket_deltas
        ),
    }


def _aggregate_diagnostics(bucket_results: list[dict[str, Any]]) -> dict[str, Any]:
    exposure_rows = {ARM_CI: [], ARM_EXACT: []}
    graph_rows = {ARM_CI: [], ARM_EXACT: []}
    oracle_rows = {ARM_CI: [], ARM_EXACT: []}
    unresolved_rows = {ARM_CI: [], ARM_EXACT: []}
    coverage_rows = {ARM_CI: [], ARM_EXACT: []}
    for seed_payload in bucket_results:
        for bucket in seed_payload["buckets"]:
            for arm in (ARM_CI, ARM_EXACT):
                arm_payload = bucket["arms"][arm]
                exposure_rows[arm].append(arm_payload["positive_exposure"])
                graph_rows[arm].append(arm_payload["pair_graph"])
                oracle_rows[arm].append(arm_payload["oracle_bounds"])
                unresolved_rows[arm].append(
                    int(arm_payload["confidence_bound_unresolved_count"])
                )
                coverage_rows[arm].append(
                    arm_payload["scheduler_diagnostics"].get("coverage", {})
                )
    return {
        "confidence_bound_unresolved_count": {
            arm: _summary(unresolved_rows[arm]) for arm in unresolved_rows
        },
        "unique_future_positives_touched": {
            arm: {
                "total": sum(
                    int(row["unique_future_positives_touched"])
                    for row in exposure_rows[arm]
                ),
                "mean_touch_rate": _mean(
                    [
                        float(row["unique_future_positive_touch_rate"])
                        for row in exposure_rows[arm]
                    ]
                ),
            }
            for arm in exposure_rows
        },
        "graph_connectivity": {
            arm: {
                "mean_largest_component_size": _mean(
                    [int(row["largest_component_size"]) for row in graph_rows[arm]]
                ),
                "mean_component_count": _mean(
                    [int(row["component_count"]) for row in graph_rows[arm]]
                ),
                "mean_future_positive_degree": _mean(
                    [
                        float(row["future_positive_degree"]["mean"])
                        for row in graph_rows[arm]
                    ]
                ),
            }
            for arm in graph_rows
        },
        "oracle_caps": {
            arm: {
                "mean_pointwise_plus_touched_recall_cap": _mean(
                    [
                        float(
                            row["pointwise_plus_touched_positive_upper_bound"][
                                "recall_cap"
                            ]
                        )
                        for row in oracle_rows[arm]
                    ]
                ),
                "mean_positive_negative_pair_recall_cap": _mean(
                    [
                        float(
                            row["positive_negative_pair_label_oracle_upper_bound"][
                                "recall_cap"
                            ]
                        )
                        for row in oracle_rows[arm]
                    ]
                ),
                "mean_observed_positive_winner_recall_cap": _mean(
                    [
                        float(row["observed_positive_winner_upper_bound"]["recall_cap"])
                        for row in oracle_rows[arm]
                    ]
                ),
            }
            for arm in oracle_rows
        },
        "randomized_coverage": {
            arm: {
                "random_floor_pairs": sum(
                    int(row.get("random_floor_pairs", 0)) for row in coverage_rows[arm]
                ),
                "random_floor_rate": _mean(
                    [float(row.get("random_floor_rate", 0.0)) for row in coverage_rows[arm]]
                ),
            }
            for arm in coverage_rows
        },
        "weak_bucket_deltas": _weak_bucket_deltas(bucket_results),
    }


def _weak_bucket_deltas(bucket_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for seed_payload in bucket_results:
        for bucket in seed_payload["buckets"]:
            exact = bucket["arms"][ARM_EXACT]
            ci = bucket["arms"][ARM_CI]
            exact_hits = int(
                exact["top_k_error_decomposition"]["selected_positive_count"]
            )
            if exact_hits >= int(bucket["k"]):
                continue
            ci_hits = int(ci["top_k_error_decomposition"]["selected_positive_count"])
            exact_exposure = exact["positive_exposure"]
            ci_exposure = ci["positive_exposure"]
            exact_oracle = exact["oracle_bounds"]
            ci_oracle = ci["oracle_bounds"]
            rows.append(
                {
                    "seed": int(seed_payload["seed"]),
                    "bucket": bucket["bucket"],
                    "exact_selected_positive_count": exact_hits,
                    "ci_selected_positive_count": ci_hits,
                    "selected_positive_delta": ci_hits - exact_hits,
                    "unique_future_positives_touched_delta": int(
                        ci_exposure["unique_future_positives_touched"]
                    )
                    - int(exact_exposure["unique_future_positives_touched"]),
                    "pointwise_plus_touched_recall_cap_delta": round(
                        float(
                            ci_oracle["pointwise_plus_touched_positive_upper_bound"][
                                "recall_cap"
                            ]
                        )
                        - float(
                            exact_oracle[
                                "pointwise_plus_touched_positive_upper_bound"
                            ]["recall_cap"]
                        ),
                        8,
                    ),
                    "positive_negative_pair_recall_cap_delta": round(
                        float(
                            ci_oracle[
                                "positive_negative_pair_label_oracle_upper_bound"
                            ]["recall_cap"]
                        )
                        - float(
                            exact_oracle[
                                "positive_negative_pair_label_oracle_upper_bound"
                            ]["recall_cap"]
                        ),
                        8,
                    ),
                }
            )
    return {
        "definition": (
            "Weak buckets are seed/bucket rows where cached exact-pool random "
            "posterior top-K selected fewer than K future positives."
        ),
        "row_count": len(rows),
        "selected_positive_delta_total": sum(
            int(row["selected_positive_delta"]) for row in rows
        ),
        "unique_future_positives_touched_delta_total": sum(
            int(row["unique_future_positives_touched_delta"]) for row in rows
        ),
        "mean_pointwise_plus_touched_recall_cap_delta": _mean(
            [float(row["pointwise_plus_touched_recall_cap_delta"]) for row in rows]
        ),
        "mean_positive_negative_pair_recall_cap_delta": _mean(
            [float(row["positive_negative_pair_recall_cap_delta"]) for row in rows]
        ),
        "rows": rows,
    }


def _gate_criteria(*, random_floor_fraction: float) -> dict[str, Any]:
    return {
        "coverage_floor_required": True,
        "minimum_random_floor_rate": min(0.15, max(0.0, random_floor_fraction)),
        "credible_metric_improvement": (
            "mean recall delta >= 0.025, mean nDCG delta >= 0, and mean AP "
            "delta >= -0.01 versus cached exact-pool random"
        ),
        "strong_weak_bucket_oracle_headroom": (
            "weak-bucket mean pointwise-plus-touched or positive-negative-pair "
            "oracle recall-cap delta >= 0.025 with nonnegative selected-positive delta"
        ),
        "no_paid_followup_if_cost_accounting_uncertain": True,
    }


def _gate_verdict(
    *,
    paired_deltas: dict[str, Any],
    aggregate_diagnostics: dict[str, Any],
    criteria: dict[str, Any],
) -> dict[str, Any]:
    metric_deltas = paired_deltas["metric_deltas"]
    recall_delta = float(metric_deltas["recall_at_k"]["mean"])
    ndcg_delta = float(metric_deltas["ndcg_at_k"]["mean"])
    ap_delta = float(metric_deltas["average_precision"]["mean"])
    weak = aggregate_diagnostics["weak_bucket_deltas"]
    coverage = aggregate_diagnostics["randomized_coverage"][ARM_CI]
    random_floor_rate = float(coverage["random_floor_rate"])
    coverage_preserved = random_floor_rate >= float(
        criteria["minimum_random_floor_rate"]
    )
    credible_metric = (
        recall_delta >= 0.025 and ndcg_delta >= 0.0 and ap_delta >= -0.01
    )
    strong_oracle = (
        int(weak["selected_positive_delta_total"]) >= 0
        and (
            float(weak["mean_pointwise_plus_touched_recall_cap_delta"]) >= 0.025
            or float(weak["mean_positive_negative_pair_recall_cap_delta"]) >= 0.025
        )
    )
    paid_allowed = bool(coverage_preserved and (credible_metric or strong_oracle))
    reasons = []
    if not coverage_preserved:
        reasons.append("randomized coverage floor was not preserved")
    if not credible_metric:
        reasons.append("posterior top-K metrics did not clear the improvement gate")
    if not strong_oracle:
        reasons.append("weak-bucket oracle headroom did not clear the fallback gate")
    if paid_allowed:
        recommendation = (
            "A guarded paid pairwise-only follow-up may be considered, but must "
            "start with a dry-run estimate, provider-prefixed model availability "
            "check, JSONL ledger, separate artifact directory, and --max-usd <= 10."
        )
    else:
        recommendation = (
            "Do not spend on a paid CI partition arm from this gate; keep exact-pool "
            "or historical random plus posterior top-K as the small-budget baseline."
        )
    return {
        "paid_followup_allowed": paid_allowed,
        "coverage_preserved": coverage_preserved,
        "credible_metric_improvement": credible_metric,
        "strong_weak_bucket_oracle_headroom": strong_oracle,
        "mean_recall_delta": recall_delta,
        "mean_ndcg_delta": ndcg_delta,
        "mean_average_precision_delta": ap_delta,
        "random_floor_rate": random_floor_rate,
        "blocking_reasons": reasons,
        "recommended_next_action": recommendation,
    }


def _comparison_from_artifact(payload: dict[str, Any]) -> PairwiseComparison | None:
    comparison_payload = payload.get("comparison")
    if isinstance(comparison_payload, dict):
        try:
            return PairwiseComparison.from_dict(comparison_payload)
        except ValueError:
            return None
    subject = payload.get("subject") or {}
    response = payload.get("response") or {}
    left_id = subject.get("left_id")
    right_id = subject.get("right_id")
    if left_id is None or right_id is None or not isinstance(response, dict):
        return None
    try:
        return PairwiseComparison.from_dict(
            {
                "left_id": left_id,
                "right_id": right_id,
                "winner": response.get("winner", "uncertain"),
                "soft_probability": response.get("soft_probability"),
                "confidence": response.get("confidence", 0.5),
                "reasons": response.get("reasons", []),
            }
        )
    except ValueError:
        return None


def _orient_cached_comparison(
    cached: CachedPairwiseLabel,
    pair: ScheduledPair,
) -> PairwiseComparison:
    comparison = cached.comparison
    if comparison.left_id == pair.left_id and comparison.right_id == pair.right_id:
        winner = comparison.winner
    elif comparison.left_id == pair.right_id and comparison.right_id == pair.left_id:
        winner = _invert_winner(comparison.winner)
    else:
        raise ValueError("cached comparison does not reference scheduled pair")
    return PairwiseComparison(
        left_id=pair.left_id,
        right_id=pair.right_id,
        winner=winner,  # type: ignore[arg-type]
        soft_probability=comparison.soft_probability,
        confidence=comparison.confidence,
        reasons=list(comparison.reasons),
        order=pair.order,
        metadata={
            **comparison.metadata,
            "cached_artifact_path": str(cached.artifact_path),
            "cached_artifact_kind": cached.kind,
            "scheduled_pair_purpose": pair.purpose,
        },
    )


def _pairwise_cache_dirs(
    source_artifact_dir: Path,
    *,
    phase: str,
    explicit_dirs: Sequence[Path] | None,
) -> list[Path]:
    if explicit_dirs:
        dirs = [source_artifact_dir, *explicit_dirs]
    else:
        dirs = [
            path
            for path in sorted((REPO_ROOT / "artifacts").glob("backtest-arxiv-*-live"))
            if (path / phase).exists()
        ]
        if source_artifact_dir not in dirs:
            dirs.insert(0, source_artifact_dir)
    deduped = []
    seen = set()
    for path in dirs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _manifest_label_lookup(payload: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    lookup: dict[str, dict[str, dict[str, Any]]] = {}
    for bucket in payload.get("buckets", []):
        bucket_name = str(bucket.get("name"))
        rows: dict[str, dict[str, Any]] = {}
        for paper in bucket.get("papers", []):
            paper_id = str(paper.get("paper_id") or paper.get("id") or "")
            labels = paper.get("labels") or {}
            rows[paper_id] = {
                "title": paper.get("title"),
                "citation_count": labels.get("citation_count"),
                "citation_rank": labels.get("citation_rank"),
                "citation_percentile_within_bucket": labels.get(
                    "citation_percentile_within_bucket"
                ),
                "citation_positive": labels.get("citation_positive"),
            }
        lookup[bucket_name] = rows
    return lookup


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _top_k_ids(predictions: Sequence[Prediction], *, k: int) -> list[str]:
    return [
        prediction.paper_id
        for prediction in sorted(
            predictions,
            key=lambda item: (item.score, item.paper_id),
            reverse=True,
        )[:k]
    ]


def _mean_metric_rows(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        return {
            "bucket_count": 0,
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "average_precision": 0.0,
        }
    keys = [
        key
        for key in ("recall_at_k", "precision_at_k", "ndcg_at_k", "average_precision")
        if key in rows[0]
    ]
    return {
        "bucket_count": len(rows),
        **{key: _mean([float(row[key]) for row in rows]) for key in keys},
    }


def _summary(values: Sequence[int | float]) -> dict[str, float | int]:
    items = [float(value) for value in values]
    if not items:
        return {"count": 0, "mean": 0.0, "stddev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(items),
        "mean": round(mean(items), 8),
        "stddev": round(pstdev(items), 8) if len(items) > 1 else 0.0,
        "min": round(min(items), 8),
        "max": round(max(items), 8),
    }


def _mean(values: Sequence[int | float]) -> float:
    items = [float(value) for value in values]
    return round(sum(items) / len(items), 8) if items else 0.0


def _pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    return tuple(sorted((left_id, right_id)))


def _invert_winner(winner: str) -> str:
    if winner == "left":
        return "right"
    if winner == "right":
        return "left"
    return winner


def _stdout_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_type": payload["artifact_type"],
        "artifact_path": payload["artifact_path"],
        "paid_calls_made": payload["paid_calls_made"],
        "paid_spend_usd": payload["paid_spend_usd"],
        "gate_verdict": payload["gate_verdict"],
        "aggregate_metrics": payload["aggregate_metrics"],
        "paired_deltas_vs_exact_pool_random": payload[
            "paired_deltas_vs_exact_pool_random"
        ],
        "aggregate_diagnostics": payload["aggregate_diagnostics"],
        "limitations": payload["limitations"],
        "recommended_next_action": payload["recommended_next_action"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
