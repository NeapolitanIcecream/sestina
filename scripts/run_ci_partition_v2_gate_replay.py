#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ci_partition_gate import (  # noqa: E402
    DEFAULT_SEEDS,
    POSTERIOR_STRATEGY,
    _arm_result_payload,
    _config_for_phase,
    _manifest_label_lookup,
    _orient_cached_comparison,
    _pair_key,
    _pairwise_cache_dirs,
    _parse_seeds,
    _top_k_ids,
    load_cached_pairwise_labels,
)
from sestina.active_arm_gate import (  # noqa: E402
    CURRENT_KNOWN_SPEND_USD,
    DEFAULT_PAID_CAP_USD,
    build_active_arm_gate,
    validate_active_arm_gate_artifact_schema,
)
from sestina.backtest import Prediction, compare_strategies  # noqa: E402
from sestina.backtest_budget import load_config  # noqa: E402
from sestina.backtest_runner import (  # noqa: E402
    load_dataset_manifest,
    validate_model_names,
)
from sestina.candidates import select_candidates  # noqa: E402
from sestina.ci_partition_gate import (  # noqa: E402
    ReliabilityAwareCIPartitionV2Config,
    confidence_interval_partition,
    replay_reliability_aware_ci_partition_v2_gate,
    schedule_cached_exact_pool_random,
)
from sestina.diagnostics import write_json_artifact  # noqa: E402
from sestina.scheduler import resolve_pairwise_budget  # noqa: E402
from sestina.scheduler_followup import load_pointwise_papers_from_artifacts  # noqa: E402


ARTIFACT_TYPE = "sestina-ci-partition-v2-gate-replay"
SCHEMA_VERSION = 1
ARM_V2 = "reliability_aware_ci_partition_v2_cached_replay"
ARM_EXACT = "exact_pool_random_cached_replay"
REQUIRED_TOP_LEVEL_KEYS = {
    "artifact_type",
    "schema_version",
    "paid_calls_made",
    "paid_spend_usd",
    "active_arm_name",
    "candidate_random_control_baseline",
    "gate_verdict",
    "aggregate_metrics",
    "paired_deltas_vs_exact_pool_random",
    "seed_level_metric_intervals",
    "aggregate_diagnostics",
    "bucket_results",
    "limitations",
    "active_arm_gate",
}
REQUIRED_AGGREGATE_DIAGNOSTICS = {
    "confidence_bound_unresolved_count",
    "graph_connectivity",
    "oracle_caps",
    "unique_future_positives_touched",
    "weak_bucket_deltas",
    "ci_v2_reliability",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a no-paid reliability-aware CI partition v2 cached replay and "
            "evaluate it with the active-arm gate harness."
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
        "--random-variance-artifact",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-full-random-variance-completion"
            / "full-random-variance-completion.json"
        ),
    )
    parser.add_argument(
        "--original-ci-artifact",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-ci-partition-gate"
            / "ci-partition-gate-analysis.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-ci-partition-v2-gate-replay"
            / "ci-partition-v2-gate-replay.json"
        ),
    )
    parser.add_argument(
        "--active-gate-output",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-ci-partition-v2-gate-replay"
            / "active-arm-gate.json"
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
    parser.add_argument("--min-cached-incident-support", type=int, default=4)
    parser.add_argument("--min-effective-pairwise-n", type=float, default=1.25)
    parser.add_argument("--reliable-pair-threshold", type=float, default=0.55)
    parser.add_argument(
        "--low-reliability-random-floor-fraction",
        type=float,
        default=0.5,
    )
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

    payload = run_ci_partition_v2_gate_replay(
        config_path=args.config,
        manifest_path=args.manifest,
        source_artifact_dir=args.source_artifact_dir,
        random_variance_artifact_path=args.random_variance_artifact,
        original_ci_artifact_path=args.original_ci_artifact,
        output_path=args.output,
        active_gate_output_path=args.active_gate_output,
        phase=args.phase,
        seeds=_parse_seeds(args.seeds),
        scheduler_samples=args.scheduler_samples,
        posterior_samples=args.posterior_samples,
        pairwise_strength=args.pairwise_strength,
        confidence_z=args.confidence_z,
        random_floor_fraction=args.random_floor_fraction,
        batch_size=args.batch_size,
        min_cached_incident_support=args.min_cached_incident_support,
        min_effective_pairwise_n=args.min_effective_pairwise_n,
        reliable_pair_threshold=args.reliable_pair_threshold,
        low_reliability_random_floor_fraction=(
            args.low_reliability_random_floor_fraction
        ),
        pairwise_cache_artifact_dirs=args.pairwise_cache_artifact_dir,
    )
    sys.stdout.write(json.dumps(_stdout_summary(payload), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def run_ci_partition_v2_gate_replay(
    *,
    config_path: Path,
    manifest_path: Path,
    source_artifact_dir: Path,
    random_variance_artifact_path: Path,
    original_ci_artifact_path: Path,
    output_path: Path,
    active_gate_output_path: Path,
    phase: str = "pilot",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    scheduler_samples: int = 800,
    posterior_samples: int = 1200,
    pairwise_strength: float = 2.5,
    confidence_z: float = 1.96,
    random_floor_fraction: float = 0.25,
    batch_size: int = 5,
    min_cached_incident_support: int = 4,
    min_effective_pairwise_n: float = 1.25,
    reliable_pair_threshold: float = 0.55,
    low_reliability_random_floor_fraction: float = 0.5,
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
    random_variance_artifact = _read_json(random_variance_artifact_path)
    original_ci_artifact = _read_json(original_ci_artifact_path)
    ci_config = ReliabilityAwareCIPartitionV2Config(
        confidence_z=confidence_z,
        pairwise_strength=pairwise_strength,
        posterior_samples=scheduler_samples,
        random_floor_fraction=random_floor_fraction,
        batch_size=batch_size,
        min_cached_incident_support=min_cached_incident_support,
        min_effective_pairwise_n=min_effective_pairwise_n,
        reliable_pair_threshold=reliable_pair_threshold,
        low_reliability_random_floor_fraction=(
            low_reliability_random_floor_fraction
        ),
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
            comparison_map = {key: row.comparison for key, row in cached.items()}
            pointwise_predictions = [
                Prediction(paper.paper_id, paper.pointwise.good_probability)
                for paper in papers
            ]
            pointwise_top_k_ids = _top_k_ids(pointwise_predictions, k=bucket.k)
            v2_replay = replay_reliability_aware_ci_partition_v2_gate(
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
                _orient_cached_comparison(
                    cached[_pair_key(pair.left_id, pair.right_id)],
                    pair,
                )
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
                    ARM_V2: _arm_result_payload(
                        papers,
                        relevant_ids=bucket.relevant_ids,
                        k=bucket.k,
                        schedule=v2_replay.schedule,
                        comparisons=v2_replay.comparisons,
                        pointwise_predictions=pointwise_predictions,
                        pointwise_top_k_ids=pointwise_top_k_ids,
                        labels_by_id=labels_by_bucket.get(bucket.name, {}),
                        posterior_samples=posterior_samples,
                        pairwise_strength=pairwise_strength,
                        interval_config=ci_config,
                        seed=seed,
                        comparison_source={
                            "source": "cached_pairwise_replay",
                            "scheduled_pairwise_total": len(v2_replay.schedule),
                            "cached_pairwise_labels_available": len(
                                v2_replay.comparisons
                            ),
                            "missing_pairwise_labels": (
                                len(v2_replay.schedule)
                                - len(v2_replay.comparisons)
                            ),
                            "partial": (
                                len(v2_replay.schedule)
                                != len(v2_replay.comparisons)
                            ),
                        },
                        scheduler_diagnostics=v2_replay.diagnostics,
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
                                len(exact_schedule.pairs)
                                - len(exact_comparisons)
                            ),
                            "partial": (
                                len(exact_schedule.pairs)
                                != len(exact_comparisons)
                            ),
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
    replay_gate = _ci_v2_replay_gate_verdict(
        paired_deltas=paired_deltas,
        aggregate_diagnostics=aggregate_diagnostics,
        minimum_random_floor_rate=min(0.15, max(0.0, random_floor_fraction)),
    )
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "manifest_path": str(manifest_path),
        "source_artifact_dir": str(source_artifact_dir),
        "output_path": str(output_path),
        "active_arm_gate_output_path": str(active_gate_output_path),
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 0,
        "known_paid_spend_before_workflow_usd": CURRENT_KNOWN_SPEND_USD,
        "paid_cap_usd": DEFAULT_PAID_CAP_USD,
        "spend_policy": (
            "no-paid offline replay over existing reviewed pointwise and cached "
            "pairwise artifacts only; no Sestina paid LLM calls, pointwise "
            "calls, paid labeling, random-baseline spending, ledger rewrites, "
            "or paid-call artifact rewrites are made"
        ),
        "active_arm_name": ARM_V2,
        "candidate_random_control_baseline": ARM_EXACT,
        "pairwise_model_validated_from_config": pairwise_model,
        "label_policy": {
            "future_labels_used_as_model_features": False,
            "future_labels_used_for_scheduling": False,
            "future_labels_used_for_retrospective_diagnostics_only": True,
            "cached_label_values_used_before_scheduling": False,
            "cache_availability_used_for_scheduling": True,
            "pairwise_labels_used_for_replay": (
                "cached historical/follow-up pairwise labels only"
            ),
            "pointwise_paid_calls_made": 0,
        },
        "method": {
            "name": "reliability-aware CI partition v2 cached replay",
            "design": (
                "Use the existing CI partition replay, but treat boundary CI "
                "decisions as unreliable unless endpoints have cached incident "
                "support and revealed pairwise evidence. When unresolved "
                "boundary reliability is low, raise the randomized fallback "
                "floor instead of spending all budget on unstable CI priorities."
            ),
            "conservative_changes": [
                "Reliability weighting uses cached feasible incident support and "
                "revealed effective pairwise n; cached label values are not "
                "read until a scheduled pair is replayed.",
                "Pairs below the reliability threshold are excluded from active "
                "CI priority selection and can only enter through randomized "
                "coverage or low-reliability random fallback.",
                "If nearly the whole bucket remains unresolved, the random floor "
                "is increased to the configured low-reliability fallback rate.",
            ],
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
            "min_cached_incident_support": min_cached_incident_support,
            "min_effective_pairwise_n": min_effective_pairwise_n,
            "reliable_pair_threshold": reliable_pair_threshold,
            "low_reliability_random_floor_fraction": (
                low_reliability_random_floor_fraction
            ),
            "seed_17_policy": (
                "seed-17 is retained in the required 20-seed paired set but is "
                "not used as a standalone comparator"
            ),
        },
        "pairwise_cache_artifact_dirs": [str(path) for path in cache_dirs],
        "pairwise_cache_stats_by_bucket": cache_stats_by_bucket,
        "arms": [
            {
                "name": ARM_V2,
                "method": "reliability_aware_ci_partition_v2_cached_replay",
                "randomized_coverage_floor": True,
            },
            {
                "name": ARM_EXACT,
                "method": "cached_exact_pool_random",
                "randomized_coverage_floor": True,
            },
        ],
        "ci_partition_v2_replay_gate_verdict": replay_gate,
        "aggregate_metrics": aggregate_metrics,
        "paired_deltas_vs_exact_pool_random": paired_deltas,
        "seed_level_metric_intervals": _seed_level_metric_intervals(
            aggregate_metrics=aggregate_metrics,
            paired_deltas=paired_deltas,
        ),
        "aggregate_diagnostics": aggregate_diagnostics,
        "bucket_results": bucket_results,
        "comparison_to_reviewed_artifacts": _comparison_to_reviewed_artifacts(
            current_metrics=aggregate_metrics,
            current_paired_deltas=paired_deltas,
            original_ci_artifact=original_ci_artifact,
            original_ci_artifact_path=original_ci_artifact_path,
            random_variance_artifact=random_variance_artifact,
            random_variance_artifact_path=random_variance_artifact_path,
        ),
        "cache_and_label_caveats": {
            "missing_pairwise_labels": _missing_label_totals(bucket_results),
            "cache_reuse": (
                "The replay is constrained to labels already present in reviewed "
                "artifacts. This is acceptable for the no-paid gate but does not "
                "prove fresh paid acquisition performance."
            ),
        },
        "limitations": [
            "This is an offline cached-label replay, not a fresh paid acquisition run.",
            (
                "The exact comparison pool is approximated by cached labels "
                "intersected with the exact EVSI feasible proposal pool."
            ),
            (
                "The v2 reliability policy uses cache availability and revealed "
                "support, not new labels or retrospective citation labels."
            ),
            (
                "Retrospective citation labels are used only for evaluation, "
                "weak-bucket deltas, and oracle caps."
            ),
            "The pilot remains 8 historical arXiv buckets; paired seed variance is still material.",
        ],
    }
    active_gate = build_active_arm_gate(
        payload,
        random_variance_artifact,
        active_artifact_path=str(output_path),
        random_variance_artifact_path=str(random_variance_artifact_path),
        active_arm_name=ARM_V2,
        candidate_random_control_baseline=ARM_EXACT,
        paid_followup_estimate_usd=0.0,
        known_spend_usd=CURRENT_KNOWN_SPEND_USD,
        paid_cap_usd=DEFAULT_PAID_CAP_USD,
    )
    validate_active_arm_gate_artifact_schema(active_gate)
    payload["gate_verdict"] = active_gate["gate_verdict"]
    payload["paid_followup_allowed"] = active_gate["paid_followup_allowed"]
    payload["active_arm_gate"] = {
        "artifact_path": str(active_gate_output_path),
        "artifact_type": active_gate["artifact_type"],
        "paid_followup_allowed": active_gate["paid_followup_allowed"],
        "gate_verdict": active_gate["gate_verdict"],
        "seed_level_confidence_intervals": active_gate[
            "seed_level_confidence_intervals"
        ],
        "caveats": active_gate["caveats"],
        "random_variance_reference": active_gate["random_variance_reference"],
        "recommended_next_action": active_gate["recommended_next_action"],
    }
    payload["recommended_next_action"] = _recommendation(active_gate)
    validate_ci_partition_v2_artifact_schema(payload)
    write_json_artifact(output_path, payload)
    write_json_artifact(active_gate_output_path, active_gate)
    return {**payload, "artifact_path": str(output_path)}


def validate_ci_partition_v2_artifact_schema(payload: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(payload))
    if missing:
        raise ValueError(
            "CI partition v2 artifact missing top-level keys: "
            + ", ".join(missing)
        )
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("CI partition v2 artifact has unexpected artifact_type")
    if payload.get("paid_calls_made") != 0 or payload.get("paid_spend_usd") != 0.0:
        raise ValueError("CI partition v2 artifact must be zero-paid")
    diagnostics = payload.get("aggregate_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("CI partition v2 aggregate_diagnostics must be an object")
    missing_diagnostics = sorted(REQUIRED_AGGREGATE_DIAGNOSTICS - set(diagnostics))
    if missing_diagnostics:
        raise ValueError(
            "CI partition v2 artifact missing aggregate diagnostics: "
            + ", ".join(missing_diagnostics)
        )
    paired = payload.get("paired_deltas_vs_exact_pool_random")
    if not isinstance(paired, Mapping) or not isinstance(
        paired.get("seed_deltas"),
        Mapping,
    ):
        raise ValueError("CI partition v2 artifact missing paired seed deltas")


def _aggregate_metrics(bucket_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows_by_arm: dict[str, list[dict[str, float | int]]] = {
        ARM_V2: [],
        ARM_EXACT: [],
    }
    seed_rows: dict[str, dict[int, list[dict[str, float | int]]]] = {
        ARM_V2: {},
        ARM_EXACT: {},
    }
    for seed_payload in bucket_results:
        seed = int(seed_payload["seed"])
        for bucket in seed_payload["buckets"]:
            for arm in (ARM_V2, ARM_EXACT):
                row = bucket["arms"][arm]["metrics"][POSTERIOR_STRATEGY]
                rows_by_arm[arm].append(row)
                seed_rows[arm].setdefault(seed, []).append(row)
    output = {}
    for arm, rows in rows_by_arm.items():
        arm_seed_rows = {
            str(seed): _mean_metric_rows(rows_for_seed)
            for seed, rows_for_seed in sorted(seed_rows[arm].items())
        }
        output[arm] = {
            **_mean_metric_rows(rows),
            "seed_count": len(seed_rows[arm]),
            "seed_metric_rows": arm_seed_rows,
            "seed_level_intervals": {
                metric: _summary_with_ci(
                    [
                        float(row[metric])
                        for row in arm_seed_rows.values()
                        if metric in row
                    ]
                )
                for metric in ("recall_at_k", "ndcg_at_k", "average_precision")
            },
        }
    return output


def _paired_deltas(bucket_results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("recall_at_k", "ndcg_at_k", "average_precision")
    seed_deltas: dict[int, dict[str, float]] = {}
    bucket_deltas = []
    for seed_payload in bucket_results:
        seed = int(seed_payload["seed"])
        per_metric_totals = Counter()
        bucket_count = 0
        for bucket in seed_payload["buckets"]:
            active = bucket["arms"][ARM_V2]["metrics"][POSTERIOR_STRATEGY]
            exact = bucket["arms"][ARM_EXACT]["metrics"][POSTERIOR_STRATEGY]
            bucket_count += 1
            row = {"bucket": bucket["bucket"], "seed": seed}
            for metric in metrics:
                delta = round(float(active[metric]) - float(exact[metric]), 8)
                per_metric_totals[metric] += delta
                row[f"{metric}_delta"] = delta
            active_hits = int(
                bucket["arms"][ARM_V2]["top_k_error_decomposition"][
                    "selected_positive_count"
                ]
            )
            exact_hits = int(
                bucket["arms"][ARM_EXACT]["top_k_error_decomposition"][
                    "selected_positive_count"
                ]
            )
            row["selected_positive_delta"] = active_hits - exact_hits
            bucket_deltas.append(row)
        seed_deltas[seed] = {
            metric: round(per_metric_totals[metric] / bucket_count, 8)
            if bucket_count
            else 0.0
            for metric in metrics
        }
    return {
        "reference_arm": ARM_EXACT,
        "comparison_arm": ARM_V2,
        "metric_deltas": {
            metric: _summary_with_ci(
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
    exposure_rows = {ARM_V2: [], ARM_EXACT: []}
    graph_rows = {ARM_V2: [], ARM_EXACT: []}
    oracle_rows = {ARM_V2: [], ARM_EXACT: []}
    unresolved_rows = {ARM_V2: [], ARM_EXACT: []}
    coverage_rows = {ARM_V2: [], ARM_EXACT: []}
    reliability_rows = []
    for seed_payload in bucket_results:
        for bucket in seed_payload["buckets"]:
            for arm in (ARM_V2, ARM_EXACT):
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
            replay_diagnostics = bucket["arms"][ARM_V2]["scheduler_diagnostics"]
            reliability = replay_diagnostics.get("ci_reliability")
            added_round_reliability = False
            for round_row in replay_diagnostics.get(
                "round_history",
                [],
            ):
                if not isinstance(round_row, Mapping):
                    continue
                round_reliability = round_row.get("ci_reliability")
                if isinstance(round_reliability, Mapping):
                    reliability_rows.append(round_reliability)
                    added_round_reliability = True
            if isinstance(reliability, Mapping) and not added_round_reliability:
                reliability_rows.append(reliability)
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
                    [
                        float(row.get("random_floor_rate", 0.0))
                        for row in coverage_rows[arm]
                    ]
                ),
                "low_reliability_random_fallback_pairs": sum(
                    int(row.get("low_reliability_random_fallback_pairs", 0))
                    for row in coverage_rows[arm]
                ),
            }
            for arm in coverage_rows
        },
        "ci_v2_reliability": _ci_v2_reliability_aggregate(reliability_rows),
        "weak_bucket_deltas": _weak_bucket_deltas(bucket_results),
    }


def _weak_bucket_deltas(bucket_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for seed_payload in bucket_results:
        for bucket in seed_payload["buckets"]:
            exact = bucket["arms"][ARM_EXACT]
            active = bucket["arms"][ARM_V2]
            exact_hits = int(
                exact["top_k_error_decomposition"]["selected_positive_count"]
            )
            if exact_hits >= int(bucket["k"]):
                continue
            active_hits = int(
                active["top_k_error_decomposition"]["selected_positive_count"]
            )
            exact_exposure = exact["positive_exposure"]
            active_exposure = active["positive_exposure"]
            exact_oracle = exact["oracle_bounds"]
            active_oracle = active["oracle_bounds"]
            rows.append(
                {
                    "seed": int(seed_payload["seed"]),
                    "bucket": bucket["bucket"],
                    "exact_selected_positive_count": exact_hits,
                    "v2_selected_positive_count": active_hits,
                    "selected_positive_delta": active_hits - exact_hits,
                    "unique_future_positives_touched_delta": int(
                        active_exposure["unique_future_positives_touched"]
                    )
                    - int(exact_exposure["unique_future_positives_touched"]),
                    "pointwise_plus_touched_recall_cap_delta": round(
                        float(
                            active_oracle[
                                "pointwise_plus_touched_positive_upper_bound"
                            ]["recall_cap"]
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
                            active_oracle[
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


def _ci_v2_reliability_aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    fallback_rates = []
    for row in rows:
        if "low_reliability_fallback_active" in row:
            fallback_rates.append(
                1.0 if row.get("low_reliability_fallback_active") else 0.0
            )
        elif "low_reliability_fallback_round_rate" in row:
            fallback_rates.append(_number(row.get("low_reliability_fallback_round_rate")))
    return {
        "row_count": len(rows),
        "mean_boundary_item_reliability": _mean(
            [_number(row.get("mean_boundary_item_reliability")) for row in rows]
        ),
        "mean_low_reliability_boundary_item_rate": _mean(
            [_number(row.get("low_reliability_boundary_item_rate")) for row in rows]
        ),
        "low_reliability_fallback_row_rate": _mean(fallback_rates),
        "mean_unresolved_fraction": _mean(
            [_number(row.get("unresolved_fraction")) for row in rows]
        ),
        "mean_stable_ci_candidate_count": _mean(
            [_number(row.get("stable_ci_candidate_count")) for row in rows]
        ),
    }


def _ci_v2_replay_gate_verdict(
    *,
    paired_deltas: Mapping[str, Any],
    aggregate_diagnostics: Mapping[str, Any],
    minimum_random_floor_rate: float,
) -> dict[str, Any]:
    metric_deltas = paired_deltas["metric_deltas"]
    recall_delta = float(metric_deltas["recall_at_k"]["mean"])
    ndcg_delta = float(metric_deltas["ndcg_at_k"]["mean"])
    ap_delta = float(metric_deltas["average_precision"]["mean"])
    weak = aggregate_diagnostics["weak_bucket_deltas"]
    coverage = aggregate_diagnostics["randomized_coverage"][ARM_V2]
    reliability = aggregate_diagnostics["ci_v2_reliability"]
    random_floor_rate = float(coverage["random_floor_rate"])
    coverage_preserved = random_floor_rate >= minimum_random_floor_rate
    secondary_nonnegative = ndcg_delta >= 0.0 and ap_delta >= 0.0
    credible_metric = recall_delta >= 0.025 and secondary_nonnegative
    reliability_not_worse = (
        float(reliability["low_reliability_fallback_row_rate"]) > 0.0
    )
    oracle_preserved = (
        float(weak["mean_positive_negative_pair_recall_cap_delta"]) >= 0.0
    )
    paid_allowed = bool(
        coverage_preserved
        and credible_metric
        and reliability_not_worse
        and oracle_preserved
    )
    reasons = []
    if not coverage_preserved:
        reasons.append("randomized coverage floor was not preserved")
    if not credible_metric:
        reasons.append(
            "posterior top-K metrics did not clear +0.025 Recall@K with "
            "nonnegative nDCG/AP"
        )
    if not oracle_preserved:
        reasons.append("positive-negative oracle recall cap fell versus exact-pool random")
    if not reliability_not_worse:
        reasons.append("v2 reliability fallback did not activate on low-support rows")
    return {
        "paid_followup_allowed": paid_allowed,
        "coverage_preserved": coverage_preserved,
        "credible_metric_improvement": credible_metric,
        "secondary_metrics_nonnegative": secondary_nonnegative,
        "positive_negative_oracle_cap_preserved": oracle_preserved,
        "reliability_fallback_observed": reliability_not_worse,
        "mean_recall_delta": recall_delta,
        "mean_ndcg_delta": ndcg_delta,
        "mean_average_precision_delta": ap_delta,
        "random_floor_rate": random_floor_rate,
        "blocking_reasons": reasons,
    }


def _comparison_to_reviewed_artifacts(
    *,
    current_metrics: Mapping[str, Any],
    current_paired_deltas: Mapping[str, Any],
    original_ci_artifact: Mapping[str, Any],
    original_ci_artifact_path: Path,
    random_variance_artifact: Mapping[str, Any],
    random_variance_artifact_path: Path,
) -> dict[str, Any]:
    original_metrics = original_ci_artifact.get("aggregate_metrics")
    original_metrics = original_metrics if isinstance(original_metrics, Mapping) else {}
    original_deltas = original_ci_artifact.get("paired_deltas_vs_exact_pool_random")
    original_deltas = original_deltas if isinstance(original_deltas, Mapping) else {}
    return {
        "original_ci_partition_replay": {
            "artifact_path": str(original_ci_artifact_path),
            "artifact_type": original_ci_artifact.get("artifact_type"),
            "aggregate_metrics": original_metrics,
            "paired_deltas_vs_exact_pool_random": original_deltas.get(
                "metric_deltas",
                {},
            ),
            "current_v2_minus_original_ci": _metric_delta_between_arms(
                current_metrics.get(ARM_V2, {}),
                original_metrics.get("ci_partition_elimination", {}),
            ),
        },
        "reviewed_random_variance_reference": {
            "artifact_path": str(random_variance_artifact_path),
            "artifact_type": random_variance_artifact.get("artifact_type"),
            "paid_calls_made_in_reference": random_variance_artifact.get(
                "paid_calls_made",
            ),
            "paid_spend_usd_in_reference": random_variance_artifact.get(
                "paid_spend_usd",
            ),
            "historical_random_full_schedule": _random_reference_metrics(
                random_variance_artifact,
                "historical_random_full_schedule",
            ),
            "exact_pool_random_full_schedule": _random_reference_metrics(
                random_variance_artifact,
                "exact_pool_random_full_schedule",
            ),
        },
        "current_v2_paired_deltas_vs_exact_pool_random": (
            current_paired_deltas.get("metric_deltas", {})
        ),
    }


def _seed_level_metric_intervals(
    *,
    aggregate_metrics: Mapping[str, Any],
    paired_deltas: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "unit": (
            "seed-level means across the 8 buckets; bucket rows are not treated "
            "as independent for headline intervals"
        ),
        "arms": {
            arm: (aggregate_metrics.get(arm, {}) or {}).get(
                "seed_level_intervals",
                {},
            )
            for arm in (ARM_V2, ARM_EXACT)
        },
        "paired_active_minus_exact_pool_random": paired_deltas.get(
            "metric_deltas",
            {},
        ),
    }


def _missing_label_totals(bucket_results: list[dict[str, Any]]) -> dict[str, int]:
    totals = {ARM_V2: 0, ARM_EXACT: 0}
    partial_rows = {ARM_V2: 0, ARM_EXACT: 0}
    for seed_payload in bucket_results:
        for bucket in seed_payload["buckets"]:
            for arm in (ARM_V2, ARM_EXACT):
                source = bucket["arms"][arm]["comparison_source"]
                totals[arm] += int(source["missing_pairwise_labels"])
                partial_rows[arm] += int(bool(source["partial"]))
    return {
        "active_missing_pairwise_labels": totals[ARM_V2],
        "random_control_missing_pairwise_labels": totals[ARM_EXACT],
        "active_partial_rows": partial_rows[ARM_V2],
        "random_control_partial_rows": partial_rows[ARM_EXACT],
    }


def _recommendation(active_gate: Mapping[str, Any]) -> str:
    if active_gate.get("paid_followup_allowed") is True:
        return (
            "The no-paid active-arm gate passed. Treat v2 only as a candidate "
            "for a later pairwise-only paid workflow after a separate dry-run "
            "estimate; this workflow intentionally made zero paid calls."
        )
    return (
        "Do not spend on reliability-aware CI partition v2. Use this artifact "
        "as the no-paid blocker record and revise the simulator or policy "
        "before any paid labels."
    )


def _random_reference_metrics(
    artifact: Mapping[str, Any],
    arm: str,
) -> dict[str, Any]:
    aggregate = artifact.get("aggregate_metrics")
    aggregate = aggregate if isinstance(aggregate, Mapping) else {}
    payload = aggregate.get(arm)
    payload = payload if isinstance(payload, Mapping) else {}
    return {
        "seed_level_intervals": payload.get("seed_level_intervals", {}),
        "bucket_seed_row_mean": payload.get("bucket_seed_row_mean", {}),
    }


def _metric_delta_between_arms(
    active: Any,
    reference: Any,
) -> dict[str, float | None]:
    if not isinstance(active, Mapping) or not isinstance(reference, Mapping):
        return {}
    output = {}
    for metric in ("recall_at_k", "ndcg_at_k", "average_precision"):
        if metric in active and metric in reference:
            output[metric] = round(float(active[metric]) - float(reference[metric]), 8)
        else:
            output[metric] = None
    return output


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
        "stddev": round(stdev(items), 8) if len(items) > 1 else 0.0,
        "min": round(min(items), 8),
        "max": round(max(items), 8),
    }


def _summary_with_ci(values: Sequence[int | float]) -> dict[str, Any]:
    items = [float(value) for value in values]
    if not items:
        return {
            "count": 0,
            "mean": 0.0,
            "stddev": 0.0,
            "standard_error": 0.0,
            "min": 0.0,
            "max": 0.0,
            "normal_approx_95_ci": [None, None],
        }
    value_mean = mean(items)
    value_stddev = stdev(items) if len(items) > 1 else 0.0
    standard_error = value_stddev / math.sqrt(len(items)) if len(items) > 1 else 0.0
    return {
        "count": len(items),
        "mean": round(value_mean, 8),
        "stddev": round(value_stddev, 8),
        "standard_error": round(standard_error, 8),
        "min": round(min(items), 8),
        "max": round(max(items), 8),
        "normal_approx_95_ci": [
            round(value_mean - (1.96 * standard_error), 8),
            round(value_mean + (1.96 * standard_error), 8),
        ],
    }


def _mean(values: Sequence[int | float]) -> float:
    items = [float(value) for value in values]
    return round(sum(items) / len(items), 8) if items else 0.0


def _number(value: Any) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _stdout_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_type": payload["artifact_type"],
        "artifact_path": payload["artifact_path"],
        "active_arm_gate_artifact_path": payload["active_arm_gate_output_path"],
        "paid_calls_made": payload["paid_calls_made"],
        "paid_spend_usd": payload["paid_spend_usd"],
        "active_arm_name": payload["active_arm_name"],
        "candidate_random_control_baseline": payload[
            "candidate_random_control_baseline"
        ],
        "paid_followup_allowed": payload["paid_followup_allowed"],
        "gate_verdict": payload["gate_verdict"],
        "paired_deltas_vs_exact_pool_random": payload[
            "paired_deltas_vs_exact_pool_random"
        ]["metric_deltas"],
        "ci_v2_reliability": payload["aggregate_diagnostics"][
            "ci_v2_reliability"
        ],
        "cache_and_label_caveats": payload["cache_and_label_caveats"],
        "recommended_next_action": payload["recommended_next_action"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
