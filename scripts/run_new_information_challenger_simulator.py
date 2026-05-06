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
    build_budget_completeness_caveat,
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
    CIPartitionConfig,
    confidence_interval_partition,
    schedule_cached_exact_pool_random,
)
from sestina.diagnostics import write_json_artifact  # noqa: E402
from sestina.models import PairwiseComparison  # noqa: E402
from sestina.new_information_challenger import (  # noqa: E402
    NewInformationChallengerConfig,
    replay_new_information_challenger,
)
from sestina.scheduler import resolve_pairwise_budget  # noqa: E402
from sestina.scheduler_followup import load_pointwise_papers_from_artifacts  # noqa: E402


ARTIFACT_TYPE = "sestina-new-information-challenger-simulator"
SCHEMA_VERSION = 1
ARM_NEW_INFO = "new_information_challenger_cached_replay"
ARM_EXACT = "exact_pool_random_cached_replay"
REQUIRED_TOP_LEVEL_KEYS = {
    "artifact_type",
    "schema_version",
    "paid_calls_made",
    "paid_spend_usd",
    "pointwise_calls_made",
    "active_arm_name",
    "candidate_random_control_baseline",
    "gate_verdict",
    "aggregate_metrics",
    "paired_deltas_vs_exact_pool_random",
    "seed_level_metric_intervals",
    "aggregate_diagnostics",
    "bucket_results",
    "cache_and_label_caveats",
    "budget_fill",
    "limitations",
    "active_arm_gate",
}
REQUIRED_AGGREGATE_DIAGNOSTICS = {
    "confidence_bound_unresolved_count",
    "graph_connectivity",
    "oracle_caps",
    "unique_future_positives_touched",
    "weak_bucket_deltas",
    "new_information_challenger",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a no-paid new-information challenger cached replay and "
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
        "--random-control-gap-artifact",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-random-control-diagnosis"
            / "random-control-gap-analysis.json"
        ),
    )
    parser.add_argument(
        "--ci-artifact",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-ci-partition-gate"
            / "ci-partition-gate-analysis.json"
        ),
    )
    parser.add_argument(
        "--ci-v2-artifact",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-ci-partition-v2-gate-replay"
            / "ci-partition-v2-gate-replay.json"
        ),
    )
    parser.add_argument(
        "--prior-incomplete-artifact",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-new-information-challenger-simulator"
            / "new-information-challenger-simulator.json"
        ),
        help=(
            "Prior incomplete new-information simulator artifact used only for "
            "retrospective comparison."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-new-information-challenger-simulator"
            / "new-information-challenger-simulator.json"
        ),
    )
    parser.add_argument(
        "--active-gate-output",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-new-information-challenger-simulator"
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
    parser.add_argument("--random-floor-fraction", type=float, default=0.2)
    parser.add_argument("--anchor-multiplier", type=int, default=2)
    parser.add_argument("--challenger-multiplier", type=int, default=3)
    parser.add_argument("--min-challengers", type=int, default=8)
    parser.add_argument("--minimum-rubric-residual", type=float, default=0.02)
    parser.add_argument("--per-item-cap", type=int, default=6)
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

    payload = run_new_information_challenger_simulator(
        config_path=args.config,
        manifest_path=args.manifest,
        source_artifact_dir=args.source_artifact_dir,
        random_variance_artifact_path=args.random_variance_artifact,
        random_control_gap_artifact_path=args.random_control_gap_artifact,
        ci_artifact_path=args.ci_artifact,
        ci_v2_artifact_path=args.ci_v2_artifact,
        prior_incomplete_artifact_path=args.prior_incomplete_artifact,
        output_path=args.output,
        active_gate_output_path=args.active_gate_output,
        phase=args.phase,
        seeds=_parse_seeds(args.seeds),
        scheduler_samples=args.scheduler_samples,
        posterior_samples=args.posterior_samples,
        pairwise_strength=args.pairwise_strength,
        confidence_z=args.confidence_z,
        random_floor_fraction=args.random_floor_fraction,
        anchor_multiplier=args.anchor_multiplier,
        challenger_multiplier=args.challenger_multiplier,
        min_challengers=args.min_challengers,
        minimum_rubric_residual=args.minimum_rubric_residual,
        per_item_cap=args.per_item_cap,
        pairwise_cache_artifact_dirs=args.pairwise_cache_artifact_dir,
    )
    sys.stdout.write(json.dumps(_stdout_summary(payload), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def run_new_information_challenger_simulator(
    *,
    config_path: Path,
    manifest_path: Path,
    source_artifact_dir: Path,
    random_variance_artifact_path: Path,
    random_control_gap_artifact_path: Path,
    ci_artifact_path: Path,
    ci_v2_artifact_path: Path,
    output_path: Path,
    active_gate_output_path: Path,
    phase: str = "pilot",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    scheduler_samples: int = 800,
    posterior_samples: int = 1200,
    pairwise_strength: float = 2.5,
    confidence_z: float = 1.96,
    random_floor_fraction: float = 0.2,
    anchor_multiplier: int = 2,
    challenger_multiplier: int = 3,
    min_challengers: int = 8,
    minimum_rubric_residual: float = 0.02,
    per_item_cap: int | None = 6,
    pairwise_cache_artifact_dirs: Sequence[Path] | None = None,
    prior_incomplete_artifact_path: Path | None = None,
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
    random_control_gap_artifact = _read_optional_json(random_control_gap_artifact_path)
    ci_artifact = _read_optional_json(ci_artifact_path)
    ci_v2_artifact = _read_optional_json(ci_v2_artifact_path)
    prior_incomplete_artifact = (
        _read_optional_json(prior_incomplete_artifact_path)
        if prior_incomplete_artifact_path is not None
        else None
    )
    new_info_config = NewInformationChallengerConfig(
        pairwise_strength=pairwise_strength,
        posterior_samples=scheduler_samples,
        random_floor_fraction=random_floor_fraction,
        anchor_multiplier=anchor_multiplier,
        challenger_multiplier=challenger_multiplier,
        min_challengers=min_challengers,
        minimum_rubric_residual=minimum_rubric_residual,
        per_item_cap=per_item_cap,
    )
    interval_config = CIPartitionConfig(
        confidence_z=confidence_z,
        pairwise_strength=pairwise_strength,
        posterior_samples=scheduler_samples,
        per_item_cap=per_item_cap,
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
            new_info_replay = replay_new_information_challenger(
                papers,
                comparison_map,
                k=bucket.k,
                budget=budget,
                seed=seed,
                config=new_info_config,
            )
            exact_schedule = schedule_cached_exact_pool_random(
                papers,
                [],
                k=bucket.k,
                budget=budget,
                seed=seed,
                config=interval_config,
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
                    ARM_NEW_INFO: _arm_result_payload(
                        papers,
                        relevant_ids=bucket.relevant_ids,
                        k=bucket.k,
                        schedule=new_info_replay.schedule,
                        comparisons=new_info_replay.comparisons,
                        pointwise_predictions=pointwise_predictions,
                        pointwise_top_k_ids=pointwise_top_k_ids,
                        labels_by_id=labels_by_bucket.get(bucket.name, {}),
                        posterior_samples=posterior_samples,
                        pairwise_strength=pairwise_strength,
                        interval_config=interval_config,
                        seed=seed,
                        comparison_source={
                            **_comparison_source_payload(
                                source="new_information_challenger_cached_replay",
                                scheduled_pairwise_total=len(
                                    new_info_replay.schedule
                                ),
                                cached_pairwise_labels_available=len(
                                    new_info_replay.comparisons
                                ),
                                resolved_pairwise_budget=budget.budget,
                            ),
                        },
                        scheduler_diagnostics={
                            **new_info_replay.diagnostics,
                            "final_ci_partition": confidence_interval_partition(
                                papers,
                                new_info_replay.comparisons,
                                k=bucket.k,
                                config=interval_config,
                            ).to_dict(),
                        },
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
                        interval_config=interval_config,
                        seed=seed,
                        comparison_source={
                            **_comparison_source_payload(
                                source="cached_exact_pool_random_replay",
                                scheduled_pairwise_total=len(exact_schedule.pairs),
                                cached_pairwise_labels_available=len(
                                    exact_comparisons
                                ),
                                resolved_pairwise_budget=budget.budget,
                            ),
                        },
                        scheduler_diagnostics={
                            **exact_schedule.diagnostics,
                            "final_ci_partition": confidence_interval_partition(
                                papers,
                                exact_comparisons,
                                k=bucket.k,
                                config=interval_config,
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
    budget_completeness_caveat = build_budget_completeness_caveat(
        {"bucket_results": bucket_results},
        active_arm=ARM_NEW_INFO,
        random_control=ARM_EXACT,
    )
    replay_gate = _new_information_replay_gate_verdict(
        paired_deltas=paired_deltas,
        aggregate_diagnostics=aggregate_diagnostics,
        minimum_random_floor_rate=min(0.10, max(0.0, random_floor_fraction)),
        budget_completeness_caveat=budget_completeness_caveat,
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
        "active_arm_name": ARM_NEW_INFO,
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
            "name": "new-information challenger cached simulator",
            "design": (
                "Expose possible pointwise false negatives by pairing incumbent "
                "top/boundary anchors against papers below or near the pointwise "
                "decision boundary whose pointwise rubric components, lexical "
                "novelty, uncertainty, or metadata diversity are stronger than "
                "the scalar pointwise probability suggests."
            ),
            "conservative_changes": [
                "Selection uses only pointwise artifacts and paper metadata/text "
                "available before scheduling; retrospective citation labels are "
                "used only after the replay for metrics and diagnostics.",
                "Replay pairs are restricted to cached reviewed pairwise labels "
                "to produce a complete no-paid active-gate input.",
                "Rows that cannot fill the resolved budget from the primary "
                "challenger pool use a predeclared cached frontier fallback "
                "that keeps at least one preselected challenger endpoint and "
                "does not inspect label values.",
                "The policy keeps a randomized cached coverage floor and uses "
                "the paired exact-pool random cached replay as the primary "
                "control.",
            ],
            "distinct_from_prior_candidate_construction": [
                "Not naive expanded-pool random: the proposal pool is ranked by "
                "rubric residual and model-visible false-negative cues rather "
                "than widened uniformly.",
                "Not targeted-outside random: challengers are selected by "
                "pointwise rubric disagreement, lexical novelty, and metadata "
                "diversity rather than posterior anchor/outsider scores with "
                "random within-pool selection.",
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
            "anchor_multiplier": anchor_multiplier,
            "challenger_multiplier": challenger_multiplier,
            "min_challengers": min_challengers,
            "minimum_rubric_residual": minimum_rubric_residual,
            "per_item_cap": per_item_cap,
            "cached_fallback_enabled": new_info_config.cached_fallback_enabled,
            "cached_fallback_frontier_multiplier": (
                new_info_config.cached_fallback_frontier_multiplier
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
                "name": ARM_NEW_INFO,
                "method": "new_information_challenger_cached_replay",
                "randomized_coverage_floor": True,
            },
            {
                "name": ARM_EXACT,
                "method": "cached_exact_pool_random",
                "randomized_coverage_floor": True,
            },
        ],
        "new_information_replay_gate_verdict": replay_gate,
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
            random_variance_artifact=random_variance_artifact,
            random_variance_artifact_path=random_variance_artifact_path,
            random_control_gap_artifact=random_control_gap_artifact,
            random_control_gap_artifact_path=random_control_gap_artifact_path,
            ci_artifact=ci_artifact,
            ci_artifact_path=ci_artifact_path,
            ci_v2_artifact=ci_v2_artifact,
            ci_v2_artifact_path=ci_v2_artifact_path,
        ),
        "budget_fill": _budget_fill_summary(
            bucket_results=bucket_results,
            config=new_info_config,
            source_artifact_dir=source_artifact_dir,
            cache_dirs=cache_dirs,
            prior_incomplete_artifact=prior_incomplete_artifact,
            prior_incomplete_artifact_path=prior_incomplete_artifact_path,
            current_metrics=aggregate_metrics,
            current_paired_deltas=paired_deltas,
        ),
        "cache_and_label_caveats": {
            "missing_pairwise_labels": _missing_label_totals(bucket_results),
            "budget_completeness": budget_completeness_caveat,
            "cache_reuse": (
                "The replay is constrained to labels already present in reviewed "
                "artifacts. This creates a complete no-paid gate prerequisite "
                "but does not prove fresh paid acquisition coverage."
            ),
        },
        "limitations": [
            "This is an offline cached-label simulator, not a fresh paid acquisition run.",
            (
                "Cache availability is used to keep the replay complete; future "
                "paid execution would need a dry-run estimate and fresh pairwise "
                "availability is not guaranteed."
            ),
            (
                "Any row that schedules fewer comparisons than the resolved "
                "pairwise budget is a blocking completeness caveat for later "
                "paid follow-up unless filled by a predeclared no-future-label "
                "cached fallback."
            ),
            (
                "The exact comparison pool is approximated by cached labels "
                "intersected with the exact EVSI feasible proposal pool."
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
        active_arm_name=ARM_NEW_INFO,
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
        "spend_estimate": active_gate["spend_estimate"],
        "random_variance_reference": active_gate["random_variance_reference"],
        "recommended_next_action": active_gate["recommended_next_action"],
    }
    payload["spend_estimate"] = active_gate["spend_estimate"]
    payload["recommended_next_action"] = _recommendation(
        active_gate,
        replay_gate=replay_gate,
    )
    payload["budget_fill"]["gate_verdict"] = active_gate["gate_verdict"]
    payload["budget_fill"]["paid_followup_allowed"] = active_gate[
        "paid_followup_allowed"
    ]
    payload["budget_fill"]["recommendation"] = payload["recommended_next_action"]
    validate_new_information_artifact_schema(payload)
    write_json_artifact(output_path, payload)
    write_json_artifact(active_gate_output_path, active_gate)
    return {**payload, "artifact_path": str(output_path)}


def validate_new_information_artifact_schema(payload: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(payload))
    if missing:
        raise ValueError(
            "new-information simulator artifact missing top-level keys: "
            + ", ".join(missing)
        )
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(
            "new-information simulator artifact has unexpected artifact_type"
        )
    if payload.get("paid_calls_made") != 0 or payload.get("paid_spend_usd") != 0.0:
        raise ValueError("new-information simulator artifact must be zero-paid")
    if payload.get("pointwise_calls_made") != 0:
        raise ValueError(
            "new-information simulator artifact must make zero pointwise calls"
        )
    diagnostics = payload.get("aggregate_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError(
            "new-information simulator aggregate_diagnostics must be an object"
        )
    missing_diagnostics = sorted(REQUIRED_AGGREGATE_DIAGNOSTICS - set(diagnostics))
    if missing_diagnostics:
        raise ValueError(
            "new-information simulator artifact missing aggregate diagnostics: "
            + ", ".join(missing_diagnostics)
        )
    paired = payload.get("paired_deltas_vs_exact_pool_random")
    if not isinstance(paired, Mapping) or not isinstance(
        paired.get("seed_deltas"),
        Mapping,
    ):
        raise ValueError(
            "new-information simulator artifact missing paired seed deltas"
        )
    gate = payload.get("active_arm_gate")
    if not isinstance(gate, Mapping) or "paid_followup_allowed" not in gate:
        raise ValueError("new-information simulator artifact missing gate summary")
    budget_fill = payload.get("budget_fill")
    if not isinstance(budget_fill, Mapping):
        raise ValueError("new-information simulator artifact missing budget_fill")
    missing_budget_fill = sorted(
        {
            "method",
            "fallback_policy",
            "inputs",
            "shortfall_summary",
            "prior_incomplete_comparison",
            "recommendation",
        }
        - set(budget_fill)
    )
    if missing_budget_fill:
        raise ValueError(
            "new-information simulator budget_fill missing keys: "
            + ", ".join(missing_budget_fill)
        )
    active_arm = str(payload.get("active_arm_name") or ARM_NEW_INFO)
    random_control = str(
        payload.get("candidate_random_control_baseline") or ARM_EXACT
    )
    budget_caveat = build_budget_completeness_caveat(
        payload,
        active_arm=active_arm,
        random_control=random_control,
    )
    if budget_caveat["present"]:
        cache_caveats = payload.get("cache_and_label_caveats")
        if not isinstance(cache_caveats, Mapping):
            raise ValueError(
                "new-information simulator budget shortfall missing cache caveats"
            )
        reported = cache_caveats.get("budget_completeness")
        if not _budget_caveat_matches_shortfall(reported, budget_caveat):
            raise ValueError(
                "new-information simulator budget shortfall must be reported "
                "as a blocking budget_completeness caveat"
            )
        if payload.get("paid_followup_allowed") is True:
            raise ValueError(
                "new-information simulator budget shortfall must block paid follow-up"
            )
        gate_verdict = payload.get("gate_verdict")
        if isinstance(gate_verdict, Mapping) and (
            gate_verdict.get("paid_followup_allowed") is True
        ):
            raise ValueError(
                "new-information simulator budget shortfall must block gate verdict"
            )
        if gate.get("paid_followup_allowed") is True:
            raise ValueError(
                "new-information simulator budget shortfall must block active gate"
            )


def _budget_caveat_matches_shortfall(
    reported: Any,
    expected: Mapping[str, Any],
) -> bool:
    if not isinstance(reported, Mapping):
        return False
    required_fields = (
        "active_budget_shortfall",
        "random_control_budget_shortfall",
        "active_under_budget_rows",
        "random_control_under_budget_rows",
    )
    return bool(
        reported.get("present") is True
        and reported.get("blocking") is True
        and all(
            int(reported.get(field, -1) or 0) == int(expected.get(field, -2) or 0)
            for field in required_fields
        )
    )


def _aggregate_metrics(bucket_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows_by_arm: dict[str, list[dict[str, float | int]]] = {
        ARM_NEW_INFO: [],
        ARM_EXACT: [],
    }
    seed_rows: dict[str, dict[int, list[dict[str, float | int]]]] = {
        ARM_NEW_INFO: {},
        ARM_EXACT: {},
    }
    for seed_payload in bucket_results:
        seed = int(seed_payload["seed"])
        for bucket in seed_payload["buckets"]:
            for arm in (ARM_NEW_INFO, ARM_EXACT):
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
            active = bucket["arms"][ARM_NEW_INFO]["metrics"][POSTERIOR_STRATEGY]
            exact = bucket["arms"][ARM_EXACT]["metrics"][POSTERIOR_STRATEGY]
            bucket_count += 1
            row = {"bucket": bucket["bucket"], "seed": seed}
            for metric in metrics:
                delta = round(float(active[metric]) - float(exact[metric]), 8)
                per_metric_totals[metric] += delta
                row[f"{metric}_delta"] = delta
            active_hits = int(
                bucket["arms"][ARM_NEW_INFO]["top_k_error_decomposition"][
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
        "comparison_arm": ARM_NEW_INFO,
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
    exposure_rows = {ARM_NEW_INFO: [], ARM_EXACT: []}
    graph_rows = {ARM_NEW_INFO: [], ARM_EXACT: []}
    oracle_rows = {ARM_NEW_INFO: [], ARM_EXACT: []}
    unresolved_rows = {ARM_NEW_INFO: [], ARM_EXACT: []}
    coverage_rows = {ARM_NEW_INFO: [], ARM_EXACT: []}
    new_info_rows = []
    for seed_payload in bucket_results:
        for bucket in seed_payload["buckets"]:
            for arm in (ARM_NEW_INFO, ARM_EXACT):
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
            new_info = bucket["arms"][ARM_NEW_INFO]["scheduler_diagnostics"].get(
                "new_information_challenger",
            )
            if isinstance(new_info, Mapping):
                new_info_rows.append(new_info)
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
                "scheduled_unique_papers": sum(
                    int(row.get("scheduled_unique_papers", 0))
                    for row in coverage_rows[arm]
                ),
            }
            for arm in coverage_rows
        },
        "new_information_challenger": _new_information_aggregate(new_info_rows),
        "weak_bucket_deltas": _weak_bucket_deltas(bucket_results),
    }


def _new_information_aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "mean_anchor_count": _mean([_number(row.get("anchor_count")) for row in rows]),
        "mean_challenger_count": _mean(
            [_number(row.get("challenger_count")) for row in rows]
        ),
        "mean_scheduled_challenger_count": _mean(
            [_number(row.get("scheduled_challenger_count")) for row in rows]
        ),
        "mean_challenger_rubric_residual": _mean(
            [_number(row.get("mean_challenger_rubric_residual")) for row in rows]
        ),
        "mean_challenger_lexical_novelty": _mean(
            [_number(row.get("mean_challenger_lexical_novelty")) for row in rows]
        ),
        "mean_challenger_metadata_diversity": _mean(
            [_number(row.get("mean_challenger_metadata_diversity")) for row in rows]
        ),
        "uses_future_labels_for_scheduling": any(
            bool(row.get("uses_future_labels_for_scheduling")) for row in rows
        ),
        "cached_label_values_used_before_scheduling": any(
            bool(row.get("cached_label_values_used_before_scheduling"))
            for row in rows
        ),
    }


def _weak_bucket_deltas(bucket_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for seed_payload in bucket_results:
        for bucket in seed_payload["buckets"]:
            exact = bucket["arms"][ARM_EXACT]
            active = bucket["arms"][ARM_NEW_INFO]
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
                    "new_information_selected_positive_count": active_hits,
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


def _new_information_replay_gate_verdict(
    *,
    paired_deltas: Mapping[str, Any],
    aggregate_diagnostics: Mapping[str, Any],
    minimum_random_floor_rate: float,
    budget_completeness_caveat: Mapping[str, Any],
) -> dict[str, Any]:
    metric_deltas = paired_deltas["metric_deltas"]
    recall_delta = float(metric_deltas["recall_at_k"]["mean"])
    ndcg_delta = float(metric_deltas["ndcg_at_k"]["mean"])
    ap_delta = float(metric_deltas["average_precision"]["mean"])
    weak = aggregate_diagnostics["weak_bucket_deltas"]
    coverage = aggregate_diagnostics["randomized_coverage"][ARM_NEW_INFO]
    new_info = aggregate_diagnostics["new_information_challenger"]
    random_floor_rate = float(coverage["random_floor_rate"])
    coverage_preserved = random_floor_rate >= minimum_random_floor_rate
    false_negative_signal_present = (
        float(new_info["mean_challenger_rubric_residual"]) > 0.0
        and float(new_info["mean_scheduled_challenger_count"]) > 0.0
        and new_info["uses_future_labels_for_scheduling"] is False
        and new_info["cached_label_values_used_before_scheduling"] is False
    )
    secondary_nonnegative = ndcg_delta >= 0.0 and ap_delta >= 0.0
    credible_metric = recall_delta >= 0.025 and secondary_nonnegative
    weak_oracle_preserved = (
        float(weak["mean_pointwise_plus_touched_recall_cap_delta"]) >= 0.0
        and float(weak["mean_positive_negative_pair_recall_cap_delta"]) >= 0.0
    )
    budget_complete = budget_completeness_caveat.get("present") is not True
    paid_allowed = bool(
        coverage_preserved
        and false_negative_signal_present
        and weak_oracle_preserved
        and credible_metric
        and budget_complete
    )
    reasons = []
    if not coverage_preserved:
        reasons.append("randomized coverage floor was not preserved")
    if not false_negative_signal_present:
        reasons.append("model-visible false-negative challenger signal was absent")
    if not weak_oracle_preserved:
        reasons.append("weak-bucket oracle headroom fell versus exact-pool random")
    if not credible_metric:
        reasons.append(
            "posterior top-K metrics did not clear +0.025 Recall@K with "
            "nonnegative nDCG/AP"
        )
    if not budget_complete:
        reasons.append("resolved pairwise budget shortfall is present")
    return {
        "paid_followup_allowed": paid_allowed,
        "coverage_preserved": coverage_preserved,
        "false_negative_signal_present": false_negative_signal_present,
        "weak_oracle_headroom_preserved": weak_oracle_preserved,
        "credible_metric_improvement": credible_metric,
        "budget_complete": budget_complete,
        "secondary_metrics_nonnegative": secondary_nonnegative,
        "mean_recall_delta": recall_delta,
        "mean_ndcg_delta": ndcg_delta,
        "mean_average_precision_delta": ap_delta,
        "random_floor_rate": random_floor_rate,
        "blocking_reasons": reasons,
    }


def _comparison_source_payload(
    *,
    source: str,
    scheduled_pairwise_total: int,
    cached_pairwise_labels_available: int,
    resolved_pairwise_budget: int,
) -> dict[str, Any]:
    missing_pairwise_labels = (
        scheduled_pairwise_total - cached_pairwise_labels_available
    )
    budget_shortfall = max(0, resolved_pairwise_budget - scheduled_pairwise_total)
    return {
        "source": source,
        "resolved_pairwise_budget": resolved_pairwise_budget,
        "scheduled_pairwise_total": scheduled_pairwise_total,
        "cached_pairwise_labels_available": cached_pairwise_labels_available,
        "missing_pairwise_labels": missing_pairwise_labels,
        "partial": missing_pairwise_labels != 0,
        "budget_complete": budget_shortfall == 0,
        "scheduled_pairwise_shortfall": budget_shortfall,
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
            for arm in (ARM_NEW_INFO, ARM_EXACT)
        },
        "paired_active_minus_exact_pool_random": paired_deltas.get(
            "metric_deltas",
            {},
        ),
    }


def _missing_label_totals(bucket_results: list[dict[str, Any]]) -> dict[str, int]:
    totals = {ARM_NEW_INFO: 0, ARM_EXACT: 0}
    partial_rows = {ARM_NEW_INFO: 0, ARM_EXACT: 0}
    for seed_payload in bucket_results:
        for bucket in seed_payload["buckets"]:
            for arm in (ARM_NEW_INFO, ARM_EXACT):
                source = bucket["arms"][arm]["comparison_source"]
                totals[arm] += int(source["missing_pairwise_labels"])
                partial_rows[arm] += int(bool(source["partial"]))
    return {
        "active_missing_pairwise_labels": totals[ARM_NEW_INFO],
        "random_control_missing_pairwise_labels": totals[ARM_EXACT],
        "active_partial_rows": partial_rows[ARM_NEW_INFO],
        "random_control_partial_rows": partial_rows[ARM_EXACT],
    }


def _comparison_to_reviewed_artifacts(
    *,
    current_metrics: Mapping[str, Any],
    current_paired_deltas: Mapping[str, Any],
    random_variance_artifact: Mapping[str, Any],
    random_variance_artifact_path: Path,
    random_control_gap_artifact: Mapping[str, Any] | None,
    random_control_gap_artifact_path: Path,
    ci_artifact: Mapping[str, Any] | None,
    ci_artifact_path: Path,
    ci_v2_artifact: Mapping[str, Any] | None,
    ci_v2_artifact_path: Path,
) -> dict[str, Any]:
    return {
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
            "historical_random_comparison_is_unpaired": True,
        },
        "prior_candidate_construction_context": {
            "artifact_path": str(random_control_gap_artifact_path),
            "artifact_type": _mapping_get(random_control_gap_artifact, "artifact_type"),
            "posterior_topk_metrics": _prior_candidate_metrics(
                random_control_gap_artifact,
            ),
            "methodological_note": (
                "Prior expanded-pool and targeted-outsider rows are single-seed "
                "context, not complete paired active-gate controls."
            ),
        },
        "original_ci_partition_replay": _compact_prior_active_artifact(
            ci_artifact,
            ci_artifact_path,
            active_arm="ci_partition_elimination",
        ),
        "ci_partition_v2_replay": _compact_prior_active_artifact(
            ci_v2_artifact,
            ci_v2_artifact_path,
            active_arm="reliability_aware_ci_partition_v2_cached_replay",
        ),
        "current_new_information_paired_deltas_vs_exact_pool_random": (
            current_paired_deltas.get("metric_deltas", {})
        ),
        "current_new_information_minus_prior_context": {
            "minus_ci_partition_replay": _metric_delta_between_arms(
                current_metrics.get(ARM_NEW_INFO, {}),
                _mapping_get(_mapping_get(ci_artifact, "aggregate_metrics"), "ci_partition_elimination"),
            ),
            "minus_ci_partition_v2_replay": _metric_delta_between_arms(
                current_metrics.get(ARM_NEW_INFO, {}),
                _mapping_get(
                    _mapping_get(ci_v2_artifact, "aggregate_metrics"),
                    "reliability_aware_ci_partition_v2_cached_replay",
                ),
            ),
        },
    }


def _budget_fill_summary(
    *,
    bucket_results: list[dict[str, Any]],
    config: NewInformationChallengerConfig,
    source_artifact_dir: Path,
    cache_dirs: Sequence[Path],
    prior_incomplete_artifact: Mapping[str, Any] | None,
    prior_incomplete_artifact_path: Path | None,
    current_metrics: Mapping[str, Any],
    current_paired_deltas: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    totals = Counter()
    for seed_payload in bucket_results:
        seed = int(seed_payload["seed"])
        for bucket in seed_payload["buckets"]:
            active = bucket["arms"][ARM_NEW_INFO]
            source = active["comparison_source"]
            fallback = active["scheduler_diagnostics"].get(
                "cached_frontier_fallback",
                {},
            )
            primary_shortfall = int(
                fallback.get(
                    "primary_scheduled_pairwise_shortfall",
                    source.get("scheduled_pairwise_shortfall", 0),
                )
                or 0
            )
            selected_total = int(fallback.get("selected_total", 0) or 0)
            remaining_shortfall = int(
                fallback.get(
                    "remaining_shortfall",
                    source.get("scheduled_pairwise_shortfall", 0),
                )
                or 0
            )
            totals["row_count"] += 1
            totals["primary_active_budget_shortfall"] += primary_shortfall
            totals["fallback_completed_shortfall"] += selected_total
            totals["remaining_active_budget_shortfall"] += remaining_shortfall
            totals["fallback_cached_feasible_proposals"] += int(
                fallback.get("cached_feasible_proposals", 0) or 0
            )
            if primary_shortfall:
                totals["rows_with_primary_shortfall"] += 1
            if selected_total:
                totals["rows_filled_by_fallback"] += 1
            if remaining_shortfall:
                totals["rows_with_remaining_shortfall"] += 1
            if primary_shortfall or selected_total or remaining_shortfall:
                rows.append(
                    {
                        "seed": seed,
                        "bucket": bucket["bucket"],
                        "resolved_pairwise_budget": int(
                            source["resolved_pairwise_budget"]
                        ),
                        "primary_scheduled_total": int(
                            fallback.get(
                                "primary_scheduled_total",
                                source["scheduled_pairwise_total"],
                            )
                            or 0
                        ),
                        "primary_scheduled_pairwise_shortfall": primary_shortfall,
                        "fallback_selected_total": selected_total,
                        "scheduled_total_after_fallback": int(
                            source["scheduled_pairwise_total"]
                        ),
                        "remaining_shortfall": remaining_shortfall,
                        "fallback_cached_feasible_proposals": int(
                            fallback.get("cached_feasible_proposals", 0) or 0
                        ),
                        "fallback_frontier_pair_candidates": int(
                            fallback.get("frontier_pair_candidates", 0) or 0
                        ),
                        "fallback_cap_relaxed_to_fill_shortfall": bool(
                            fallback.get("cap_relaxed_to_fill_shortfall")
                        ),
                    }
                )
    return {
        "method": {
            "name": "budget-filled new-information challenger cached replay",
            "predeclared_fallback": True,
            "zero_paid_calls": True,
            "description": (
                "Run the primary new-information challenger cached replay, then "
                "fill only active rows still under the resolved budget using a "
                "cached frontier fallback selected before any label values or "
                "future citation outcomes are inspected."
            ),
        },
        "fallback_policy": {
            "name": "predeclared_cached_frontier_challenger_fallback",
            "purpose": "new_information_cached_frontier_fallback",
            "enabled": config.cached_fallback_enabled,
            "frontier_multiplier": config.cached_fallback_frontier_multiplier,
            "endpoint_rule": (
                "At least one endpoint must be in the preselected "
                "new-information challenger set; the comparator must be in the "
                "top frontier_multiplier*K boundary-proximate pointwise "
                "frontier."
            ),
            "exclusions": [
                "already-seen comparison keys",
                "all primary anchor-challenger proposal keys",
            ],
            "selection_order": (
                "deterministic ranked score from challenger score, comparator "
                "boundary proximity, pointwise probability, uncertainty, "
                "lexical novelty, and metadata diversity; cache-key "
                "availability is used only as an offline feasibility filter"
            ),
            "future_labels_used_for_scheduling": False,
            "future_citation_labels_used_for_scheduling": False,
            "cached_label_values_used_before_scheduling": False,
            "cache_availability_used_for_scheduling": True,
        },
        "inputs": {
            "source_artifact_dir": str(source_artifact_dir),
            "pairwise_cache_artifact_dirs": [str(path) for path in cache_dirs],
            "prior_incomplete_artifact_path": (
                str(prior_incomplete_artifact_path)
                if prior_incomplete_artifact_path is not None
                else None
            ),
        },
        "shortfall_summary": {
            "row_count": int(totals["row_count"]),
            "rows_with_primary_shortfall": int(
                totals["rows_with_primary_shortfall"]
            ),
            "primary_active_budget_shortfall": int(
                totals["primary_active_budget_shortfall"]
            ),
            "rows_filled_by_fallback": int(totals["rows_filled_by_fallback"]),
            "fallback_completed_shortfall": int(
                totals["fallback_completed_shortfall"]
            ),
            "rows_with_remaining_shortfall": int(
                totals["rows_with_remaining_shortfall"]
            ),
            "remaining_active_budget_shortfall": int(
                totals["remaining_active_budget_shortfall"]
            ),
            "budget_complete_after_fallback": (
                int(totals["remaining_active_budget_shortfall"]) == 0
            ),
            "fallback_cached_feasible_proposals": int(
                totals["fallback_cached_feasible_proposals"]
            ),
        },
        "filled_or_blocked_rows": rows,
        "current_metrics": current_metrics,
        "current_paired_deltas_vs_exact_pool_random": (
            current_paired_deltas.get("metric_deltas", {})
        ),
        "prior_incomplete_comparison": _prior_incomplete_comparison(
            prior_incomplete_artifact,
            prior_incomplete_artifact_path=prior_incomplete_artifact_path,
            current_metrics=current_metrics,
            current_paired_deltas=current_paired_deltas,
            current_shortfall_summary={
                "active_budget_shortfall": int(
                    totals["remaining_active_budget_shortfall"]
                ),
                "active_under_budget_rows": int(
                    totals["rows_with_remaining_shortfall"]
                ),
            },
        ),
        "gate_verdict": None,
        "paid_followup_allowed": False,
        "recommendation": "pending active-arm gate evaluation",
    }


def _prior_incomplete_comparison(
    prior_artifact: Mapping[str, Any] | None,
    *,
    prior_incomplete_artifact_path: Path | None,
    current_metrics: Mapping[str, Any],
    current_paired_deltas: Mapping[str, Any],
    current_shortfall_summary: Mapping[str, int],
) -> dict[str, Any]:
    if not isinstance(prior_artifact, Mapping):
        return {
            "available": False,
            "artifact_path": (
                str(prior_incomplete_artifact_path)
                if prior_incomplete_artifact_path is not None
                else None
            ),
        }
    prior_metrics = _mapping_get(prior_artifact.get("aggregate_metrics"), ARM_NEW_INFO)
    prior_paired = prior_artifact.get("paired_deltas_vs_exact_pool_random")
    prior_paired = prior_paired if isinstance(prior_paired, Mapping) else {}
    prior_budget = _mapping_get(
        prior_artifact.get("cache_and_label_caveats"),
        "budget_completeness",
    )
    prior_budget = prior_budget if isinstance(prior_budget, Mapping) else {}
    current_active = current_metrics.get(ARM_NEW_INFO, {})
    return {
        "available": True,
        "artifact_path": (
            str(prior_incomplete_artifact_path)
            if prior_incomplete_artifact_path is not None
            else None
        ),
        "artifact_type": prior_artifact.get("artifact_type"),
        "prior_active_metrics": prior_metrics if isinstance(prior_metrics, Mapping) else {},
        "current_active_metrics": current_active,
        "budget_filled_minus_prior_active_metrics": _metric_delta_between_arms(
            current_active,
            prior_metrics,
        ),
        "prior_paired_deltas_vs_exact_pool_random": prior_paired.get(
            "metric_deltas",
            {},
        ),
        "current_paired_deltas_vs_exact_pool_random": current_paired_deltas.get(
            "metric_deltas",
            {},
        ),
        "paired_delta_mean_change_budget_filled_minus_prior": (
            _paired_delta_mean_change(current_paired_deltas, prior_paired)
        ),
        "prior_budget_shortfall": {
            "active_budget_shortfall": int(
                prior_budget.get("active_budget_shortfall", 0) or 0
            ),
            "active_under_budget_rows": int(
                prior_budget.get("active_under_budget_rows", 0) or 0
            ),
        },
        "current_remaining_budget_shortfall": dict(current_shortfall_summary),
    }


def _paired_delta_mean_change(
    current: Mapping[str, Any],
    prior: Mapping[str, Any],
) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    current_metrics = current.get("metric_deltas")
    prior_metrics = prior.get("metric_deltas")
    current_metrics = current_metrics if isinstance(current_metrics, Mapping) else {}
    prior_metrics = prior_metrics if isinstance(prior_metrics, Mapping) else {}
    for metric in ("recall_at_k", "ndcg_at_k", "average_precision"):
        current_metric = current_metrics.get(metric)
        prior_metric = prior_metrics.get(metric)
        if isinstance(current_metric, Mapping) and isinstance(prior_metric, Mapping):
            output[metric] = round(
                float(current_metric.get("mean", 0.0) or 0.0)
                - float(prior_metric.get("mean", 0.0) or 0.0),
                8,
            )
        else:
            output[metric] = None
    return output


def _prior_candidate_metrics(artifact: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        return {}
    aggregate = artifact.get("aggregate_metrics")
    aggregate = aggregate if isinstance(aggregate, Mapping) else {}
    output = {}
    for arm in (
        "exact_pool_random",
        "historical_random",
        "expanded_pool_random",
        "targeted_outsider_random",
    ):
        arm_payload = aggregate.get(arm)
        if isinstance(arm_payload, Mapping) and isinstance(
            arm_payload.get("posterior_topk"),
            Mapping,
        ):
            output[arm] = arm_payload["posterior_topk"]
    return output


def _compact_prior_active_artifact(
    artifact: Mapping[str, Any] | None,
    artifact_path: Path,
    *,
    active_arm: str,
) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        return {"artifact_path": str(artifact_path), "available": False}
    aggregate = artifact.get("aggregate_metrics")
    aggregate = aggregate if isinstance(aggregate, Mapping) else {}
    paired = artifact.get("paired_deltas_vs_exact_pool_random")
    paired = paired if isinstance(paired, Mapping) else {}
    gate = artifact.get("gate_verdict")
    gate = gate if isinstance(gate, Mapping) else {}
    return {
        "artifact_path": str(artifact_path),
        "available": True,
        "artifact_type": artifact.get("artifact_type"),
        "active_arm": active_arm,
        "aggregate_metrics": aggregate.get(active_arm, {}),
        "paired_deltas_vs_exact_pool_random": paired.get("metric_deltas", {}),
        "paid_followup_allowed": gate.get("paid_followup_allowed"),
        "blocking_reasons": gate.get("blocking_reasons", []),
    }


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


def _recommendation(
    active_gate: Mapping[str, Any],
    *,
    replay_gate: Mapping[str, Any],
) -> str:
    active_gate_allowed = active_gate.get("paid_followup_allowed") is True
    replay_gate_allowed = replay_gate.get("paid_followup_allowed") is True
    if active_gate_allowed and replay_gate_allowed:
        return (
            "The no-paid active-arm gate passed. Treat the new-information "
            "challenger only as a candidate for a later pairwise-only paid "
            "workflow after a separate dry-run estimate; this workflow made "
            "zero paid calls."
        )
    if active_gate_allowed:
        return (
            "The reviewed active-arm gate passed on paired Recall@K/nDCG/AP, "
            "so this is a candidate for later review, but do not treat it as an "
            "unconditional paid-run approval: the replay-local false-negative "
            "diagnostic blocked because weak-bucket oracle headroom fell versus "
            "exact-pool random. Any future paid workflow must explicitly accept "
            "that caveat, run a separate dry-run estimate, and remain pairwise-only."
        )
    return (
        "Do not spend on the new-information challenger. Use this artifact as "
        "the no-paid replay record and revise the simulator or policy before "
        "any paid labels."
    )


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


def _mapping_get(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


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
        "new_information_challenger": payload["aggregate_diagnostics"][
            "new_information_challenger"
        ],
        "budget_fill": payload["budget_fill"]["shortfall_summary"],
        "cache_and_label_caveats": payload["cache_and_label_caveats"],
        "recommended_next_action": payload["recommended_next_action"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
