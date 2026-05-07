#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_random_control_gap import (  # noqa: E402
    oracle_bounds,
    pair_graph_diagnostics,
    positive_exposure_diagnostics,
    top_k_error_decomposition,
)
from scripts.run_ci_partition_gate import (  # noqa: E402
    DEFAULT_SEEDS,
    POSTERIOR_STRATEGY,
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
    _config_for_phase,
    _random_pair_schedule,
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
from sestina.experiment_protocol import (  # noqa: E402
    build_next_experiment_protocol,
    validate_next_experiment_protocol,
)
from sestina.evsi_scheduler import posterior_top_k_predictions  # noqa: E402
from sestina.models import PairwiseComparison, Paper, ScheduledPair  # noqa: E402
from sestina.new_information_challenger import (  # noqa: E402
    NewInformationChallengerConfig,
    replay_new_information_challenger,
)
from sestina.no_paid_algorithm_sweep import (  # noqa: E402
    BordaLCBConfig,
    HybridScheduleConfig,
    choose_best_candidate,
    paper_borda_lcb_predictions,
    paired_seed_metric_deltas,
    schedule_model_visible_hybrid_pairs,
    summarize_values,
)
from sestina.scheduler import PairwiseBudget, resolve_pairwise_budget  # noqa: E402
from sestina.scheduler_followup import load_pointwise_papers_from_artifacts  # noqa: E402


ARTIFACT_TYPE = "sestina-no-paid-algorithm-sweep"
BLOCKED_PROTOCOL_ARTIFACT_TYPE = "sestina-no-paid-algorithm-sweep-blocked-protocol"
SCHEMA_VERSION = 1

ARM_EXACT = "exact_pool_random_cached_replay"
ARM_HISTORICAL = "historical_random_cached_replay"
ARM_POSTERIOR_PRIOR = "posterior_topk_pointwise_prior_control"
ARM_CI = "ci_partition_elimination_cached_replay"
ARM_BORDA = "paper_borda_lcb_cached_replay"
ARM_COVERAGE = "randomized_coverage_floor_hybrid_cached_replay"
ARM_CHALLENGER = "challenger_outsider_hybrid_cached_replay"

CONTROL_ARMS = (ARM_EXACT, ARM_HISTORICAL, ARM_POSTERIOR_PRIOR)
CANDIDATE_ARMS = (ARM_CI, ARM_BORDA, ARM_COVERAGE, ARM_CHALLENGER)
ALL_ARMS = (*CONTROL_ARMS, *CANDIDATE_ARMS)
PRIMARY_STRATEGY_BY_ARM = {
    ARM_BORDA: "borda_lcb_topk",
    ARM_EXACT: POSTERIOR_STRATEGY,
    ARM_HISTORICAL: POSTERIOR_STRATEGY,
    ARM_POSTERIOR_PRIOR: POSTERIOR_STRATEGY,
    ARM_CI: POSTERIOR_STRATEGY,
    ARM_COVERAGE: POSTERIOR_STRATEGY,
    ARM_CHALLENGER: POSTERIOR_STRATEGY,
}
METRICS = ("recall_at_k", "ndcg_at_k", "average_precision")
REQUIRED_TOP_LEVEL_KEYS = {
    "artifact_type",
    "schema_version",
    "phase",
    "paid_calls_made",
    "paid_spend_usd",
    "pointwise_calls_made",
    "active_arm_name",
    "candidate_random_control_baseline",
    "gate_verdict",
    "aggregate_metrics",
    "paired_deltas_vs_exact_pool_random",
    "paired_deltas_by_candidate",
    "seed_level_metric_intervals",
    "aggregate_diagnostics",
    "bucket_results",
    "candidate_arms_tried",
    "control_arms",
    "limitations",
    "active_arm_gate",
    "protocol_outcome",
}
REQUIRED_AGGREGATE_DIAGNOSTICS = {
    "confidence_bound_unresolved_count",
    "graph_connectivity",
    "oracle_caps",
    "unique_future_positives_touched",
    "weak_bucket_deltas",
    "weak_bucket_deltas_by_candidate",
    "randomized_coverage",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the no-paid Sestina algorithm sweep over local cached historical "
            "arXiv artifacts. This makes zero paid LLM calls, zero pointwise calls, "
            "and writes only sanitized derived artifacts."
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
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-no-paid-algorithm-sweep"
            / "no-paid-algorithm-sweep.json"
        ),
    )
    parser.add_argument(
        "--blocked-protocol-output",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-no-paid-algorithm-sweep"
            / "blocked-protocol.json"
        ),
    )
    parser.add_argument(
        "--active-gate-output",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-no-paid-algorithm-sweep"
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
    parser.add_argument("--posterior-samples", type=int, default=900)
    parser.add_argument("--pairwise-strength", type=float, default=2.5)
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument("--random-floor-fraction", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument(
        "--pairwise-cache-artifact-dir",
        action="append",
        type=Path,
        default=None,
        help=(
            "Additional artifact directories to scan for cached pairwise labels. "
            "Defaults to live arXiv dirs plus the completed full-random variance "
            "cache directory."
        ),
    )
    args = parser.parse_args(argv)

    payload = run_no_paid_algorithm_sweep(
        config_path=args.config,
        manifest_path=args.manifest,
        source_artifact_dir=args.source_artifact_dir,
        random_variance_artifact_path=args.random_variance_artifact,
        output_path=args.output,
        blocked_protocol_output_path=args.blocked_protocol_output,
        active_gate_output_path=args.active_gate_output,
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


def run_no_paid_algorithm_sweep(
    *,
    config_path: Path,
    manifest_path: Path,
    source_artifact_dir: Path,
    random_variance_artifact_path: Path,
    output_path: Path,
    blocked_protocol_output_path: Path,
    active_gate_output_path: Path,
    phase: str = "pilot",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    scheduler_samples: int = 800,
    posterior_samples: int = 900,
    pairwise_strength: float = 2.5,
    confidence_z: float = 1.96,
    random_floor_fraction: float = 0.30,
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
    random_variance_artifact = _read_json(random_variance_artifact_path)
    cache_dirs = _sweep_pairwise_cache_dirs(
        source_artifact_dir,
        phase=phase,
        random_variance_artifact_path=random_variance_artifact_path,
        explicit_dirs=pairwise_cache_artifact_dirs,
    )
    ci_config = CIPartitionConfig(
        confidence_z=confidence_z,
        pairwise_strength=pairwise_strength,
        posterior_samples=scheduler_samples,
        random_floor_fraction=random_floor_fraction,
        batch_size=batch_size,
    )
    challenger_config = NewInformationChallengerConfig(
        pairwise_strength=pairwise_strength,
        posterior_samples=scheduler_samples,
        random_floor_fraction=max(0.20, random_floor_fraction),
        anchor_multiplier=2,
        challenger_multiplier=3,
        min_challengers=8,
        per_item_cap=6,
    )
    borda_schedule_config = HybridScheduleConfig(
        name=ARM_BORDA,
        random_floor_fraction=0.20,
        anchor_multiplier=3,
        challenger_multiplier=3,
    )
    coverage_schedule_config = HybridScheduleConfig(
        name=ARM_COVERAGE,
        random_floor_fraction=max(0.35, random_floor_fraction),
        anchor_multiplier=2,
        challenger_multiplier=5,
    )

    cache_stats_by_bucket: dict[str, dict[str, Any]] = {}
    bucket_results = []
    for seed in seeds:
        seed_payload = {"seed": int(seed), "buckets": []}
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
            arms = _run_bucket_arms(
                papers,
                relevant_ids=bucket.relevant_ids,
                labels_by_id=labels_by_bucket.get(bucket.name, {}),
                k=bucket.k,
                budget=budget,
                selection=selection,
                cached=cached,
                comparison_map=comparison_map,
                available_pair_keys=available,
                seed=int(seed),
                pointwise_predictions=pointwise_predictions,
                pointwise_top_k_ids=pointwise_top_k_ids,
                ci_config=ci_config,
                challenger_config=challenger_config,
                borda_schedule_config=borda_schedule_config,
                coverage_schedule_config=coverage_schedule_config,
                posterior_samples=posterior_samples,
                pairwise_strength=pairwise_strength,
            )
            seed_payload["buckets"].append(
                {
                    "bucket": bucket.name,
                    "seed": int(seed),
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
                    "arms": arms,
                }
            )
        bucket_results.append(seed_payload)

    aggregate_metrics, seed_metric_rows = _aggregate_metrics(bucket_results)
    paired_deltas_by_candidate = {
        arm: _paired_deltas(bucket_results, comparison_arm=arm)
        for arm in CANDIDATE_ARMS
    }
    weak_bucket_deltas_by_candidate = {
        arm: _weak_bucket_deltas(bucket_results, comparison_arm=arm)
        for arm in CANDIDATE_ARMS
    }
    aggregate_diagnostics_base = _aggregate_diagnostics(
        bucket_results,
        weak_bucket_deltas_by_candidate=weak_bucket_deltas_by_candidate,
    )
    gate_summaries = _candidate_gate_summaries(
        bucket_results=bucket_results,
        aggregate_metrics=aggregate_metrics,
        paired_deltas_by_candidate=paired_deltas_by_candidate,
        aggregate_diagnostics_base=aggregate_diagnostics_base,
        weak_bucket_deltas_by_candidate=weak_bucket_deltas_by_candidate,
        random_variance_artifact=random_variance_artifact,
        random_variance_artifact_path=random_variance_artifact_path,
        output_path=output_path,
    )
    best_candidate = choose_best_candidate(gate_summaries)
    aggregate_diagnostics = {
        **aggregate_diagnostics_base,
        "weak_bucket_deltas": weak_bucket_deltas_by_candidate[best_candidate],
    }
    active_gate_input = _active_gate_input_payload(
        phase=phase,
        manifest_path=manifest_path,
        source_artifact_dir=source_artifact_dir,
        output_path=output_path,
        pairwise_model=pairwise_model,
        seeds=seeds,
        scheduler_samples=scheduler_samples,
        posterior_samples=posterior_samples,
        pairwise_strength=pairwise_strength,
        confidence_z=confidence_z,
        random_floor_fraction=random_floor_fraction,
        batch_size=batch_size,
        cache_dirs=cache_dirs,
        cache_stats_by_bucket=cache_stats_by_bucket,
        aggregate_metrics=aggregate_metrics,
        paired_deltas_by_candidate=paired_deltas_by_candidate,
        aggregate_diagnostics=aggregate_diagnostics,
        bucket_results=bucket_results,
        active_arm_name=best_candidate,
    )
    best_gate = build_active_arm_gate(
        active_gate_input,
        random_variance_artifact,
        active_artifact_path=str(output_path),
        random_variance_artifact_path=str(random_variance_artifact_path),
        active_arm_name=best_candidate,
        candidate_random_control_baseline=ARM_EXACT,
        paid_followup_estimate_usd=0.0,
        known_spend_usd=CURRENT_KNOWN_SPEND_USD,
        paid_cap_usd=DEFAULT_PAID_CAP_USD,
    )
    validate_active_arm_gate_artifact_schema(best_gate)
    protocol_outcome: dict[str, Any]
    active_gate_summary: dict[str, Any]
    if best_gate["paid_followup_allowed"] is True:
        write_json_artifact(active_gate_output_path, best_gate)
        if blocked_protocol_output_path.exists():
            blocked_protocol_output_path.unlink()
        protocol = build_next_experiment_protocol(
            no_paid_gate_artifact=best_gate,
            priority_direction=_priority_direction_for_candidate(best_candidate),
        )
        validate_next_experiment_protocol(protocol)
        protocol_outcome = {
            "status": "passed_no_paid_gate",
            "blocked_protocol_artifact_path": None,
            "next_experiment_protocol": protocol,
        }
        active_gate_summary = {
            "artifact_written": True,
            "artifact_path": str(active_gate_output_path),
            "artifact_type": best_gate["artifact_type"],
            "paid_followup_allowed": best_gate["paid_followup_allowed"],
            "gate_verdict": best_gate["gate_verdict"],
            "recommended_next_action": best_gate["recommended_next_action"],
        }
    else:
        stale_active_gate_removed = _remove_stale_active_gate_artifact(
            active_gate_output_path
        )
        blocked = _blocked_protocol_report(
            sweep_artifact_path=output_path,
            best_candidate=best_candidate,
            best_gate=best_gate,
            candidate_gate_summaries=gate_summaries,
        )
        validate_blocked_protocol_report(blocked)
        write_json_artifact(blocked_protocol_output_path, blocked)
        protocol_outcome = {
            "status": "blocked_no_candidate_passed",
            "blocked_protocol_artifact_path": str(blocked_protocol_output_path),
            "stale_active_gate_artifact_removed": stale_active_gate_removed,
            "blocking_reasons": best_gate["gate_verdict"]["blocking_reasons"],
            "next_experiment_protocol": blocked["next_experiment_protocol"],
        }
        active_gate_summary = {
            "artifact_written": False,
            "artifact_path": None,
            "artifact_type": best_gate["artifact_type"],
            "paid_followup_allowed": False,
            "stale_artifact_removed": stale_active_gate_removed,
            "gate_verdict": best_gate["gate_verdict"],
            "recommended_next_action": best_gate["recommended_next_action"],
            "not_written_reason": (
                "No candidate legitimately passed the merged active-arm gate."
            ),
        }

    payload = {
        **active_gate_input,
        "gate_verdict": best_gate["gate_verdict"],
        "paid_followup_allowed": best_gate["paid_followup_allowed"],
        "active_arm_gate": active_gate_summary,
        "candidate_gate_summaries": gate_summaries,
        "best_candidate_selection": {
            "best_candidate": best_candidate,
            "selection_rule": (
                "prefer passing gate, then mean Recall@K delta, Recall@K CI "
                "lower bound, nDCG@K delta, and AP delta"
            ),
        },
        "random_variance_reference_baselines": _random_variance_context(
            random_variance_artifact,
            random_variance_artifact_path=random_variance_artifact_path,
        ),
        "seed_level_metric_intervals": _seed_level_metric_intervals(
            aggregate_metrics=aggregate_metrics,
            paired_deltas_by_candidate=paired_deltas_by_candidate,
        ),
        "protocol_outcome": protocol_outcome,
        "recommended_next_action": best_gate["recommended_next_action"],
    }
    validate_no_paid_sweep_artifact_schema(payload)
    write_json_artifact(output_path, payload)
    return {**payload, "artifact_path": str(output_path)}


def validate_no_paid_sweep_artifact_schema(payload: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(payload))
    if missing:
        raise ValueError(
            "no-paid algorithm sweep artifact missing top-level keys: "
            + ", ".join(missing)
        )
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("no-paid algorithm sweep artifact has unexpected artifact_type")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("no-paid algorithm sweep artifact has unexpected schema_version")
    if payload.get("paid_calls_made") != 0:
        raise ValueError("no-paid algorithm sweep must make zero paid calls")
    if float(payload.get("paid_spend_usd") or 0.0) != 0.0:
        raise ValueError("no-paid algorithm sweep must spend zero USD")
    if int(payload.get("pointwise_calls_made") or 0) != 0:
        raise ValueError("no-paid algorithm sweep must make zero pointwise calls")
    candidate_names = {
        str(row.get("name"))
        for row in payload["candidate_arms_tried"]
        if isinstance(row, Mapping)
    }
    missing_candidates = sorted(set(CANDIDATE_ARMS) - candidate_names)
    if missing_candidates:
        raise ValueError(
            "no-paid algorithm sweep did not try required candidate arms: "
            + ", ".join(missing_candidates)
        )
    control_names = {
        str(row.get("name"))
        for row in payload["control_arms"]
        if isinstance(row, Mapping)
    }
    missing_controls = sorted({ARM_EXACT, ARM_HISTORICAL} - control_names)
    if missing_controls:
        raise ValueError(
            "no-paid algorithm sweep missing required controls: "
            + ", ".join(missing_controls)
        )
    diagnostics = payload.get("aggregate_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("no-paid algorithm sweep aggregate_diagnostics must be object")
    missing_diagnostics = sorted(REQUIRED_AGGREGATE_DIAGNOSTICS - set(diagnostics))
    if missing_diagnostics:
        raise ValueError(
            "no-paid algorithm sweep missing aggregate diagnostics: "
            + ", ".join(missing_diagnostics)
        )
    active_gate = payload.get("active_arm_gate")
    if not isinstance(active_gate, Mapping):
        raise ValueError("no-paid algorithm sweep active_arm_gate must be object")
    if (
        active_gate.get("paid_followup_allowed") is not True
        and active_gate.get("artifact_written") is True
    ):
        raise ValueError("blocked no-paid sweep must not write an active-arm gate")


def validate_blocked_protocol_report(payload: Mapping[str, Any]) -> None:
    if payload.get("artifact_type") != BLOCKED_PROTOCOL_ARTIFACT_TYPE:
        raise ValueError("blocked protocol report has unexpected artifact_type")
    if payload.get("paid_calls_made") != 0 or payload.get("paid_spend_usd") != 0.0:
        raise ValueError("blocked protocol report must be zero-paid")
    if payload.get("pointwise_calls_made") != 0:
        raise ValueError("blocked protocol report must make zero pointwise calls")
    if payload.get("active_arm_gate_artifact_produced") is not False:
        raise ValueError("blocked protocol report cannot produce an active gate")
    protocol = payload.get("next_experiment_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("blocked protocol report missing next experiment protocol")
    validate_next_experiment_protocol(protocol)
    gate = protocol["future_experiment_gate"]["no_paid_gate"]
    if gate["passed"] is not False:
        raise ValueError("blocked protocol report must keep no-paid gate blocked")


def _remove_stale_active_gate_artifact(active_gate_output_path: Path) -> bool:
    if not active_gate_output_path.exists():
        return False
    active_gate_output_path.unlink()
    return True


def _run_bucket_arms(
    papers: list[Paper],
    *,
    relevant_ids: set[str],
    labels_by_id: dict[str, dict[str, Any]],
    k: int,
    budget: PairwiseBudget,
    selection: Any,
    cached: Mapping[tuple[str, str], Any],
    comparison_map: Mapping[tuple[str, str], PairwiseComparison],
    available_pair_keys: set[tuple[str, str]],
    seed: int,
    pointwise_predictions: list[Prediction],
    pointwise_top_k_ids: list[str],
    ci_config: CIPartitionConfig,
    challenger_config: NewInformationChallengerConfig,
    borda_schedule_config: HybridScheduleConfig,
    coverage_schedule_config: HybridScheduleConfig,
    posterior_samples: int,
    pairwise_strength: float,
) -> dict[str, Any]:
    arms: dict[str, Any] = {}

    prior_predictions, prior_posterior = posterior_top_k_predictions(
        papers,
        [],
        k=k,
        samples=posterior_samples,
        seed=seed,
        pairwise_strength=pairwise_strength,
    )
    arms[ARM_POSTERIOR_PRIOR] = _arm_result_payload(
        papers,
        relevant_ids=relevant_ids,
        k=k,
        schedule=[],
        comparisons=[],
        selected_predictions=prior_predictions,
        selected_strategy=POSTERIOR_STRATEGY,
        pointwise_predictions=pointwise_predictions,
        pointwise_top_k_ids=pointwise_top_k_ids,
        labels_by_id=labels_by_id,
        posterior_samples=posterior_samples,
        pairwise_strength=pairwise_strength,
        interval_config=ci_config,
        seed=seed,
        comparison_source=_comparison_source(
            source="pointwise_prior_posterior_control",
            schedule=[],
            comparisons=[],
            budget=budget,
            budget_applicable=False,
        ),
        scheduler_diagnostics={
            "method": "posterior_topk_pointwise_prior_control",
            "posterior_topk_diagnostics": prior_posterior.diagnostics,
            "coverage": {"random_floor_pairs": 0, "random_floor_rate": 0.0},
        },
    )

    exact_schedule = schedule_cached_exact_pool_random(
        papers,
        [],
        k=k,
        budget=budget,
        seed=seed,
        config=ci_config,
        available_pair_keys=available_pair_keys,
    )
    exact_comparisons = _reveal_cached_schedule(exact_schedule.pairs, cached)
    arms[ARM_EXACT] = _arm_result_payload(
        papers,
        relevant_ids=relevant_ids,
        k=k,
        schedule=exact_schedule.pairs,
        comparisons=exact_comparisons,
        selected_predictions=None,
        selected_strategy=POSTERIOR_STRATEGY,
        pointwise_predictions=pointwise_predictions,
        pointwise_top_k_ids=pointwise_top_k_ids,
        labels_by_id=labels_by_id,
        posterior_samples=posterior_samples,
        pairwise_strength=pairwise_strength,
        interval_config=ci_config,
        seed=seed,
        comparison_source=_comparison_source(
            source="cached_exact_pool_random_replay",
            schedule=exact_schedule.pairs,
            comparisons=exact_comparisons,
            budget=budget,
        ),
        scheduler_diagnostics=exact_schedule.diagnostics,
    )

    historical_schedule = _random_pair_schedule(
        selection,
        budget=budget,
        seed=seed + 7919,
    )
    historical_comparisons = _reveal_cached_schedule(historical_schedule, cached)
    arms[ARM_HISTORICAL] = _arm_result_payload(
        papers,
        relevant_ids=relevant_ids,
        k=k,
        schedule=historical_schedule,
        comparisons=historical_comparisons,
        selected_predictions=None,
        selected_strategy=POSTERIOR_STRATEGY,
        pointwise_predictions=pointwise_predictions,
        pointwise_top_k_ids=pointwise_top_k_ids,
        labels_by_id=labels_by_id,
        posterior_samples=posterior_samples,
        pairwise_strength=pairwise_strength,
        interval_config=ci_config,
        seed=seed,
        comparison_source=_comparison_source(
            source="cached_historical_random_replay",
            schedule=historical_schedule,
            comparisons=historical_comparisons,
            budget=budget,
        ),
        scheduler_diagnostics={
            "method": "historical_random_cached_replay",
            "scheduled_total": len(historical_schedule),
            "coverage": {
                "random_floor_pairs": len(historical_schedule),
                "random_floor_rate": 1.0 if historical_schedule else 0.0,
            },
        },
    )

    ci_replay = replay_ci_partition_gate(
        papers,
        comparison_map,
        k=k,
        budget=budget,
        seed=seed,
        config=ci_config,
    )
    arms[ARM_CI] = _arm_result_payload(
        papers,
        relevant_ids=relevant_ids,
        k=k,
        schedule=ci_replay.schedule,
        comparisons=ci_replay.comparisons,
        selected_predictions=None,
        selected_strategy=POSTERIOR_STRATEGY,
        pointwise_predictions=pointwise_predictions,
        pointwise_top_k_ids=pointwise_top_k_ids,
        labels_by_id=labels_by_id,
        posterior_samples=posterior_samples,
        pairwise_strength=pairwise_strength,
        interval_config=ci_config,
        seed=seed,
        comparison_source=_comparison_source(
            source="cached_ci_partition_replay",
            schedule=ci_replay.schedule,
            comparisons=ci_replay.comparisons,
            budget=budget,
        ),
        scheduler_diagnostics=ci_replay.diagnostics,
    )

    borda_schedule, borda_schedule_diagnostics = schedule_model_visible_hybrid_pairs(
        papers,
        k=k,
        budget=budget,
        seed=seed,
        available_pair_keys=available_pair_keys,
        config=borda_schedule_config,
    )
    borda_comparisons = _reveal_cached_schedule(borda_schedule, cached)
    borda_predictions, borda_diagnostics = paper_borda_lcb_predictions(
        papers,
        borda_comparisons,
        config=BordaLCBConfig(),
    )
    arms[ARM_BORDA] = _arm_result_payload(
        papers,
        relevant_ids=relevant_ids,
        k=k,
        schedule=borda_schedule,
        comparisons=borda_comparisons,
        selected_predictions=borda_predictions,
        selected_strategy=PRIMARY_STRATEGY_BY_ARM[ARM_BORDA],
        pointwise_predictions=pointwise_predictions,
        pointwise_top_k_ids=pointwise_top_k_ids,
        labels_by_id=labels_by_id,
        posterior_samples=posterior_samples,
        pairwise_strength=pairwise_strength,
        interval_config=ci_config,
        seed=seed,
        comparison_source=_comparison_source(
            source="cached_borda_lcb_replay",
            schedule=borda_schedule,
            comparisons=borda_comparisons,
            budget=budget,
        ),
        scheduler_diagnostics={
            **borda_schedule_diagnostics,
            "ranking_policy": borda_diagnostics,
        },
    )

    coverage_schedule, coverage_diagnostics = schedule_model_visible_hybrid_pairs(
        papers,
        k=k,
        budget=budget,
        seed=seed,
        available_pair_keys=available_pair_keys,
        config=coverage_schedule_config,
    )
    coverage_comparisons = _reveal_cached_schedule(coverage_schedule, cached)
    arms[ARM_COVERAGE] = _arm_result_payload(
        papers,
        relevant_ids=relevant_ids,
        k=k,
        schedule=coverage_schedule,
        comparisons=coverage_comparisons,
        selected_predictions=None,
        selected_strategy=POSTERIOR_STRATEGY,
        pointwise_predictions=pointwise_predictions,
        pointwise_top_k_ids=pointwise_top_k_ids,
        labels_by_id=labels_by_id,
        posterior_samples=posterior_samples,
        pairwise_strength=pairwise_strength,
        interval_config=ci_config,
        seed=seed,
        comparison_source=_comparison_source(
            source="cached_randomized_coverage_floor_hybrid_replay",
            schedule=coverage_schedule,
            comparisons=coverage_comparisons,
            budget=budget,
        ),
        scheduler_diagnostics=coverage_diagnostics,
    )

    challenger_replay = replay_new_information_challenger(
        papers,
        comparison_map,
        k=k,
        budget=budget,
        seed=seed,
        config=challenger_config,
    )
    arms[ARM_CHALLENGER] = _arm_result_payload(
        papers,
        relevant_ids=relevant_ids,
        k=k,
        schedule=challenger_replay.schedule,
        comparisons=challenger_replay.comparisons,
        selected_predictions=None,
        selected_strategy=POSTERIOR_STRATEGY,
        pointwise_predictions=pointwise_predictions,
        pointwise_top_k_ids=pointwise_top_k_ids,
        labels_by_id=labels_by_id,
        posterior_samples=posterior_samples,
        pairwise_strength=pairwise_strength,
        interval_config=ci_config,
        seed=seed,
        comparison_source=_comparison_source(
            source="cached_challenger_outsider_hybrid_replay",
            schedule=challenger_replay.schedule,
            comparisons=challenger_replay.comparisons,
            budget=budget,
        ),
        scheduler_diagnostics=challenger_replay.diagnostics,
    )
    return arms


def _arm_result_payload(
    papers: list[Paper],
    *,
    relevant_ids: set[str],
    k: int,
    schedule: list[ScheduledPair],
    comparisons: list[PairwiseComparison],
    selected_predictions: list[Prediction] | None,
    selected_strategy: str,
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
    strategies = {
        "pointwise_only": pointwise_predictions,
        POSTERIOR_STRATEGY: posterior_predictions,
    }
    if selected_predictions is not None:
        strategies[selected_strategy] = selected_predictions
    metrics = compare_strategies(strategies, relevant_ids=relevant_ids, k=k)
    chosen_predictions = strategies[selected_strategy]
    chosen_top_k_ids = _top_k_ids(chosen_predictions, k=k)
    posterior_top_k_ids = _top_k_ids(posterior_predictions, k=k)
    graph = pair_graph_diagnostics(
        schedule,
        relevant_ids=relevant_ids,
        posterior_top_k_ids=posterior_top_k_ids,
        pointwise_top_k_ids=pointwise_top_k_ids,
    )
    degree = _degree_map(schedule)
    graph["candidate_top_k_degree"] = _degree_summary(chosen_top_k_ids, degree)
    graph["degree_around_future_positives"] = graph["future_positive_degree"]
    graph["degree_around_posterior_top_k"] = graph["posterior_top_k_degree"]
    ci_state = confidence_interval_partition(
        papers,
        comparisons,
        k=k,
        config=interval_config,
    )
    return {
        "selected_strategy": selected_strategy,
        "comparison_source": comparison_source,
        "metrics": {name: metric.to_dict() for name, metric in metrics.items()},
        "positive_exposure": positive_exposure_diagnostics(
            schedule,
            relevant_ids=relevant_ids,
            paper_count=len(papers),
        ),
        "pair_graph": graph,
        "oracle_bounds": oracle_bounds(
            k=k,
            relevant_ids=relevant_ids,
            pointwise_top_k_ids=pointwise_top_k_ids,
            schedule=schedule,
            comparisons=comparisons,
        ),
        "top_k_error_decomposition": top_k_error_decomposition(
            predictions=chosen_predictions,
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


def _active_gate_input_payload(
    *,
    phase: str,
    manifest_path: Path,
    source_artifact_dir: Path,
    output_path: Path,
    pairwise_model: str,
    seeds: Sequence[int],
    scheduler_samples: int,
    posterior_samples: int,
    pairwise_strength: float,
    confidence_z: float,
    random_floor_fraction: float,
    batch_size: int,
    cache_dirs: Sequence[Path],
    cache_stats_by_bucket: dict[str, dict[str, Any]],
    aggregate_metrics: dict[str, Any],
    paired_deltas_by_candidate: dict[str, Any],
    aggregate_diagnostics: dict[str, Any],
    bucket_results: list[dict[str, Any]],
    active_arm_name: str,
) -> dict[str, Any]:
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "manifest_path": str(manifest_path),
        "source_artifact_dir": str(source_artifact_dir),
        "output_path": str(output_path),
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 0,
        "known_paid_spend_before_workflow_usd": CURRENT_KNOWN_SPEND_USD,
        "paid_cap_usd": DEFAULT_PAID_CAP_USD,
        "spend_policy": (
            "no-paid offline algorithm sweep over existing reviewed pointwise "
            "and cached/historical pairwise artifacts only; no Sestina paid LLM "
            "calls, pointwise calls, paid labeling, fresh holdout, ledger "
            "rewrites, or paid-call artifact rewrites are made"
        ),
        "current_result_boundary": {
            "campaign_status": "stopped",
            "claim_scope": (
                "cached/no-paid replay plus guarded cache-only execution; not a "
                "fresh holdout validation and not a publication-ready paid-label "
                "result"
            ),
            "paid_label_purchase_authorized": False,
        },
        "active_arm_name": active_arm_name,
        "candidate_random_control_baseline": ARM_EXACT,
        "pairwise_model_validated_from_config": pairwise_model,
        "label_policy": {
            "future_labels_used_as_model_features": False,
            "future_labels_used_for_scheduling": False,
            "future_labels_used_for_retrospective_diagnostics_only": True,
            "cached_label_values_used_before_scheduling": False,
            "cache_availability_used_for_scheduling": True,
            "pairwise_labels_used_for_replay": (
                "cached historical/follow-up pairwise labels revealed only after "
                "a policy schedules the corresponding pair"
            ),
            "pointwise_paid_calls_made": 0,
        },
        "development_replay_set": {
            "name": "old_8_historical_arxiv_buckets",
            "bucket_count": len(bucket_results[0]["buckets"]) if bucket_results else 0,
            "use": "development_replay_only_not_fresh_holdout",
        },
        "analysis_parameters": {
            "seeds": [int(seed) for seed in seeds],
            "seed_count": len(seeds),
            "scheduler_samples": scheduler_samples,
            "posterior_samples": posterior_samples,
            "pairwise_strength": pairwise_strength,
            "confidence_z": confidence_z,
            "random_floor_fraction": random_floor_fraction,
            "batch_size": batch_size,
            "seed_17_policy": (
                "seed-17 is retained in the paired 20-seed replay but is not "
                "used as a standalone comparator"
            ),
        },
        "pairwise_cache_artifact_dirs": [str(path) for path in cache_dirs],
        "pairwise_cache_stats_by_bucket": cache_stats_by_bucket,
        "control_arms": _arm_descriptions(CONTROL_ARMS),
        "candidate_arms_tried": _arm_descriptions(CANDIDATE_ARMS),
        "aggregate_metrics": aggregate_metrics,
        "paired_deltas_vs_exact_pool_random": paired_deltas_by_candidate[
            active_arm_name
        ],
        "paired_deltas_by_candidate": paired_deltas_by_candidate,
        "aggregate_diagnostics": aggregate_diagnostics,
        "bucket_results": bucket_results,
        "limitations": [
            "This is an offline cached-label replay, not a fresh paid acquisition run.",
            (
                "The replay uses old 8-bucket historical arXiv development data; "
                "it is not a new holdout."
            ),
            (
                "Cache availability can constrain schedules, but cached label "
                "values are not used before scheduling."
            ),
            (
                "Retrospective citation labels are used only for evaluation, "
                "weak-bucket deltas, and oracle caps."
            ),
            (
                "The current campaign remains stopped; passing this sweep would "
                "only justify a separately reviewed dry-run/preflight, not label "
                "purchase."
            ),
        ],
        "reproduction_commands": [
            "uv run python scripts/run_no_paid_algorithm_sweep.py",
            "uv run pytest tests/test_no_paid_algorithm_sweep.py",
            "uv run python scripts/validate_next_experiment_protocol.py",
            "git diff --check",
        ],
    }


def _candidate_gate_summaries(
    *,
    bucket_results: list[dict[str, Any]],
    aggregate_metrics: dict[str, Any],
    paired_deltas_by_candidate: dict[str, Any],
    aggregate_diagnostics_base: dict[str, Any],
    weak_bucket_deltas_by_candidate: dict[str, Any],
    random_variance_artifact: Mapping[str, Any],
    random_variance_artifact_path: Path,
    output_path: Path,
) -> dict[str, dict[str, Any]]:
    summaries = {}
    for candidate in CANDIDATE_ARMS:
        diagnostics = {
            **aggregate_diagnostics_base,
            "weak_bucket_deltas": weak_bucket_deltas_by_candidate[candidate],
        }
        temp = {
            "artifact_type": ARTIFACT_TYPE,
            "schema_version": SCHEMA_VERSION,
            "paid_calls_made": 0,
            "paid_spend_usd": 0.0,
            "pointwise_calls_made": 0,
            "active_arm_name": candidate,
            "candidate_random_control_baseline": ARM_EXACT,
            "aggregate_metrics": aggregate_metrics,
            "paired_deltas_vs_exact_pool_random": paired_deltas_by_candidate[
                candidate
            ],
            "aggregate_diagnostics": diagnostics,
            "bucket_results": bucket_results,
            "label_policy": {
                "future_labels_used_for_scheduling": False,
                "cached_label_values_used_before_scheduling": False,
            },
        }
        gate = build_active_arm_gate(
            temp,
            random_variance_artifact,
            active_artifact_path=str(output_path),
            random_variance_artifact_path=str(random_variance_artifact_path),
            active_arm_name=candidate,
            candidate_random_control_baseline=ARM_EXACT,
            paid_followup_estimate_usd=0.0,
            known_spend_usd=CURRENT_KNOWN_SPEND_USD,
            paid_cap_usd=DEFAULT_PAID_CAP_USD,
        )
        verdict = gate["gate_verdict"]
        summaries[candidate] = {
            "paid_followup_allowed": gate["paid_followup_allowed"],
            "blocking_reasons": verdict["blocking_reasons"],
            "mean_recall_delta": verdict["mean_recall_delta"],
            "recall_delta_ci": verdict["recall_delta_ci"],
            "mean_ndcg_delta": verdict["mean_ndcg_delta"],
            "mean_average_precision_delta": verdict["mean_average_precision_delta"],
            "seed_count": verdict["seed_count"],
            "credible_recall_gate_passed": verdict["credible_recall_gate_passed"],
            "mean_margin_gate_passed": verdict["mean_margin_gate_passed"],
            "missing_label_caveat_present": verdict[
                "missing_label_caveat_present"
            ],
            "budget_completeness_caveat_present": verdict[
                "budget_completeness_caveat_present"
            ],
            "core_diagnostics_complete": verdict["core_diagnostics_complete"],
            "randomized_floor_or_paired_control_present": verdict[
                "randomized_floor_or_paired_control_present"
            ],
        }
    return summaries


def _aggregate_metrics(
    bucket_results: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, float | int]]]]:
    rows_by_arm: dict[str, list[dict[str, float | int]]] = {
        arm: [] for arm in ALL_ARMS
    }
    seed_rows: dict[str, dict[int, list[dict[str, float | int]]]] = {
        arm: {} for arm in ALL_ARMS
    }
    seed_metric_rows: dict[str, dict[str, dict[str, float | int]]] = {}
    for seed_payload in bucket_results:
        seed = int(seed_payload["seed"])
        seed_metric_rows.setdefault(str(seed), {})
        for bucket in seed_payload["buckets"]:
            for arm in ALL_ARMS:
                selected_strategy = bucket["arms"][arm]["selected_strategy"]
                row = bucket["arms"][arm]["metrics"][selected_strategy]
                rows_by_arm[arm].append(row)
                seed_rows[arm].setdefault(seed, []).append(row)
        for arm in ALL_ARMS:
            seed_metric_rows[str(seed)][arm] = _mean_metric_rows(seed_rows[arm][seed])
    aggregate = {}
    for arm in ALL_ARMS:
        arm_seed_rows = {
            str(seed): _mean_metric_rows(rows)
            for seed, rows in sorted(seed_rows[arm].items())
        }
        aggregate[arm] = {
            **_mean_metric_rows(rows_by_arm[arm]),
            "seed_count": len(seed_rows[arm]),
            "selected_strategy": PRIMARY_STRATEGY_BY_ARM[arm],
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


def _paired_deltas(
    bucket_results: list[dict[str, Any]],
    *,
    comparison_arm: str,
) -> dict[str, Any]:
    seed_metric_rows: dict[str, dict[str, dict[str, float | int]]] = {}
    bucket_deltas = []
    for seed_payload in bucket_results:
        seed = int(seed_payload["seed"])
        per_arm_rows: dict[str, list[dict[str, float | int]]] = {
            comparison_arm: [],
            ARM_EXACT: [],
        }
        for bucket in seed_payload["buckets"]:
            active = bucket["arms"][comparison_arm]
            exact = bucket["arms"][ARM_EXACT]
            active_metric = active["metrics"][active["selected_strategy"]]
            exact_metric = exact["metrics"][exact["selected_strategy"]]
            per_arm_rows[comparison_arm].append(active_metric)
            per_arm_rows[ARM_EXACT].append(exact_metric)
            row = {"bucket": bucket["bucket"], "seed": seed}
            for metric in METRICS:
                row[f"{metric}_delta"] = round(
                    float(active_metric[metric]) - float(exact_metric[metric]),
                    8,
                )
            active_hits = int(
                active["top_k_error_decomposition"]["selected_positive_count"]
            )
            exact_hits = int(
                exact["top_k_error_decomposition"]["selected_positive_count"]
            )
            row["selected_positive_delta"] = active_hits - exact_hits
            bucket_deltas.append(row)
        seed_metric_rows[str(seed)] = {
            comparison_arm: _mean_metric_rows(per_arm_rows[comparison_arm]),
            ARM_EXACT: _mean_metric_rows(per_arm_rows[ARM_EXACT]),
        }
    payload = paired_seed_metric_deltas(
        seed_metric_rows,
        comparison_arm=comparison_arm,
        reference_arm=ARM_EXACT,
    )
    payload["bucket_deltas"] = bucket_deltas
    payload["selected_positive_total_delta"] = sum(
        int(row["selected_positive_delta"]) for row in bucket_deltas
    )
    return payload


def _aggregate_diagnostics(
    bucket_results: list[dict[str, Any]],
    *,
    weak_bucket_deltas_by_candidate: dict[str, Any],
) -> dict[str, Any]:
    exposure_rows = {arm: [] for arm in ALL_ARMS}
    graph_rows = {arm: [] for arm in ALL_ARMS}
    oracle_rows = {arm: [] for arm in ALL_ARMS}
    unresolved_rows = {arm: [] for arm in ALL_ARMS}
    coverage_rows = {arm: [] for arm in ALL_ARMS}
    comparison_source_rows = {arm: [] for arm in ALL_ARMS}
    for seed_payload in bucket_results:
        for bucket in seed_payload["buckets"]:
            for arm in ALL_ARMS:
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
                comparison_source_rows[arm].append(arm_payload["comparison_source"])
    return {
        "confidence_bound_unresolved_count": {
            arm: summarize_values(unresolved_rows[arm]) for arm in ALL_ARMS
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
            for arm in ALL_ARMS
        },
        "graph_connectivity": {
            arm: {
                "mean_largest_component_size": _mean(
                    [int(row["largest_component_size"]) for row in graph_rows[arm]]
                ),
                "mean_component_count": _mean(
                    [int(row["component_count"]) for row in graph_rows[arm]]
                ),
                "degree_around_future_positives": _aggregate_degree_rows(
                    graph_rows[arm],
                    "future_positive_degree",
                ),
                "posterior_top_k_degree": _aggregate_degree_rows(
                    graph_rows[arm],
                    "posterior_top_k_degree",
                ),
                "degree_around_posterior_top_k": _aggregate_degree_rows(
                    graph_rows[arm],
                    "posterior_top_k_degree",
                ),
                "candidate_top_k_degree": _aggregate_degree_rows(
                    graph_rows[arm],
                    "candidate_top_k_degree",
                ),
            }
            for arm in ALL_ARMS
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
            for arm in ALL_ARMS
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
            }
            for arm in ALL_ARMS
        },
        "comparison_source_completeness": {
            arm: _comparison_source_summary(comparison_source_rows[arm])
            for arm in ALL_ARMS
        },
        "weak_bucket_deltas_by_candidate": weak_bucket_deltas_by_candidate,
    }


def _weak_bucket_deltas(
    bucket_results: list[dict[str, Any]],
    *,
    comparison_arm: str,
) -> dict[str, Any]:
    rows = []
    for seed_payload in bucket_results:
        for bucket in seed_payload["buckets"]:
            exact = bucket["arms"][ARM_EXACT]
            active = bucket["arms"][comparison_arm]
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
                    "comparison_arm": comparison_arm,
                    "exact_selected_positive_count": exact_hits,
                    "candidate_selected_positive_count": active_hits,
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
        "comparison_arm": comparison_arm,
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


def _blocked_protocol_report(
    *,
    sweep_artifact_path: Path,
    best_candidate: str,
    best_gate: Mapping[str, Any],
    candidate_gate_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    protocol = build_next_experiment_protocol()
    validate_next_experiment_protocol(protocol)
    return {
        "artifact_type": BLOCKED_PROTOCOL_ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 0,
        "sweep_artifact_path": str(sweep_artifact_path),
        "active_arm_gate_artifact_produced": False,
        "best_candidate": best_candidate,
        "blocking_reason": "no_candidate_legitimately_passed_merged_active_arm_gate",
        "best_candidate_gate_verdict": best_gate["gate_verdict"],
        "candidate_gate_summaries": candidate_gate_summaries,
        "next_experiment_protocol": protocol,
        "campaign_stop_boundary": protocol["current_result_boundary"],
        "recommended_next_action": (
            "Keep the current campaign stopped. Treat the sweep as a blocked "
            "no-paid replay report and do not buy labels or start fresh holdout "
            "validation from it."
        ),
    }


def _priority_direction_for_candidate(candidate: str) -> str:
    if candidate == ARM_COVERAGE:
        return "no_paid_replay_gate_randomized_coverage_floor"
    return "confidence_interval_top_k_partition_elimination"


def _comparison_source(
    *,
    source: str,
    schedule: Sequence[ScheduledPair],
    comparisons: Sequence[PairwiseComparison],
    budget: PairwiseBudget,
    budget_applicable: bool = True,
) -> dict[str, Any]:
    scheduled_total = len(schedule)
    revealed_total = len(comparisons)
    return {
        "source": source,
        "scheduled_pairwise_total": scheduled_total,
        "cached_pairwise_labels_available": revealed_total,
        "missing_pairwise_labels": scheduled_total - revealed_total,
        "partial": scheduled_total != revealed_total,
        "resolved_pairwise_budget": budget.budget if budget_applicable else None,
        "budget_applicable": budget_applicable,
    }


def _reveal_cached_schedule(
    schedule: Sequence[ScheduledPair],
    cached: Mapping[tuple[str, str], Any],
) -> list[PairwiseComparison]:
    comparisons = []
    for pair in schedule:
        key = _pair_key(pair.left_id, pair.right_id)
        if key not in cached:
            continue
        comparisons.append(_orient_cached_comparison(cached[key], pair))
    return comparisons


def _sweep_pairwise_cache_dirs(
    source_artifact_dir: Path,
    *,
    phase: str,
    random_variance_artifact_path: Path,
    explicit_dirs: Sequence[Path] | None,
) -> list[Path]:
    base = _pairwise_cache_dirs(
        source_artifact_dir,
        phase=phase,
        explicit_dirs=explicit_dirs,
    )
    extra_candidates = [random_variance_artifact_path.parent]
    dirs = [*base]
    for path in extra_candidates:
        if (path / phase).exists():
            dirs.append(path)
    deduped = []
    seen = set()
    for path in dirs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _seed_level_metric_intervals(
    *,
    aggregate_metrics: Mapping[str, Any],
    paired_deltas_by_candidate: Mapping[str, Any],
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
            for arm in ALL_ARMS
        },
        "paired_active_minus_exact_pool_random_by_candidate": {
            arm: paired.get("metric_deltas", {})
            for arm, paired in paired_deltas_by_candidate.items()
        },
    }


def _random_variance_context(
    artifact: Mapping[str, Any],
    *,
    random_variance_artifact_path: Path,
) -> dict[str, Any]:
    aggregate = artifact.get("aggregate_metrics")
    aggregate = aggregate if isinstance(aggregate, Mapping) else {}
    return {
        "artifact_path": str(random_variance_artifact_path),
        "artifact_type": artifact.get("artifact_type"),
        "paid_calls_made_in_reference": artifact.get("paid_calls_made"),
        "paid_spend_usd_in_reference": artifact.get("paid_spend_usd"),
        "seed_count": (artifact.get("analysis_parameters") or {}).get("seed_count")
        if isinstance(artifact.get("analysis_parameters"), Mapping)
        else None,
        "historical_random_full_schedule": aggregate.get(
            "historical_random_full_schedule",
            {},
        ),
        "exact_pool_random_full_schedule": aggregate.get(
            "exact_pool_random_full_schedule",
            {},
        ),
        "context_note": (
            "The sweep itself makes zero paid calls; this local historical "
            "reference is read only as a completed variance/control artifact."
        ),
    }


def _comparison_source_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scheduled = sum(int(row.get("scheduled_pairwise_total", 0) or 0) for row in rows)
    revealed = sum(
        int(row.get("cached_pairwise_labels_available", 0) or 0) for row in rows
    )
    missing = sum(int(row.get("missing_pairwise_labels", 0) or 0) for row in rows)
    partial_rows = sum(1 for row in rows if row.get("partial") is True)
    shortfall = 0
    under_budget_rows = 0
    for row in rows:
        budget = row.get("resolved_pairwise_budget")
        if not isinstance(budget, int):
            continue
        current_shortfall = max(
            0,
            budget - int(row.get("scheduled_pairwise_total", 0) or 0),
        )
        shortfall += current_shortfall
        under_budget_rows += int(current_shortfall > 0)
    return {
        "row_count": len(rows),
        "scheduled_pairwise_total": scheduled,
        "cached_pairwise_labels_available": revealed,
        "missing_pairwise_labels": missing,
        "partial_rows": partial_rows,
        "budget_shortfall": shortfall,
        "under_budget_rows": under_budget_rows,
    }


def _aggregate_degree_rows(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    degree_rows = [
        row.get(key)
        for row in rows
        if isinstance(row.get(key), Mapping)
    ]
    return {
        "row_count": len(degree_rows),
        "mean": _mean([float(row.get("mean", 0.0)) for row in degree_rows]),
        "mean_zero_degree_count": _mean(
            [int(row.get("zero_degree_count", 0)) for row in degree_rows]
        ),
        "mean_max": _mean([int(row.get("max", 0)) for row in degree_rows]),
    }


def _degree_map(schedule: Sequence[ScheduledPair]) -> Counter[str]:
    degree: Counter[str] = Counter()
    for pair in schedule:
        degree[pair.left_id] += 1
        degree[pair.right_id] += 1
    return degree


def _degree_summary(ids: Sequence[str] | set[str], degree: Counter[str]) -> dict[str, Any]:
    values = [int(degree.get(paper_id, 0)) for paper_id in ids]
    if not values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "zero_degree_count": 0,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": _mean(values),
        "zero_degree_count": sum(1 for value in values if value == 0),
    }


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


def _mean(values: Sequence[int | float]) -> float:
    items = [float(value) for value in values]
    return round(sum(items) / len(items), 8) if items else 0.0


def _arm_descriptions(arms: Sequence[str]) -> list[dict[str, Any]]:
    descriptions = {
        ARM_EXACT: {
            "method": "cached exact-pool random",
            "role": "paired random control",
            "selected_strategy": POSTERIOR_STRATEGY,
        },
        ARM_HISTORICAL: {
            "method": "cached historical random",
            "role": "historical random control",
            "selected_strategy": POSTERIOR_STRATEGY,
        },
        ARM_POSTERIOR_PRIOR: {
            "method": "posterior top-K over pointwise priors only",
            "role": "posterior top-K control",
            "selected_strategy": POSTERIOR_STRATEGY,
        },
        ARM_CI: {
            "method": "confidence-interval top-K partition/elimination",
            "role": "candidate",
            "selected_strategy": POSTERIOR_STRATEGY,
        },
        ARM_BORDA: {
            "method": "paper-level Borda/win-rate lower-confidence-bound ranking",
            "role": "candidate",
            "selected_strategy": PRIMARY_STRATEGY_BY_ARM[ARM_BORDA],
        },
        ARM_COVERAGE: {
            "method": "randomized coverage-floor hybrid",
            "role": "candidate",
            "selected_strategy": POSTERIOR_STRATEGY,
        },
        ARM_CHALLENGER: {
            "method": "challenger/outsider hybrid using model-visible signals",
            "role": "candidate",
            "selected_strategy": POSTERIOR_STRATEGY,
        },
    }
    return [{"name": arm, **descriptions[arm]} for arm in arms]


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
        "paid_calls_made": payload["paid_calls_made"],
        "paid_spend_usd": payload["paid_spend_usd"],
        "pointwise_calls_made": payload["pointwise_calls_made"],
        "active_arm_name": payload["active_arm_name"],
        "candidate_random_control_baseline": payload[
            "candidate_random_control_baseline"
        ],
        "paid_followup_allowed": payload["paid_followup_allowed"],
        "gate_verdict": payload["gate_verdict"],
        "aggregate_metrics": {
            arm: {
                metric: payload["aggregate_metrics"][arm][metric]
                for metric in METRICS
            }
            for arm in ALL_ARMS
        },
        "candidate_gate_summaries": payload["candidate_gate_summaries"],
        "active_arm_gate": payload["active_arm_gate"],
        "protocol_outcome": payload["protocol_outcome"],
        "recommended_next_action": payload["recommended_next_action"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
