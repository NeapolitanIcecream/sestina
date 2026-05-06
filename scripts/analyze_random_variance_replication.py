#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_posterior_decision_shrinkage import (  # noqa: E402
    _comparison_label_stats,
    _mean_metric_rows,
)
from scripts.analyze_random_control_gap import (  # noqa: E402
    oracle_bounds,
    pair_graph_diagnostics,
    positive_exposure_diagnostics,
    top_k_error_decomposition,
)
from scripts.run_ci_partition_gate import (  # noqa: E402
    _manifest_label_lookup,
    _orient_cached_comparison,
    _pair_key,
    _pairwise_cache_dirs,
    load_cached_pairwise_labels,
)
from sestina.backtest import Prediction, compare_strategies  # noqa: E402
from sestina.backtest_budget import load_config  # noqa: E402
from sestina.backtest_runner import (  # noqa: E402
    _call_estimate,
    _config_for_phase,
    _random_pair_schedule,
    load_dataset_manifest,
    validate_model_names,
)
from sestina.diagnostics import DiagnosticRecorder, write_json_artifact  # noqa: E402
from sestina.evsi_scheduler import (  # noqa: E402
    EVSISchedulerConfig,
    _build_evsi_context,
    _evsi_coverage,
    _evsi_score_distribution,
    _proposal_pool_profile,
    _select_random_evsi_pairs,
    posterior_top_k_predictions,
    schedule_exact_pool_random,
)
from sestina.models import (  # noqa: E402
    PairwiseComparison,
    PairwiseOrderMetadata,
    ScheduledPair,
)
from sestina.scheduler import PairSchedule, PairwiseBudget, resolve_pairwise_budget  # noqa: E402
from sestina.scheduler_followup import (  # noqa: E402
    legacy_select_candidates,
    load_pointwise_papers_from_artifacts,
)

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
KNOWN_PAID_SPEND_BEFORE_WORKFLOW_USD = 1.476685
POSTERIOR_STRATEGY = "posterior_topk"
ARM_HISTORICAL_CACHED = "historical_random_cached_replay"
ARM_EXACT_CACHED = "exact_pool_random_cached_replay"
ARM_HISTORICAL_FULL = "historical_random_full_schedule_probe"
ARM_EXACT_FULL = "exact_pool_random_full_schedule_probe"
REQUIRED_ARTIFACT_KEYS = {
    "artifact_type",
    "schema_version",
    "paid_calls_made",
    "paid_spend_usd",
    "analysis_parameters",
    "full_schedule_cache_probe",
    "cached_replay",
    "uncertainty_summary",
    "recommendation",
    "limitations",
}


@dataclass(frozen=True, slots=True)
class BucketCache:
    labels: dict[tuple[str, str], Any]
    stats: dict[str, Any]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify random/exact-pool random seed variance for the historical "
            "arXiv pilot using cached labels only. This script never calls an LLM."
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
            / "backtest-arxiv-random-variance-replication"
            / "random-variance-replication.json"
        ),
    )
    parser.add_argument("--phase", default="pilot")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="comma-separated replay seeds",
    )
    parser.add_argument("--scheduler-samples", type=int, default=1200)
    parser.add_argument("--posterior-samples", type=int, default=1200)
    parser.add_argument("--pairwise-strength", type=float, default=2.5)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
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

    payload = analyze_random_variance_replication(
        config_path=args.config,
        manifest_path=args.manifest,
        source_artifact_dir=args.source_artifact_dir,
        output_path=args.output,
        phase=args.phase,
        seeds=_parse_seeds(args.seeds),
        scheduler_samples=args.scheduler_samples,
        posterior_samples=args.posterior_samples,
        pairwise_strength=args.pairwise_strength,
        bootstrap_samples=args.bootstrap_samples,
        pairwise_cache_artifact_dirs=args.pairwise_cache_artifact_dir,
    )
    sys.stdout.write(json.dumps(_stdout_summary(payload), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def analyze_random_variance_replication(
    *,
    config_path: Path,
    manifest_path: Path,
    source_artifact_dir: Path,
    output_path: Path,
    phase: str = "pilot",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    scheduler_samples: int = 1200,
    posterior_samples: int = 1200,
    pairwise_strength: float = 2.5,
    bootstrap_samples: int = 4000,
    pairwise_cache_artifact_dirs: Sequence[Path] | None = None,
) -> dict[str, Any]:
    raw_config = load_config(config_path)
    phase_config = _config_for_phase(raw_config, phase=phase)["phases"][0]
    pairwise_model = str(phase_config["pairwise_model"])
    validate_model_names([pairwise_model])
    pairwise_estimate = _call_estimate(
        "pairwise",
        pairwise_model,
        raw_config["token_assumptions"],
        raw_config["rate_card"],
    )
    manifest = load_dataset_manifest(manifest_path)
    buckets = manifest.buckets_for_phase(phase)
    labels_by_bucket = _manifest_label_lookup(manifest.payload)
    cache_dirs = _pairwise_cache_dirs(
        source_artifact_dir,
        phase=phase,
        explicit_dirs=pairwise_cache_artifact_dirs,
    )
    cache_by_bucket = {
        bucket.name: BucketCache(
            labels=labels,
            stats=stats,
        )
        for bucket in buckets
        for labels, stats in [
            load_cached_pairwise_labels(
                bucket.name,
                artifact_dirs=cache_dirs,
                phase=phase,
            )
        ]
    }

    full_probe_rows = {ARM_HISTORICAL_FULL: [], ARM_EXACT_FULL: []}
    cached_seed_results = []
    unique_missing_keys: dict[str, set[tuple[str, str]]] = {
        bucket.name: set() for bucket in buckets
    }

    for seed in seeds:
        seed_payload = {"seed": int(seed), "buckets": []}
        for bucket in buckets:
            papers = load_pointwise_papers_from_artifacts(
                bucket,
                source_artifact_dir=source_artifact_dir,
                phase=phase,
            )
            selection = legacy_select_candidates(papers, k=bucket.k)
            budget = resolve_pairwise_budget(
                n=len(papers),
                candidate_size=len(selection.candidate_ids),
            )
            cache = cache_by_bucket[bucket.name]
            available_pair_keys = set(cache.labels)
            pointwise_predictions = [
                Prediction(paper.paper_id, paper.pointwise.good_probability)
                for paper in papers
            ]
            pointwise_top_k_ids = _top_k_ids(pointwise_predictions, k=bucket.k)

            historical_full = _random_pair_schedule(
                selection,
                budget=budget,
                seed=seed + 7919,
            )
            exact_full = schedule_exact_pool_random(
                papers,
                [],
                k=bucket.k,
                budget=budget,
                seed=seed,
                config=EVSISchedulerConfig(
                    samples=scheduler_samples,
                    pairwise_strength=pairwise_strength,
                ),
            )
            for arm_name, schedule in (
                (ARM_HISTORICAL_FULL, historical_full),
                (ARM_EXACT_FULL, exact_full.pairs),
            ):
                row = _schedule_cache_probe(
                    arm_name,
                    bucket=bucket.name,
                    seed=seed,
                    schedule=schedule,
                    cached=cache.labels,
                )
                full_probe_rows[arm_name].append(row)
                for key in row["missing_pair_keys"]:
                    unique_missing_keys[bucket.name].add(tuple(key))

            historical_cached = schedule_cached_historical_random(
                selection.candidate_ids,
                available_pair_keys=available_pair_keys,
                budget=budget,
                seed=seed + 7919,
            )
            exact_cached = schedule_cached_exact_pool_random_replay(
                papers,
                [],
                k=bucket.k,
                budget=budget,
                seed=seed,
                config=EVSISchedulerConfig(
                    samples=scheduler_samples,
                    pairwise_strength=pairwise_strength,
                ),
                available_pair_keys=available_pair_keys,
            )
            historical_comparisons = _cached_comparisons_for_schedule(
                historical_cached,
                cached=cache.labels,
            )
            exact_comparisons = _cached_comparisons_for_schedule(
                exact_cached.pairs,
                cached=cache.labels,
            )
            seed_payload["buckets"].append(
                {
                    "bucket": bucket.name,
                    "seed": int(seed),
                    "k": bucket.k,
                    "papers_total": len(papers),
                    "positive_labels_total": len(bucket.relevant_ids),
                    "budget": budget.to_dict(),
                    "pairwise_cache": cache.stats,
                    "pointwise_metrics": compare_strategies(
                        {"pointwise_only": pointwise_predictions},
                        relevant_ids=bucket.relevant_ids,
                        k=bucket.k,
                    )["pointwise_only"].to_dict(),
                    "arms": {
                        ARM_HISTORICAL_CACHED: _arm_payload(
                            papers,
                            relevant_ids=bucket.relevant_ids,
                            k=bucket.k,
                            schedule=historical_cached,
                            comparisons=historical_comparisons,
                            pointwise_predictions=pointwise_predictions,
                            pointwise_top_k_ids=pointwise_top_k_ids,
                            labels_by_id=labels_by_bucket.get(bucket.name, {}),
                            posterior_samples=posterior_samples,
                            pairwise_strength=pairwise_strength,
                            seed=seed,
                            comparison_source={
                                "source": "cached_historical_random_replay",
                                "scheduled_pairwise_total": len(historical_cached),
                                "cached_pairwise_labels_available": len(
                                    historical_comparisons
                                ),
                                "missing_pairwise_labels": (
                                    len(historical_cached)
                                    - len(historical_comparisons)
                                ),
                                "partial": len(historical_cached)
                                != len(historical_comparisons),
                            },
                            scheduler_diagnostics={
                                "method": "historical_random_cached_replay",
                                "candidate_count": len(selection.candidate_ids),
                                "candidate_pair_count": _candidate_pair_count(
                                    len(selection.candidate_ids)
                                ),
                                "cached_candidate_pair_count": _cached_candidate_count(
                                    selection.candidate_ids,
                                    available_pair_keys,
                                ),
                                "scheduled_total": len(historical_cached),
                                "budget": budget.budget,
                                "selection_policy": (
                                    "random_within_cached_legacy_candidate_pairs"
                                ),
                            },
                        ),
                        ARM_EXACT_CACHED: _arm_payload(
                            papers,
                            relevant_ids=bucket.relevant_ids,
                            k=bucket.k,
                            schedule=exact_cached.pairs,
                            comparisons=exact_comparisons,
                            pointwise_predictions=pointwise_predictions,
                            pointwise_top_k_ids=pointwise_top_k_ids,
                            labels_by_id=labels_by_bucket.get(bucket.name, {}),
                            posterior_samples=posterior_samples,
                            pairwise_strength=pairwise_strength,
                            seed=seed,
                            comparison_source={
                                "source": "cached_exact_pool_random_replay",
                                "scheduled_pairwise_total": len(exact_cached.pairs),
                                "cached_pairwise_labels_available": len(
                                    exact_comparisons
                                ),
                                "missing_pairwise_labels": (
                                    len(exact_cached.pairs) - len(exact_comparisons)
                                ),
                                "partial": len(exact_cached.pairs)
                                != len(exact_comparisons),
                            },
                            scheduler_diagnostics=exact_cached.diagnostics,
                        ),
                    },
                }
            )
        cached_seed_results.append(seed_payload)

    cached_replay = {
        "aggregate_metrics": _aggregate_cached_metrics(
            cached_seed_results,
            bootstrap_samples=bootstrap_samples,
        ),
        "paired_deltas": _paired_deltas(
            cached_seed_results,
            bootstrap_samples=bootstrap_samples,
        ),
        "seed_results": cached_seed_results,
    }
    single_seed_references = _single_seed_references(
        cached_replay,
        bootstrap_samples=bootstrap_samples,
    )
    full_schedule_cache_probe = {
        "interpretation": (
            "These rows probe cache completeness for the original full schedules. "
            "Metrics from incomplete rows are intentionally not used for the "
            "multi-seed interval estimates."
        ),
        "arms": {
            arm: _aggregate_full_probe(rows) for arm, rows in full_probe_rows.items()
        },
        "seed_bucket_rows": {
            arm: _compact_probe_rows(rows) for arm, rows in full_probe_rows.items()
        },
        "combined_unique_missing_pair_keys_total": sum(
            len(keys) for keys in unique_missing_keys.values()
        ),
        "combined_unique_missing_pair_keys_by_bucket": {
            bucket: len(keys) for bucket, keys in sorted(unique_missing_keys.items())
        },
        "estimated_paid_pairwise_completion_cost_usd": round(
            sum(len(keys) for keys in unique_missing_keys.values())
            * pairwise_estimate.cost_usd,
            6,
        ),
        "paid_completion_not_run_reason": (
            "Cached constrained replays already quantify the single-seed variance "
            "risk. Completing 20 full random schedules would be a new paid "
            "baseline-labeling workflow, so this audit leaves it as an explicit "
            "costed option rather than spending by default."
        ),
    }
    uncertainty_summary = _uncertainty_summary(
        cached_replay,
        full_schedule_cache_probe=full_schedule_cache_probe,
        single_seed_references=single_seed_references,
    )
    payload = {
        "artifact_type": "sestina-random-variance-replication",
        "schema_version": 1,
        "phase": phase,
        "manifest_path": str(manifest_path),
        "source_artifact_dir": str(source_artifact_dir),
        "output_path": str(output_path),
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "known_paid_spend_before_workflow_usd": KNOWN_PAID_SPEND_BEFORE_WORKFLOW_USD,
        "spend_policy": (
            "offline cached-label variance audit; no LLM calls are made"
        ),
        "label_policy": {
            "pointwise_paid_calls_made": 0,
            "pairwise_paid_calls_made": 0,
            "pairwise_labels_used_for_replay": (
                "cached historical/follow-up pairwise labels only"
            ),
            "future_labels_used_as_model_features": False,
            "future_labels_used_for_scheduling": False,
            "future_labels_used_for_retrospective_metrics_only": True,
        },
        "pairwise_model_validated_from_config": pairwise_model,
        "analysis_parameters": {
            "seeds": [int(seed) for seed in seeds],
            "seed_count": len(seeds),
            "scheduler_samples": scheduler_samples,
            "posterior_samples": posterior_samples,
            "pairwise_strength": pairwise_strength,
            "bootstrap_samples": bootstrap_samples,
            "confidence_level": 0.95,
            "interval_unit": (
                "seed-level means across the 8 buckets; bucket rows are not "
                "treated as independent for headline intervals"
            ),
        },
        "pairwise_cache_artifact_dirs": [str(path) for path in cache_dirs],
        "pairwise_cache_stats_by_bucket": {
            bucket: cache.stats for bucket, cache in sorted(cache_by_bucket.items())
        },
        "arms": [
            {
                "name": ARM_HISTORICAL_CACHED,
                "method": "legacy candidate-pool random constrained to cached labels",
                "aggregate_metrics_role": "headline_cached_variance_estimate",
            },
            {
                "name": ARM_EXACT_CACHED,
                "method": (
                    "current exact-pool random scheduler policy applied after "
                    "filtering exact EVSI feasible proposals to cached labels"
                ),
                "aggregate_metrics_role": "headline_cached_variance_estimate",
            },
            {
                "name": ARM_HISTORICAL_FULL,
                "method": "legacy full random schedule cache-completeness probe",
                "aggregate_metrics_role": "label_missingness_diagnostic_only",
            },
            {
                "name": ARM_EXACT_FULL,
                "method": "exact-pool full random schedule cache-completeness probe",
                "aggregate_metrics_role": "label_missingness_diagnostic_only",
            },
        ],
        "full_schedule_cache_probe": full_schedule_cache_probe,
        "cached_replay": cached_replay,
        "single_seed_complete_reference_context": single_seed_references,
        "uncertainty_summary": uncertainty_summary,
        "recommendation": uncertainty_summary["recommendation"],
        "limitations": [
            "This audit makes no paid calls and therefore does not complete every full random schedule generated for non-17 seeds.",
            "The headline intervals use cached-label constrained replays, not fresh full random schedules.",
            "The exact-pool cached replay uses the current exact-pool random scheduler policy after filtering to cached labels, but the filtered feasible pool can still differ from a fresh full exact-pool random schedule.",
            "Cached pairwise labels come from multiple prior active and random workflows, so the cached feasible pool is not a perfect draw from the original proposal distributions.",
            "Future citation labels are used only for retrospective metrics and diagnostics, not for scheduling or posterior scoring.",
            "Only 8 historical arXiv buckets are available; one selected positive changes mean Recall@K by 0.025.",
        ],
    }
    validate_random_variance_artifact_schema(payload)
    write_json_artifact(output_path, payload)
    return {**payload, "artifact_path": str(output_path)}


def schedule_cached_historical_random(
    candidate_ids: Sequence[str],
    *,
    available_pair_keys: set[tuple[str, str]],
    budget: PairwiseBudget,
    seed: int,
) -> list[ScheduledPair]:
    rng = random.Random(seed)
    candidates = [
        tuple(pair)
        for pair in itertools.combinations(candidate_ids, 2)
        if _pair_key(pair[0], pair[1]) in available_pair_keys
    ]
    rng.shuffle(candidates)
    scheduled = []
    for index, (left_id, right_id) in enumerate(candidates[: budget.budget], start=1):
        if rng.random() < 0.5:
            shown_first_id, shown_second_id = left_id, right_id
        else:
            shown_first_id, shown_second_id = right_id, left_id
        scheduled.append(
            ScheduledPair(
                left_id=left_id,
                right_id=right_id,
                priority=0.0,
                purpose=ARM_HISTORICAL_CACHED,
                order=PairwiseOrderMetadata(
                    shown_first_id=shown_first_id,
                    shown_second_id=shown_second_id,
                    randomized=True,
                    seed=seed,
                    position_bias_audit=(index % 5 == 0),
                ),
            )
        )
    return scheduled


def schedule_cached_exact_pool_random_replay(
    papers: list[Any],
    comparisons: list[PairwiseComparison],
    *,
    k: int,
    budget: PairwiseBudget,
    seed: int,
    config: EVSISchedulerConfig | None = None,
    available_pair_keys: set[tuple[str, str]] | None = None,
    diagnostics: DiagnosticRecorder | None = None,
) -> PairSchedule:
    """Replay exact-pool random after filtering to proposal pairs with cached labels."""
    cfg = config or EVSISchedulerConfig()
    recorder = diagnostics or DiagnosticRecorder()
    paper_by_id = {paper.paper_id: paper for paper in papers}
    if budget.budget <= 0 or len(paper_by_id) < 2 or k <= 0:
        payload = _empty_cached_exact_pool_diagnostics(
            k=k,
            budget=budget.budget,
            seed=seed,
            config=cfg,
            all_proposals=[],
            feasible=[],
            available_pair_keys=available_pair_keys,
        )
        recorder.record(
            step="pair_scheduling",
            code="exact_pool_random_cached_replay_pair_scheduling_empty",
            message="no cached exact-pool random comparisons scheduled",
            data=payload,
        )
        return PairSchedule(pairs=[], budget=budget, diagnostics=payload)

    seen_pairs = {
        _pair_key(comparison.left_id, comparison.right_id)
        for comparison in comparisons
    }
    context = _build_evsi_context(
        papers,
        comparisons=comparisons,
        k=k,
        seed=seed,
        config=cfg,
        seen_pairs=seen_pairs,
        diagnostics=recorder,
    )
    all_proposals = list(context.proposals)
    feasible = (
        all_proposals
        if available_pair_keys is None
        else [
            proposal
            for proposal in all_proposals
            if _pair_key(proposal.left_id, proposal.right_id) in available_pair_keys
        ]
    )
    scheduled = _select_random_evsi_pairs(
        feasible,
        budget=budget.budget,
        seed=seed,
        per_item_cap=cfg.per_item_cap,
        purpose_override=ARM_EXACT_CACHED,
    )
    effective_cap = _effective_random_pool_per_item_cap(
        proposal_count=len(feasible),
        budget=budget.budget,
        per_item_cap=cfg.per_item_cap,
    )
    payload = {
        "candidate_count": len(context.pool),
        "scheduled_total": len(scheduled),
        "pairs_considered": len(all_proposals),
        "unique_pairs_considered": len(
            {
                _pair_key(proposal.left_id, proposal.right_id)
                for proposal in all_proposals
            }
        ),
        "budget": budget.budget,
        "k": k,
        "posterior": {
            "samples": context.posterior.samples,
            "average_top_k_probability": context.posterior.diagnostics.get(
                "average_top_k_probability"
            ),
        },
        "acquisition": _cached_exact_pool_acquisition_payload(
            seed=seed,
            config=cfg,
            effective_per_item_cap=effective_cap,
        ),
        "purpose_counts": dict(
            sorted(Counter(pair.purpose for pair in scheduled).items())
        ),
        "coverage": _evsi_coverage(
            scheduled,
            item_by_id={item.paper_id: item for item in context.items},
        ),
        "proposal_pool_profile": _proposal_pool_profile(
            items=context.items,
            pool=context.pool,
            scheduled=scheduled,
            k=k,
        ),
        "evsi_score_distribution": _evsi_score_distribution(all_proposals),
        **_cached_exact_pool_filter_payload(
            all_proposals=all_proposals,
            feasible=feasible,
            available_pair_keys=available_pair_keys,
        ),
        "batch_history": [
            {
                "batch_index": 1,
                "selected_total": len(scheduled),
                "cached_label_revealed_total": len(scheduled),
                "novel_pairs_total": 0,
                "top_k_entropy_reduction": None,
                "top_k_set_churn": None,
                "note": (
                    "cached replay schedules one offline batch after removing "
                    "proposal pairs without cached labels"
                ),
            }
        ],
    }
    recorder.record(
        step="pair_scheduling",
        code="exact_pool_random_cached_replay_pair_scheduling_completed",
        message=(
            "randomly sampled cached exact EVSI feasible proposal pool with the "
            "current exact-pool random policy"
        ),
        data=payload,
    )
    return PairSchedule(pairs=scheduled, budget=budget, diagnostics=payload)


def _empty_cached_exact_pool_diagnostics(
    *,
    k: int,
    budget: int,
    seed: int,
    config: EVSISchedulerConfig,
    all_proposals: Sequence[Any],
    feasible: Sequence[Any],
    available_pair_keys: set[tuple[str, str]] | None,
) -> dict[str, Any]:
    return {
        "candidate_count": 0,
        "scheduled_total": 0,
        "pairs_considered": len(all_proposals),
        "unique_pairs_considered": 0,
        "budget": budget,
        "k": k,
        "posterior": {"samples": 0, "average_top_k_probability": None},
        "acquisition": _cached_exact_pool_acquisition_payload(
            seed=seed,
            config=config,
            effective_per_item_cap=None,
        ),
        "purpose_counts": {},
        "coverage": {
            "random_floor_pairs": 0,
            "random_floor_rate": 0.0,
            "scheduled_unique_papers": 0,
            "scheduled_relevant_papers": 0,
            "scheduled_pointwise_top_k_papers": 0,
            "scheduled_boundary_papers": 0,
        },
        **_cached_exact_pool_filter_payload(
            all_proposals=all_proposals,
            feasible=feasible,
            available_pair_keys=available_pair_keys,
        ),
    }


def _cached_exact_pool_acquisition_payload(
    *,
    seed: int,
    config: EVSISchedulerConfig,
    effective_per_item_cap: int | None,
) -> dict[str, Any]:
    return {
        "method": ARM_EXACT_CACHED,
        "source_method": "exact_pool_random",
        "random_seed": seed,
        "posterior_samples": config.samples,
        "pairwise_strength": config.pairwise_strength,
        "per_item_cap": config.per_item_cap,
        "effective_per_item_cap": effective_per_item_cap,
        "pool_multiplier": config.pool_multiplier,
        "diverse_outsider_count": config.diverse_outsider_count,
        "selection_policy": "schedule_exact_pool_random_after_cached_label_filter",
    }


def _cached_exact_pool_filter_payload(
    *,
    all_proposals: Sequence[Any],
    feasible: Sequence[Any],
    available_pair_keys: set[tuple[str, str]] | None,
) -> dict[str, Any]:
    if available_pair_keys is None:
        cached_total = None
        filtered = 0
    else:
        cached_total = len(available_pair_keys)
        filtered = len(all_proposals) - len(feasible)
    return {
        "available_label_filter": {
            "enabled": available_pair_keys is not None,
            "cached_pair_keys_total": cached_total,
            "exact_feasible_pairs_total": len(all_proposals),
            "cached_feasible_pairs_total": len(feasible),
            "filtered_uncached_feasible_pairs_total": filtered,
            "cached_feasible_pair_rate": _rate(len(feasible), len(all_proposals)),
        }
    }


def _effective_random_pool_per_item_cap(
    *,
    proposal_count: int,
    budget: int,
    per_item_cap: int | None,
) -> int | None:
    if budget <= 0:
        return None
    return per_item_cap or max(
        2,
        math.ceil((2.5 * budget) / max(1, proposal_count**0.5)),
    )


def validate_random_variance_artifact_schema(payload: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_ARTIFACT_KEYS - set(payload))
    if missing:
        raise ValueError(
            "random variance artifact missing top-level keys: " + ", ".join(missing)
        )
    if payload.get("artifact_type") != "sestina-random-variance-replication":
        raise ValueError("random variance artifact has unexpected artifact_type")
    cached = payload.get("cached_replay")
    if not isinstance(cached, dict):
        raise ValueError("random variance artifact cached_replay must be an object")
    for key in ("aggregate_metrics", "paired_deltas", "seed_results"):
        if key not in cached:
            raise ValueError(f"random variance cached_replay missing {key}")


def _arm_payload(
    papers: list[Any],
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
    return {
        "comparison_source": comparison_source,
        "metrics": {name: metric.to_dict() for name, metric in metrics.items()},
        "comparison_label_stats": _comparison_label_stats(comparisons),
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
        "posterior_topk_diagnostics": posterior.diagnostics,
        "scheduler_diagnostics": scheduler_diagnostics,
    }


def _cached_comparisons_for_schedule(
    schedule: list[ScheduledPair],
    *,
    cached: dict[tuple[str, str], Any],
) -> list[PairwiseComparison]:
    comparisons = []
    for pair in schedule:
        label = cached.get(_pair_key(pair.left_id, pair.right_id))
        if label is None:
            continue
        comparisons.append(_orient_cached_comparison(label, pair))
    return comparisons


def _schedule_cache_probe(
    arm_name: str,
    *,
    bucket: str,
    seed: int,
    schedule: list[ScheduledPair],
    cached: dict[tuple[str, str], Any],
) -> dict[str, Any]:
    scheduled_keys = [_pair_key(pair.left_id, pair.right_id) for pair in schedule]
    missing = [key for key in scheduled_keys if key not in cached]
    return {
        "arm": arm_name,
        "bucket": bucket,
        "seed": int(seed),
        "scheduled_pairwise_total": len(schedule),
        "cached_pairwise_labels_available": len(schedule) - len(missing),
        "missing_pairwise_labels": len(missing),
        "cache_reuse_rate": _rate(len(schedule) - len(missing), len(schedule)),
        "complete": len(missing) == 0,
        "missing_pair_keys": [list(key) for key in missing],
        "sample_missing_pair_keys": [list(key) for key in missing[:10]],
    }


def _aggregate_full_probe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scheduled = sum(int(row["scheduled_pairwise_total"]) for row in rows)
    cached = sum(int(row["cached_pairwise_labels_available"]) for row in rows)
    missing = sum(int(row["missing_pairwise_labels"]) for row in rows)
    complete_rows = [row for row in rows if bool(row["complete"])]
    return {
        "seed_bucket_row_count": len(rows),
        "complete_seed_bucket_row_count": len(complete_rows),
        "complete_seed_bucket_row_rate": _rate(len(complete_rows), len(rows)),
        "scheduled_pairwise_total": scheduled,
        "cached_pairwise_labels_available": cached,
        "missing_pairwise_labels": missing,
        "cache_reuse_rate": _rate(cached, scheduled),
        "complete_seeds_by_bucket": _complete_seeds_by_bucket(rows),
    }


def _compact_probe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in row.items()
            if key not in {"missing_pair_keys"}
        }
        for row in rows
    ]


def _complete_seeds_by_bucket(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    by_bucket: dict[str, list[int]] = {}
    for row in rows:
        if bool(row["complete"]):
            by_bucket.setdefault(str(row["bucket"]), []).append(int(row["seed"]))
    return {bucket: sorted(seeds) for bucket, seeds in sorted(by_bucket.items())}


def _aggregate_cached_metrics(
    seed_results: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    arms = (ARM_HISTORICAL_CACHED, ARM_EXACT_CACHED)
    output = {}
    for arm in arms:
        seed_rows: dict[int, dict[str, float | int]] = {}
        bucket_rows: dict[str, list[dict[str, float | int]]] = {}
        all_rows = []
        for seed_payload in seed_results:
            seed = int(seed_payload["seed"])
            rows = []
            for bucket in seed_payload["buckets"]:
                row = bucket["arms"][arm]["metrics"][POSTERIOR_STRATEGY]
                rows.append(row)
                all_rows.append(row)
                bucket_rows.setdefault(str(bucket["bucket"]), []).append(row)
            seed_rows[seed] = _mean_metric_rows(rows)
        output[arm] = {
            "bucket_seed_row_mean": _mean_metric_rows(all_rows),
            "seed_metric_rows": {
                str(seed): row for seed, row in sorted(seed_rows.items())
            },
            "seed_level_intervals": _metric_intervals(
                list(seed_rows.values()),
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=3001,
            ),
            "bucket_level_intervals": {
                bucket: _metric_intervals(
                    rows,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=4001 + index,
                )
                for index, (bucket, rows) in enumerate(sorted(bucket_rows.items()))
            },
        }
    return output


def _paired_deltas(
    seed_results: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    metrics = (
        "recall_at_k",
        "precision_at_k",
        "ndcg_at_k",
        "average_precision",
        "brier_score",
        "near_miss_positive_rate",
    )
    seed_deltas = {}
    bucket_deltas = []
    for seed_payload in seed_results:
        seed = int(seed_payload["seed"])
        rows_by_metric = {metric: [] for metric in metrics}
        selected_positive_deltas = []
        for bucket in seed_payload["buckets"]:
            historical = bucket["arms"][ARM_HISTORICAL_CACHED]
            exact = bucket["arms"][ARM_EXACT_CACHED]
            historical_metrics = historical["metrics"][POSTERIOR_STRATEGY]
            exact_metrics = exact["metrics"][POSTERIOR_STRATEGY]
            row = {"seed": seed, "bucket": bucket["bucket"]}
            for metric in metrics:
                delta = round(
                    float(historical_metrics[metric]) - float(exact_metrics[metric]),
                    8,
                )
                rows_by_metric[metric].append(delta)
                row[f"{metric}_delta"] = delta
            historical_hits = int(
                historical["top_k_error_decomposition"]["selected_positive_count"]
            )
            exact_hits = int(
                exact["top_k_error_decomposition"]["selected_positive_count"]
            )
            hit_delta = historical_hits - exact_hits
            selected_positive_deltas.append(hit_delta)
            row["selected_positive_delta"] = hit_delta
            bucket_deltas.append(row)
        seed_deltas[seed] = {
            metric: round(sum(values) / len(values), 8)
            for metric, values in rows_by_metric.items()
        }
        seed_deltas[seed]["selected_positive_delta_total"] = sum(
            selected_positive_deltas
        )
    return {
        "reference_arm": ARM_EXACT_CACHED,
        "comparison_arm": ARM_HISTORICAL_CACHED,
        "interpretation": (
            "Positive deltas mean historical cached random exceeded exact-pool "
            "cached random on the same seed."
        ),
        "seed_deltas": {str(seed): row for seed, row in sorted(seed_deltas.items())},
        "bucket_deltas": bucket_deltas,
        "metric_delta_intervals": {
            metric: summarize_values(
                [float(row[metric]) for row in seed_deltas.values()],
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=5001 + index,
            )
            for index, metric in enumerate(metrics)
        },
        "selected_positive_total_delta": sum(
            int(row["selected_positive_delta"]) for row in bucket_deltas
        ),
    }


def _single_seed_references(
    cached_replay: dict[str, Any],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    source_path = (
        REPO_ROOT
        / "artifacts"
        / "backtest-arxiv-random-control-diagnosis"
        / "random-control-gap-analysis.json"
    )
    if not source_path.exists():
        return {
            "source_artifact_path": str(source_path),
            "missing": True,
            "reference_metrics": {},
            "cached_replay_vs_references": {},
        }
    payload = json.loads(source_path.read_text())
    aggregate_metrics = payload.get("aggregate_metrics") or {}
    references = {
        arm: metrics["posterior_topk"]
        for arm, metrics in aggregate_metrics.items()
        if isinstance(metrics, dict) and "posterior_topk" in metrics
    }
    replay_vs_references = {}
    for replay_arm, replay_metrics in cached_replay["aggregate_metrics"].items():
        seed_rows = replay_metrics["seed_metric_rows"]
        replay_vs_references[replay_arm] = {}
        for ref_arm, ref_metrics in sorted(references.items()):
            replay_vs_references[replay_arm][ref_arm] = _seed_delta_vs_reference(
                seed_rows,
                ref_metrics,
                bootstrap_samples=bootstrap_samples,
            )
    return {
        "source_artifact_path": str(source_path),
        "missing": False,
        "reference_metrics": references,
        "cached_replay_vs_references": replay_vs_references,
        "reference_caveat": (
            "Reference metrics are the existing complete-label seed-17 posterior "
            "top-K rows from the random-control diagnostic artifact."
        ),
    }


def _seed_delta_vs_reference(
    seed_rows: dict[str, dict[str, float | int]],
    reference_metrics: dict[str, float | int],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    metrics = (
        "recall_at_k",
        "precision_at_k",
        "ndcg_at_k",
        "average_precision",
    )
    result = {}
    for index, metric in enumerate(metrics):
        ref_value = float(reference_metrics.get(metric, 0.0))
        deltas = [
            float(row.get(metric, 0.0)) - ref_value for row in seed_rows.values()
        ]
        result[metric] = {
            **summarize_values(
                deltas,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=6001 + index,
            ),
            "reference_value": ref_value,
            "seed_fraction_above_reference": _rate(
                sum(1 for delta in deltas if delta > 0.0),
                len(deltas),
            ),
            "seed_fraction_equal_or_above_reference": _rate(
                sum(1 for delta in deltas if delta >= 0.0),
                len(deltas),
            ),
        }
    return result


def _uncertainty_summary(
    cached_replay: dict[str, Any],
    *,
    full_schedule_cache_probe: dict[str, Any],
    single_seed_references: dict[str, Any],
) -> dict[str, Any]:
    exact_interval = cached_replay["aggregate_metrics"][ARM_EXACT_CACHED][
        "seed_level_intervals"
    ]["recall_at_k"]
    historical_interval = cached_replay["aggregate_metrics"][ARM_HISTORICAL_CACHED][
        "seed_level_intervals"
    ]["recall_at_k"]
    paired_recall = cached_replay["paired_deltas"]["metric_delta_intervals"][
        "recall_at_k"
    ]
    full_historical = full_schedule_cache_probe["arms"][ARM_HISTORICAL_FULL]
    full_exact = full_schedule_cache_probe["arms"][ARM_EXACT_FULL]
    reference_metrics = single_seed_references.get("reference_metrics") or {}
    exact_reference = (
        reference_metrics.get("exact_pool_random", {}).get("recall_at_k")
        if reference_metrics
        else None
    )
    historical_reference = (
        reference_metrics.get("historical_random", {}).get("recall_at_k")
        if reference_metrics
        else None
    )
    return {
        "headline": (
            "The random/exact-pool baseline advantage should be treated as "
            "real enough to keep as the baseline, but not robust enough to use "
            "as a single-seed production claim."
        ),
        "findings": [
            {
                "finding": "full_schedule_replication_is_not_cached_complete",
                "evidence": {
                    ARM_HISTORICAL_FULL: {
                        "complete_seed_bucket_row_rate": full_historical[
                            "complete_seed_bucket_row_rate"
                        ],
                        "cache_reuse_rate": full_historical["cache_reuse_rate"],
                    },
                    ARM_EXACT_FULL: {
                        "complete_seed_bucket_row_rate": full_exact[
                            "complete_seed_bucket_row_rate"
                        ],
                        "cache_reuse_rate": full_exact["cache_reuse_rate"],
                    },
                    "estimated_paid_pairwise_completion_cost_usd": (
                        full_schedule_cache_probe[
                            "estimated_paid_pairwise_completion_cost_usd"
                        ]
                    ),
                },
            },
            {
                "finding": "cached_replay_seed_variance_is_material",
                "evidence": {
                    ARM_HISTORICAL_CACHED: historical_interval,
                    ARM_EXACT_CACHED: exact_interval,
                    "historical_minus_exact_paired_recall_delta": paired_recall,
                    "single_selected_positive_changes_mean_recall_by": 0.025,
                },
            },
            {
                "finding": "seed_17_complete_random_rows_are_not_enough_by_themselves",
                "evidence": {
                    "historical_random_seed17_complete_recall": historical_reference,
                    "exact_pool_random_seed17_complete_recall": exact_reference,
                    "complete_reference_source": single_seed_references.get(
                        "source_artifact_path"
                    ),
                },
            },
        ],
        "randomized_floor_mandatory": True,
        "recommendation": {
            "baseline_for_next_active_comparison": (
                "Use exact-pool random or historical random with posterior top-K, "
                "but compare active arms against paired random seeds rather than a "
                "single seed-17 reference."
            ),
            "randomized_floor_should_be_mandatory": True,
            "minimum_future_reporting": [
                "per-seed and per-bucket metrics",
                "seed-unit 95% confidence intervals for active-minus-random deltas",
                "label cache reuse and missing-label counts",
                "new paid calls and ledger spend by artifact directory",
                "weak-bucket exposure/oracle-cap diagnostics",
            ],
            "claim_threshold": (
                "Do not claim an active arm beats random unless the paired "
                "active-minus-random Recall@K interval is positive or the mean "
                "Recall@K gain is at least 0.025 with nonnegative nDCG/AP deltas "
                "and no missing-label caveat."
            ),
            "paid_followup": (
                "No paid pairwise calls were needed for this audit. If the team "
                "wants complete full-schedule variance instead of cached replay, "
                "the artifact gives a separate costed completion estimate that "
                "must be run through guarded pairwise-only labeling."
            ),
        },
    }


def _metric_intervals(
    rows: list[dict[str, float | int]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not rows:
        return {}
    keys = [
        key
        for key in (
            "recall_at_k",
            "precision_at_k",
            "ndcg_at_k",
            "average_precision",
            "brier_score",
            "near_miss_positive_rate",
        )
        if key in rows[0]
    ]
    return {
        key: summarize_values(
            [float(row[key]) for row in rows],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index,
        )
        for index, key in enumerate(keys)
    }


def summarize_values(
    values: Sequence[float | int],
    *,
    bootstrap_samples: int = 4000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    items = [float(value) for value in values]
    if not items:
        return {
            "count": 0,
            "mean": 0.0,
            "stddev": 0.0,
            "standard_error": 0.0,
            "normal_approx_95_ci": [0.0, 0.0],
            "bootstrap_percentile_95_ci": [0.0, 0.0],
            "min": 0.0,
            "max": 0.0,
        }
    avg = mean(items)
    sample_std = stdev(items) if len(items) > 1 else 0.0
    se = sample_std / (len(items) ** 0.5) if items else 0.0
    bootstrap = _bootstrap_mean_ci(
        items,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return {
        "count": len(items),
        "mean": round(avg, 8),
        "stddev": round(sample_std, 8),
        "standard_error": round(se, 8),
        "normal_approx_95_ci": [
            round(avg - (1.96 * se), 8),
            round(avg + (1.96 * se), 8),
        ],
        "bootstrap_percentile_95_ci": bootstrap,
        "min": round(min(items), 8),
        "max": round(max(items), 8),
    }


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1 or samples <= 0:
        value = round(float(values[0]), 8)
        return [value, value]
    rng = random.Random(seed)
    means = []
    count = len(values)
    for _ in range(samples):
        total = 0.0
        for _ in range(count):
            total += values[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    low_index = int(0.025 * (len(means) - 1))
    high_index = int(0.975 * (len(means) - 1))
    return [round(means[low_index], 8), round(means[high_index], 8)]


def _top_k_ids(predictions: Sequence[Prediction], *, k: int) -> list[str]:
    return [
        prediction.paper_id
        for prediction in sorted(
            predictions,
            key=lambda item: (item.score, item.paper_id),
            reverse=True,
        )[:k]
    ]


def _candidate_pair_count(candidate_count: int) -> int:
    return int(candidate_count * max(0, candidate_count - 1) / 2)


def _cached_candidate_count(
    candidate_ids: Sequence[str],
    available_pair_keys: set[tuple[str, str]],
) -> int:
    return sum(
        1
        for left_id, right_id in itertools.combinations(candidate_ids, 2)
        if _pair_key(left_id, right_id) in available_pair_keys
    )


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _rate(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 8) if denominator else 0.0


def _stdout_summary(payload: dict[str, Any]) -> dict[str, Any]:
    aggregate = payload["cached_replay"]["aggregate_metrics"]
    return {
        "artifact_type": payload["artifact_type"],
        "artifact_path": payload["artifact_path"],
        "paid_calls_made": payload["paid_calls_made"],
        "paid_spend_usd": payload["paid_spend_usd"],
        "known_paid_spend_before_workflow_usd": payload[
            "known_paid_spend_before_workflow_usd"
        ],
        "full_schedule_cache_probe": payload["full_schedule_cache_probe"]["arms"],
        "cached_replay_seed_level_intervals": {
            arm: {
                metric: aggregate[arm]["seed_level_intervals"][metric]
                for metric in ("recall_at_k", "ndcg_at_k", "average_precision")
            }
            for arm in (ARM_HISTORICAL_CACHED, ARM_EXACT_CACHED)
        },
        "historical_minus_exact_paired_deltas": {
            metric: payload["cached_replay"]["paired_deltas"][
                "metric_delta_intervals"
            ][metric]
            for metric in ("recall_at_k", "ndcg_at_k", "average_precision")
        },
        "recommendation": payload["recommendation"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
