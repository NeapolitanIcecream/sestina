#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_posterior_decision_shrinkage import (  # noqa: E402
    ArmSpec,
    _aggregate_comparison_sources,
    _aggregate_score_predictions,
    _comparison_label_stats,
    _default_arm_specs,
    _load_historical_schedule_comparisons,
    _mean_metric_rows,
    _validate_aggregate_comparison_sources,
)
from sestina.backtest import Prediction, compare_strategies  # noqa: E402
from sestina.backtest_budget import load_config  # noqa: E402
from sestina.backtest_runner import (  # noqa: E402
    BacktestBucket,
    _config_for_phase,
    _random_pair_schedule,
    load_dataset_manifest,
    validate_model_names,
)
from sestina.candidates import select_candidates  # noqa: E402
from sestina.diagnostics import write_json_artifact  # noqa: E402
from sestina.evsi_scheduler import posterior_top_k_predictions  # noqa: E402
from sestina.models import PairwiseComparison, Paper, ScheduledPair  # noqa: E402
from sestina.scheduler import resolve_pairwise_budget  # noqa: E402
from sestina.scheduler_followup import (  # noqa: E402
    _cached_schedule_comparisons,
    build_scheduler_only_bucket_plan,
    legacy_schedule_pairs,
    legacy_select_candidates,
    load_pointwise_papers_from_artifacts,
)


@dataclass(frozen=True, slots=True)
class ArmRun:
    spec: ArmSpec
    schedule: list[ScheduledPair]
    comparisons: list[PairwiseComparison]
    comparison_source: dict[str, Any]
    scheduler_diagnostics: dict[str, Any]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline random-control gap diagnostic for the historical arXiv pilot. "
            "This reads cached artifacts only and never calls an LLM."
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
            / "backtest-arxiv-random-control-diagnosis"
            / "random-control-gap-analysis.json"
        ),
    )
    parser.add_argument("--phase", default="pilot")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--pairwise-strength", type=float, default=2.5)
    args = parser.parse_args(argv)

    payload = analyze_random_control_gap(
        config_path=args.config,
        manifest_path=args.manifest,
        source_artifact_dir=args.source_artifact_dir,
        output_path=args.output,
        phase=args.phase,
        seed=args.seed,
        samples=args.samples,
        pairwise_strength=args.pairwise_strength,
    )
    sys.stdout.write(json.dumps(_stdout_summary(payload), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def analyze_random_control_gap(
    *,
    config_path: Path,
    manifest_path: Path,
    source_artifact_dir: Path,
    output_path: Path,
    phase: str = "pilot",
    seed: int = 17,
    samples: int = 2000,
    pairwise_strength: float = 2.5,
) -> dict[str, Any]:
    raw_config = load_config(config_path)
    phase_config = _config_for_phase(raw_config, phase=phase)["phases"][0]
    pairwise_model = str(phase_config["pairwise_model"])
    validate_model_names([pairwise_model])

    manifest = load_dataset_manifest(manifest_path)
    buckets = manifest.buckets_for_phase(phase)
    labels_by_bucket = _manifest_label_lookup(manifest.payload)
    arms = _default_arm_specs()
    bucket_results: list[dict[str, Any]] = []
    aggregate_inputs: dict[str, dict[str, list[dict[str, float | int]]]] = {
        arm.name: {} for arm in arms
    }

    for bucket in buckets:
        papers = load_pointwise_papers_from_artifacts(
            bucket,
            source_artifact_dir=source_artifact_dir,
            phase=phase,
        )
        pointwise_predictions = [
            Prediction(paper.paper_id, paper.pointwise.good_probability)
            for paper in papers
        ]
        pointwise_top_k_ids = _top_k_ids(pointwise_predictions, k=bucket.k)
        bucket_payload: dict[str, Any] = {
            "bucket": bucket.name,
            "k": bucket.k,
            "papers_total": len(papers),
            "positive_labels_total": len(bucket.relevant_ids),
            "candidate_context": _candidate_context(
                papers,
                bucket=bucket,
                pointwise_top_k_ids=pointwise_top_k_ids,
            ),
            "arms": {},
        }
        for arm in arms:
            run = _load_arm_run(
                arm,
                bucket=bucket,
                papers=papers,
                source_artifact_dir=source_artifact_dir,
                phase=phase,
                seed=seed,
            )
            score_predictions = _aggregate_score_predictions(
                papers,
                run.comparisons,
                pairwise_strength=pairwise_strength,
            )
            posterior_predictions, posterior = posterior_top_k_predictions(
                papers,
                run.comparisons,
                k=bucket.k,
                pairwise_strength=pairwise_strength,
                samples=samples,
                seed=seed,
            )
            metrics = compare_strategies(
                {
                    "pointwise_only": pointwise_predictions,
                    "pairwise_score": score_predictions,
                    "posterior_topk": posterior_predictions,
                },
                relevant_ids=bucket.relevant_ids,
                k=bucket.k,
            )
            metrics_payload = {
                name: metric.to_dict() for name, metric in metrics.items()
            }
            for strategy, metric in metrics_payload.items():
                aggregate_inputs[arm.name].setdefault(strategy, []).append(metric)

            posterior_top_k_ids = _top_k_ids(posterior_predictions, k=bucket.k)
            top_k_decomposition = top_k_error_decomposition(
                predictions=posterior_predictions,
                relevant_ids=bucket.relevant_ids,
                k=bucket.k,
                schedule=run.schedule,
                pointwise_top_k_ids=pointwise_top_k_ids,
                labels_by_id=labels_by_bucket.get(bucket.name, {}),
            )
            arm_payload = {
                "comparison_source": run.comparison_source,
                "metrics": metrics_payload,
                "comparison_label_stats": _comparison_label_stats(run.comparisons),
                "positive_exposure": positive_exposure_diagnostics(
                    run.schedule,
                    relevant_ids=bucket.relevant_ids,
                    paper_count=len(papers),
                ),
                "pair_graph": pair_graph_diagnostics(
                    run.schedule,
                    relevant_ids=bucket.relevant_ids,
                    posterior_top_k_ids=posterior_top_k_ids,
                    pointwise_top_k_ids=pointwise_top_k_ids,
                ),
                "pair_label_alignment": pair_label_alignment_diagnostics(
                    run.comparisons,
                    relevant_ids=bucket.relevant_ids,
                ),
                "oracle_bounds": oracle_bounds(
                    k=bucket.k,
                    relevant_ids=bucket.relevant_ids,
                    pointwise_top_k_ids=pointwise_top_k_ids,
                    schedule=run.schedule,
                    comparisons=run.comparisons,
                ),
                "top_k_error_decomposition": top_k_decomposition,
                "posterior_topk_diagnostics": posterior.diagnostics,
                "scheduler_diagnostics_summary": _scheduler_diagnostics_summary(
                    run.scheduler_diagnostics
                ),
            }
            bucket_payload["arms"][arm.name] = arm_payload
        bucket_results.append(bucket_payload)

    aggregate_comparison_sources = _aggregate_comparison_sources(
        bucket_results,
        arm_names=[arm.name for arm in arms],
    )
    aggregate_metric_exclusions = {
        arm: summary
        for arm, summary in sorted(aggregate_comparison_sources.items())
        if not summary["aggregate_metrics_included"]
    }
    aggregate_metrics = {
        arm: {
            strategy: _mean_metric_rows(rows)
            for strategy, rows in sorted(strategies.items())
        }
        for arm, strategies in sorted(aggregate_inputs.items())
        if aggregate_comparison_sources[arm]["aggregate_metrics_included"]
    }
    _validate_aggregate_comparison_sources(
        aggregate_metrics,
        aggregate_comparison_sources,
    )
    aggregate_diagnostics = aggregate_complete_arm_diagnostics(
        bucket_results,
        comparison_sources=aggregate_comparison_sources,
    )
    decision_variant_artifacts = _load_decision_variant_artifacts()
    random_control_gap = diagnose_random_control_gap(
        bucket_results=bucket_results,
        aggregate_metrics=aggregate_metrics,
        aggregate_diagnostics=aggregate_diagnostics,
        decision_variant_artifacts=decision_variant_artifacts,
    )

    payload = {
        "artifact_type": "sestina-random-control-gap-analysis",
        "phase": phase,
        "manifest_path": str(manifest_path),
        "source_artifact_dir": str(source_artifact_dir),
        "output_path": str(output_path),
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "spend_policy": (
            "offline reconstruction from existing pointwise, pairwise, scheduler, "
            "and retrospective citation-label artifacts; no LLM calls are made"
        ),
        "label_policy": {
            "future_labels_used_as_model_features": False,
            "future_labels_used_for_scheduling": False,
            "future_labels_used_for_retrospective_diagnostics_only": True,
        },
        "pairwise_model_validated_from_config": pairwise_model,
        "analysis_parameters": {
            "seed": seed,
            "posterior_samples": samples,
            "pairwise_strength": pairwise_strength,
        },
        "arms": [_arm_payload(arm) for arm in arms],
        "aggregate_comparison_sources": aggregate_comparison_sources,
        "aggregate_metric_exclusions": aggregate_metric_exclusions,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_diagnostics": aggregate_diagnostics,
        "decision_variant_artifacts": decision_variant_artifacts,
        "random_control_gap_diagnosis": random_control_gap,
        "bucket_results": bucket_results,
        "limitations": [
            "One seed and 8 historical buckets; differences of one or two papers are not production claims.",
            "Future citation labels are used only after scheduling and scoring for diagnostic decomposition.",
            "Revised active and one-shot posterior EVSI are excluded from aggregate reconstruction metrics because the current cached-label reconstruction is partial.",
            "Oracle bounds are retrospective ceilings, not attainable model-visible features.",
            "Posterior top-K Brier scores are top-K membership probabilities, not calibrated citation-good probabilities.",
        ],
        "recommended_next_experiment": {
            "direction": (
                "Do not spend on another one-shot acquisition-score tweak. The next "
                "credible paid arm needs a pre-registered coverage and reliability "
                "change that raises retrospective exposure/oracle headroom in weak "
                "buckets before labels are purchased."
            ),
            "offline_gate_before_paid_calls": [
                "Show higher pointwise-plus-touched-positive and positive-negative pair oracle caps than exact-pool random on existing buckets.",
                "Preserve a randomized floor or run paired exact-pool random seeds so a two-positive swing is not overread.",
                "Log scheduled_pair diagnostics for every reused and paid label so future complete-arm reconstruction does not depend on scheduler drift.",
            ],
            "avoid_next": [
                "another pure EVSI within-pool score variant",
                "naive proposal-pool widening",
                "simple shrinkage toward the pointwise prior",
                "soft-strength calibration as the default decision rule",
            ],
        },
    }
    write_json_artifact(output_path, payload)
    return {**payload, "artifact_path": str(output_path)}


def _load_arm_run(
    arm: ArmSpec,
    *,
    bucket: BacktestBucket,
    papers: list[Paper],
    source_artifact_dir: Path,
    phase: str,
    seed: int,
) -> ArmRun:
    if arm.source == "historical_pilot":
        selection = legacy_select_candidates(papers, k=bucket.k)
        budget = resolve_pairwise_budget(
            n=len(papers),
            candidate_size=len(selection.candidate_ids),
        )
        if arm.historical_kind == "pairwise_active":
            scheduled = legacy_schedule_pairs(
                papers,
                candidate_selection=selection,
                k=bucket.k,
                budget=budget,
                seed=seed,
            )
            schedule = scheduled.pairs
            scheduler_diagnostics = scheduled.diagnostics
        elif arm.historical_kind == "pairwise_random":
            schedule = _random_pair_schedule(
                selection,
                budget=budget,
                seed=seed + 7919,
            )
            scheduler_diagnostics = {
                "method": "historical_random_pair_schedule",
                "scheduled_total": len(schedule),
                "candidate_count": len(selection.candidate_ids),
            }
        else:
            raise ValueError(f"unsupported historical_kind {arm.historical_kind!r}")
        comparisons = _load_historical_schedule_comparisons(
            bucket,
            schedule=schedule,
            source_artifact_dir=source_artifact_dir,
            phase=phase,
            kind=str(arm.historical_kind),
        )
        return ArmRun(
            spec=arm,
            schedule=schedule,
            comparisons=comparisons,
            comparison_source={
                "source": arm.source,
                "historical_kind": arm.historical_kind,
                "artifact_dir": str(source_artifact_dir),
                "scheduled_pairwise_total": len(schedule),
                "cached_pairwise_labels_available": len(comparisons),
                "missing_pairwise_labels": len(schedule) - len(comparisons),
                "partial": len(comparisons) != len(schedule),
            },
            scheduler_diagnostics=scheduler_diagnostics,
        )

    if arm.source != "followup" or arm.scheduler_kind is None:
        raise ValueError(f"unsupported arm source for {arm.name}")
    if arm.artifact_dir is None:
        raise ValueError(f"missing artifact_dir for {arm.name}")
    plan = build_scheduler_only_bucket_plan(
        bucket,
        source_artifact_dir=source_artifact_dir,
        followup_artifact_dir=arm.artifact_dir,
        phase=phase,
        seed=seed,
        scheduler_kind=arm.scheduler_kind,
    )
    comparisons = _cached_schedule_comparisons(plan)
    missing = len(plan.schedule) - len(comparisons)
    return ArmRun(
        spec=arm,
        schedule=plan.schedule,
        comparisons=comparisons,
        comparison_source={
            "source": arm.source,
            "scheduler_kind": arm.scheduler_kind,
            "artifact_dir": str(arm.artifact_dir),
            "scheduled_pairwise_total": len(plan.schedule),
            "cached_pairwise_labels_available": len(comparisons),
            "missing_pairwise_labels": missing,
            "partial": missing > 0,
            "reuse_stats": plan.reusable_stats,
        },
        scheduler_diagnostics=plan.diagnostics,
    )


def positive_exposure_diagnostics(
    schedule: Sequence[ScheduledPair],
    *,
    relevant_ids: set[str],
    paper_count: int,
) -> dict[str, Any]:
    touched = _touched_ids(schedule)
    positives_touched = touched & relevant_ids
    pairs_touching_positive = 0
    positive_negative_pairs = 0
    positive_positive_pairs = 0
    negative_negative_pairs = 0
    for pair in schedule:
        left_positive = pair.left_id in relevant_ids
        right_positive = pair.right_id in relevant_ids
        if left_positive or right_positive:
            pairs_touching_positive += 1
        if left_positive and right_positive:
            positive_positive_pairs += 1
        elif left_positive or right_positive:
            positive_negative_pairs += 1
        else:
            negative_negative_pairs += 1
    return {
        "scheduled_pairs_total": len(schedule),
        "unique_papers_touched": len(touched),
        "unique_paper_touch_rate": _safe_rate(len(touched), paper_count),
        "pairs_touching_future_positive": pairs_touching_positive,
        "pair_exposure_rate": _safe_rate(pairs_touching_positive, len(schedule)),
        "unique_future_positives_touched": len(positives_touched),
        "unique_future_positive_touch_rate": _safe_rate(
            len(positives_touched),
            len(relevant_ids),
        ),
        "positive_negative_pairs": positive_negative_pairs,
        "positive_positive_pairs": positive_positive_pairs,
        "negative_negative_pairs": negative_negative_pairs,
        "touched_future_positive_ids": sorted(positives_touched),
    }


def pair_graph_diagnostics(
    schedule: Sequence[ScheduledPair],
    *,
    relevant_ids: set[str],
    posterior_top_k_ids: Sequence[str],
    pointwise_top_k_ids: Sequence[str],
) -> dict[str, Any]:
    degree = _degree_map(schedule)
    touched = set(degree)
    components = _connected_components(schedule)
    largest = max(components, key=len, default=set())
    posterior_top_k = set(posterior_top_k_ids)
    pointwise_top_k = set(pointwise_top_k_ids)
    return {
        "scheduled_pairs_total": len(schedule),
        "unique_papers_touched": len(touched),
        "component_count": len(components),
        "largest_component_size": len(largest),
        "components_with_future_positive": sum(
            1 for component in components if component & relevant_ids
        ),
        "components_with_posterior_top_k": sum(
            1 for component in components if component & posterior_top_k
        ),
        "future_positives_in_largest_component": len(largest & relevant_ids),
        "posterior_top_k_in_largest_component": len(largest & posterior_top_k),
        "pointwise_top_k_in_largest_component": len(largest & pointwise_top_k),
        "future_positive_degree": _degree_summary(relevant_ids, degree),
        "posterior_top_k_degree": _degree_summary(posterior_top_k, degree),
        "pointwise_top_k_degree": _degree_summary(pointwise_top_k, degree),
    }


def pair_label_alignment_diagnostics(
    comparisons: Sequence[PairwiseComparison],
    *,
    relevant_ids: set[str],
) -> dict[str, Any]:
    positive_negative_pairs = 0
    positive_wins = 0
    negative_wins = 0
    ties_or_uncertain = 0
    pairs_touching_positive = 0
    winner_counts = Counter(comparison.winner for comparison in comparisons)
    decisive_confidences: list[float] = []
    decisive_soft_probabilities: list[float] = []
    for comparison in comparisons:
        left_positive = comparison.left_id in relevant_ids
        right_positive = comparison.right_id in relevant_ids
        if left_positive or right_positive:
            pairs_touching_positive += 1
        if left_positive == right_positive:
            continue
        positive_negative_pairs += 1
        if comparison.winner == "tie" or comparison.winner == "uncertain":
            ties_or_uncertain += 1
            continue
        if (
            comparison.winner == "left"
            and left_positive
            or comparison.winner == "right"
            and right_positive
        ):
            positive_wins += 1
        else:
            negative_wins += 1
        decisive_confidences.append(comparison.confidence)
        if comparison.soft_probability is not None:
            decisive_soft_probabilities.append(comparison.soft_probability)
    return {
        "comparisons_total": len(comparisons),
        "winner_counts": dict(sorted(winner_counts.items())),
        "pairs_touching_future_positive": pairs_touching_positive,
        "positive_negative_pairs_with_label": positive_negative_pairs,
        "positive_wins": positive_wins,
        "negative_wins": negative_wins,
        "ties_or_uncertain": ties_or_uncertain,
        "positive_win_rate": _safe_rate(positive_wins, positive_negative_pairs),
        "negative_win_rate": _safe_rate(negative_wins, positive_negative_pairs),
        "mean_decisive_confidence_on_positive_negative_pairs": _mean(
            decisive_confidences
        ),
        "mean_decisive_soft_probability_on_positive_negative_pairs": _mean(
            decisive_soft_probabilities
        ),
    }


def oracle_bounds(
    *,
    k: int,
    relevant_ids: set[str],
    pointwise_top_k_ids: Sequence[str],
    schedule: Sequence[ScheduledPair],
    comparisons: Sequence[PairwiseComparison],
) -> dict[str, Any]:
    pointwise_positives = set(pointwise_top_k_ids) & relevant_ids
    touched_positives = _touched_ids(schedule) & relevant_ids
    posneg_exposed = _positive_negative_exposed_ids(schedule, relevant_ids)
    observed_positive_winners = _observed_positive_winner_ids(
        comparisons,
        relevant_ids,
    )
    touched_cap = set(touched_positives)
    pointwise_plus_touched = pointwise_positives | touched_positives
    pair_label_oracle = pointwise_positives | posneg_exposed
    observed_winner_cap = pointwise_positives | observed_positive_winners
    return {
        "positive_labels_total": len(relevant_ids),
        "pointwise_top_k_positive_ids": sorted(pointwise_positives),
        "pointwise_recall_at_k": _recall_cap(pointwise_positives, relevant_ids, k),
        "touched_positive_upper_bound": _oracle_row(
            touched_cap,
            relevant_ids,
            k,
        ),
        "pointwise_plus_touched_positive_upper_bound": _oracle_row(
            pointwise_plus_touched,
            relevant_ids,
            k,
        ),
        "positive_negative_pair_label_oracle_upper_bound": _oracle_row(
            pair_label_oracle,
            relevant_ids,
            k,
        ),
        "observed_positive_winner_upper_bound": _oracle_row(
            observed_winner_cap,
            relevant_ids,
            k,
        ),
        "label_interpretation_gap_vs_pair_label_oracle": round(
            _recall_cap(pair_label_oracle, relevant_ids, k)
            - _recall_cap(observed_winner_cap, relevant_ids, k),
            8,
        ),
    }


def top_k_error_decomposition(
    *,
    predictions: Sequence[Prediction],
    relevant_ids: set[str],
    k: int,
    schedule: Sequence[ScheduledPair],
    pointwise_top_k_ids: Sequence[str],
    labels_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ranked = _ranked_predictions(predictions)
    top_k = ranked[:k]
    top_k_ids = [prediction.paper_id for prediction in top_k]
    selected = set(top_k_ids)
    false_positive_ids = [paper_id for paper_id in top_k_ids if paper_id not in relevant_ids]
    false_negative_ids = sorted(relevant_ids - selected)
    degree = _degree_map(schedule)
    score_by_id = {prediction.paper_id: prediction.score for prediction in ranked}
    pointwise_rank = {
        paper_id: rank for rank, paper_id in enumerate(pointwise_top_k_ids, start=1)
    }
    return {
        "selected_positive_count": len(selected & relevant_ids),
        "selected_false_positive_count": len(false_positive_ids),
        "false_negative_count": len(false_negative_ids),
        "posterior_top_k_ids": top_k_ids,
        "selected_future_positive_ids": sorted(selected & relevant_ids),
        "false_positive_rows": [
            _paper_error_row(
                paper_id,
                score_by_id=score_by_id,
                degree=degree,
                pointwise_rank=pointwise_rank,
                labels_by_id=labels_by_id,
            )
            for paper_id in false_positive_ids
        ],
        "false_negative_rows": [
            _paper_error_row(
                paper_id,
                score_by_id=score_by_id,
                degree=degree,
                pointwise_rank=pointwise_rank,
                labels_by_id=labels_by_id,
            )
            for paper_id in false_negative_ids
        ],
    }


def aggregate_complete_arm_diagnostics(
    bucket_results: list[dict[str, Any]],
    *,
    comparison_sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    included_arms = [
        arm
        for arm, source in sorted(comparison_sources.items())
        if bool(source.get("aggregate_metrics_included"))
    ]
    return {
        "positive_exposure": {
            arm: _aggregate_positive_exposure(
                [bucket["arms"][arm]["positive_exposure"] for bucket in bucket_results]
            )
            for arm in included_arms
        },
        "pair_graph": {
            arm: _aggregate_pair_graph(
                [bucket["arms"][arm]["pair_graph"] for bucket in bucket_results]
            )
            for arm in included_arms
        },
        "pair_label_alignment": {
            arm: _aggregate_pair_label_alignment(
                [
                    bucket["arms"][arm]["pair_label_alignment"]
                    for bucket in bucket_results
                ]
            )
            for arm in included_arms
        },
        "oracle_bounds": {
            arm: _aggregate_oracle_bounds(
                [bucket["arms"][arm]["oracle_bounds"] for bucket in bucket_results]
            )
            for arm in included_arms
        },
        "top_k_error_decomposition": {
            arm: _aggregate_top_k_errors(
                [
                    bucket["arms"][arm]["top_k_error_decomposition"]
                    for bucket in bucket_results
                ]
            )
            for arm in included_arms
        },
    }


def diagnose_random_control_gap(
    *,
    bucket_results: list[dict[str, Any]],
    aggregate_metrics: dict[str, dict[str, dict[str, float | int]]],
    aggregate_diagnostics: dict[str, Any],
    decision_variant_artifacts: dict[str, Any],
) -> dict[str, Any]:
    posterior_rows = {
        arm: metrics["posterior_topk"]
        for arm, metrics in aggregate_metrics.items()
        if "posterior_topk" in metrics
    }
    ranked_by_recall = sorted(
        posterior_rows,
        key=lambda arm: (
            float(posterior_rows[arm]["recall_at_k"]),
            float(posterior_rows[arm]["ndcg_at_k"]),
            float(posterior_rows[arm]["average_precision"]),
        ),
        reverse=True,
    )
    exposure = aggregate_diagnostics["positive_exposure"]
    oracle = aggregate_diagnostics["oracle_bounds"]
    label_alignment = aggregate_diagnostics["pair_label_alignment"]
    graph = aggregate_diagnostics["pair_graph"]

    exact_vs = {
        arm: _positive_hit_gap(bucket_results, "exact_pool_random", arm)
        for arm in posterior_rows
        if arm != "exact_pool_random"
    }
    historical_vs = {
        arm: _positive_hit_gap(bucket_results, "historical_random", arm)
        for arm in posterior_rows
        if arm != "historical_random"
    }
    exposure_rank = _rank_arms_by(
        exposure,
        "unique_future_positive_touch_rate",
    )
    pair_oracle_rank = _rank_arms_by(
        oracle,
        "mean_positive_negative_pair_label_oracle_recall_cap",
    )
    positive_win_rank = _rank_arms_by(label_alignment, "positive_win_rate")
    largest_component_rank = _rank_arms_by(
        graph,
        "mean_largest_component_size",
    )
    strength_notes = _decision_variant_notes(decision_variant_artifacts)
    return {
        "best_complete_posterior_topk_arms_by_recall": ranked_by_recall,
        "posterior_topk_metric_table": posterior_rows,
        "exact_pool_random_vs_other_complete_arms": exact_vs,
        "historical_random_vs_other_complete_arms": historical_vs,
        "exposure_rank_by_unique_future_positive_touch_rate": exposure_rank,
        "pair_oracle_rank_by_positive_negative_pair_cap": pair_oracle_rank,
        "pair_label_rank_by_positive_win_rate": positive_win_rank,
        "graph_rank_by_largest_component_size": largest_component_rank,
        "interpretation": _interpretation_rows(
            posterior_rows=posterior_rows,
            exposure=exposure,
            oracle=oracle,
            label_alignment=label_alignment,
            exact_vs=exact_vs,
            strength_notes=strength_notes,
        ),
        "next_direction": {
            "summary": (
                "Random/exact-pool random remains the right small-budget baseline. "
                "The active arms do not show a consistent exposure, graph, or "
                "label-interpretation advantage over random controls, and the "
                "observed recall gaps are only a few positives across 40 labels."
            ),
            "credible_change_before_more_spend": (
                "A future paid arm should first demonstrate, offline, that it "
                "raises weak-bucket exposure and positive-negative oracle headroom "
                "without discarding the randomized floor."
            ),
        },
    }


def _interpretation_rows(
    *,
    posterior_rows: dict[str, dict[str, float | int]],
    exposure: dict[str, dict[str, Any]],
    oracle: dict[str, dict[str, Any]],
    label_alignment: dict[str, dict[str, Any]],
    exact_vs: dict[str, dict[str, Any]],
    strength_notes: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exact = posterior_rows.get("exact_pool_random", {})
    historical = posterior_rows.get("historical_random", {})
    if exact and historical:
        rows.append(
            {
                "finding": "random_controls_are_still_the_best_complete_small_budget_rows",
                "evidence": {
                    "exact_pool_random_recall": exact.get("recall_at_k"),
                    "historical_random_recall": historical.get("recall_at_k"),
                    "exact_pool_random_ndcg": exact.get("ndcg_at_k"),
                    "historical_random_ndcg": historical.get("ndcg_at_k"),
                    "exact_pool_random_ap": exact.get("average_precision"),
                    "historical_random_ap": historical.get("average_precision"),
                },
            }
        )
    exact_exposure = exposure.get("exact_pool_random", {})
    higher_exposure_arms = [
        arm
        for arm, row in exposure.items()
        if arm != "exact_pool_random"
        and float(row.get("unique_future_positive_touch_rate", 0.0))
        > float(exact_exposure.get("unique_future_positive_touch_rate", 0.0))
    ]
    rows.append(
        {
            "finding": "exact_pool_random_is_not_explained_by_positive_exposure_alone",
            "evidence": {
                "exact_pool_random_unique_future_positive_touch_rate": (
                    exact_exposure.get("unique_future_positive_touch_rate")
                ),
                "arms_with_higher_positive_touch_rate": higher_exposure_arms,
            },
        }
    )
    exact_oracle = oracle.get("exact_pool_random", {})
    rows.append(
        {
            "finding": "oracle_headroom_exists_but_active_selection_does_not_convert_it",
            "evidence": {
                "exact_pool_pointwise_plus_touched_recall_cap": exact_oracle.get(
                    "mean_pointwise_plus_touched_positive_recall_cap"
                ),
                "exact_pool_pair_label_oracle_recall_cap": exact_oracle.get(
                    "mean_positive_negative_pair_label_oracle_recall_cap"
                ),
                "exact_pool_observed_positive_winner_recall_cap": exact_oracle.get(
                    "mean_observed_positive_winner_recall_cap"
                ),
            },
        }
    )
    rows.append(
        {
            "finding": "label_interpretation_is_not_a_sufficient_fix",
            "evidence": {
                "historical_random_positive_win_rate": label_alignment.get(
                    "historical_random",
                    {},
                ).get("positive_win_rate"),
                "exact_pool_random_positive_win_rate": label_alignment.get(
                    "exact_pool_random",
                    {},
                ).get("positive_win_rate"),
                "posterior_decision_and_strength_notes": strength_notes,
            },
        }
    )
    rows.append(
        {
            "finding": "random_variance_remains_material",
            "evidence": {
                arm: {
                    "positive_hit_gap_vs_exact_pool_random": row["positive_hit_gap"],
                    "bucket_count_with_positive_hit_gap": row[
                        "bucket_count_with_nonzero_gap"
                    ],
                    "dominant_bucket_share_abs_gap": row[
                        "dominant_bucket_share_abs_gap"
                    ],
                }
                for arm, row in exact_vs.items()
                if arm
                in {
                    "historical_active",
                    "sequential_evsi",
                    "cctd_gf",
                    "expanded_pool_random",
                    "targeted_outsider_random",
                }
            },
        }
    )
    return rows


def _load_decision_variant_artifacts() -> dict[str, Any]:
    specs = {
        "posterior_decision_shrinkage": (
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-posterior-decision-shrinkage"
            / "decision-shrinkage-analysis.json"
        ),
        "pairwise_strength_calibration": (
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-pairwise-strength-calibration"
            / "strength-calibration-analysis.json"
        ),
    }
    loaded: dict[str, Any] = {}
    for name, path in specs.items():
        if not path.exists():
            loaded[name] = {"artifact_path": str(path), "missing": True}
            continue
        payload = json.loads(path.read_text())
        loaded[name] = {
            "artifact_path": str(path),
            "missing": False,
            "artifact_type": payload.get("artifact_type"),
            "aggregate_metric_exclusions": payload.get(
                "aggregate_metric_exclusions",
                {},
            ),
            "aggregate_metrics": payload.get("aggregate_metrics", {}),
            "aggregate_deltas_vs_posterior_topk": payload.get(
                "aggregate_deltas_vs_posterior_topk",
                {},
            ),
            "aggregate_strength_diagnostics": payload.get(
                "aggregate_strength_diagnostics",
                {},
            ),
        }
    return loaded


def _decision_variant_notes(artifacts: dict[str, Any]) -> dict[str, Any]:
    notes: dict[str, Any] = {}
    shrinkage = artifacts.get("posterior_decision_shrinkage") or {}
    strength = artifacts.get("pairwise_strength_calibration") or {}
    if not shrinkage.get("missing"):
        notes["posterior_decision_shrinkage"] = {
            "recall_improved_complete_arms": _arms_with_positive_delta(
                shrinkage,
                metric="recall_at_k",
            ),
            "exact_pool_random_delta": (
                shrinkage.get("aggregate_deltas_vs_posterior_topk", {})
                .get("exact_pool_random", {})
            ),
            "historical_random_delta": (
                shrinkage.get("aggregate_deltas_vs_posterior_topk", {})
                .get("historical_random", {})
            ),
        }
    if not strength.get("missing"):
        notes["pairwise_strength_calibration"] = {
            "recall_improved_complete_arms": _arms_with_positive_delta(
                strength,
                metric="recall_at_k",
            ),
            "exact_pool_random_delta": (
                strength.get("aggregate_deltas_vs_posterior_topk", {})
                .get("exact_pool_random", {})
            ),
            "historical_random_delta": (
                strength.get("aggregate_deltas_vs_posterior_topk", {})
                .get("historical_random", {})
            ),
        }
    return notes


def _arms_with_positive_delta(artifact: dict[str, Any], *, metric: str) -> list[str]:
    deltas = artifact.get("aggregate_deltas_vs_posterior_topk") or {}
    return sorted(
        arm for arm, row in deltas.items() if float(row.get(metric, 0.0)) > 0.0
    )


def _positive_hit_gap(
    bucket_results: list[dict[str, Any]],
    reference_arm: str,
    comparison_arm: str,
) -> dict[str, Any]:
    rows = []
    total_gap = 0
    abs_gap = 0
    for bucket in bucket_results:
        ref = bucket["arms"][reference_arm]["top_k_error_decomposition"]
        comp = bucket["arms"][comparison_arm]["top_k_error_decomposition"]
        gap = int(ref["selected_positive_count"]) - int(comp["selected_positive_count"])
        total_gap += gap
        abs_gap += abs(gap)
        if gap:
            rows.append(
                {
                    "bucket": bucket["bucket"],
                    "reference_selected_positive_count": ref[
                        "selected_positive_count"
                    ],
                    "comparison_selected_positive_count": comp[
                        "selected_positive_count"
                    ],
                    "positive_hit_gap": gap,
                }
            )
    dominant = max((abs(int(row["positive_hit_gap"])) for row in rows), default=0)
    return {
        "reference_arm": reference_arm,
        "comparison_arm": comparison_arm,
        "positive_hit_gap": total_gap,
        "abs_positive_hit_gap": abs_gap,
        "bucket_count_with_nonzero_gap": len(rows),
        "dominant_bucket_share_abs_gap": _safe_rate(dominant, abs_gap),
        "bucket_deltas": rows,
    }


def _candidate_context(
    papers: list[Paper],
    *,
    bucket: BacktestBucket,
    pointwise_top_k_ids: Sequence[str],
) -> dict[str, Any]:
    legacy = legacy_select_candidates(papers, k=bucket.k)
    current = select_candidates(papers, k=bucket.k)
    pointwise_positive_ids = set(pointwise_top_k_ids) & bucket.relevant_ids
    return {
        "pointwise_top_k_positive_count": len(pointwise_positive_ids),
        "pointwise_top_k_positive_ids": sorted(pointwise_positive_ids),
        "legacy_candidate_positive_count": len(
            set(legacy.candidate_ids) & bucket.relevant_ids
        ),
        "legacy_candidate_positive_recall_cap": _safe_rate(
            len(set(legacy.candidate_ids) & bucket.relevant_ids),
            len(bucket.relevant_ids),
        ),
        "legacy_candidate_count": len(legacy.candidate_ids),
        "current_candidate_positive_count": len(
            set(current.candidate_ids) & bucket.relevant_ids
        ),
        "current_candidate_positive_recall_cap": _safe_rate(
            len(set(current.candidate_ids) & bucket.relevant_ids),
            len(bucket.relevant_ids),
        ),
        "current_candidate_count": len(current.candidate_ids),
    }


def _aggregate_positive_exposure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives_total = sum(int(row["unique_future_positives_touched"]) for row in rows)
    scheduled_total = sum(int(row["scheduled_pairs_total"]) for row in rows)
    pairs_touching = sum(int(row["pairs_touching_future_positive"]) for row in rows)
    positive_negative_pairs = sum(int(row["positive_negative_pairs"]) for row in rows)
    unique_papers = [int(row["unique_papers_touched"]) for row in rows]
    positive_touch_rates = [
        float(row["unique_future_positive_touch_rate"]) for row in rows
    ]
    return {
        "bucket_count": len(rows),
        "scheduled_pairs_total": scheduled_total,
        "pairs_touching_future_positive": pairs_touching,
        "pair_exposure_rate": _safe_rate(pairs_touching, scheduled_total),
        "unique_future_positives_touched": positives_total,
        "unique_future_positive_touch_rate": _mean(positive_touch_rates),
        "positive_negative_pairs": positive_negative_pairs,
        "mean_unique_papers_touched": _mean(unique_papers),
        "mean_unique_paper_touch_rate": _mean(
            [float(row["unique_paper_touch_rate"]) for row in rows]
        ),
    }


def _aggregate_pair_graph(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "bucket_count": len(rows),
        "mean_component_count": _mean([int(row["component_count"]) for row in rows]),
        "mean_largest_component_size": _mean(
            [int(row["largest_component_size"]) for row in rows]
        ),
        "mean_components_with_future_positive": _mean(
            [int(row["components_with_future_positive"]) for row in rows]
        ),
        "mean_future_positive_degree": _mean(
            [
                float(row["future_positive_degree"]["mean"])
                for row in rows
            ]
        ),
        "mean_future_positive_zero_degree_count": _mean(
            [
                int(row["future_positive_degree"]["zero_degree_count"])
                for row in rows
            ]
        ),
        "mean_posterior_top_k_degree": _mean(
            [float(row["posterior_top_k_degree"]["mean"]) for row in rows]
        ),
        "mean_posterior_top_k_zero_degree_count": _mean(
            [
                int(row["posterior_top_k_degree"]["zero_degree_count"])
                for row in rows
            ]
        ),
    }


def _aggregate_pair_label_alignment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    posneg = sum(int(row["positive_negative_pairs_with_label"]) for row in rows)
    positive_wins = sum(int(row["positive_wins"]) for row in rows)
    negative_wins = sum(int(row["negative_wins"]) for row in rows)
    ties = sum(int(row["ties_or_uncertain"]) for row in rows)
    return {
        "bucket_count": len(rows),
        "positive_negative_pairs_with_label": posneg,
        "positive_wins": positive_wins,
        "negative_wins": negative_wins,
        "ties_or_uncertain": ties,
        "positive_win_rate": _safe_rate(positive_wins, posneg),
        "negative_win_rate": _safe_rate(negative_wins, posneg),
        "mean_decisive_confidence_on_positive_negative_pairs": _mean(
            [
                float(row["mean_decisive_confidence_on_positive_negative_pairs"])
                for row in rows
            ]
        ),
        "mean_decisive_soft_probability_on_positive_negative_pairs": _mean(
            [
                float(row["mean_decisive_soft_probability_on_positive_negative_pairs"])
                for row in rows
            ]
        ),
    }


def _aggregate_oracle_bounds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "bucket_count": len(rows),
        "mean_pointwise_recall_at_k": _mean(
            [float(row["pointwise_recall_at_k"]) for row in rows]
        ),
        "mean_touched_positive_recall_cap": _mean(
            [
                float(row["touched_positive_upper_bound"]["recall_cap"])
                for row in rows
            ]
        ),
        "mean_pointwise_plus_touched_positive_recall_cap": _mean(
            [
                float(
                    row["pointwise_plus_touched_positive_upper_bound"]["recall_cap"]
                )
                for row in rows
            ]
        ),
        "mean_positive_negative_pair_label_oracle_recall_cap": _mean(
            [
                float(
                    row[
                        "positive_negative_pair_label_oracle_upper_bound"
                    ]["recall_cap"]
                )
                for row in rows
            ]
        ),
        "mean_observed_positive_winner_recall_cap": _mean(
            [
                float(row["observed_positive_winner_upper_bound"]["recall_cap"])
                for row in rows
            ]
        ),
        "mean_label_interpretation_gap_vs_pair_label_oracle": _mean(
            [
                float(row["label_interpretation_gap_vs_pair_label_oracle"])
                for row in rows
            ]
        ),
    }


def _aggregate_top_k_errors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "bucket_count": len(rows),
        "selected_positive_total": sum(
            int(row["selected_positive_count"]) for row in rows
        ),
        "selected_false_positive_total": sum(
            int(row["selected_false_positive_count"]) for row in rows
        ),
        "false_negative_total": sum(int(row["false_negative_count"]) for row in rows),
        "mean_selected_positive_count": _mean(
            [int(row["selected_positive_count"]) for row in rows]
        ),
    }


def _rank_arms_by(rows_by_arm: dict[str, dict[str, Any]], key: str) -> list[str]:
    return sorted(
        rows_by_arm,
        key=lambda arm: float(rows_by_arm[arm].get(key, 0.0)),
        reverse=True,
    )


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


def _scheduler_diagnostics_summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    coverage = diagnostics.get("coverage") or {}
    isolation = diagnostics.get("isolation") or {}
    return {
        "scheduled_total": diagnostics.get("scheduled_total"),
        "candidate_count": diagnostics.get("candidate_count"),
        "pairs_considered": diagnostics.get("pairs_considered"),
        "purpose_counts": diagnostics.get("purpose_counts")
        or coverage.get("purpose_counts"),
        "coverage": coverage,
        "isolation": {
            key: isolation.get(key)
            for key in (
                "unique_papers_touched",
                "plausible_top_k_papers_touched",
                "plausible_top_k_degree_distribution",
                "connected_components",
                "retrospective_future_positive_exposure",
                "positive_vs_negative_pairwise_win_rate",
            )
            if key in isolation
        },
    }


def _paper_error_row(
    paper_id: str,
    *,
    score_by_id: dict[str, float],
    degree: Counter[str],
    pointwise_rank: dict[str, int],
    labels_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    labels = labels_by_id.get(paper_id, {})
    return {
        "paper_id": paper_id,
        "posterior_top_k_score": round(float(score_by_id.get(paper_id, 0.0)), 8),
        "pair_degree": int(degree.get(paper_id, 0)),
        "touched_by_scheduled_pair": int(degree.get(paper_id, 0)) > 0,
        "pointwise_top_k_rank": pointwise_rank.get(paper_id),
        "title": labels.get("title"),
        "citation_count": labels.get("citation_count"),
        "citation_rank": labels.get("citation_rank"),
        "citation_percentile_within_bucket": labels.get(
            "citation_percentile_within_bucket"
        ),
        "citation_positive": labels.get("citation_positive"),
    }


def _connected_components(schedule: Sequence[ScheduledPair]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = {}
    for pair in schedule:
        adjacency.setdefault(pair.left_id, set()).add(pair.right_id)
        adjacency.setdefault(pair.right_id, set()).add(pair.left_id)
    seen: set[str] = set()
    components: list[set[str]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        component: set[str] = set()
        queue = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            component.add(node)
            for neighbor in adjacency.get(node, set()):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        components.append(component)
    return components


def _degree_map(schedule: Sequence[ScheduledPair]) -> Counter[str]:
    degree: Counter[str] = Counter()
    for pair in schedule:
        degree[pair.left_id] += 1
        degree[pair.right_id] += 1
    return degree


def _degree_summary(ids: Iterable[str], degree: Counter[str]) -> dict[str, Any]:
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


def _positive_negative_exposed_ids(
    schedule: Sequence[ScheduledPair],
    relevant_ids: set[str],
) -> set[str]:
    exposed: set[str] = set()
    for pair in schedule:
        left_positive = pair.left_id in relevant_ids
        right_positive = pair.right_id in relevant_ids
        if left_positive == right_positive:
            continue
        exposed.add(pair.left_id if left_positive else pair.right_id)
    return exposed


def _observed_positive_winner_ids(
    comparisons: Sequence[PairwiseComparison],
    relevant_ids: set[str],
) -> set[str]:
    winners: set[str] = set()
    for comparison in comparisons:
        left_positive = comparison.left_id in relevant_ids
        right_positive = comparison.right_id in relevant_ids
        if left_positive == right_positive:
            continue
        if comparison.winner == "left" and left_positive:
            winners.add(comparison.left_id)
        elif comparison.winner == "right" and right_positive:
            winners.add(comparison.right_id)
    return winners


def _oracle_row(ids: set[str], relevant_ids: set[str], k: int) -> dict[str, Any]:
    positive_ids = sorted(ids & relevant_ids)
    return {
        "recoverable_positive_count": min(k, len(positive_ids)),
        "recoverable_positive_ids": positive_ids,
        "recall_cap": _recall_cap(set(positive_ids), relevant_ids, k),
    }


def _recall_cap(ids: set[str], relevant_ids: set[str], k: int) -> float:
    return round(min(k, len(ids & relevant_ids)) / len(relevant_ids), 8) if relevant_ids else 0.0


def _touched_ids(schedule: Sequence[ScheduledPair]) -> set[str]:
    return {
        paper_id
        for pair in schedule
        for paper_id in (pair.left_id, pair.right_id)
    }


def _top_k_ids(predictions: Sequence[Prediction], *, k: int) -> list[str]:
    return [prediction.paper_id for prediction in _ranked_predictions(predictions)[:k]]


def _ranked_predictions(predictions: Sequence[Prediction]) -> list[Prediction]:
    return sorted(predictions, key=lambda item: item.score, reverse=True)


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 8) if denominator else 0.0


def _mean(values: Iterable[int | float]) -> float:
    items = [float(value) for value in values]
    return round(sum(items) / len(items), 8) if items else 0.0


def _arm_payload(arm: ArmSpec) -> dict[str, Any]:
    return {
        "name": arm.name,
        "source": arm.source,
        "scheduler_kind": arm.scheduler_kind,
        "historical_kind": arm.historical_kind,
        "artifact_dir": str(arm.artifact_dir) if arm.artifact_dir else None,
    }


def _stdout_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_type": payload["artifact_type"],
        "artifact_path": payload["artifact_path"],
        "paid_calls_made": payload["paid_calls_made"],
        "paid_spend_usd": payload["paid_spend_usd"],
        "aggregate_metric_exclusions": payload["aggregate_metric_exclusions"],
        "aggregate_metrics": payload["aggregate_metrics"],
        "random_control_gap_diagnosis": payload["random_control_gap_diagnosis"],
        "recommended_next_experiment": payload["recommended_next_experiment"],
        "limitations": payload["limitations"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
