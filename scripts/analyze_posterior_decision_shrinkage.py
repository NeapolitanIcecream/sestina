#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sestina.aggregation import AggregationConfig, aggregate  # noqa: E402
from sestina.backtest import Prediction, compare_strategies  # noqa: E402
from sestina.backtest_budget import load_config  # noqa: E402
from sestina.backtest_runner import (  # noqa: E402
    _config_for_phase,
    _random_pair_schedule,
    load_dataset_manifest,
    validate_model_names,
)
from sestina.diagnostics import write_json_artifact  # noqa: E402
from sestina.evsi_scheduler import posterior_top_k_predictions  # noqa: E402
from sestina.models import PairwiseComparison, Paper  # noqa: E402
from sestina.posterior_decision import (  # noqa: E402
    SparsePairwiseShrinkageConfig,
    sparse_pairwise_shrunk_top_k_predictions,
)
from sestina.scheduler import resolve_pairwise_budget  # noqa: E402
from sestina.scheduler_followup import (  # noqa: E402
    _cached_schedule_comparisons,
    _load_historical_schedule_comparisons,
    build_scheduler_only_bucket_plan,
    legacy_schedule_pairs,
    legacy_select_candidates,
    load_pointwise_papers_from_artifacts,
)


@dataclass(frozen=True, slots=True)
class ArmSpec:
    name: str
    source: str
    scheduler_kind: str | None = None
    artifact_dir: Path | None = None
    historical_kind: str | None = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline analysis of degree-aware posterior top-K shrinkage. "
            "This reads cached pointwise/pairwise artifacts and never calls an LLM."
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
            / "backtest-arxiv-posterior-decision-shrinkage"
            / "decision-shrinkage-analysis.json"
        ),
    )
    parser.add_argument("--phase", default="pilot")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--prior-degree", type=float, default=2.0)
    parser.add_argument("--pairwise-strength", type=float, default=2.5)
    args = parser.parse_args(argv)

    payload = analyze_posterior_decision_shrinkage(
        config_path=args.config,
        manifest_path=args.manifest,
        source_artifact_dir=args.source_artifact_dir,
        output_path=args.output,
        phase=args.phase,
        seed=args.seed,
        samples=args.samples,
        prior_degree=args.prior_degree,
        pairwise_strength=args.pairwise_strength,
    )
    sys.stdout.write(json.dumps(_stdout_summary(payload), indent=2, sort_keys=True) + "\n")
    return 0


def analyze_posterior_decision_shrinkage(
    *,
    config_path: Path,
    manifest_path: Path,
    source_artifact_dir: Path,
    output_path: Path,
    phase: str = "pilot",
    seed: int = 17,
    samples: int = 2000,
    prior_degree: float = 2.0,
    pairwise_strength: float = 2.5,
) -> dict[str, Any]:
    raw_config = load_config(config_path)
    phase_config = _config_for_phase(raw_config, phase=phase)["phases"][0]
    pairwise_model = str(phase_config["pairwise_model"])
    validate_model_names([pairwise_model])
    manifest = load_dataset_manifest(manifest_path)
    buckets = manifest.buckets_for_phase(phase)
    arms = _default_arm_specs()
    decision_config = SparsePairwiseShrinkageConfig(
        prior_degree=prior_degree,
        pairwise_strength=pairwise_strength,
        samples=samples,
        seed=seed,
    )
    bucket_results = []
    aggregate_inputs: dict[str, dict[str, list[dict[str, float | int]]]] = {
        arm.name: {} for arm in arms
    }
    diagnostic_inputs: dict[str, list[dict[str, Any]]] = {arm.name: [] for arm in arms}

    for bucket in buckets:
        papers = load_pointwise_papers_from_artifacts(
            bucket,
            source_artifact_dir=source_artifact_dir,
            phase=phase,
        )
        bucket_payload = {
            "bucket": bucket.name,
            "k": bucket.k,
            "papers_total": len(papers),
            "positive_labels_total": len(bucket.relevant_ids),
            "arms": {},
        }
        for arm in arms:
            comparisons, comparison_source = _load_arm_comparisons(
                arm,
                bucket=bucket,
                papers=papers,
                source_artifact_dir=source_artifact_dir,
                phase=phase,
                seed=seed,
            )
            pointwise_predictions = [
                Prediction(paper.paper_id, paper.pointwise.good_probability)
                for paper in papers
            ]
            posterior_predictions, posterior = posterior_top_k_predictions(
                papers,
                comparisons,
                k=bucket.k,
                pairwise_strength=pairwise_strength,
                samples=samples,
                seed=seed,
            )
            shrinkage = sparse_pairwise_shrunk_top_k_predictions(
                papers,
                comparisons,
                k=bucket.k,
                config=decision_config,
            )
            score_predictions = _aggregate_score_predictions(
                papers,
                comparisons,
                pairwise_strength=pairwise_strength,
            )
            metrics = compare_strategies(
                {
                    "pointwise_only": pointwise_predictions,
                    "pairwise_score": score_predictions,
                    "posterior_topk": posterior_predictions,
                    "degree_shrunk_posterior_topk": shrinkage.predictions,
                },
                relevant_ids=bucket.relevant_ids,
                k=bucket.k,
            )
            metrics_payload = {
                name: metric.to_dict() for name, metric in metrics.items()
            }
            for strategy, metric in metrics_payload.items():
                aggregate_inputs[arm.name].setdefault(strategy, []).append(metric)
            arm_payload = {
                "comparison_source": comparison_source,
                "comparison_label_stats": _comparison_label_stats(comparisons),
                "posterior_topk_diagnostics": posterior.diagnostics,
                "decision_diagnostics": shrinkage.diagnostics,
                "metrics": metrics_payload,
            }
            diagnostic_inputs[arm.name].append(shrinkage.diagnostics)
            bucket_payload["arms"][arm.name] = arm_payload
        bucket_results.append(bucket_payload)

    aggregate_metrics = {
        arm: {
            strategy: _mean_metric_rows(rows)
            for strategy, rows in sorted(strategies.items())
        }
        for arm, strategies in sorted(aggregate_inputs.items())
    }
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
        arm: metrics
        for arm, metrics in sorted(aggregate_metrics.items())
        if aggregate_comparison_sources[arm]["aggregate_metrics_included"]
    }
    _validate_aggregate_comparison_sources(
        aggregate_metrics,
        aggregate_comparison_sources,
    )
    aggregate_deltas = {
        arm: _decision_delta(metrics)
        for arm, metrics in aggregate_metrics.items()
    }
    aggregate_diagnostics = {
        arm: _aggregate_decision_diagnostics(rows)
        for arm, rows in sorted(diagnostic_inputs.items())
        if aggregate_comparison_sources[arm]["aggregate_metrics_included"]
    }
    payload = {
        "artifact_type": "sestina-posterior-decision-shrinkage-analysis",
        "phase": phase,
        "manifest_path": str(manifest_path),
        "source_artifact_dir": str(source_artifact_dir),
        "output_path": str(output_path),
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "spend_policy": (
            "offline re-aggregation of existing pointwise and pairwise artifacts; "
            "no LLM calls are made"
        ),
        "pairwise_model_validated_from_config": pairwise_model,
        "decision_rule": {
            "method": "degree_shrunk_posterior_topk_membership",
            "prior_degree": prior_degree,
            "pairwise_strength": pairwise_strength,
            "samples": samples,
            "seed": seed,
            "uses_future_labels_for_decision": False,
            "retrospective_labels_used_only_for_metrics": True,
        },
        "arms": [_arm_payload(arm) for arm in arms],
        "aggregate_comparison_sources": aggregate_comparison_sources,
        "aggregate_metric_exclusions": aggregate_metric_exclusions,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_deltas_vs_posterior_topk": aggregate_deltas,
        "aggregate_decision_diagnostics": aggregate_diagnostics,
        "bucket_results": bucket_results,
        "limitations": [
            "One seed and 8 historical buckets; results are not a production claim.",
            "The rule reuses existing paid pairwise labels and does not test a new acquisition schedule.",
            "The prior-degree value was selected before this run as a conservative sparse-evidence default, not tuned by bucket labels.",
            "Brier scores for top-K membership rules are decision probabilities, not calibrated paper-good probabilities.",
        ],
    }
    write_json_artifact(output_path, payload)
    return {**payload, "artifact_path": str(output_path)}


def _default_arm_specs() -> list[ArmSpec]:
    return [
        ArmSpec(
            name="historical_active",
            source="historical_pilot",
            historical_kind="pairwise_active",
        ),
        ArmSpec(
            name="historical_random",
            source="historical_pilot",
            historical_kind="pairwise_random",
        ),
        ArmSpec(
            name="revised_active",
            source="followup",
            scheduler_kind="quota",
            artifact_dir=REPO_ROOT / "artifacts" / "backtest-arxiv-scheduler-followup-live",
        ),
        ArmSpec(
            name="posterior_topk_evsi",
            source="followup",
            scheduler_kind="evsi",
            artifact_dir=REPO_ROOT / "artifacts" / "backtest-arxiv-evsi-followup-live",
        ),
        ArmSpec(
            name="exact_pool_random",
            source="followup",
            scheduler_kind="exact_pool_random",
            artifact_dir=REPO_ROOT / "artifacts" / "backtest-arxiv-exact-pool-random-live",
        ),
        ArmSpec(
            name="sequential_evsi",
            source="followup",
            scheduler_kind="sequential_evsi",
            artifact_dir=REPO_ROOT / "artifacts" / "backtest-arxiv-sequential-evsi-live",
        ),
        ArmSpec(
            name="cctd_gf",
            source="followup",
            scheduler_kind="cctd_gf",
            artifact_dir=REPO_ROOT / "artifacts" / "backtest-arxiv-cctd-gf-live",
        ),
        ArmSpec(
            name="expanded_pool_random",
            source="followup",
            scheduler_kind="expanded_pool_random",
            artifact_dir=REPO_ROOT / "artifacts" / "backtest-arxiv-expanded-pool-random-live",
        ),
        ArmSpec(
            name="targeted_outsider_random",
            source="followup",
            scheduler_kind="targeted_outsider_random",
            artifact_dir=REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-targeted-outsider-random-live",
        ),
    ]


def _load_arm_comparisons(
    arm: ArmSpec,
    *,
    bucket: Any,
    papers: list[Paper],
    source_artifact_dir: Path,
    phase: str,
    seed: int,
) -> tuple[list[PairwiseComparison], dict[str, Any]]:
    if arm.source == "historical_pilot":
        comparisons = _load_historical_arm_comparisons(
            arm,
            bucket=bucket,
            papers=papers,
            source_artifact_dir=source_artifact_dir,
            phase=phase,
            seed=seed,
        )
        return comparisons, {
            "source": arm.source,
            "historical_kind": arm.historical_kind,
            "artifact_dir": str(source_artifact_dir),
            "scheduled_pairwise_total": len(comparisons),
            "cached_pairwise_labels_available": len(comparisons),
            "missing_pairwise_labels": 0,
            "partial": False,
        }
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
    return comparisons, {
        "source": arm.source,
        "scheduler_kind": arm.scheduler_kind,
        "artifact_dir": str(arm.artifact_dir),
        "scheduled_pairwise_total": len(plan.schedule),
        "cached_pairwise_labels_available": len(comparisons),
        "missing_pairwise_labels": missing,
        "partial": missing > 0,
        "reuse_stats": plan.reusable_stats,
    }


def _load_historical_arm_comparisons(
    arm: ArmSpec,
    *,
    bucket: Any,
    papers: list[Paper],
    source_artifact_dir: Path,
    phase: str,
    seed: int,
) -> list[PairwiseComparison]:
    selection = legacy_select_candidates(papers, k=bucket.k)
    budget = resolve_pairwise_budget(
        n=len(papers),
        candidate_size=len(selection.candidate_ids),
    )
    if arm.historical_kind == "pairwise_active":
        schedule = legacy_schedule_pairs(
            papers,
            candidate_selection=selection,
            k=bucket.k,
            budget=budget,
            seed=seed,
        ).pairs
    elif arm.historical_kind == "pairwise_random":
        schedule = _random_pair_schedule(
            selection,
            budget=budget,
            seed=seed + 7919,
        )
    else:
        raise ValueError(f"unsupported historical_kind {arm.historical_kind!r}")
    return _load_historical_schedule_comparisons(
        bucket,
        schedule=schedule,
        source_artifact_dir=source_artifact_dir,
        phase=phase,
        kind=arm.historical_kind,
    )


def _aggregate_score_predictions(
    papers: list[Paper],
    comparisons: list[PairwiseComparison],
    *,
    pairwise_strength: float,
) -> list[Prediction]:
    result = aggregate(
        papers,
        comparisons,
        config=AggregationConfig(pairwise_strength=pairwise_strength),
    )
    return [
        Prediction(paper_id, estimate.posterior_good_probability)
        for paper_id, estimate in result.estimates.items()
    ]


def _comparison_label_stats(comparisons: list[PairwiseComparison]) -> dict[str, Any]:
    winner_counts = Counter(comparison.winner for comparison in comparisons)
    compared_ids = {
        paper_id
        for comparison in comparisons
        for paper_id in (comparison.left_id, comparison.right_id)
    }
    confidences = [comparison.confidence for comparison in comparisons]
    soft_probabilities = [
        comparison.soft_probability
        for comparison in comparisons
        if comparison.soft_probability is not None
    ]
    return {
        "comparison_count": len(comparisons),
        "winner_counts": dict(sorted(winner_counts.items())),
        "tie_or_uncertain_count": winner_counts["tie"] + winner_counts["uncertain"],
        "unique_compared_paper_count": len(compared_ids),
        "mean_confidence": round(sum(confidences) / len(confidences), 6)
        if confidences
        else 0.0,
        "mean_soft_probability": round(
            sum(soft_probabilities) / len(soft_probabilities),
            6,
        )
        if soft_probabilities
        else None,
    }


def _aggregate_comparison_sources(
    bucket_results: list[dict[str, Any]],
    *,
    arm_names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for arm_name in arm_names:
        sources = [
            bucket_payload["arms"][arm_name]["comparison_source"]
            for bucket_payload in bucket_results
        ]
        partial_bucket_count = sum(bool(source.get("partial")) for source in sources)
        scheduled_total = sum(
            int(source.get("scheduled_pairwise_total") or 0) for source in sources
        )
        cached_total = sum(
            int(source.get("cached_pairwise_labels_available") or 0)
            for source in sources
        )
        missing_total = sum(
            int(source.get("missing_pairwise_labels") or 0) for source in sources
        )
        cached_counts = [
            int(source.get("cached_pairwise_labels_available") or 0)
            for source in sources
        ]
        partial = partial_bucket_count > 0
        summary: dict[str, Any] = {
            "bucket_count": len(sources),
            "source": _sorted_non_null(source.get("source") for source in sources),
            "scheduler_kind": _sorted_non_null(
                source.get("scheduler_kind") for source in sources
            ),
            "historical_kind": _sorted_non_null(
                source.get("historical_kind") for source in sources
            ),
            "artifact_dir": _sorted_non_null(
                source.get("artifact_dir") for source in sources
            ),
            "scheduled_pairwise_total": scheduled_total,
            "cached_pairwise_labels_available": cached_total,
            "missing_pairwise_labels": missing_total,
            "cached_pairwise_labels_available_min": min(cached_counts)
            if cached_counts
            else 0,
            "cached_pairwise_labels_available_max": max(cached_counts)
            if cached_counts
            else 0,
            "partial": partial,
            "partial_bucket_count": partial_bucket_count,
            "aggregate_metrics_included": not partial,
        }
        if partial:
            summary["explicit_partial_caveat"] = True
            summary["aggregate_caveat"] = (
                "excluded_from_aggregate_metrics_due_to_partial_cached_pairwise_labels"
            )
        summaries[arm_name] = summary
    return summaries


def _validate_aggregate_comparison_sources(
    aggregate_metrics: dict[str, dict[str, dict[str, float | int]]],
    aggregate_comparison_sources: dict[str, dict[str, Any]],
) -> None:
    for arm_name in aggregate_metrics:
        if arm_name not in aggregate_comparison_sources:
            raise ValueError(f"missing aggregate comparison source for {arm_name}")
        summary = aggregate_comparison_sources[arm_name]
        if bool(summary.get("partial")) and not _has_explicit_partial_caveat(summary):
            raise ValueError(
                f"aggregate arm {arm_name} uses partial pairwise labels without "
                "an explicit partial caveat"
            )


def _has_explicit_partial_caveat(summary: dict[str, Any]) -> bool:
    return bool(summary.get("explicit_partial_caveat")) and bool(
        summary.get("aggregate_caveat")
    )


def _sorted_non_null(values: Any) -> list[str]:
    return sorted({str(value) for value in values if value is not None})


def _mean_metric_rows(
    rows: list[dict[str, float | int]],
) -> dict[str, float | int]:
    if not rows:
        return {}
    keys = sorted(rows[0])
    result: dict[str, float | int] = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        if key == "k":
            result[key] = int(round(sum(values) / len(values)))
        else:
            result[key] = round(sum(values) / len(values), 8)
    result["bucket_count"] = len(rows)
    return result


def _decision_delta(
    metrics: dict[str, dict[str, float | int]],
) -> dict[str, float]:
    posterior = metrics.get("posterior_topk") or {}
    shrunk = metrics.get("degree_shrunk_posterior_topk") or {}
    keys = [
        "recall_at_k",
        "precision_at_k",
        "ndcg_at_k",
        "average_precision",
        "brier_score",
        "near_miss_positive_rate",
    ]
    return {
        key: round(float(shrunk.get(key, 0.0)) - float(posterior.get(key, 0.0)), 8)
        for key in keys
    }


def _aggregate_decision_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return {
        "bucket_count": len(rows),
        "mean_changed_vs_pairwise_topk_count": _mean_nested(
            rows,
            "top_k_comparison",
            "changed_vs_pairwise_topk_count",
        ),
        "mean_overlap_with_pairwise_topk": _mean_nested(
            rows,
            "top_k_comparison",
            "overlap_with_pairwise_topk",
        ),
        "mean_overlap_with_prior_topk": _mean_nested(
            rows,
            "top_k_comparison",
            "overlap_with_prior_topk",
        ),
        "mean_selected_shrinkage_weight": _mean_nested(
            rows,
            "coverage",
            "selected_mean_shrinkage_weight",
        ),
        "mean_selected_comparisons_used": _mean_nested(
            rows,
            "coverage",
            "selected_mean_comparisons_used",
        ),
        "selected_zero_degree_total": int(
            sum(
                int((row.get("coverage") or {}).get("selected_zero_degree_count", 0))
                for row in rows
            )
        ),
        "mean_boundary_tie_count": _mean_nested(
            rows,
            "tie_statistics",
            "boundary_tie_count",
        ),
        "mean_abs_pairwise_prior_topk_delta": _mean_nested(
            rows,
            "uncertainty",
            "mean_abs_pairwise_prior_topk_delta",
        ),
        "selected_mean_abs_pairwise_prior_topk_delta": _mean_nested(
            rows,
            "uncertainty",
            "selected_mean_abs_pairwise_prior_topk_delta",
        ),
    }


def _mean_nested(
    rows: list[dict[str, Any]],
    section: str,
    key: str,
) -> float:
    values = [
        float((row.get(section) or {}).get(key, 0.0))
        for row in rows
    ]
    return round(sum(values) / len(values), 8) if values else 0.0


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
        "decision_rule": payload["decision_rule"],
        "aggregate_comparison_sources": payload["aggregate_comparison_sources"],
        "aggregate_metric_exclusions": payload["aggregate_metric_exclusions"],
        "aggregate_metrics": payload["aggregate_metrics"],
        "aggregate_deltas_vs_posterior_topk": payload[
            "aggregate_deltas_vs_posterior_topk"
        ],
        "aggregate_decision_diagnostics": payload["aggregate_decision_diagnostics"],
        "limitations": payload["limitations"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
