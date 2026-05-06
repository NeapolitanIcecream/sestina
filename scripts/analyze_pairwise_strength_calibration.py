#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_posterior_decision_shrinkage import (  # noqa: E402
    _aggregate_comparison_sources,
    _aggregate_score_predictions,
    _comparison_label_stats,
    _default_arm_specs,
    _load_arm_comparisons,
    _mean_metric_rows,
    _validate_aggregate_comparison_sources,
)
from sestina.backtest import Prediction, compare_strategies  # noqa: E402
from sestina.backtest_budget import load_config  # noqa: E402
from sestina.backtest_runner import (  # noqa: E402
    _config_for_phase,
    load_dataset_manifest,
    validate_model_names,
)
from sestina.diagnostics import write_json_artifact  # noqa: E402
from sestina.evsi_scheduler import posterior_top_k_predictions  # noqa: E402
from sestina.pairwise_strength import (  # noqa: E402
    PairwiseStrengthCalibrationConfig,
    soft_strength_calibrated_comparisons,
)
from sestina.scheduler_followup import (  # noqa: E402
    load_pointwise_papers_from_artifacts,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline analysis of pairwise soft-strength calibration. "
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
            / "backtest-arxiv-pairwise-strength-calibration"
            / "strength-calibration-analysis.json"
        ),
    )
    parser.add_argument("--phase", default="pilot")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--pairwise-strength", type=float, default=2.5)
    parser.add_argument("--minimum-win-multiplier", type=float, default=0.5)
    parser.add_argument("--margin-exponent", type=float, default=1.0)
    parser.add_argument("--default-soft-probability", type=float, default=0.75)
    args = parser.parse_args(argv)

    payload = analyze_pairwise_strength_calibration(
        config_path=args.config,
        manifest_path=args.manifest,
        source_artifact_dir=args.source_artifact_dir,
        output_path=args.output,
        phase=args.phase,
        seed=args.seed,
        samples=args.samples,
        pairwise_strength=args.pairwise_strength,
        minimum_win_multiplier=args.minimum_win_multiplier,
        margin_exponent=args.margin_exponent,
        default_soft_probability=args.default_soft_probability,
    )
    sys.stdout.write(
        json.dumps(_stdout_summary(payload), indent=2, sort_keys=True) + "\n"
    )
    return 0


def analyze_pairwise_strength_calibration(
    *,
    config_path: Path,
    manifest_path: Path,
    source_artifact_dir: Path,
    output_path: Path,
    phase: str = "pilot",
    seed: int = 17,
    samples: int = 2000,
    pairwise_strength: float = 2.5,
    minimum_win_multiplier: float = 0.5,
    margin_exponent: float = 1.0,
    default_soft_probability: float = 0.75,
) -> dict[str, Any]:
    raw_config = load_config(config_path)
    phase_config = _config_for_phase(raw_config, phase=phase)["phases"][0]
    pairwise_model = str(phase_config["pairwise_model"])
    validate_model_names([pairwise_model])
    manifest = load_dataset_manifest(manifest_path)
    buckets = manifest.buckets_for_phase(phase)
    arms = _default_arm_specs()
    calibration_config = PairwiseStrengthCalibrationConfig(
        minimum_win_multiplier=minimum_win_multiplier,
        margin_exponent=margin_exponent,
        default_soft_probability=default_soft_probability,
    )
    bucket_results: list[dict[str, Any]] = []
    aggregate_inputs: dict[str, dict[str, list[dict[str, float | int]]]] = {
        arm.name: {} for arm in arms
    }
    strength_diagnostic_inputs: dict[str, list[dict[str, Any]]] = {
        arm.name: [] for arm in arms
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
        bucket_payload: dict[str, Any] = {
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
            posterior_predictions, posterior = posterior_top_k_predictions(
                papers,
                comparisons,
                k=bucket.k,
                pairwise_strength=pairwise_strength,
                samples=samples,
                seed=seed,
            )
            calibration = soft_strength_calibrated_comparisons(
                comparisons,
                config=calibration_config,
            )
            (
                calibrated_predictions,
                calibrated_posterior,
            ) = posterior_top_k_predictions(
                papers,
                calibration.comparisons,
                k=bucket.k,
                pairwise_strength=pairwise_strength,
                samples=samples,
                seed=seed,
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
                    "soft_strength_calibrated_posterior_topk": (
                        calibrated_predictions
                    ),
                },
                relevant_ids=bucket.relevant_ids,
                k=bucket.k,
            )
            metrics_payload = {
                name: metric.to_dict() for name, metric in metrics.items()
            }
            for strategy, metric in metrics_payload.items():
                aggregate_inputs[arm.name].setdefault(strategy, []).append(metric)
            strength_diagnostic_inputs[arm.name].append(calibration.diagnostics)
            bucket_payload["arms"][arm.name] = {
                "comparison_source": comparison_source,
                "comparison_label_stats": _comparison_label_stats(comparisons),
                "strength_calibration_diagnostics": calibration.diagnostics,
                "posterior_topk_diagnostics": posterior.diagnostics,
                "calibrated_posterior_topk_diagnostics": (
                    calibrated_posterior.diagnostics
                ),
                "decision_outputs": _decision_outputs(
                    baseline_posterior=posterior,
                    calibrated_posterior=calibrated_posterior,
                    k=bucket.k,
                ),
                "metrics": metrics_payload,
            }
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
    aggregate_strength_diagnostics = {
        arm: _aggregate_strength_diagnostics(rows)
        for arm, rows in sorted(strength_diagnostic_inputs.items())
        if aggregate_comparison_sources[arm]["aggregate_metrics_included"]
    }
    payload = {
        "artifact_type": "sestina-pairwise-strength-calibration-analysis",
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
        "calibration_rule": {
            "method": "soft_probability_strength_calibration",
            "pairwise_strength": pairwise_strength,
            "posterior_samples": samples,
            "posterior_seed": seed,
            "minimum_win_multiplier": minimum_win_multiplier,
            "margin_exponent": margin_exponent,
            "default_soft_probability": default_soft_probability,
            "uses_future_labels_for_calibration": False,
            "retrospective_labels_used_only_for_metrics": True,
        },
        "arms": [
            {
                "name": arm.name,
                "source": arm.source,
                "scheduler_kind": arm.scheduler_kind,
                "historical_kind": arm.historical_kind,
                "artifact_dir": str(arm.artifact_dir) if arm.artifact_dir else None,
            }
            for arm in arms
        ],
        "aggregate_comparison_sources": aggregate_comparison_sources,
        "aggregate_metric_exclusions": aggregate_metric_exclusions,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_deltas_vs_posterior_topk": {
            arm: _calibration_delta(metrics)
            for arm, metrics in aggregate_metrics.items()
        },
        "aggregate_strength_diagnostics": aggregate_strength_diagnostics,
        "bucket_results": bucket_results,
        "limitations": [
            "One seed and 8 historical buckets; results are not a production claim.",
            (
                "The rule reuses existing paid pairwise labels and does not test "
                "a new acquisition schedule."
            ),
            (
                "The fixed strength rule was selected as a conservative "
                "soft-label interpretation, not tuned by future citation labels."
            ),
            (
                "The calibration changes pairwise likelihood strength only; "
                "it does not repair missing or weak candidate exposure."
            ),
            (
                "Brier scores for top-K membership rules are decision "
                "probabilities, not calibrated paper-good probabilities."
            ),
        ],
    }
    write_json_artifact(output_path, payload)
    return {**payload, "artifact_path": str(output_path)}


def _decision_outputs(
    *,
    baseline_posterior: Any,
    calibrated_posterior: Any,
    k: int,
) -> list[dict[str, Any]]:
    paper_ids = sorted(
        set(baseline_posterior.top_k_probabilities)
        | set(calibrated_posterior.top_k_probabilities)
    )
    baseline_ranked = _ranked_probability_ids(baseline_posterior)
    calibrated_ranked = _ranked_probability_ids(calibrated_posterior)
    baseline_rank = {
        paper_id: rank for rank, paper_id in enumerate(baseline_ranked, start=1)
    }
    calibrated_rank = {
        paper_id: rank for rank, paper_id in enumerate(calibrated_ranked, start=1)
    }
    baseline_selected = set(baseline_ranked[:k])
    calibrated_selected = set(calibrated_ranked[:k])
    rows = []
    for paper_id in paper_ids:
        rows.append(
            {
                "paper_id": paper_id,
                "baseline_top_k_probability": (
                    baseline_posterior.top_k_probabilities.get(paper_id, 0.0)
                ),
                "calibrated_top_k_probability": (
                    calibrated_posterior.top_k_probabilities.get(paper_id, 0.0)
                ),
                "probability_delta": round(
                    calibrated_posterior.top_k_probabilities.get(paper_id, 0.0)
                    - baseline_posterior.top_k_probabilities.get(paper_id, 0.0),
                    8,
                ),
                "baseline_mean_sampled_rank": (
                    baseline_posterior.mean_sampled_rank.get(paper_id, 0.0)
                ),
                "calibrated_mean_sampled_rank": (
                    calibrated_posterior.mean_sampled_rank.get(paper_id, 0.0)
                ),
                "baseline_decision_rank": baseline_rank.get(paper_id),
                "calibrated_decision_rank": calibrated_rank.get(paper_id),
                "selected_by_baseline_posterior_topk": paper_id in baseline_selected,
                "selected_by_calibrated_posterior_topk": (
                    paper_id in calibrated_selected
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["calibrated_top_k_probability"]),
            -float(row["calibrated_mean_sampled_rank"]),
            str(row["paper_id"]),
        ),
        reverse=True,
    )
    return rows


def _ranked_probability_ids(posterior: Any) -> list[str]:
    return [
        paper_id
        for paper_id, _ in sorted(
            posterior.top_k_probabilities.items(),
            key=lambda item: (
                item[1],
                -posterior.mean_sampled_rank.get(item[0], 0.0),
                item[0],
            ),
            reverse=True,
        )
    ]


def _calibration_delta(
    metrics: dict[str, dict[str, float | int]],
) -> dict[str, float]:
    baseline = metrics.get("posterior_topk") or {}
    calibrated = metrics.get("soft_strength_calibrated_posterior_topk") or {}
    keys = [
        "recall_at_k",
        "precision_at_k",
        "ndcg_at_k",
        "average_precision",
        "brier_score",
        "near_miss_positive_rate",
    ]
    return {
        key: round(float(calibrated.get(key, 0.0)) - float(baseline.get(key, 0.0)), 8)
        for key in keys
    }


def _aggregate_strength_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [row.get("summary") or {} for row in rows]
    if not summaries:
        return {}
    return {
        "bucket_count": len(summaries),
        "comparison_count": int(
            sum(int(summary.get("comparison_count", 0)) for summary in summaries)
        ),
        "decisive_count": int(
            sum(int(summary.get("decisive_count", 0)) for summary in summaries)
        ),
        "tie_count": int(
            sum(int(summary.get("tie_count", 0)) for summary in summaries)
        ),
        "uncertain_count": int(
            sum(int(summary.get("uncertain_count", 0)) for summary in summaries)
        ),
        "missing_soft_probability_count": int(
            sum(
                int(summary.get("missing_soft_probability_count", 0))
                for summary in summaries
            )
        ),
        "mean_original_confidence": _mean_summary(
            summaries,
            "mean_original_confidence",
        ),
        "mean_calibrated_confidence": _mean_summary(
            summaries,
            "mean_calibrated_confidence",
        ),
        "mean_strength_multiplier": _mean_summary(
            summaries,
            "mean_strength_multiplier",
        ),
        "mean_decisive_strength_multiplier": _mean_summary(
            summaries,
            "mean_decisive_strength_multiplier",
        ),
        "mean_decisive_soft_margin": _mean_summary(
            summaries,
            "mean_decisive_soft_margin",
        ),
    }


def _mean_summary(summaries: list[dict[str, Any]], key: str) -> float:
    values = [float(summary.get(key, 0.0)) for summary in summaries]
    return round(sum(values) / len(values), 8) if values else 0.0


def _stdout_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_type": payload["artifact_type"],
        "artifact_path": payload["artifact_path"],
        "paid_calls_made": payload["paid_calls_made"],
        "paid_spend_usd": payload["paid_spend_usd"],
        "calibration_rule": payload["calibration_rule"],
        "aggregate_comparison_sources": payload["aggregate_comparison_sources"],
        "aggregate_metric_exclusions": payload["aggregate_metric_exclusions"],
        "aggregate_metrics": payload["aggregate_metrics"],
        "aggregate_deltas_vs_posterior_topk": payload[
            "aggregate_deltas_vs_posterior_topk"
        ],
        "aggregate_strength_diagnostics": payload[
            "aggregate_strength_diagnostics"
        ],
        "limitations": payload["limitations"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
