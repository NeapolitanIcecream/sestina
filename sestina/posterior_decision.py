from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sestina.aggregation import (
    AggregationConfig,
    AggregationResult,
    PaperEstimate,
    aggregate,
)
from sestina.backtest import Prediction
from sestina.diagnostics import DiagnosticRecorder
from sestina.models import PairwiseComparison, Paper
from sestina.posterior import TopKPosterior, estimate_top_k_probabilities


@dataclass(frozen=True, slots=True)
class SparsePairwiseShrinkageConfig:
    """Decision rule parameters for sparse pairwise posterior shrinkage."""

    prior_degree: float = 2.0
    pairwise_strength: float = 2.5
    samples: int = 2000
    seed: int = 17


@dataclass(frozen=True, slots=True)
class SparsePairwiseShrinkageResult:
    predictions: list[Prediction]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    pairwise_posterior: TopKPosterior | None = None
    prior_posterior: TopKPosterior | None = None
    pairwise_aggregation: AggregationResult | None = None
    prior_aggregation: AggregationResult | None = None


def sparse_pairwise_shrunk_top_k_predictions(
    papers: list[Paper],
    comparisons: list[PairwiseComparison],
    *,
    k: int,
    config: SparsePairwiseShrinkageConfig | None = None,
    diagnostics: DiagnosticRecorder | None = None,
) -> SparsePairwiseShrinkageResult:
    """Blend pairwise posterior top-K membership with pointwise-only membership.

    The blend weight is per paper:

    `comparisons_used / (comparisons_used + prior_degree)`.

    Low-degree papers therefore stay closer to the pointwise-only posterior,
    while papers with repeated pairwise evidence can move toward the pairwise
    posterior. The rule uses only model-visible pointwise assessments and
    already collected pairwise labels; it does not inspect evaluation labels.
    """
    cfg = config or SparsePairwiseShrinkageConfig()
    recorder = diagnostics or DiagnosticRecorder()
    prior_aggregation = aggregate(
        papers,
        [],
        config=AggregationConfig(pairwise_strength=cfg.pairwise_strength),
        diagnostics=recorder,
    )
    pairwise_aggregation = aggregate(
        papers,
        comparisons,
        config=AggregationConfig(pairwise_strength=cfg.pairwise_strength),
        diagnostics=recorder,
    )
    prior_posterior = estimate_top_k_probabilities(
        prior_aggregation,
        k=k,
        samples=cfg.samples,
        seed=cfg.seed,
        diagnostics=recorder,
    )
    pairwise_posterior = estimate_top_k_probabilities(
        pairwise_aggregation,
        k=k,
        samples=cfg.samples,
        seed=cfg.seed,
        diagnostics=recorder,
    )
    predictions, payload = shrunk_top_k_membership_predictions(
        prior_posterior=prior_posterior,
        pairwise_posterior=pairwise_posterior,
        pairwise_aggregation=pairwise_aggregation,
        prior_degree=cfg.prior_degree,
        k=k,
    )
    payload["rule_parameters"].update(
        {
            "pairwise_strength": cfg.pairwise_strength,
            "posterior_samples": cfg.samples,
            "posterior_seed": cfg.seed,
        }
    )
    recorder.record(
        step="posterior_decision",
        code="sparse_pairwise_shrinkage_completed",
        message="computed sparse-pairwise shrunk posterior top-K decisions",
        data={
            "method": payload["method"],
            "k": k,
            "paper_count": len(predictions),
            "prior_degree": cfg.prior_degree,
            "changed_top_k_count": payload["top_k_comparison"][
                "changed_vs_pairwise_topk_count"
            ],
        },
    )
    return SparsePairwiseShrinkageResult(
        predictions=predictions,
        diagnostics=payload,
        pairwise_posterior=pairwise_posterior,
        prior_posterior=prior_posterior,
        pairwise_aggregation=pairwise_aggregation,
        prior_aggregation=prior_aggregation,
    )


def shrunk_top_k_membership_predictions(
    *,
    prior_posterior: TopKPosterior,
    pairwise_posterior: TopKPosterior,
    pairwise_aggregation: AggregationResult,
    prior_degree: float,
    k: int,
) -> tuple[list[Prediction], dict[str, Any]]:
    """Return predictions and diagnostics for the shrinkage decision rule."""
    estimates = pairwise_aggregation.estimates
    paper_ids = sorted(
        set(prior_posterior.top_k_probabilities)
        | set(pairwise_posterior.top_k_probabilities)
        | set(estimates)
    )
    if not paper_ids or k <= 0:
        payload = _empty_decision_payload(k=k, prior_degree=prior_degree)
        return [], payload

    rows = []
    predictions = []
    denominator_prior = max(0.0, prior_degree)
    for paper_id in paper_ids:
        estimate = estimates.get(paper_id)
        comparisons_used = estimate.comparisons_used if estimate is not None else 0
        weight = (
            comparisons_used / (comparisons_used + denominator_prior)
            if comparisons_used + denominator_prior > 0.0
            else 1.0
        )
        prior_probability = prior_posterior.top_k_probabilities.get(paper_id, 0.0)
        pairwise_probability = pairwise_posterior.top_k_probabilities.get(
            paper_id,
            0.0,
        )
        score = (
            (weight * pairwise_probability)
            + ((1.0 - weight) * prior_probability)
        )
        row = _decision_row(
            paper_id=paper_id,
            estimate=estimate,
            prior_probability=prior_probability,
            pairwise_probability=pairwise_probability,
            shrinkage_weight=weight,
            shrunk_probability=score,
            prior_posterior=prior_posterior,
            pairwise_posterior=pairwise_posterior,
        )
        rows.append(row)
        predictions.append(Prediction(paper_id, round(score, 8)))

    row_by_id = {row["paper_id"]: row for row in rows}
    predictions.sort(
        key=lambda item: (
            item.score,
            row_by_id[item.paper_id]["pairwise_top_k_probability"],
            row_by_id[item.paper_id]["prior_top_k_probability"],
            -float(row_by_id[item.paper_id]["pairwise_mean_sampled_rank"]),
            row_by_id[item.paper_id]["posterior_logit"],
            item.paper_id,
        ),
        reverse=True,
    )
    ranked_ids = [prediction.paper_id for prediction in predictions]
    selected_ids = set(ranked_ids[:k])
    pairwise_ranked_ids = _rank_probability_rows(
        pairwise_posterior,
        estimates=estimates,
    )
    prior_ranked_ids = _rank_probability_rows(
        prior_posterior,
        estimates=estimates,
    )
    pairwise_selected = set(pairwise_ranked_ids[:k])
    prior_selected = set(prior_ranked_ids[:k])
    for rank, paper_id in enumerate(ranked_ids, start=1):
        row = row_by_id[paper_id]
        row["decision_rank"] = rank
        row["selected_by_shrunk_rule"] = paper_id in selected_ids
        row["selected_by_pairwise_posterior_topk"] = paper_id in pairwise_selected
        row["selected_by_prior_posterior_topk"] = paper_id in prior_selected

    ordered_rows = [row_by_id[paper_id] for paper_id in ranked_ids]
    payload = {
        "method": "degree_shrunk_posterior_topk_membership",
        "k": k,
        "rule_parameters": {
            "prior_degree": prior_degree,
            "weight_formula": (
                "comparisons_used / (comparisons_used + prior_degree)"
            ),
            "score_formula": (
                "weight * pairwise_top_k_probability + "
                "(1 - weight) * prior_top_k_probability"
            ),
            "uses_future_labels_for_decision": False,
        },
        "posterior_inputs": {
            "prior_method": prior_posterior.diagnostics.get("method"),
            "pairwise_method": pairwise_posterior.diagnostics.get("method"),
            "prior_samples": prior_posterior.samples,
            "pairwise_samples": pairwise_posterior.samples,
        },
        "coverage": _coverage_payload(ordered_rows, selected_ids=selected_ids),
        "uncertainty": _uncertainty_payload(ordered_rows, selected_ids=selected_ids),
        "tie_statistics": _tie_payload(ordered_rows, k=k),
        "top_k_comparison": {
            "shrunk_top_k_ids": ranked_ids[:k],
            "pairwise_posterior_top_k_ids": pairwise_ranked_ids[:k],
            "prior_posterior_top_k_ids": prior_ranked_ids[:k],
            "overlap_with_pairwise_topk": len(selected_ids & pairwise_selected),
            "overlap_with_prior_topk": len(selected_ids & prior_selected),
            "changed_vs_pairwise_topk_count": k - len(selected_ids & pairwise_selected),
            "changed_vs_prior_topk_count": k - len(selected_ids & prior_selected),
        },
        "decision_outputs": ordered_rows,
    }
    return predictions, payload


def _decision_row(
    *,
    paper_id: str,
    estimate: PaperEstimate | None,
    prior_probability: float,
    pairwise_probability: float,
    shrinkage_weight: float,
    shrunk_probability: float,
    prior_posterior: TopKPosterior,
    pairwise_posterior: TopKPosterior,
) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "prior_top_k_probability": round(prior_probability, 8),
        "pairwise_top_k_probability": round(pairwise_probability, 8),
        "shrunk_top_k_probability": round(shrunk_probability, 8),
        "shrinkage_weight": round(shrinkage_weight, 8),
        "comparisons_used": int(estimate.comparisons_used) if estimate else 0,
        "comparisons_won": estimate.comparisons_won if estimate else 0.0,
        "comparisons_lost": estimate.comparisons_lost if estimate else 0.0,
        "posterior_logit": estimate.posterior_logit if estimate else 0.0,
        "posterior_good_probability": (
            estimate.posterior_good_probability if estimate else 0.0
        ),
        "posterior_variance": estimate.variance if estimate else 0.0,
        "prior_mean_sampled_rank": prior_posterior.mean_sampled_rank.get(
            paper_id,
            0.0,
        ),
        "pairwise_mean_sampled_rank": pairwise_posterior.mean_sampled_rank.get(
            paper_id,
            0.0,
        ),
        "decision_rank": None,
        "selected_by_shrunk_rule": False,
        "selected_by_pairwise_posterior_topk": False,
        "selected_by_prior_posterior_topk": False,
    }


def _rank_probability_rows(
    posterior: TopKPosterior,
    *,
    estimates: dict[str, PaperEstimate],
) -> list[str]:
    return [
        paper_id
        for paper_id, _ in sorted(
            posterior.top_k_probabilities.items(),
            key=lambda item: (
                item[1],
                -posterior.mean_sampled_rank.get(item[0], 0.0),
                estimates.get(item[0]).posterior_logit
                if estimates.get(item[0]) is not None
                else 0.0,
                item[0],
            ),
            reverse=True,
        )
    ]


def _coverage_payload(
    rows: list[dict[str, Any]],
    *,
    selected_ids: set[str],
) -> dict[str, Any]:
    degrees = [int(row["comparisons_used"]) for row in rows]
    selected_degrees = [
        int(row["comparisons_used"])
        for row in rows
        if row["paper_id"] in selected_ids
    ]
    weights = [float(row["shrinkage_weight"]) for row in rows]
    selected_weights = [
        float(row["shrinkage_weight"])
        for row in rows
        if row["paper_id"] in selected_ids
    ]
    return {
        "paper_count": len(rows),
        "compared_paper_count": sum(1 for value in degrees if value > 0),
        "zero_degree_paper_count": sum(1 for value in degrees if value == 0),
        "min_comparisons_used": min(degrees) if degrees else 0,
        "max_comparisons_used": max(degrees) if degrees else 0,
        "mean_comparisons_used": round(sum(degrees) / len(degrees), 6)
        if degrees
        else 0.0,
        "selected_min_comparisons_used": min(selected_degrees)
        if selected_degrees
        else 0,
        "selected_mean_comparisons_used": round(
            sum(selected_degrees) / len(selected_degrees),
            6,
        )
        if selected_degrees
        else 0.0,
        "selected_zero_degree_count": sum(
            1 for value in selected_degrees if value == 0
        ),
        "mean_shrinkage_weight": round(sum(weights) / len(weights), 6)
        if weights
        else 0.0,
        "selected_mean_shrinkage_weight": round(
            sum(selected_weights) / len(selected_weights),
            6,
        )
        if selected_weights
        else 0.0,
        "degree_histogram": {
            str(degree): count for degree, count in sorted(Counter(degrees).items())
        },
    }


def _uncertainty_payload(
    rows: list[dict[str, Any]],
    *,
    selected_ids: set[str],
) -> dict[str, Any]:
    variances = [float(row["posterior_variance"]) for row in rows]
    selected_variances = [
        float(row["posterior_variance"])
        for row in rows
        if row["paper_id"] in selected_ids
    ]
    probability_deltas = [
        abs(
            float(row["pairwise_top_k_probability"])
            - float(row["prior_top_k_probability"])
        )
        for row in rows
    ]
    selected_deltas = [
        abs(
            float(row["pairwise_top_k_probability"])
            - float(row["prior_top_k_probability"])
        )
        for row in rows
        if row["paper_id"] in selected_ids
    ]
    return {
        "mean_posterior_variance": round(sum(variances) / len(variances), 8)
        if variances
        else 0.0,
        "selected_mean_posterior_variance": round(
            sum(selected_variances) / len(selected_variances),
            8,
        )
        if selected_variances
        else 0.0,
        "mean_abs_pairwise_prior_topk_delta": round(
            sum(probability_deltas) / len(probability_deltas),
            8,
        )
        if probability_deltas
        else 0.0,
        "selected_mean_abs_pairwise_prior_topk_delta": round(
            sum(selected_deltas) / len(selected_deltas),
            8,
        )
        if selected_deltas
        else 0.0,
    }


def _tie_payload(rows: list[dict[str, Any]], *, k: int) -> dict[str, Any]:
    scores = [float(row["shrunk_top_k_probability"]) for row in rows]
    score_counts = Counter(scores)
    duplicate_scores = sum(count for count in score_counts.values() if count > 1)
    kth_score = scores[k - 1] if 0 < k <= len(scores) else None
    boundary_tie_count = (
        sum(1 for score in scores if score == kth_score)
        if kth_score is not None
        else 0
    )
    return {
        "duplicate_score_groups": sum(1 for count in score_counts.values() if count > 1),
        "duplicate_score_paper_count": duplicate_scores,
        "kth_score": kth_score,
        "boundary_tie_count": boundary_tie_count,
    }


def _empty_decision_payload(*, k: int, prior_degree: float) -> dict[str, Any]:
    return {
        "method": "degree_shrunk_posterior_topk_membership",
        "k": k,
        "rule_parameters": {
            "prior_degree": prior_degree,
            "weight_formula": (
                "comparisons_used / (comparisons_used + prior_degree)"
            ),
            "score_formula": (
                "weight * pairwise_top_k_probability + "
                "(1 - weight) * prior_top_k_probability"
            ),
            "uses_future_labels_for_decision": False,
        },
        "posterior_inputs": {
            "prior_method": None,
            "pairwise_method": None,
            "prior_samples": 0,
            "pairwise_samples": 0,
        },
        "coverage": {
            "paper_count": 0,
            "compared_paper_count": 0,
            "zero_degree_paper_count": 0,
        },
        "uncertainty": {},
        "tie_statistics": {},
        "top_k_comparison": {
            "shrunk_top_k_ids": [],
            "pairwise_posterior_top_k_ids": [],
            "prior_posterior_top_k_ids": [],
            "overlap_with_pairwise_topk": 0,
            "overlap_with_prior_topk": 0,
            "changed_vs_pairwise_topk_count": 0,
            "changed_vs_prior_topk_count": 0,
        },
        "decision_outputs": [],
    }
