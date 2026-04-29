from __future__ import annotations

import math
from dataclasses import dataclass, field

from sestina.diagnostics import DiagnosticRecorder
from sestina.models import PairwiseComparison, Paper


@dataclass(frozen=True, slots=True)
class PaperEstimate:
    paper_id: str
    prior_logit: float
    posterior_logit: float
    posterior_good_probability: float
    variance: float
    comparisons_used: int
    comparisons_won: float
    comparisons_lost: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "paper_id": self.paper_id,
            "prior_logit": self.prior_logit,
            "posterior_logit": self.posterior_logit,
            "posterior_good_probability": self.posterior_good_probability,
            "variance": self.variance,
            "comparisons_used": self.comparisons_used,
            "comparisons_won": self.comparisons_won,
            "comparisons_lost": self.comparisons_lost,
        }


@dataclass(frozen=True, slots=True)
class AggregationConfig:
    prior_strength: float = 3.0
    pairwise_strength: float = 2.5
    max_iterations: int = 300
    tolerance: float = 1e-7
    learning_rate: float = 0.9


@dataclass(frozen=True, slots=True)
class AggregationResult:
    estimates: dict[str, PaperEstimate]
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "estimates": {
                paper_id: estimate.to_dict()
                for paper_id, estimate in self.estimates.items()
            },
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class _Evidence:
    left_id: str
    right_id: str
    y_left: float
    weight: float
    kind: str


def aggregate(
    papers: list[Paper],
    comparisons: list[PairwiseComparison],
    *,
    config: AggregationConfig | None = None,
    diagnostics: DiagnosticRecorder | None = None,
) -> AggregationResult:
    cfg = config or AggregationConfig()
    recorder = diagnostics or DiagnosticRecorder()
    paper_by_id = {paper.paper_id: paper for paper in papers}
    priors = {
        paper.paper_id: _logit(paper.pointwise.good_probability) for paper in papers
    }
    precision = {
        paper.paper_id: _prior_precision(paper, cfg.prior_strength) for paper in papers
    }
    theta = dict(priors)
    evidence, ingestion_diag = _prepare_evidence(
        comparisons,
        known_ids=set(paper_by_id),
        pairwise_strength=cfg.pairwise_strength,
    )
    for event in ingestion_diag:
        recorder.record(**event)

    converged = False
    max_step = 0.0
    for iteration in range(cfg.max_iterations):
        gradient = {
            paper_id: -precision[paper_id] * (theta[paper_id] - priors[paper_id])
            for paper_id in theta
        }
        curvature = dict(precision)
        for item in evidence:
            delta = theta[item.left_id] - theta[item.right_id]
            probability = _sigmoid(delta)
            residual = item.y_left - probability
            gradient[item.left_id] += item.weight * residual
            gradient[item.right_id] -= item.weight * residual
            pair_curvature = item.weight * probability * (1.0 - probability)
            curvature[item.left_id] += pair_curvature
            curvature[item.right_id] += pair_curvature
        max_step = 0.0
        for paper_id in theta:
            step = cfg.learning_rate * gradient[paper_id] / max(curvature[paper_id], 1e-9)
            step = max(-1.0, min(1.0, step))
            theta[paper_id] += step
            max_step = max(max_step, abs(step))
        if max_step < cfg.tolerance:
            converged = True
            break
    else:
        iteration = cfg.max_iterations - 1

    final_curvature = dict(precision)
    counts = {
        paper.paper_id: {"used": 0, "won": 0.0, "lost": 0.0} for paper in papers
    }
    for item in evidence:
        probability = _sigmoid(theta[item.left_id] - theta[item.right_id])
        pair_curvature = item.weight * probability * (1.0 - probability)
        final_curvature[item.left_id] += pair_curvature
        final_curvature[item.right_id] += pair_curvature
        counts[item.left_id]["used"] += 1
        counts[item.right_id]["used"] += 1
        counts[item.left_id]["won"] += item.y_left * item.weight
        counts[item.left_id]["lost"] += (1.0 - item.y_left) * item.weight
        counts[item.right_id]["won"] += (1.0 - item.y_left) * item.weight
        counts[item.right_id]["lost"] += item.y_left * item.weight

    estimates = {
        paper_id: PaperEstimate(
            paper_id=paper_id,
            prior_logit=round(priors[paper_id], 8),
            posterior_logit=round(theta[paper_id], 8),
            posterior_good_probability=round(_sigmoid(theta[paper_id]), 8),
            variance=round(1.0 / max(final_curvature[paper_id], 1e-9), 8),
            comparisons_used=int(counts[paper_id]["used"]),
            comparisons_won=round(counts[paper_id]["won"], 8),
            comparisons_lost=round(counts[paper_id]["lost"], 8),
        )
        for paper_id in paper_by_id
    }
    payload = {
        "papers_total": len(papers),
        "comparisons_input_total": len(comparisons),
        "comparisons_used_total": len(evidence),
        "iterations": iteration + 1,
        "converged": converged,
        "max_step": round(max_step, 10),
        "method": "map_bayesian_bradley_terry_fractional",
        "tie_total": sum(1 for item in evidence if item.kind == "tie"),
        "uncertain_total": sum(1 for item in evidence if item.kind == "uncertain"),
        "prior_strength": cfg.prior_strength,
        "pairwise_strength": cfg.pairwise_strength,
    }
    recorder.record(
        step="aggregation",
        code="aggregation_completed",
        message="aggregated pointwise priors and pairwise evidence",
        data=payload,
    )
    return AggregationResult(estimates=estimates, diagnostics=payload)


def _prepare_evidence(
    comparisons: list[PairwiseComparison],
    *,
    known_ids: set[str],
    pairwise_strength: float,
) -> tuple[list[_Evidence], list[dict[str, object]]]:
    evidence: list[_Evidence] = []
    diagnostics: list[dict[str, object]] = []
    skipped_unknown = 0
    skipped_duplicate = 0
    seen: set[tuple[str, str, str]] = set()
    for comparison in comparisons:
        if comparison.left_id not in known_ids or comparison.right_id not in known_ids:
            skipped_unknown += 1
            continue
        if comparison.left_id == comparison.right_id:
            skipped_duplicate += 1
            continue
        key = (comparison.left_id, comparison.right_id, comparison.winner)
        if key in seen:
            skipped_duplicate += 1
            continue
        seen.add(key)
        y_left, kind = _comparison_target(comparison)
        weight = pairwise_strength * _comparison_weight(comparison)
        if weight <= 0.0:
            continue
        evidence.append(
            _Evidence(
                left_id=comparison.left_id,
                right_id=comparison.right_id,
                y_left=y_left,
                weight=weight,
                kind=kind,
            )
        )
    diagnostics.append(
        {
            "step": "comparison_ingestion",
            "code": "comparison_ingestion_completed",
            "level": "info",
            "message": "ingested optional pairwise comparisons",
            "data": {
                "input_total": len(comparisons),
                "used_total": len(evidence),
                "skipped_unknown_total": skipped_unknown,
                "skipped_duplicate_total": skipped_duplicate,
            },
        }
    )
    if skipped_unknown:
        diagnostics.append(
            {
                "step": "comparison_ingestion",
                "code": "comparison_unknown_paper",
                "level": "warning",
                "message": "skipped comparisons that reference unknown papers",
                "data": {"skipped_unknown_total": skipped_unknown},
            }
        )
    return evidence, diagnostics


def _comparison_target(comparison: PairwiseComparison) -> tuple[float, str]:
    if comparison.winner == "tie":
        return 0.5, "tie"
    if comparison.winner == "uncertain":
        return 0.5, "uncertain"
    soft = comparison.soft_probability
    if soft is None:
        soft = 0.75
    soft = max(0.5, min(0.999, soft))
    if comparison.winner == "left":
        return soft, "win"
    return 1.0 - soft, "win"


def _comparison_weight(comparison: PairwiseComparison) -> float:
    confidence = max(0.0, min(1.0, comparison.confidence))
    if comparison.winner == "tie":
        return 0.35 * confidence
    if comparison.winner == "uncertain":
        return 0.15 * confidence
    return confidence


def _prior_precision(paper: Paper, prior_strength: float) -> float:
    uncertainty_discount = 1.0 - (0.75 * paper.pointwise.uncertainty)
    return max(0.1, prior_strength * uncertainty_discount)


def _logit(probability: float) -> float:
    clipped = max(0.001, min(0.999, probability))
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0:
        scale = math.exp(-value)
        return 1.0 / (1.0 + scale)
    scale = math.exp(value)
    return scale / (1.0 + scale)

