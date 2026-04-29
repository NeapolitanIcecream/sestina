from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sestina.aggregation import AggregationResult
from sestina.diagnostics import DiagnosticRecorder
from sestina.models import Paper
from sestina.posterior import TopKPosterior


@dataclass(frozen=True, slots=True)
class PaperRecommendation:
    paper_id: str
    title: str
    posterior_good_probability: float
    top_k_probability: float
    tier: str
    reasons: list[str]
    caveats: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "posterior_good_probability": self.posterior_good_probability,
            "top_k_probability": self.top_k_probability,
            "tier": self.tier,
            "reasons": list(self.reasons),
            "caveats": list(self.caveats),
        }


@dataclass(frozen=True, slots=True)
class RecommendationOutput:
    recommended_good_papers: list[PaperRecommendation]
    near_misses: list[PaperRecommendation]
    all_tiers: list[PaperRecommendation]
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "recommended_good_papers": [
                item.to_dict() for item in self.recommended_good_papers
            ],
            "near_misses": [item.to_dict() for item in self.near_misses],
            "all_tiers": [item.to_dict() for item in self.all_tiers],
            "diagnostics": dict(self.diagnostics),
        }


def build_recommendations(
    papers: list[Paper],
    aggregation: AggregationResult,
    posterior: TopKPosterior,
    *,
    k: int,
    diagnostics: DiagnosticRecorder | None = None,
) -> RecommendationOutput:
    recorder = diagnostics or DiagnosticRecorder()
    paper_by_id = {paper.paper_id: paper for paper in papers}
    rows: list[PaperRecommendation] = []
    ranked_ids = sorted(
        aggregation.estimates,
        key=lambda paper_id: (
            posterior.top_k_probabilities.get(paper_id, 0.0),
            aggregation.estimates[paper_id].posterior_good_probability,
        ),
        reverse=True,
    )
    recommended_ids = set(ranked_ids[:k])
    near_miss_limit = max(1, min(len(papers), k + int(len(papers) ** 0.5)))
    near_miss_ids = set(ranked_ids[k:near_miss_limit])
    for paper_id in ranked_ids:
        estimate = aggregation.estimates[paper_id]
        top_k_probability = posterior.top_k_probabilities.get(paper_id, 0.0)
        paper = paper_by_id[paper_id]
        tier = _tier_for(
            paper_id,
            recommended_ids=recommended_ids,
            near_miss_ids=near_miss_ids,
            top_k_probability=top_k_probability,
        )
        rows.append(
            PaperRecommendation(
                paper_id=paper_id,
                title=paper.title,
                posterior_good_probability=estimate.posterior_good_probability,
                top_k_probability=top_k_probability,
                tier=tier,
                reasons=_reasons_for(paper, estimate.comparisons_used),
                caveats=_caveats_for(paper, estimate.comparisons_used, top_k_probability),
            )
        )
    recommended = [row for row in rows if row.paper_id in recommended_ids]
    near_misses = [row for row in rows if row.paper_id in near_miss_ids]
    payload = {
        "recommended_total": len(recommended),
        "near_miss_total": len(near_misses),
        "tier_counts": _tier_counts(rows),
    }
    recorder.record(
        step="output",
        code="recommendations_built",
        message="built tiered good-paper recommendations",
        data=payload,
    )
    return RecommendationOutput(
        recommended_good_papers=recommended,
        near_misses=near_misses,
        all_tiers=rows,
        diagnostics=payload,
    )


def _tier_for(
    paper_id: str,
    *,
    recommended_ids: set[str],
    near_miss_ids: set[str],
    top_k_probability: float,
) -> str:
    if top_k_probability >= 0.75:
        return "strong_yes"
    if paper_id in recommended_ids:
        return "yes"
    if top_k_probability >= 0.20 or paper_id in near_miss_ids:
        return "near_miss"
    return "unlikely"


def _reasons_for(paper: Paper, comparisons_used: int) -> list[str]:
    reasons = list(paper.pointwise.reasons[:3])
    if paper.pointwise.summary and len(reasons) < 3:
        reasons.append(paper.pointwise.summary)
    if comparisons_used:
        reasons.append(f"pairwise evidence used: {comparisons_used}")
    return reasons[:4]


def _caveats_for(
    paper: Paper,
    comparisons_used: int,
    top_k_probability: float,
) -> list[str]:
    caveats: list[str] = []
    if paper.pointwise.uncertainty >= 0.65:
        caveats.append("high pointwise uncertainty")
    if comparisons_used == 0:
        caveats.append("no pairwise comparisons ingested")
    if 0.20 <= top_k_probability <= 0.60:
        caveats.append("near-boundary posterior mass")
    return caveats


def _tier_counts(rows: list[PaperRecommendation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.tier] = counts.get(row.tier, 0) + 1
    return counts

