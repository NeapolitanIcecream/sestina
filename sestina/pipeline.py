from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sestina.aggregation import AggregationConfig, aggregate
from sestina.candidates import CandidateSelectionConfig, select_candidates
from sestina.diagnostics import DiagnosticRecorder, fingerprint
from sestina.models import PairwiseComparison, Paper, SelectionMode, TargetSpec
from sestina.output import build_recommendations
from sestina.posterior import estimate_top_k_probabilities
from sestina.scheduler import resolve_pairwise_budget, schedule_pairs
from sestina.targets import resolve_target


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    mode: SelectionMode = "content_only"
    candidate_size: int | None = None
    pairwise_budget: int | None = None
    seed: int = 0
    posterior_samples: int = 2000
    prior_strength: float = 3.0
    pairwise_strength: float = 2.5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PipelineConfig":
        payload = data or {}
        return cls(
            mode=payload.get("mode", "content_only"),
            candidate_size=(
                int(payload["candidate_size"])
                if payload.get("candidate_size") is not None
                else None
            ),
            pairwise_budget=(
                int(payload["pairwise_budget"])
                if payload.get("pairwise_budget") is not None
                else None
            ),
            seed=int(payload.get("seed", 0)),
            posterior_samples=int(payload.get("posterior_samples", 2000)),
            prior_strength=float(payload.get("prior_strength", 3.0)),
            pairwise_strength=float(payload.get("pairwise_strength", 2.5)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "candidate_size": self.candidate_size,
            "pairwise_budget": self.pairwise_budget,
            "seed": self.seed,
            "posterior_samples": self.posterior_samples,
            "prior_strength": self.prior_strength,
            "pairwise_strength": self.pairwise_strength,
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    payload: dict[str, Any]
    diagnostics: DiagnosticRecorder = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def run_pipeline(
    papers: list[Paper],
    target: TargetSpec,
    *,
    comparisons: list[PairwiseComparison] | None = None,
    config: PipelineConfig | dict[str, Any] | None = None,
    diagnostics: DiagnosticRecorder | None = None,
) -> PipelineResult:
    cfg = config if isinstance(config, PipelineConfig) else PipelineConfig.from_dict(config)
    recorder = diagnostics or DiagnosticRecorder()
    _validate_unique_papers(papers, diagnostics=recorder)
    resolved = resolve_target(len(papers), target, diagnostics=recorder)
    selection = select_candidates(
        papers,
        k=resolved.k,
        config=CandidateSelectionConfig(
            candidate_size=cfg.candidate_size,
            mode=cfg.mode,
        ),
        diagnostics=recorder,
    )
    budget = resolve_pairwise_budget(
        n=len(papers),
        candidate_size=len(selection.candidate_ids),
        override=cfg.pairwise_budget,
        diagnostics=recorder,
    )
    schedule = schedule_pairs(
        papers,
        candidate_selection=selection,
        k=resolved.k,
        budget=budget,
        seed=cfg.seed,
        diagnostics=recorder,
    )
    aggregation = aggregate(
        papers,
        comparisons or [],
        config=AggregationConfig(
            prior_strength=cfg.prior_strength,
            pairwise_strength=cfg.pairwise_strength,
        ),
        diagnostics=recorder,
    )
    posterior = estimate_top_k_probabilities(
        aggregation,
        k=resolved.k,
        samples=cfg.posterior_samples,
        seed=cfg.seed,
        diagnostics=recorder,
    )
    recommendations = build_recommendations(
        papers,
        aggregation,
        posterior,
        k=resolved.k,
        diagnostics=recorder,
    )
    caveats = _pipeline_caveats(
        papers_total=len(papers),
        comparisons_total=len(comparisons or []),
        scheduled_total=len(schedule.pairs),
    )
    payload = {
        "product": "Sestina",
        "target": resolved.to_dict(),
        "config": cfg.to_dict(),
        "candidate_selection": selection.to_dict(),
        "pairwise_schedule": schedule.to_dict(),
        "aggregation": aggregation.to_dict(),
        "posterior": posterior.to_dict(),
        "recommendations": recommendations.to_dict(),
        "caveats": caveats,
        "diagnostics": recorder.to_dict(),
    }
    return PipelineResult(payload=payload, diagnostics=recorder)


def _validate_unique_papers(
    papers: list[Paper],
    *,
    diagnostics: DiagnosticRecorder,
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for paper in papers:
        if paper.paper_id in seen:
            duplicates.add(paper.paper_id)
        seen.add(paper.paper_id)
    if duplicates:
        diagnostics.record(
            step="input_validation",
            code="duplicate_paper_id",
            level="error",
            message="paper ids must be unique",
            data={
                "duplicate_id_fingerprints": [
                    fingerprint(paper_id) for paper_id in sorted(duplicates)
                ]
            },
        )
        raise ValueError("paper ids must be unique")
    diagnostics.record(
        step="input_validation",
        code="input_validated",
        message="validated paper ids and input counts",
        data={"papers_total": len(papers)},
    )


def _pipeline_caveats(
    *,
    papers_total: int,
    comparisons_total: int,
    scheduled_total: int,
) -> list[str]:
    caveats: list[str] = []
    if comparisons_total == 0 and scheduled_total:
        caveats.append("pairwise schedule was produced but no judged comparisons were ingested")
    if papers_total < 10:
        caveats.append("small paper sets have coarse posterior top-K probabilities")
    return caveats
