from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

from sestina.aggregation import AggregationConfig, aggregate
from sestina.backtest import Prediction
from sestina.diagnostics import DiagnosticRecorder
from sestina.models import (
    PairwiseComparison,
    PairwiseOrderMetadata,
    Paper,
    ScheduledPair,
)
from sestina.posterior import TopKPosterior, estimate_top_k_probabilities
from sestina.scheduler import PairSchedule, PairwiseBudget


@dataclass(frozen=True, slots=True)
class EVSISchedulerConfig:
    pairwise_strength: float = 2.5
    samples: int = 1200
    temperature: float = 1.0
    ucb_lambda: float = 1.0
    boundary_window: float = 4.0
    pool_multiplier: int = 2
    diverse_outsider_count: int | None = None
    calibration_fraction: float = 0.20
    per_item_cap: int | None = None


@dataclass(frozen=True, slots=True)
class _PosteriorItem:
    paper_id: str
    mean: float
    variance: float
    sigma: float
    top_k_probability: float
    mean_rank: float
    boundary_mass: float
    ucb: float
    metadata_bucket: str


@dataclass(frozen=True, slots=True)
class _AcquisitionProposal:
    left_id: str
    right_id: str
    score: float
    purpose: str
    diagnostics: dict[str, Any]


def posterior_top_k_predictions(
    papers: list[Paper],
    comparisons: list[PairwiseComparison],
    *,
    k: int,
    samples: int = 2000,
    seed: int = 0,
    pairwise_strength: float = 2.5,
    diagnostics: DiagnosticRecorder | None = None,
) -> tuple[list[Prediction], TopKPosterior]:
    """Return predictions scored by posterior top-K membership probability."""
    aggregation = aggregate(
        papers,
        comparisons,
        config=AggregationConfig(pairwise_strength=pairwise_strength),
        diagnostics=diagnostics,
    )
    posterior = estimate_top_k_probabilities(
        aggregation,
        k=k,
        samples=samples,
        seed=seed,
        diagnostics=diagnostics,
    )
    predictions = [
        Prediction(paper_id, probability)
        for paper_id, probability in posterior.top_k_probabilities.items()
    ]
    predictions.sort(key=lambda item: (item.score, item.paper_id), reverse=True)
    return predictions, posterior


def schedule_evsi_boundary_duels(
    papers: list[Paper],
    comparisons: list[PairwiseComparison],
    *,
    k: int,
    budget: PairwiseBudget,
    seed: int = 0,
    config: EVSISchedulerConfig | None = None,
    diagnostics: DiagnosticRecorder | None = None,
) -> PairSchedule:
    cfg = config or EVSISchedulerConfig()
    recorder = diagnostics or DiagnosticRecorder()
    paper_by_id = {paper.paper_id: paper for paper in papers}
    if budget.budget <= 0 or len(paper_by_id) < 2 or k <= 0:
        payload = _empty_schedule_diagnostics(k=k, budget=budget.budget)
        recorder.record(
            step="pair_scheduling",
            code="evsi_pair_scheduling_empty",
            message="no EVSI boundary-duel comparisons scheduled",
            data=payload,
        )
        return PairSchedule(pairs=[], budget=budget, diagnostics=payload)

    aggregation = aggregate(
        papers,
        comparisons,
        config=AggregationConfig(pairwise_strength=cfg.pairwise_strength),
        diagnostics=recorder,
    )
    posterior = estimate_top_k_probabilities(
        aggregation,
        k=k,
        samples=cfg.samples,
        seed=seed,
        diagnostics=recorder,
    )
    items = _posterior_items(
        papers,
        aggregation=aggregation,
        posterior=posterior,
        k=k,
        config=cfg,
    )
    pool = _dynamic_proposal_pool(items, k=k, config=cfg)
    seen_pairs = {
        _pair_key(comparison.left_id, comparison.right_id)
        for comparison in comparisons
    }
    proposals = _evsi_proposals(
        pool,
        paper_by_id=paper_by_id,
        seen_pairs=seen_pairs,
        config=cfg,
    )
    scheduled = _select_evsi_pairs(
        proposals,
        budget=budget.budget,
        seed=seed,
        calibration_fraction=cfg.calibration_fraction,
        per_item_cap=cfg.per_item_cap,
    )
    payload = {
        "candidate_count": len(pool),
        "scheduled_total": len(scheduled),
        "pairs_considered": len(proposals),
        "unique_pairs_considered": len(
            {_pair_key(proposal.left_id, proposal.right_id) for proposal in proposals}
        ),
        "budget": budget.budget,
        "k": k,
        "posterior": {
            "samples": posterior.samples,
            "average_top_k_probability": posterior.diagnostics.get(
                "average_top_k_probability"
            ),
        },
        "acquisition": {
            "method": "top_k_evsi_approximation",
            "temperature": cfg.temperature,
            "calibration_fraction": cfg.calibration_fraction,
            "pool_multiplier": cfg.pool_multiplier,
        },
        "purpose_counts": dict(
            sorted(Counter(pair.purpose for pair in scheduled).items())
        ),
        "coverage": _evsi_coverage(
            scheduled,
            item_by_id={item.paper_id: item for item in items},
        ),
    }
    recorder.record(
        step="pair_scheduling",
        code="evsi_pair_scheduling_completed",
        message="scheduled posterior top-K EVSI boundary-duel comparisons",
        data=payload,
    )
    return PairSchedule(pairs=scheduled, budget=budget, diagnostics=payload)


def _posterior_items(
    papers: list[Paper],
    *,
    aggregation: Any,
    posterior: TopKPosterior,
    k: int,
    config: EVSISchedulerConfig,
) -> list[_PosteriorItem]:
    items = []
    for paper in papers:
        estimate = aggregation.estimates[paper.paper_id]
        sigma = math.sqrt(max(estimate.variance, 1e-9))
        mean_rank = posterior.mean_sampled_rank.get(paper.paper_id, float(len(papers)))
        boundary_mass = math.exp(
            -abs(mean_rank - max(1, k)) / max(config.boundary_window, 1e-9)
        )
        items.append(
            _PosteriorItem(
                paper_id=paper.paper_id,
                mean=estimate.posterior_logit,
                variance=estimate.variance,
                sigma=sigma,
                top_k_probability=posterior.top_k_probabilities.get(
                    paper.paper_id,
                    0.0,
                ),
                mean_rank=mean_rank,
                boundary_mass=boundary_mass,
                ucb=estimate.posterior_logit + (config.ucb_lambda * sigma),
                metadata_bucket=_metadata_bucket(paper),
            )
        )
    return items


def _dynamic_proposal_pool(
    items: list[_PosteriorItem],
    *,
    k: int,
    config: EVSISchedulerConfig,
) -> list[_PosteriorItem]:
    n = len(items)
    group_size = max(k, min(n, config.pool_multiplier * max(k, 1)))
    top_by_probability = sorted(
        items,
        key=lambda item: (item.top_k_probability, item.mean, item.paper_id),
        reverse=True,
    )[:group_size]
    top_by_ucb = sorted(
        items,
        key=lambda item: (item.ucb, item.top_k_probability, item.paper_id),
        reverse=True,
    )[:group_size]
    boundary = sorted(
        items,
        key=lambda item: (item.boundary_mass, item.top_k_probability, item.paper_id),
        reverse=True,
    )[:group_size]
    selected = _ordered_unique(top_by_probability, top_by_ucb, boundary)
    outsider_count = (
        config.diverse_outsider_count
        if config.diverse_outsider_count is not None
        else max(k, group_size)
    )
    top_ids = {item.paper_id for item in top_by_probability[: max(k, 1)]}
    outsiders = [
        item
        for item in sorted(
            items,
            key=lambda item: (item.ucb, item.boundary_mass, item.paper_id),
            reverse=True,
        )
        if item.paper_id not in top_ids
    ]
    selected.extend(_diverse_prefix(outsiders, limit=outsider_count))
    return _ordered_unique(selected)


def _evsi_proposals(
    pool: list[_PosteriorItem],
    *,
    paper_by_id: dict[str, Paper],
    seen_pairs: set[tuple[str, str]],
    config: EVSISchedulerConfig,
) -> list[_AcquisitionProposal]:
    sorted_by_topk = sorted(
        pool,
        key=lambda item: (item.top_k_probability, item.mean, item.paper_id),
        reverse=True,
    )
    probable_incumbents = sum(
        1 for item in pool if item.top_k_probability >= 0.5
    )
    incumbent_count = max(1, min(len(pool), probable_incumbents))
    incumbents = {item.paper_id for item in sorted_by_topk[:incumbent_count]}
    proposals: list[_AcquisitionProposal] = []
    for left, right in itertools.combinations(pool, 2):
        key = _pair_key(left.paper_id, right.paper_id)
        if key in seen_pairs:
            continue
        probability = _sigmoid((left.mean - right.mean) / max(config.temperature, 1e-9))
        entropy = _binary_entropy(probability)
        membership_swap = (
            left.top_k_probability * (1.0 - right.top_k_probability)
            + right.top_k_probability * (1.0 - left.top_k_probability)
        )
        boundary_relevance = left.boundary_mass + right.boundary_mass
        metadata_diverse = left.metadata_bucket != right.metadata_bucket
        pair_role = (
            "incumbent_challenger"
            if (left.paper_id in incumbents) != (right.paper_id in incumbents)
            else "boundary_local"
        )
        coverage_bonus = 1.0
        if metadata_diverse:
            coverage_bonus += 0.15
        if pair_role == "incumbent_challenger":
            coverage_bonus += 0.20
        score = (
            entropy
            * (0.25 + membership_swap)
            * (0.25 + boundary_relevance)
            * coverage_bonus
        )
        purpose = "evsi_boundary_duel"
        if pair_role == "incumbent_challenger" and metadata_diverse:
            purpose = "calibration_discovery"
        proposals.append(
            _AcquisitionProposal(
                left_id=left.paper_id,
                right_id=right.paper_id,
                score=score,
                purpose=purpose,
                diagnostics={
                    "acquisition_score": round(score, 8),
                    "head_to_head_probability": round(probability, 8),
                    "head_to_head_entropy": round(entropy, 8),
                    "membership_swap_probability": round(membership_swap, 8),
                    "boundary_relevance": round(boundary_relevance, 8),
                    "coverage_bonus": round(coverage_bonus, 8),
                    "pair_role": pair_role,
                    "metadata_diverse": metadata_diverse,
                    "left_top_k_probability": round(left.top_k_probability, 8),
                    "right_top_k_probability": round(right.top_k_probability, 8),
                    "left_boundary_mass": round(left.boundary_mass, 8),
                    "right_boundary_mass": round(right.boundary_mass, 8),
                    "left_metadata_bucket": left.metadata_bucket,
                    "right_metadata_bucket": right.metadata_bucket,
                    "left_title": paper_by_id[left.paper_id].title,
                    "right_title": paper_by_id[right.paper_id].title,
                },
            )
        )
    return sorted(
        proposals,
        key=lambda proposal: (proposal.score, proposal.left_id, proposal.right_id),
        reverse=True,
    )


def _select_evsi_pairs(
    proposals: list[_AcquisitionProposal],
    *,
    budget: int,
    seed: int,
    calibration_fraction: float,
    per_item_cap: int | None,
) -> list[ScheduledPair]:
    if budget <= 0:
        return []
    target_calibration = max(0, math.floor(calibration_fraction * budget))
    selected: list[_AcquisitionProposal] = []
    selected_keys: set[tuple[str, str]] = set()
    item_counts: Counter[str] = Counter()
    cap = per_item_cap or max(
        2,
        math.ceil((2.5 * budget) / max(1, len(proposals) ** 0.5)),
    )

    def take_from(candidates: list[_AcquisitionProposal], limit: int) -> None:
        for proposal in candidates:
            if len(selected) >= budget or len(selected) >= limit:
                break
            key = _pair_key(proposal.left_id, proposal.right_id)
            if key in selected_keys:
                continue
            if (
                item_counts[proposal.left_id] >= cap
                or item_counts[proposal.right_id] >= cap
            ):
                continue
            selected.append(proposal)
            selected_keys.add(key)
            item_counts[proposal.left_id] += 1
            item_counts[proposal.right_id] += 1

    calibration = [
        proposal
        for proposal in proposals
        if proposal.purpose == "calibration_discovery"
    ]
    take_from(calibration, target_calibration)
    take_from(proposals, budget)
    if len(selected) < budget:
        # Relax the per-item cap rather than returning fewer pairs when there are
        # enough nonduplicate proposals.
        for proposal in proposals:
            if len(selected) >= budget:
                break
            key = _pair_key(proposal.left_id, proposal.right_id)
            if key in selected_keys:
                continue
            selected.append(proposal)
            selected_keys.add(key)

    rng = random.Random(seed)
    scheduled: list[ScheduledPair] = []
    for index, proposal in enumerate(selected[:budget]):
        if rng.random() < 0.5:
            shown_first, shown_second = proposal.left_id, proposal.right_id
        else:
            shown_first, shown_second = proposal.right_id, proposal.left_id
        scheduled.append(
            ScheduledPair(
                left_id=proposal.left_id,
                right_id=proposal.right_id,
                priority=round(proposal.score, 8),
                purpose=proposal.purpose,
                order=PairwiseOrderMetadata(
                    shown_first_id=shown_first,
                    shown_second_id=shown_second,
                    randomized=True,
                    seed=seed,
                    position_bias_audit=(index % 5 == 0),
                    extra={
                        "canonical_left_id": proposal.left_id,
                        "canonical_right_id": proposal.right_id,
                    },
                ),
                diagnostics=proposal.diagnostics,
            )
        )
    return scheduled


def _evsi_coverage(
    scheduled: list[ScheduledPair],
    *,
    item_by_id: dict[str, _PosteriorItem],
) -> dict[str, Any]:
    purpose_counts = Counter(pair.purpose for pair in scheduled)
    pair_role_counts = Counter(
        str(pair.diagnostics.get("pair_role", "unknown")) for pair in scheduled
    )
    metadata_cross = sum(
        1 for pair in scheduled if bool(pair.diagnostics.get("metadata_diverse"))
    )
    return {
        "purpose_counts": dict(sorted(purpose_counts.items())),
        "pair_role_counts": dict(sorted(pair_role_counts.items())),
        "incumbent_challenger_pairs": pair_role_counts.get("incumbent_challenger", 0),
        "metadata_cross_bucket_pairs": metadata_cross,
        "average_acquisition_score": round(
            sum(
                float(pair.diagnostics.get("acquisition_score", 0.0))
                for pair in scheduled
            )
            / len(scheduled),
            8,
        )
        if scheduled
        else 0.0,
        "average_boundary_relevance": round(
            sum(
                float(pair.diagnostics.get("boundary_relevance", 0.0))
                for pair in scheduled
            )
            / len(scheduled),
            8,
        )
        if scheduled
        else 0.0,
        "top_k_probability_mass": round(
            sum(item.top_k_probability for item in item_by_id.values()),
            8,
        ),
    }


def _empty_schedule_diagnostics(*, k: int, budget: int) -> dict[str, Any]:
    return {
        "candidate_count": 0,
        "scheduled_total": 0,
        "pairs_considered": 0,
        "unique_pairs_considered": 0,
        "budget": budget,
        "k": k,
        "acquisition": {"method": "top_k_evsi_approximation"},
        "purpose_counts": {},
        "coverage": {
            "purpose_counts": {},
            "pair_role_counts": {},
            "incumbent_challenger_pairs": 0,
            "metadata_cross_bucket_pairs": 0,
        },
    }


def _metadata_bucket(paper: Paper) -> str:
    for key in ("primary_category", "category", "topic", "venue", "field"):
        value = paper.metadata.get(key)
        if value:
            return f"{key}:{value}"
    categories = paper.metadata.get("categories")
    if isinstance(categories, (list, tuple)) and categories:
        return f"category:{categories[0]}"
    source = paper.metadata.get("source")
    if source:
        return f"source:{source}"
    return "unknown"


def _diverse_prefix(items: list[_PosteriorItem], *, limit: int) -> list[_PosteriorItem]:
    selected: list[_PosteriorItem] = []
    used: set[str] = set()
    for item in items:
        if len(selected) >= limit:
            break
        if item.metadata_bucket in used:
            continue
        selected.append(item)
        used.add(item.metadata_bucket)
    if len(selected) < limit:
        selected_ids = {item.paper_id for item in selected}
        selected.extend(
            item
            for item in items
            if item.paper_id not in selected_ids
        )
    return selected[:limit]


def _ordered_unique(*groups: list[_PosteriorItem]) -> list[_PosteriorItem]:
    selected: list[_PosteriorItem] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if item.paper_id in seen:
                continue
            selected.append(item)
            seen.add(item.paper_id)
    return selected


def _pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    return tuple(sorted((left_id, right_id)))


def _binary_entropy(probability: float) -> float:
    p = min(0.999999, max(0.000001, probability))
    return -(p * math.log2(p)) - ((1.0 - p) * math.log2(1.0 - p))


def _sigmoid(value: float) -> float:
    if value >= 0:
        scale = math.exp(-value)
        return 1.0 / (1.0 + scale)
    scale = math.exp(value)
    return scale / (1.0 + scale)
