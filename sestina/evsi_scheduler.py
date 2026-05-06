from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
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
class SequentialEVSISchedulerConfig:
    evsi: EVSISchedulerConfig = field(default_factory=EVSISchedulerConfig)
    rounds: int = 5
    batch_size: int = 4
    stop_on_novel: bool = True


@dataclass(frozen=True, slots=True)
class CCTDGFSchedulerConfig:
    evsi: EVSISchedulerConfig = field(
        default_factory=lambda: EVSISchedulerConfig(
            samples=256,
            calibration_fraction=0.0,
            per_item_cap=6,
        )
    )
    rounds: int = 4
    batch_size: int = 5
    disagreement_pairs_per_round: int = 3
    graph_floor_pairs_per_round: int = 1
    random_floor_pairs_per_round: int = 1
    high_score_fraction: float = 0.30
    sampling_temperature: float = 0.70
    stop_on_novel: bool = True


@dataclass(frozen=True, slots=True)
class TargetedOutsiderSchedulerConfig:
    evsi: EVSISchedulerConfig = field(
        default_factory=lambda: EVSISchedulerConfig(samples=1200)
    )
    top_k_anchor_multiplier: int = 1
    boundary_anchor_multiplier: int = 1
    outsider_multiplier: int = 2
    min_outsiders: int = 8
    max_outsiders: int | None = None


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


@dataclass(frozen=True, slots=True)
class _EVSIContext:
    aggregation: Any
    posterior: TopKPosterior
    items: list[_PosteriorItem]
    pool: list[_PosteriorItem]
    proposals: list[_AcquisitionProposal]


@dataclass(frozen=True, slots=True)
class _TargetedOutsiderConstruction:
    anchors: list[_PosteriorItem]
    anchor_roles: dict[str, list[str]]
    outsiders: list[_PosteriorItem]
    outsider_scores: dict[str, dict[str, float | int]]
    proposals: list[_AcquisitionProposal]
    exact_pool: list[_PosteriorItem]
    exact_proposals: list[_AcquisitionProposal]
    expanded_pool: list[_PosteriorItem]
    expanded_proposals: list[_AcquisitionProposal]


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
        payload = _empty_schedule_diagnostics(
            k=k,
            budget=budget.budget,
            method="top_k_evsi_approximation",
        )
        recorder.record(
            step="pair_scheduling",
            code="evsi_pair_scheduling_empty",
            message="no EVSI boundary-duel comparisons scheduled",
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
    scheduled = _select_evsi_pairs(
        context.proposals,
        budget=budget.budget,
        seed=seed,
        calibration_fraction=cfg.calibration_fraction,
        per_item_cap=cfg.per_item_cap,
    )
    payload = {
        "candidate_count": len(context.pool),
        "scheduled_total": len(scheduled),
        "pairs_considered": len(context.proposals),
        "unique_pairs_considered": len(
            {
                _pair_key(proposal.left_id, proposal.right_id)
                for proposal in context.proposals
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
            item_by_id={item.paper_id: item for item in context.items},
        ),
        "proposal_pool_profile": _proposal_pool_profile(
            items=context.items,
            pool=context.pool,
            scheduled=scheduled,
            k=k,
        ),
        "evsi_score_distribution": _evsi_score_distribution(context.proposals),
    }
    recorder.record(
        step="pair_scheduling",
        code="evsi_pair_scheduling_completed",
        message="scheduled posterior top-K EVSI boundary-duel comparisons",
        data=payload,
    )
    return PairSchedule(pairs=scheduled, budget=budget, diagnostics=payload)


def schedule_exact_pool_random(
    papers: list[Paper],
    comparisons: list[PairwiseComparison],
    *,
    k: int,
    budget: PairwiseBudget,
    seed: int = 0,
    config: EVSISchedulerConfig | None = None,
    diagnostics: DiagnosticRecorder | None = None,
) -> PairSchedule:
    """Randomly sample from the exact feasible EVSI proposal pool."""
    cfg = config or EVSISchedulerConfig()
    return _schedule_random_evsi_pool(
        papers,
        comparisons,
        k=k,
        budget=budget,
        seed=seed,
        config=cfg,
        diagnostics=diagnostics,
        method="exact_pool_random",
        empty_code="exact_pool_random_pair_scheduling_empty",
        complete_code="exact_pool_random_pair_scheduling_completed",
        empty_message="no exact-pool random comparisons scheduled",
        complete_message="randomly sampled exact EVSI feasible proposal pool",
    )


def schedule_expanded_pool_random(
    papers: list[Paper],
    comparisons: list[PairwiseComparison],
    *,
    k: int,
    budget: PairwiseBudget,
    seed: int = 0,
    config: EVSISchedulerConfig | None = None,
    diagnostics: DiagnosticRecorder | None = None,
) -> PairSchedule:
    """Randomly sample from a wider posterior proposal pool.

    This isolates candidate-construction quality from EVSI/CCTD acquisition
    scoring by changing the dynamic pool and keeping within-pool selection
    random.
    """
    cfg = config or EVSISchedulerConfig(
        samples=1200,
        pool_multiplier=4,
        diverse_outsider_count=max(20, 4 * max(k, 1)),
    )
    return _schedule_random_evsi_pool(
        papers,
        comparisons,
        k=k,
        budget=budget,
        seed=seed,
        config=cfg,
        diagnostics=diagnostics,
        method="expanded_pool_random",
        empty_code="expanded_pool_random_pair_scheduling_empty",
        complete_code="expanded_pool_random_pair_scheduling_completed",
        empty_message="no expanded-pool random comparisons scheduled",
        complete_message="randomly sampled expanded posterior proposal pool",
    )


def schedule_targeted_outsider_random(
    papers: list[Paper],
    comparisons: list[PairwiseComparison],
    *,
    k: int,
    budget: PairwiseBudget,
    seed: int = 0,
    config: TargetedOutsiderSchedulerConfig | None = None,
    diagnostics: DiagnosticRecorder | None = None,
) -> PairSchedule:
    """Randomly sample anchor-vs-outsider challengers from a targeted pool."""
    cfg = config or TargetedOutsiderSchedulerConfig()
    evsi_cfg = cfg.evsi
    recorder = diagnostics or DiagnosticRecorder()
    paper_by_id = {paper.paper_id: paper for paper in papers}
    if budget.budget <= 0 or len(paper_by_id) < 2 or k <= 0:
        payload = _empty_schedule_diagnostics(
            k=k,
            budget=budget.budget,
            method="targeted_outsider_random",
        )
        payload["acquisition"].update(
            _targeted_outsider_acquisition_payload(
                config=cfg,
                seed=seed,
            )
        )
        payload["targeted_outsider"] = _empty_targeted_outsider_diagnostics()
        recorder.record(
            step="pair_scheduling",
            code="targeted_outsider_random_pair_scheduling_empty",
            message="no targeted outsider random comparisons scheduled",
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
        config=evsi_cfg,
        seen_pairs=seen_pairs,
        diagnostics=recorder,
    )
    construction = _targeted_outsider_construction(
        context,
        paper_by_id=paper_by_id,
        comparisons=comparisons,
        seen_pairs=seen_pairs,
        k=k,
        config=cfg,
    )
    scheduled = _select_random_evsi_pairs(
        construction.proposals,
        budget=budget.budget,
        seed=seed,
        per_item_cap=evsi_cfg.per_item_cap,
        purpose_override="targeted_outsider_random",
    )
    targeted_pool = _ordered_unique(construction.anchors, construction.outsiders)
    payload = {
        "candidate_count": len(targeted_pool),
        "scheduled_total": len(scheduled),
        "pairs_considered": len(construction.proposals),
        "unique_pairs_considered": len(
            {
                _pair_key(proposal.left_id, proposal.right_id)
                for proposal in construction.proposals
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
        "acquisition": _targeted_outsider_acquisition_payload(
            config=cfg,
            seed=seed,
        ),
        "purpose_counts": dict(
            sorted(Counter(pair.purpose for pair in scheduled).items())
        ),
        "coverage": {
            **_evsi_coverage(
                scheduled,
                item_by_id={item.paper_id: item for item in context.items},
            ),
            **_targeted_outsider_coverage(
                scheduled,
                anchor_ids={item.paper_id for item in construction.anchors},
                outsider_ids={item.paper_id for item in construction.outsiders},
            ),
        },
        "proposal_pool_profile": _proposal_pool_profile(
            items=context.items,
            pool=targeted_pool,
            scheduled=scheduled,
            k=k,
        ),
        "targeted_outsider": _targeted_outsider_diagnostics(
            construction,
            scheduled=scheduled,
            budget=budget.budget,
            k=k,
        ),
        "evsi_score_distribution": _evsi_score_distribution(construction.proposals),
        "batch_history": [
            {
                "batch_index": 1,
                "selected_total": len(scheduled),
                "cached_label_revealed_total": 0,
                "novel_pairs_total": 0,
                "top_k_entropy_reduction": None,
                "top_k_set_churn": None,
                "note": (
                    "targeted outsider random schedules one offline batch "
                    "without label reveal"
                ),
            }
        ],
    }
    recorder.record(
        step="pair_scheduling",
        code="targeted_outsider_random_pair_scheduling_completed",
        message="randomly sampled targeted outsider-anchor proposal pool",
        data=payload,
    )
    return PairSchedule(pairs=scheduled, budget=budget, diagnostics=payload)


def _schedule_random_evsi_pool(
    papers: list[Paper],
    comparisons: list[PairwiseComparison],
    *,
    k: int,
    budget: PairwiseBudget,
    seed: int,
    config: EVSISchedulerConfig,
    diagnostics: DiagnosticRecorder | None,
    method: str,
    empty_code: str,
    complete_code: str,
    empty_message: str,
    complete_message: str,
) -> PairSchedule:
    recorder = diagnostics or DiagnosticRecorder()
    paper_by_id = {paper.paper_id: paper for paper in papers}
    if budget.budget <= 0 or len(paper_by_id) < 2 or k <= 0:
        payload = _empty_schedule_diagnostics(
            k=k,
            budget=budget.budget,
            method=method,
        )
        payload["acquisition"].update(
            _random_pool_acquisition_payload(
                method=method,
                seed=seed,
                config=config,
            )
        )
        recorder.record(
            step="pair_scheduling",
            code=empty_code,
            message=empty_message,
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
        config=config,
        seen_pairs=seen_pairs,
        diagnostics=recorder,
    )
    scheduled = _select_random_evsi_pairs(
        context.proposals,
        budget=budget.budget,
        seed=seed,
        per_item_cap=config.per_item_cap,
        purpose_override=method,
    )
    payload = {
        "candidate_count": len(context.pool),
        "scheduled_total": len(scheduled),
        "pairs_considered": len(context.proposals),
        "unique_pairs_considered": len(
            {
                _pair_key(proposal.left_id, proposal.right_id)
                for proposal in context.proposals
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
        "acquisition": _random_pool_acquisition_payload(
            method=method,
            seed=seed,
            config=config,
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
        "evsi_score_distribution": _evsi_score_distribution(context.proposals),
        "batch_history": [
            {
                "batch_index": 1,
                "selected_total": len(scheduled),
                "cached_label_revealed_total": 0,
                "novel_pairs_total": 0,
                "top_k_entropy_reduction": None,
                "top_k_set_churn": None,
                "note": (
                    "random control schedules one offline batch without "
                    "label reveal"
                ),
            }
        ],
    }
    recorder.record(
        step="pair_scheduling",
        code=complete_code,
        message=complete_message,
        data=payload,
    )
    return PairSchedule(pairs=scheduled, budget=budget, diagnostics=payload)


def _random_pool_acquisition_payload(
    *,
    method: str,
    seed: int,
    config: EVSISchedulerConfig,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "method": method,
        "source_method": "top_k_evsi_approximation",
        "random_seed": seed,
        "per_item_cap": config.per_item_cap,
        "pool_multiplier": config.pool_multiplier,
        "diverse_outsider_count": config.diverse_outsider_count,
    }
    if method == "expanded_pool_random":
        payload["candidate_construction_change"] = "expanded_dynamic_proposal_pool"
    return payload


def _targeted_outsider_acquisition_payload(
    *,
    config: TargetedOutsiderSchedulerConfig,
    seed: int,
) -> dict[str, Any]:
    return {
        "method": "targeted_outsider_random",
        "source_method": "posterior_anchor_targeted_outsider_construction",
        "selection_policy": "random_within_constructed_anchor_outsider_pairs",
        "random_seed": seed,
        "posterior_samples": config.evsi.samples,
        "per_item_cap": config.evsi.per_item_cap,
        "top_k_anchor_multiplier": config.top_k_anchor_multiplier,
        "boundary_anchor_multiplier": config.boundary_anchor_multiplier,
        "outsider_multiplier": config.outsider_multiplier,
        "min_outsiders": config.min_outsiders,
        "max_outsiders": config.max_outsiders,
        "candidate_construction_change": "targeted_outsider_anchor_challengers",
        "model_visible_signals": [
            "posterior_top_k_probability",
            "posterior_boundary_mass",
            "posterior_ucb",
            "posterior_uncertainty",
            "metadata_bucket_diversity",
            "prior_pairwise_degree",
        ],
    }


def _empty_targeted_outsider_diagnostics() -> dict[str, Any]:
    return {
        "anchor_count": 0,
        "top_k_anchor_count": 0,
        "boundary_anchor_count": 0,
        "outsider_count": 0,
        "outsider_anchor_pair_count": 0,
        "scheduled_outsider_anchor_pair_count": 0,
        "scheduled_anchor_count": 0,
        "scheduled_outsider_count": 0,
        "pool_reference_deltas": {
            "exact_pool": {
                "pool_item_count": 0,
                "proposal_count": 0,
                "targeted_pool_item_delta": 0,
                "targeted_proposal_delta": 0,
            },
            "expanded_pool": {
                "pool_item_count": 0,
                "proposal_count": 0,
                "targeted_pool_item_delta": 0,
                "targeted_proposal_delta": 0,
            },
        },
        "budget_tradeoffs": {
            "scheduled_pairs": 0,
            "scheduled_unique_papers": 0,
            "anchor_outsider_pair_rate": 0.0,
            "scheduled_unique_papers_per_pair": 0.0,
            "anchor_touch_rate": 0.0,
            "outsider_touch_rate": 0.0,
        },
    }


def schedule_cache_aware_sequential_evsi(
    papers: list[Paper],
    comparisons: list[PairwiseComparison],
    *,
    reveal_comparison: Callable[[ScheduledPair], PairwiseComparison | None],
    k: int,
    budget: PairwiseBudget,
    seed: int = 0,
    config: SequentialEVSISchedulerConfig | None = None,
    diagnostics: DiagnosticRecorder | None = None,
) -> PairSchedule:
    """Select EVSI pairs in batches, revealing cached labels after selection."""
    cfg = config or SequentialEVSISchedulerConfig()
    evsi_cfg = cfg.evsi
    recorder = diagnostics or DiagnosticRecorder()
    paper_by_id = {paper.paper_id: paper for paper in papers}
    if budget.budget <= 0 or len(paper_by_id) < 2 or k <= 0:
        payload = _empty_schedule_diagnostics(
            k=k,
            budget=budget.budget,
            method="cache_aware_sequential_evsi",
        )
        payload.update(
            {
                "batch_history": [],
                "cached_label_revealed_total": 0,
                "novel_pairs_total": 0,
                "stopped_on_novel": False,
            }
        )
        recorder.record(
            step="pair_scheduling",
            code="sequential_evsi_pair_scheduling_empty",
            message="no sequential EVSI comparisons scheduled",
            data=payload,
        )
        return PairSchedule(pairs=[], budget=budget, diagnostics=payload)

    revealed_comparisons = list(comparisons)
    selected_pairs: list[ScheduledPair] = []
    selected_keys: set[tuple[str, str]] = {
        _pair_key(comparison.left_id, comparison.right_id)
        for comparison in comparisons
    }
    batch_history: list[dict[str, Any]] = []
    all_proposals: list[_AcquisitionProposal] = []
    cached_revealed_total = 0
    novel_total = 0
    stopped_on_novel = False
    last_items: list[_PosteriorItem] = []
    last_pool: list[_PosteriorItem] = []

    max_rounds = max(0, int(cfg.rounds))
    batch_size = max(1, int(cfg.batch_size))
    for batch_index in range(max_rounds):
        remaining_budget = budget.budget - len(selected_pairs)
        if remaining_budget <= 0:
            break
        batch_budget = min(batch_size, remaining_budget)
        batch_seed = seed + (batch_index * 9973)
        context = _build_evsi_context(
            papers,
            comparisons=revealed_comparisons,
            k=k,
            seed=batch_seed,
            config=evsi_cfg,
            seen_pairs=selected_keys,
            diagnostics=recorder,
        )
        last_items = context.items
        last_pool = context.pool
        all_proposals.extend(context.proposals)
        before_entropy = _posterior_top_k_entropy(context.posterior)
        before_top_k = _posterior_top_k_set(context.posterior, k=k)
        batch_pairs = _select_evsi_pairs(
            context.proposals,
            budget=batch_budget,
            seed=batch_seed,
            calibration_fraction=evsi_cfg.calibration_fraction,
            per_item_cap=evsi_cfg.per_item_cap,
            start_index=len(selected_pairs),
        )
        batch_pairs = [
            _annotate_batch_pair(
                pair,
                batch_index=batch_index + 1,
                global_index=len(selected_pairs) + offset,
            )
            for offset, pair in enumerate(batch_pairs)
        ]
        if not batch_pairs:
            batch_history.append(
                {
                    "batch_index": batch_index + 1,
                    "selected_total": 0,
                    "comparisons_before_batch": len(revealed_comparisons),
                    "comparisons_after_batch": len(revealed_comparisons),
                    "proposal_count": len(context.proposals),
                    "top_k_entropy_before": before_entropy,
                    "top_k_entropy_after": before_entropy,
                    "top_k_entropy_reduction": 0.0,
                    "top_k_set_churn": 0.0,
                    "stop_reason": "no_feasible_pairs",
                }
            )
            break

        selected_pairs.extend(batch_pairs)
        selected_keys.update(_pair_key(pair.left_id, pair.right_id) for pair in batch_pairs)
        cached_in_batch = 0
        novel_in_batch = 0
        for pair in batch_pairs:
            comparison = reveal_comparison(pair)
            if comparison is None:
                novel_in_batch += 1
                continue
            revealed_comparisons.append(_orient_revealed_comparison(comparison, pair))
            cached_in_batch += 1

        cached_revealed_total += cached_in_batch
        novel_total += novel_in_batch
        if cached_in_batch:
            after_context = _build_evsi_context(
                papers,
                comparisons=revealed_comparisons,
                k=k,
                seed=batch_seed + 1,
                config=evsi_cfg,
                seen_pairs=selected_keys,
                diagnostics=recorder,
            )
            after_entropy = _posterior_top_k_entropy(after_context.posterior)
            after_top_k = _posterior_top_k_set(after_context.posterior, k=k)
            last_items = after_context.items
            last_pool = after_context.pool
        else:
            after_entropy = before_entropy
            after_top_k = before_top_k
        batch_history.append(
            {
                "batch_index": batch_index + 1,
                "selected_total": len(batch_pairs),
                "cached_label_revealed_total": cached_in_batch,
                "novel_pairs_total": novel_in_batch,
                "comparisons_before_batch": len(revealed_comparisons) - cached_in_batch,
                "comparisons_after_batch": len(revealed_comparisons),
                "proposal_count": len(context.proposals),
                "top_k_entropy_before": before_entropy,
                "top_k_entropy_after": after_entropy,
                "top_k_entropy_reduction": round(before_entropy - after_entropy, 8),
                "top_k_set_churn": _top_k_set_churn(before_top_k, after_top_k),
                "stop_reason": (
                    "novel_pair_without_revealed_label"
                    if novel_in_batch and cfg.stop_on_novel
                    else None
                ),
            }
        )
        if novel_in_batch and cfg.stop_on_novel:
            stopped_on_novel = True
            break

    payload = {
        "candidate_count": len({item.paper_id for item in last_items}),
        "scheduled_total": len(selected_pairs),
        "pairs_considered": len(all_proposals),
        "unique_pairs_considered": len(
            {_pair_key(proposal.left_id, proposal.right_id) for proposal in all_proposals}
        ),
        "budget": budget.budget,
        "k": k,
        "acquisition": {
            "method": "cache_aware_sequential_evsi",
            "source_method": "top_k_evsi_approximation",
            "rounds": cfg.rounds,
            "batch_size": cfg.batch_size,
            "stop_on_novel": cfg.stop_on_novel,
            "pool_multiplier": evsi_cfg.pool_multiplier,
            "calibration_fraction": evsi_cfg.calibration_fraction,
        },
        "purpose_counts": dict(
            sorted(Counter(pair.purpose for pair in selected_pairs).items())
        ),
        "coverage": _evsi_coverage(
            selected_pairs,
            item_by_id={item.paper_id: item for item in last_items},
        ),
        "proposal_pool_profile": _proposal_pool_profile(
            items=last_items,
            pool=last_pool,
            scheduled=selected_pairs,
            k=k,
        ),
        "evsi_score_distribution": _evsi_score_distribution(all_proposals),
        "batch_history": batch_history,
        "cached_label_revealed_total": cached_revealed_total,
        "novel_pairs_total": novel_total,
        "stopped_on_novel": stopped_on_novel,
        "known_comparisons_final": len(revealed_comparisons),
    }
    recorder.record(
        step="pair_scheduling",
        code="sequential_evsi_pair_scheduling_completed",
        message="scheduled cache-aware sequential EVSI comparisons",
        data=payload,
    )
    return PairSchedule(pairs=selected_pairs, budget=budget, diagnostics=payload)


def schedule_cache_aware_cctd_gf(
    papers: list[Paper],
    comparisons: list[PairwiseComparison],
    *,
    reveal_comparison: Callable[[ScheduledPair], PairwiseComparison | None],
    k: int,
    budget: PairwiseBudget,
    seed: int = 0,
    config: CCTDGFSchedulerConfig | None = None,
    diagnostics: DiagnosticRecorder | None = None,
) -> PairSchedule:
    """Select CCTD-GF pairs in mini-batches, revealing cached labels by batch."""
    cfg = config or CCTDGFSchedulerConfig()
    evsi_cfg = cfg.evsi
    recorder = diagnostics or DiagnosticRecorder()
    paper_by_id = {paper.paper_id: paper for paper in papers}
    if budget.budget <= 0 or len(paper_by_id) < 2 or k <= 0:
        payload = _empty_schedule_diagnostics(
            k=k,
            budget=budget.budget,
            method="cctd_gf",
        )
        payload.update(
            {
                "batch_history": [],
                "cached_label_revealed_total": 0,
                "novel_pairs_total": 0,
                "stopped_on_novel": False,
                "cctd_gf_score_distribution": _evsi_score_distribution([]),
            }
        )
        recorder.record(
            step="pair_scheduling",
            code="cctd_gf_pair_scheduling_empty",
            message="no CCTD-GF comparisons scheduled",
            data=payload,
        )
        return PairSchedule(pairs=[], budget=budget, diagnostics=payload)

    revealed_comparisons = list(comparisons)
    selected_pairs: list[ScheduledPair] = []
    selected_keys: set[tuple[str, str]] = {
        _pair_key(comparison.left_id, comparison.right_id)
        for comparison in comparisons
    }
    batch_history: list[dict[str, Any]] = []
    all_evsi_proposals: list[_AcquisitionProposal] = []
    all_cctd_proposals: list[_AcquisitionProposal] = []
    cached_revealed_total = 0
    novel_total = 0
    stopped_on_novel = False
    last_items: list[_PosteriorItem] = []
    last_pool: list[_PosteriorItem] = []

    max_rounds = max(0, int(cfg.rounds))
    batch_size = max(1, int(cfg.batch_size))
    for batch_index in range(max_rounds):
        remaining_budget = budget.budget - len(selected_pairs)
        if remaining_budget <= 0:
            break
        batch_budget = min(batch_size, remaining_budget)
        batch_seed = seed + (batch_index * 9973)
        context = _build_evsi_context(
            papers,
            comparisons=revealed_comparisons,
            k=k,
            seed=batch_seed,
            config=evsi_cfg,
            seen_pairs=selected_keys,
            diagnostics=recorder,
        )
        last_items = context.items
        last_pool = context.pool
        all_evsi_proposals.extend(context.proposals)
        before_entropy = _posterior_top_k_entropy(context.posterior)
        before_top_k = _posterior_top_k_set(context.posterior, k=k)
        active_degrees = _active_degrees(
            selected_pairs=selected_pairs,
            comparisons=revealed_comparisons,
        )
        component_by_id = _active_component_by_id(
            [paper.paper_id for paper in papers],
            selected_pairs=selected_pairs,
            comparisons=revealed_comparisons,
        )
        scored_proposals = _cctd_gf_proposals(
            context,
            k=k,
            seed=batch_seed,
            config=cfg,
            active_degrees=active_degrees,
            component_by_id=component_by_id,
        )
        all_cctd_proposals.extend(scored_proposals)
        batch_proposals = _select_cctd_gf_batch(
            scored_proposals,
            budget=batch_budget,
            seed=batch_seed,
            config=cfg,
            active_degrees=active_degrees,
        )
        batch_pairs = _scheduled_pairs_from_proposals(
            batch_proposals,
            seed=batch_seed,
            start_index=len(selected_pairs),
        )
        batch_pairs = [
            _annotate_cctd_batch_pair(
                pair,
                batch_index=batch_index + 1,
                global_index=len(selected_pairs) + offset,
            )
            for offset, pair in enumerate(batch_pairs)
        ]
        if not batch_pairs:
            batch_history.append(
                {
                    "batch_index": batch_index + 1,
                    "selected_total": 0,
                    "comparisons_before_batch": len(revealed_comparisons),
                    "comparisons_after_batch": len(revealed_comparisons),
                    "proposal_count": len(context.proposals),
                    "scored_proposal_count": len(scored_proposals),
                    "top_k_entropy_before": before_entropy,
                    "top_k_entropy_after": before_entropy,
                    "top_k_entropy_reduction": 0.0,
                    "top_k_set_churn": 0.0,
                    "stop_reason": "no_feasible_pairs",
                }
            )
            break

        selected_pairs.extend(batch_pairs)
        selected_keys.update(_pair_key(pair.left_id, pair.right_id) for pair in batch_pairs)
        cached_in_batch = 0
        novel_in_batch = 0
        for pair in batch_pairs:
            comparison = reveal_comparison(pair)
            if comparison is None:
                novel_in_batch += 1
                continue
            revealed_comparisons.append(_orient_revealed_comparison(comparison, pair))
            cached_in_batch += 1

        cached_revealed_total += cached_in_batch
        novel_total += novel_in_batch
        if cached_in_batch:
            after_context = _build_evsi_context(
                papers,
                comparisons=revealed_comparisons,
                k=k,
                seed=batch_seed + 1,
                config=evsi_cfg,
                seen_pairs=selected_keys,
                diagnostics=recorder,
            )
            after_entropy = _posterior_top_k_entropy(after_context.posterior)
            after_top_k = _posterior_top_k_set(after_context.posterior, k=k)
            last_items = after_context.items
            last_pool = after_context.pool
        else:
            after_entropy = before_entropy
            after_top_k = before_top_k
        batch_history.append(
            {
                "batch_index": batch_index + 1,
                "selected_total": len(batch_pairs),
                "purpose_counts": dict(
                    sorted(Counter(pair.purpose for pair in batch_pairs).items())
                ),
                "cached_label_revealed_total": cached_in_batch,
                "novel_pairs_total": novel_in_batch,
                "comparisons_before_batch": len(revealed_comparisons) - cached_in_batch,
                "comparisons_after_batch": len(revealed_comparisons),
                "proposal_count": len(context.proposals),
                "scored_proposal_count": len(scored_proposals),
                "top_k_entropy_before": before_entropy,
                "top_k_entropy_after": after_entropy,
                "top_k_entropy_reduction": round(before_entropy - after_entropy, 8),
                "top_k_set_churn": _top_k_set_churn(before_top_k, after_top_k),
                "stop_reason": (
                    "novel_pair_without_revealed_label"
                    if novel_in_batch and cfg.stop_on_novel
                    else None
                ),
            }
        )
        if novel_in_batch and cfg.stop_on_novel:
            stopped_on_novel = True
            break

    payload = {
        "candidate_count": len({item.paper_id for item in last_pool}),
        "scheduled_total": len(selected_pairs),
        "pairs_considered": len(all_evsi_proposals),
        "unique_pairs_considered": len(
            {_pair_key(proposal.left_id, proposal.right_id) for proposal in all_evsi_proposals}
        ),
        "budget": budget.budget,
        "k": k,
        "acquisition": {
            "method": "cctd_gf",
            "source_method": "top_k_evsi_approximation",
            "rounds": cfg.rounds,
            "batch_size": cfg.batch_size,
            "disagreement_pairs_per_round": cfg.disagreement_pairs_per_round,
            "graph_floor_pairs_per_round": cfg.graph_floor_pairs_per_round,
            "random_floor_pairs_per_round": cfg.random_floor_pairs_per_round,
            "high_score_fraction": cfg.high_score_fraction,
            "sampling_temperature": cfg.sampling_temperature,
            "stop_on_novel": cfg.stop_on_novel,
            "posterior_samples": evsi_cfg.samples,
            "pool_multiplier": evsi_cfg.pool_multiplier,
            "per_item_cap": evsi_cfg.per_item_cap,
        },
        "purpose_counts": dict(
            sorted(Counter(pair.purpose for pair in selected_pairs).items())
        ),
        "coverage": {
            **_evsi_coverage(
                selected_pairs,
                item_by_id={item.paper_id: item for item in last_items},
            ),
            **_cctd_graph_coverage(selected_pairs),
        },
        "proposal_pool_profile": _proposal_pool_profile(
            items=last_items,
            pool=last_pool,
            scheduled=selected_pairs,
            k=k,
        ),
        "evsi_score_distribution": _evsi_score_distribution(all_evsi_proposals),
        "cctd_gf_score_distribution": _evsi_score_distribution(all_cctd_proposals),
        "batch_history": batch_history,
        "cached_label_revealed_total": cached_revealed_total,
        "novel_pairs_total": novel_total,
        "stopped_on_novel": stopped_on_novel,
        "known_comparisons_final": len(revealed_comparisons),
    }
    recorder.record(
        step="pair_scheduling",
        code="cctd_gf_pair_scheduling_completed",
        message="scheduled CCTD-GF active comparisons",
        data=payload,
    )
    return PairSchedule(pairs=selected_pairs, budget=budget, diagnostics=payload)


def _build_evsi_context(
    papers: list[Paper],
    *,
    comparisons: list[PairwiseComparison],
    k: int,
    seed: int,
    config: EVSISchedulerConfig,
    seen_pairs: set[tuple[str, str]],
    diagnostics: DiagnosticRecorder,
) -> _EVSIContext:
    paper_by_id = {paper.paper_id: paper for paper in papers}
    aggregation = aggregate(
        papers,
        comparisons,
        config=AggregationConfig(pairwise_strength=config.pairwise_strength),
        diagnostics=diagnostics,
    )
    posterior = estimate_top_k_probabilities(
        aggregation,
        k=k,
        samples=config.samples,
        seed=seed,
        diagnostics=diagnostics,
    )
    items = _posterior_items(
        papers,
        aggregation=aggregation,
        posterior=posterior,
        k=k,
        config=config,
    )
    pool = _dynamic_proposal_pool(items, k=k, config=config)
    proposals = _evsi_proposals(
        pool,
        paper_by_id=paper_by_id,
        seen_pairs=seen_pairs,
        config=config,
    )
    return _EVSIContext(
        aggregation=aggregation,
        posterior=posterior,
        items=items,
        pool=pool,
        proposals=proposals,
    )


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


def _targeted_outsider_construction(
    context: _EVSIContext,
    *,
    paper_by_id: dict[str, Paper],
    comparisons: list[PairwiseComparison],
    seen_pairs: set[tuple[str, str]],
    k: int,
    config: TargetedOutsiderSchedulerConfig,
) -> _TargetedOutsiderConstruction:
    exact_config = EVSISchedulerConfig(
        pairwise_strength=config.evsi.pairwise_strength,
        samples=config.evsi.samples,
        temperature=config.evsi.temperature,
        ucb_lambda=config.evsi.ucb_lambda,
        boundary_window=config.evsi.boundary_window,
        pool_multiplier=2,
        diverse_outsider_count=None,
        calibration_fraction=config.evsi.calibration_fraction,
        per_item_cap=config.evsi.per_item_cap,
    )
    expanded_config = EVSISchedulerConfig(
        pairwise_strength=config.evsi.pairwise_strength,
        samples=config.evsi.samples,
        temperature=config.evsi.temperature,
        ucb_lambda=config.evsi.ucb_lambda,
        boundary_window=config.evsi.boundary_window,
        pool_multiplier=4,
        diverse_outsider_count=max(20, 4 * max(k, 1)),
        calibration_fraction=config.evsi.calibration_fraction,
        per_item_cap=config.evsi.per_item_cap,
    )
    exact_pool = _dynamic_proposal_pool(context.items, k=k, config=exact_config)
    exact_proposals = _evsi_proposals(
        exact_pool,
        paper_by_id=paper_by_id,
        seen_pairs=seen_pairs,
        config=exact_config,
    )
    expanded_pool = _dynamic_proposal_pool(
        context.items,
        k=k,
        config=expanded_config,
    )
    expanded_proposals = _evsi_proposals(
        expanded_pool,
        paper_by_id=paper_by_id,
        seen_pairs=seen_pairs,
        config=expanded_config,
    )
    anchors, anchor_roles = _targeted_anchor_items(
        context.items,
        k=k,
        config=config,
    )
    outsiders, outsider_scores = _targeted_outsider_items(
        context.items,
        anchors=anchors,
        comparisons=comparisons,
        k=k,
        config=config,
    )
    proposals = _targeted_outsider_proposals(
        anchors=anchors,
        anchor_roles=anchor_roles,
        outsiders=outsiders,
        outsider_scores=outsider_scores,
        paper_by_id=paper_by_id,
        seen_pairs=seen_pairs,
        config=config.evsi,
    )
    return _TargetedOutsiderConstruction(
        anchors=anchors,
        anchor_roles=anchor_roles,
        outsiders=outsiders,
        outsider_scores=outsider_scores,
        proposals=proposals,
        exact_pool=exact_pool,
        exact_proposals=exact_proposals,
        expanded_pool=expanded_pool,
        expanded_proposals=expanded_proposals,
    )


def _targeted_anchor_items(
    items: list[_PosteriorItem],
    *,
    k: int,
    config: TargetedOutsiderSchedulerConfig,
) -> tuple[list[_PosteriorItem], dict[str, list[str]]]:
    if not items or k <= 0:
        return [], {}
    top_count = max(1, config.top_k_anchor_multiplier * max(k, 1))
    boundary_count = max(1, config.boundary_anchor_multiplier * max(k, 1))
    target_count = min(len(items), top_count + boundary_count)
    by_top_k = sorted(
        items,
        key=lambda item: (item.top_k_probability, item.mean, item.paper_id),
        reverse=True,
    )
    by_boundary = sorted(
        items,
        key=lambda item: (item.boundary_mass, item.top_k_probability, item.paper_id),
        reverse=True,
    )
    roles: dict[str, set[str]] = {}
    for item in by_top_k[:top_count]:
        roles.setdefault(item.paper_id, set()).add("posterior_top_k")
    for item in by_boundary[:boundary_count]:
        roles.setdefault(item.paper_id, set()).add("boundary")
    anchors = _ordered_unique(
        by_top_k[:top_count],
        by_boundary[:boundary_count],
    )
    if len(anchors) < target_count:
        filler = _ordered_unique(
            sorted(
                items,
                key=lambda item: (item.ucb, item.top_k_probability, item.paper_id),
                reverse=True,
            )
        )
        for item in filler:
            if len(anchors) >= target_count:
                break
            if item.paper_id in {anchor.paper_id for anchor in anchors}:
                continue
            anchors.append(item)
            roles.setdefault(item.paper_id, set()).add("ucb_fill")
    return anchors[:target_count], {
        paper_id: sorted(values) for paper_id, values in sorted(roles.items())
    }


def _targeted_outsider_items(
    items: list[_PosteriorItem],
    *,
    anchors: list[_PosteriorItem],
    comparisons: list[PairwiseComparison],
    k: int,
    config: TargetedOutsiderSchedulerConfig,
) -> tuple[list[_PosteriorItem], dict[str, dict[str, float | int]]]:
    anchor_ids = {item.paper_id for item in anchors}
    candidates = [item for item in items if item.paper_id not in anchor_ids]
    if not candidates:
        return [], {}
    target_count = min(
        len(candidates),
        max(config.min_outsiders, config.outsider_multiplier * max(k, 1)),
    )
    if config.max_outsiders is not None:
        target_count = min(target_count, config.max_outsiders)
    prior_degrees = _prior_pairwise_degrees(comparisons)
    anchor_buckets = {item.metadata_bucket for item in anchors}
    top_k_scores = _rank_scores(
        candidates,
        key=lambda item: (item.top_k_probability, item.mean),
    )
    ucb_scores = _rank_scores(
        candidates,
        key=lambda item: (item.ucb, item.top_k_probability),
    )
    uncertainty_scores = _rank_scores(
        candidates,
        key=lambda item: (item.sigma, item.ucb),
    )
    boundary_scores = _rank_scores(
        candidates,
        key=lambda item: (item.boundary_mass, item.top_k_probability),
    )
    scored: list[tuple[float, _PosteriorItem, dict[str, float | int]]] = []
    for item in candidates:
        metadata_diversity = (
            _rate(
                sum(1 for bucket in anchor_buckets if bucket != item.metadata_bucket),
                len(anchor_buckets),
            )
            if anchor_buckets
            else 0.0
        )
        prior_degree = prior_degrees[item.paper_id]
        exposure_score = 1.0 / (1.0 + prior_degree)
        score = (
            (0.30 * top_k_scores[item.paper_id])
            + (0.25 * ucb_scores[item.paper_id])
            + (0.20 * uncertainty_scores[item.paper_id])
            + (0.15 * boundary_scores[item.paper_id])
            + (0.05 * metadata_diversity)
            + (0.05 * exposure_score)
        )
        diagnostics: dict[str, float | int] = {
            "outsider_construction_score": round(score, 8),
            "outsider_top_k_rank_score": round(top_k_scores[item.paper_id], 8),
            "outsider_ucb_rank_score": round(ucb_scores[item.paper_id], 8),
            "outsider_uncertainty_rank_score": round(
                uncertainty_scores[item.paper_id],
                8,
            ),
            "outsider_boundary_rank_score": round(boundary_scores[item.paper_id], 8),
            "outsider_metadata_diversity": round(metadata_diversity, 8),
            "outsider_prior_pairwise_degree": int(prior_degree),
            "outsider_prior_exposure_score": round(exposure_score, 8),
        }
        scored.append((score, item, diagnostics))
    scored.sort(
        key=lambda row: (
            row[0],
            row[1].ucb,
            row[1].top_k_probability,
            row[1].paper_id,
        ),
        reverse=True,
    )
    selected = _diverse_prefix([item for _, item, _ in scored], limit=target_count)
    selected_ids = {item.paper_id for item in selected}
    scores_by_id = {
        item.paper_id: diagnostics
        for _, item, diagnostics in scored
        if item.paper_id in selected_ids
    }
    return selected, scores_by_id


def _targeted_outsider_proposals(
    *,
    anchors: list[_PosteriorItem],
    anchor_roles: dict[str, list[str]],
    outsiders: list[_PosteriorItem],
    outsider_scores: dict[str, dict[str, float | int]],
    paper_by_id: dict[str, Paper],
    seen_pairs: set[tuple[str, str]],
    config: EVSISchedulerConfig,
) -> list[_AcquisitionProposal]:
    proposals: list[_AcquisitionProposal] = []
    for anchor in anchors:
        for outsider in outsiders:
            key = _pair_key(anchor.paper_id, outsider.paper_id)
            if key in seen_pairs:
                continue
            probability = _sigmoid(
                (anchor.mean - outsider.mean) / max(config.temperature, 1e-9)
            )
            entropy = _binary_entropy(probability)
            boundary_relevance = anchor.boundary_mass + outsider.boundary_mass
            metadata_diverse = anchor.metadata_bucket != outsider.metadata_bucket
            outsider_payload = outsider_scores.get(outsider.paper_id, {})
            score = float(
                outsider_payload.get("outsider_construction_score", 0.0)
            ) * (0.5 + min(1.0, boundary_relevance / 2.0))
            if metadata_diverse:
                score *= 1.05
            proposals.append(
                _AcquisitionProposal(
                    left_id=anchor.paper_id,
                    right_id=outsider.paper_id,
                    score=score,
                    purpose="targeted_outsider_anchor",
                    diagnostics={
                        "acquisition_score": round(score, 8),
                        "construction_score": round(score, 8),
                        "head_to_head_probability": round(probability, 8),
                        "head_to_head_entropy": round(entropy, 8),
                        "boundary_relevance": round(boundary_relevance, 8),
                        "pair_role": "targeted_outsider_anchor",
                        "metadata_diverse": metadata_diverse,
                        "anchor_id": anchor.paper_id,
                        "outsider_id": outsider.paper_id,
                        "anchor_roles": anchor_roles.get(anchor.paper_id, []),
                        "anchor_top_k_probability": round(
                            anchor.top_k_probability,
                            8,
                        ),
                        "outsider_top_k_probability": round(
                            outsider.top_k_probability,
                            8,
                        ),
                        "anchor_ucb": round(anchor.ucb, 8),
                        "outsider_ucb": round(outsider.ucb, 8),
                        "anchor_mean_rank": round(anchor.mean_rank, 8),
                        "outsider_mean_rank": round(outsider.mean_rank, 8),
                        "anchor_boundary_mass": round(anchor.boundary_mass, 8),
                        "outsider_boundary_mass": round(outsider.boundary_mass, 8),
                        "anchor_metadata_bucket": anchor.metadata_bucket,
                        "outsider_metadata_bucket": outsider.metadata_bucket,
                        "anchor_title": paper_by_id[anchor.paper_id].title,
                        "outsider_title": paper_by_id[outsider.paper_id].title,
                        **outsider_payload,
                    },
                )
            )
    return sorted(
        proposals,
        key=lambda proposal: (proposal.score, proposal.left_id, proposal.right_id),
        reverse=True,
    )


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
                    "left_ucb": round(left.ucb, 8),
                    "right_ucb": round(right.ucb, 8),
                    "left_mean_rank": round(left.mean_rank, 8),
                    "right_mean_rank": round(right.mean_rank, 8),
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
    start_index: int = 0,
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

    return _scheduled_pairs_from_proposals(
        selected[:budget],
        seed=seed,
        start_index=start_index,
    )


def _select_random_evsi_pairs(
    proposals: list[_AcquisitionProposal],
    *,
    budget: int,
    seed: int,
    per_item_cap: int | None,
    purpose_override: str = "exact_pool_random",
) -> list[ScheduledPair]:
    if budget <= 0:
        return []
    rng = random.Random(seed)
    candidates = list(proposals)
    rng.shuffle(candidates)
    selected: list[_AcquisitionProposal] = []
    selected_keys: set[tuple[str, str]] = set()
    item_counts: Counter[str] = Counter()
    cap = per_item_cap or max(
        2,
        math.ceil((2.5 * budget) / max(1, len(proposals) ** 0.5)),
    )
    for proposal in candidates:
        if len(selected) >= budget:
            break
        key = _pair_key(proposal.left_id, proposal.right_id)
        if key in selected_keys:
            continue
        if item_counts[proposal.left_id] >= cap or item_counts[proposal.right_id] >= cap:
            continue
        selected.append(proposal)
        selected_keys.add(key)
        item_counts[proposal.left_id] += 1
        item_counts[proposal.right_id] += 1

    if len(selected) < budget:
        for proposal in candidates:
            if len(selected) >= budget:
                break
            key = _pair_key(proposal.left_id, proposal.right_id)
            if key in selected_keys:
                continue
            selected.append(proposal)
            selected_keys.add(key)

    return _scheduled_pairs_from_proposals(
        selected[:budget],
        seed=seed,
        purpose_override=purpose_override,
        priority_override=0.0,
        source_purpose_key="source_evsi_purpose",
    )


def _cctd_gf_proposals(
    context: _EVSIContext,
    *,
    k: int,
    seed: int,
    config: CCTDGFSchedulerConfig,
    active_degrees: Counter[str],
    component_by_id: dict[str, int],
) -> list[_AcquisitionProposal]:
    if not context.proposals:
        return []
    item_by_id = {item.paper_id: item for item in context.items}
    decision_ids = _cctd_decision_ids(context.items, k=k)
    top_k_ids = _posterior_top_k_set(context.posterior, k=k)
    samples = _posterior_latent_samples(
        context.aggregation,
        k=k,
        samples=config.evsi.samples,
        seed=seed,
    )
    proposals: list[_AcquisitionProposal] = []
    for proposal in context.proposals:
        left = item_by_id[proposal.left_id]
        right = item_by_id[proposal.right_id]
        pair_stats = _cctd_pair_stats(
            left.paper_id,
            right.paper_id,
            samples=samples,
            temperature=config.evsi.temperature,
        )
        left_degree = active_degrees[left.paper_id]
        right_degree = active_degrees[right.paper_id]
        low_degree_bonus = 0.5 * (
            (1.0 / (1.0 + left_degree)) + (1.0 / (1.0 + right_degree))
        )
        cross_component = component_by_id.get(left.paper_id) != component_by_id.get(
            right.paper_id
        )
        cross_boundary = (left.paper_id in top_k_ids) != (right.paper_id in top_k_ids)
        decision_pair = left.paper_id in decision_ids or right.paper_id in decision_ids
        boundary_relevance = left.boundary_mass + right.boundary_mass
        graph_coverage_bonus = (
            1.0
            + (0.35 * low_degree_bonus)
            + (0.20 if cross_component else 0.0)
            + (0.20 if cross_boundary else 0.0)
        )
        anti_redundancy_penalty = 1.0 / (
            1.0 + max(0.0, ((left_degree + right_degree) / 2.0) - 1.0)
        )
        score = (
            (
                0.60 * pair_stats["top_k_disagreement"]
                + 0.30 * pair_stats["pair_information"]
                + 0.10 * min(1.0, boundary_relevance / 2.0)
            )
            * graph_coverage_bonus
            * anti_redundancy_penalty
        )
        graph_floor_score = (
            (
                0.40
                * (
                    pair_stats["top_k_disagreement"]
                    + pair_stats["pair_information"]
                )
            )
            + (0.30 * low_degree_bonus)
            + (0.20 if cross_component else 0.0)
            + (0.10 if cross_boundary else 0.0)
        ) * (0.50 + min(1.0, boundary_relevance / 2.0))
        diagnostics = dict(proposal.diagnostics)
        diagnostics.update(
            {
                "acquisition_score": round(score, 8),
                "cctd_gf_score": round(score, 8),
                "graph_floor_score": round(graph_floor_score, 8),
                "top_k_disagreement": round(pair_stats["top_k_disagreement"], 8),
                "pair_information": round(pair_stats["pair_information"], 8),
                "mean_head_to_head_probability": round(
                    pair_stats["mean_head_to_head_probability"],
                    8,
                ),
                "mean_pair_entropy": round(pair_stats["mean_pair_entropy"], 8),
                "expected_pair_entropy": round(
                    pair_stats["expected_pair_entropy"],
                    8,
                ),
                "score_probability_gap": round(
                    abs(pair_stats["mean_head_to_head_probability"] - 0.5),
                    8,
                ),
                "latent_score_gap": round(abs(left.mean - right.mean), 8),
                "graph_coverage_bonus": round(graph_coverage_bonus, 8),
                "anti_redundancy_penalty": round(anti_redundancy_penalty, 8),
                "active_degree_left": int(left_degree),
                "active_degree_right": int(right_degree),
                "low_degree_bonus": round(low_degree_bonus, 8),
                "cross_component": bool(cross_component),
                "cross_decision_boundary": bool(cross_boundary),
                "decision_set_pair": bool(decision_pair),
                "left_in_decision_set": left.paper_id in decision_ids,
                "right_in_decision_set": right.paper_id in decision_ids,
                "left_in_posterior_top_k": left.paper_id in top_k_ids,
                "right_in_posterior_top_k": right.paper_id in top_k_ids,
            }
        )
        proposals.append(
            _AcquisitionProposal(
                left_id=proposal.left_id,
                right_id=proposal.right_id,
                score=score,
                purpose="cctd_gf_candidate",
                diagnostics=diagnostics,
            )
        )
    return sorted(
        proposals,
        key=lambda item: (item.score, item.left_id, item.right_id),
        reverse=True,
    )


def _select_cctd_gf_batch(
    proposals: list[_AcquisitionProposal],
    *,
    budget: int,
    seed: int,
    config: CCTDGFSchedulerConfig,
    active_degrees: Counter[str],
) -> list[_AcquisitionProposal]:
    if budget <= 0 or not proposals:
        return []
    targets = _cctd_round_targets(config, budget=budget)
    rng = random.Random(seed)
    selected: list[_AcquisitionProposal] = []
    selected_keys: set[tuple[str, str]] = set()
    item_counts: Counter[str] = Counter(active_degrees)
    cap = config.evsi.per_item_cap or max(
        2,
        math.ceil((2.5 * budget) / max(1, len(proposals) ** 0.5)),
    )

    def available(proposal: _AcquisitionProposal, *, relax_cap: bool = False) -> bool:
        key = _pair_key(proposal.left_id, proposal.right_id)
        if key in selected_keys:
            return False
        if relax_cap:
            return True
        return (
            item_counts[proposal.left_id] < cap
            and item_counts[proposal.right_id] < cap
        )

    def take(proposal: _AcquisitionProposal) -> None:
        selected.append(proposal)
        selected_keys.add(_pair_key(proposal.left_id, proposal.right_id))
        item_counts[proposal.left_id] += 1
        item_counts[proposal.right_id] += 1

    for _ in range(targets["graph"]):
        proposal = _select_cctd_graph_floor_candidate(
            proposals,
            available=available,
        )
        if proposal is not None:
            take(_with_purpose(proposal, "cctd_gf_graph_floor"))

    for _ in range(targets["random"]):
        proposal = _select_cctd_random_candidate(
            proposals,
            rng=rng,
            available=available,
        )
        if proposal is not None:
            take(_with_purpose(proposal, "cctd_gf_random_floor"))

    for _ in range(targets["disagreement"]):
        candidates = _sample_cctd_disagreement_candidates(
            proposals,
            needed=1,
            rng=rng,
            config=config,
            available=available,
        )
        if not candidates:
            break
        take(_with_purpose(candidates[0], "cctd_gf_disagreement"))

    if len(selected) < budget:
        for purpose in (
            "cctd_gf_disagreement",
            "cctd_gf_graph_floor",
            "cctd_gf_random_floor",
        ):
            for proposal in proposals:
                if len(selected) >= budget:
                    break
                if not available(proposal, relax_cap=True):
                    continue
                take(_with_purpose(proposal, purpose))
            if len(selected) >= budget:
                break

    return selected[:budget]


def _cctd_round_targets(
    config: CCTDGFSchedulerConfig,
    *,
    budget: int,
) -> dict[str, int]:
    sequence = (
        ["disagreement"] * max(0, int(config.disagreement_pairs_per_round))
        + ["graph"] * max(0, int(config.graph_floor_pairs_per_round))
        + ["random"] * max(0, int(config.random_floor_pairs_per_round))
    )
    if not sequence:
        sequence = ["disagreement"]
    selected = sequence[: max(0, int(budget))]
    counts = Counter(selected)
    remaining = max(0, int(budget)) - len(selected)
    if remaining:
        counts["disagreement"] += remaining
    return {
        "disagreement": int(counts["disagreement"]),
        "graph": int(counts["graph"]),
        "random": int(counts["random"]),
    }


def _select_cctd_graph_floor_candidate(
    proposals: list[_AcquisitionProposal],
    *,
    available: Callable[[_AcquisitionProposal], bool],
) -> _AcquisitionProposal | None:
    candidates = [
        proposal
        for proposal in proposals
        if available(proposal)
        and bool(proposal.diagnostics.get("decision_set_pair"))
        and bool(proposal.diagnostics.get("cross_decision_boundary"))
    ]
    if not candidates:
        candidates = [
            proposal
            for proposal in proposals
            if available(proposal)
            and (
                float(proposal.diagnostics.get("top_k_disagreement", 0.0)) > 0.0
                or float(proposal.diagnostics.get("pair_information", 0.0)) > 0.0
            )
        ]
    if not candidates:
        candidates = [proposal for proposal in proposals if available(proposal)]
    if not candidates:
        return None
    candidates.sort(
        key=lambda proposal: (
            float(proposal.diagnostics.get("graph_floor_score", 0.0)),
            -int(proposal.diagnostics.get("active_degree_left", 0)),
            -int(proposal.diagnostics.get("active_degree_right", 0)),
            proposal.left_id,
            proposal.right_id,
        ),
        reverse=True,
    )
    return candidates[0]


def _select_cctd_random_candidate(
    proposals: list[_AcquisitionProposal],
    *,
    rng: random.Random,
    available: Callable[[_AcquisitionProposal], bool],
) -> _AcquisitionProposal | None:
    candidates = [proposal for proposal in proposals if available(proposal)]
    if not candidates:
        return None
    return rng.choice(candidates)


def _sample_cctd_disagreement_candidates(
    proposals: list[_AcquisitionProposal],
    *,
    needed: int,
    rng: random.Random,
    config: CCTDGFSchedulerConfig,
    available: Callable[[_AcquisitionProposal], bool],
) -> list[_AcquisitionProposal]:
    selected: list[_AcquisitionProposal] = []
    while len(selected) < needed:
        candidates = [
            proposal
            for proposal in proposals
            if available(proposal) and proposal not in selected
        ]
        if not candidates:
            break
        candidates.sort(
            key=lambda proposal: (proposal.score, proposal.left_id, proposal.right_id),
            reverse=True,
        )
        high_count = max(
            1,
            min(
                len(candidates),
                math.ceil(len(candidates) * max(0.05, config.high_score_fraction)),
            ),
        )
        high_scoring = candidates[:high_count]
        selected.append(
            _weighted_cctd_choice(
                high_scoring,
                rng=rng,
                temperature=config.sampling_temperature,
            )
        )
    return selected


def _weighted_cctd_choice(
    proposals: list[_AcquisitionProposal],
    *,
    rng: random.Random,
    temperature: float,
) -> _AcquisitionProposal:
    if len(proposals) == 1:
        return proposals[0]
    scores = [proposal.score for proposal in proposals]
    low = min(scores)
    high = max(scores)
    span = max(high - low, 1e-9)
    temp = max(temperature, 1e-6)
    weights = [math.exp(((score - low) / span) / temp) for score in scores]
    total = sum(weights)
    threshold = rng.random() * total
    running = 0.0
    for proposal, weight in zip(proposals, weights, strict=True):
        running += weight
        if running >= threshold:
            return proposal
    return proposals[-1]


def _with_purpose(
    proposal: _AcquisitionProposal,
    purpose: str,
) -> _AcquisitionProposal:
    diagnostics = dict(proposal.diagnostics)
    diagnostics["source_cctd_gf_purpose"] = proposal.purpose
    diagnostics["selected_cctd_gf_purpose"] = purpose
    return _AcquisitionProposal(
        left_id=proposal.left_id,
        right_id=proposal.right_id,
        score=proposal.score,
        purpose=purpose,
        diagnostics=diagnostics,
    )


def _scheduled_pairs_from_proposals(
    proposals: list[_AcquisitionProposal],
    *,
    seed: int,
    start_index: int = 0,
    purpose_override: str | None = None,
    priority_override: float | None = None,
    source_purpose_key: str | None = None,
) -> list[ScheduledPair]:
    rng = random.Random(seed)
    scheduled: list[ScheduledPair] = []
    for offset, proposal in enumerate(proposals):
        index = start_index + offset
        if rng.random() < 0.5:
            shown_first, shown_second = proposal.left_id, proposal.right_id
        else:
            shown_first, shown_second = proposal.right_id, proposal.left_id
        diagnostics = dict(proposal.diagnostics)
        if source_purpose_key is not None:
            diagnostics[source_purpose_key] = proposal.purpose
        scheduled.append(
            ScheduledPair(
                left_id=proposal.left_id,
                right_id=proposal.right_id,
                priority=(
                    round(priority_override, 8)
                    if priority_override is not None
                    else round(proposal.score, 8)
                ),
                purpose=purpose_override or proposal.purpose,
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
                diagnostics=diagnostics,
            )
        )
    return scheduled


def _annotate_batch_pair(
    pair: ScheduledPair,
    *,
    batch_index: int,
    global_index: int,
) -> ScheduledPair:
    order_extra = dict(pair.order.extra)
    order_extra["sequential_batch_index"] = batch_index
    diagnostics = dict(pair.diagnostics)
    diagnostics["sequential_batch_index"] = batch_index
    return ScheduledPair(
        left_id=pair.left_id,
        right_id=pair.right_id,
        priority=pair.priority,
        purpose=pair.purpose,
        order=PairwiseOrderMetadata(
            shown_first_id=pair.order.shown_first_id,
            shown_second_id=pair.order.shown_second_id,
            randomized=pair.order.randomized,
            seed=pair.order.seed,
            position_bias_audit=(global_index % 5 == 0),
            extra=order_extra,
        ),
        diagnostics=diagnostics,
    )


def _annotate_cctd_batch_pair(
    pair: ScheduledPair,
    *,
    batch_index: int,
    global_index: int,
) -> ScheduledPair:
    order_extra = dict(pair.order.extra)
    order_extra["cctd_gf_batch_index"] = batch_index
    diagnostics = dict(pair.diagnostics)
    diagnostics["cctd_gf_batch_index"] = batch_index
    return ScheduledPair(
        left_id=pair.left_id,
        right_id=pair.right_id,
        priority=pair.priority,
        purpose=pair.purpose,
        order=PairwiseOrderMetadata(
            shown_first_id=pair.order.shown_first_id,
            shown_second_id=pair.order.shown_second_id,
            randomized=pair.order.randomized,
            seed=pair.order.seed,
            position_bias_audit=(global_index % 5 == 0),
            extra=order_extra,
        ),
        diagnostics=diagnostics,
    )


def _posterior_latent_samples(
    aggregation: Any,
    *,
    k: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    estimates = list(aggregation.estimates.values())
    sample_count = max(100, int(samples))
    rng = random.Random(seed)
    draws_by_id = {estimate.paper_id: [] for estimate in estimates}
    top_k_sets: list[set[str]] = []
    for _ in range(sample_count):
        draw = []
        for estimate in estimates:
            value = rng.gauss(
                estimate.posterior_logit,
                math.sqrt(max(estimate.variance, 1e-9)),
            )
            draws_by_id[estimate.paper_id].append(value)
            draw.append((value, estimate.paper_id))
        draw.sort(reverse=True)
        top_k_sets.append({paper_id for _, paper_id in draw[: max(0, k)]})
    return {
        "draws_by_id": draws_by_id,
        "top_k_sets": top_k_sets,
        "samples": sample_count,
    }


def _cctd_pair_stats(
    left_id: str,
    right_id: str,
    *,
    samples: dict[str, Any],
    temperature: float,
) -> dict[str, float]:
    left_draws = samples["draws_by_id"].get(left_id, [])
    right_draws = samples["draws_by_id"].get(right_id, [])
    top_k_sets = samples["top_k_sets"]
    sample_count = max(1, int(samples["samples"]))
    disagreement = sum(
        (left_id in top_k_set) != (right_id in top_k_set)
        for top_k_set in top_k_sets
    ) / sample_count
    probabilities = [
        _sigmoid((left_value - right_value) / max(temperature, 1e-9))
        for left_value, right_value in zip(left_draws, right_draws, strict=True)
    ]
    if not probabilities:
        mean_probability = 0.5
        expected_entropy = _binary_entropy(mean_probability)
    else:
        mean_probability = sum(probabilities) / len(probabilities)
        expected_entropy = sum(_binary_entropy(value) for value in probabilities) / len(
            probabilities
        )
    mean_entropy = _binary_entropy(mean_probability)
    information = max(0.0, mean_entropy - expected_entropy)
    return {
        "top_k_disagreement": disagreement,
        "mean_head_to_head_probability": mean_probability,
        "mean_pair_entropy": mean_entropy,
        "expected_pair_entropy": expected_entropy,
        "pair_information": information,
    }


def _cctd_decision_ids(items: list[_PosteriorItem], *, k: int) -> set[str]:
    limit = max(1, k)
    by_top_k = sorted(
        items,
        key=lambda item: (item.top_k_probability, item.mean, item.paper_id),
        reverse=True,
    )[:limit]
    by_ucb = sorted(
        items,
        key=lambda item: (item.ucb, item.top_k_probability, item.paper_id),
        reverse=True,
    )[:limit]
    by_boundary = sorted(
        items,
        key=lambda item: (item.boundary_mass, item.top_k_probability, item.paper_id),
        reverse=True,
    )[: max(limit, 2 * limit)]
    return {item.paper_id for item in _ordered_unique(by_top_k, by_ucb, by_boundary)}


def _active_degrees(
    *,
    selected_pairs: list[ScheduledPair],
    comparisons: list[PairwiseComparison],
) -> Counter[str]:
    degrees: Counter[str] = Counter()
    selected_keys = {_pair_key(pair.left_id, pair.right_id) for pair in selected_pairs}
    for pair in selected_pairs:
        degrees[pair.left_id] += 1
        degrees[pair.right_id] += 1
    for comparison in comparisons:
        if _pair_key(comparison.left_id, comparison.right_id) in selected_keys:
            continue
        degrees[comparison.left_id] += 1
        degrees[comparison.right_id] += 1
    return degrees


def _active_component_by_id(
    paper_ids: list[str],
    *,
    selected_pairs: list[ScheduledPair],
    comparisons: list[PairwiseComparison],
) -> dict[str, int]:
    graph = {paper_id: set() for paper_id in paper_ids}
    selected_keys = {_pair_key(pair.left_id, pair.right_id) for pair in selected_pairs}
    for pair in selected_pairs:
        graph.setdefault(pair.left_id, set()).add(pair.right_id)
        graph.setdefault(pair.right_id, set()).add(pair.left_id)
    for comparison in comparisons:
        if _pair_key(comparison.left_id, comparison.right_id) in selected_keys:
            continue
        graph.setdefault(comparison.left_id, set()).add(comparison.right_id)
        graph.setdefault(comparison.right_id, set()).add(comparison.left_id)
    component_by_id: dict[str, int] = {}
    seen: set[str] = set()
    for paper_id in sorted(graph):
        if paper_id in seen:
            continue
        component_id = len(set(component_by_id.values()))
        stack = [paper_id]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component_by_id[current] = component_id
            stack.extend(sorted(graph.get(current, set()) - seen))
    return component_by_id


def _orient_revealed_comparison(
    comparison: PairwiseComparison,
    pair: ScheduledPair,
) -> PairwiseComparison:
    if comparison.left_id == pair.left_id and comparison.right_id == pair.right_id:
        winner = comparison.winner
    elif comparison.left_id == pair.right_id and comparison.right_id == pair.left_id:
        winner = _invert_winner(comparison.winner)
    else:
        raise ValueError("revealed comparison does not reference selected pair")
    return PairwiseComparison(
        left_id=pair.left_id,
        right_id=pair.right_id,
        winner=winner,
        soft_probability=comparison.soft_probability,
        confidence=comparison.confidence,
        reasons=list(comparison.reasons),
        order=pair.order,
        metadata=dict(comparison.metadata),
    )


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


def _cctd_graph_coverage(scheduled: list[ScheduledPair]) -> dict[str, Any]:
    if not scheduled:
        return {
            "graph_floor_pairs": 0,
            "random_floor_pairs": 0,
            "disagreement_pairs": 0,
            "cross_component_pairs": 0,
            "decision_boundary_pairs": 0,
            "average_top_k_disagreement": 0.0,
            "average_pair_information": 0.0,
        }
    return {
        "graph_floor_pairs": sum(
            1 for pair in scheduled if pair.purpose == "cctd_gf_graph_floor"
        ),
        "random_floor_pairs": sum(
            1 for pair in scheduled if pair.purpose == "cctd_gf_random_floor"
        ),
        "disagreement_pairs": sum(
            1 for pair in scheduled if pair.purpose == "cctd_gf_disagreement"
        ),
        "cross_component_pairs": sum(
            1 for pair in scheduled if bool(pair.diagnostics.get("cross_component"))
        ),
        "decision_boundary_pairs": sum(
            1
            for pair in scheduled
            if bool(pair.diagnostics.get("cross_decision_boundary"))
        ),
        "average_top_k_disagreement": round(
            sum(
                float(pair.diagnostics.get("top_k_disagreement", 0.0))
                for pair in scheduled
            )
            / len(scheduled),
            8,
        ),
        "average_pair_information": round(
            sum(
                float(pair.diagnostics.get("pair_information", 0.0))
                for pair in scheduled
            )
            / len(scheduled),
            8,
        ),
    }


def _targeted_outsider_coverage(
    scheduled: list[ScheduledPair],
    *,
    anchor_ids: set[str],
    outsider_ids: set[str],
) -> dict[str, Any]:
    touched = {
        paper_id
        for pair in scheduled
        for paper_id in (pair.left_id, pair.right_id)
    }
    scheduled_anchor_ids = touched & anchor_ids
    scheduled_outsider_ids = touched & outsider_ids
    outsider_anchor_pairs = sum(
        1
        for pair in scheduled
        if (
            (pair.left_id in anchor_ids and pair.right_id in outsider_ids)
            or (pair.right_id in anchor_ids and pair.left_id in outsider_ids)
        )
    )
    return {
        "targeted_outsider_anchor_pairs": outsider_anchor_pairs,
        "targeted_outsider_anchor_pair_rate": _rate(
            outsider_anchor_pairs,
            len(scheduled),
        ),
        "targeted_anchor_count": len(anchor_ids),
        "targeted_outsider_count": len(outsider_ids),
        "scheduled_targeted_anchor_count": len(scheduled_anchor_ids),
        "scheduled_targeted_outsider_count": len(scheduled_outsider_ids),
        "targeted_anchor_touch_rate": _rate(
            len(scheduled_anchor_ids),
            len(anchor_ids),
        ),
        "targeted_outsider_touch_rate": _rate(
            len(scheduled_outsider_ids),
            len(outsider_ids),
        ),
    }


def _targeted_outsider_diagnostics(
    construction: _TargetedOutsiderConstruction,
    *,
    scheduled: list[ScheduledPair],
    budget: int,
    k: int,
) -> dict[str, Any]:
    targeted_pool = _ordered_unique(construction.anchors, construction.outsiders)
    anchor_ids = {item.paper_id for item in construction.anchors}
    outsider_ids = {item.paper_id for item in construction.outsiders}
    touched = {
        paper_id
        for pair in scheduled
        for paper_id in (pair.left_id, pair.right_id)
    }
    scheduled_outsider_anchor_pairs = sum(
        1
        for pair in scheduled
        if (
            (pair.left_id in anchor_ids and pair.right_id in outsider_ids)
            or (pair.right_id in anchor_ids and pair.left_id in outsider_ids)
        )
    )
    anchor_role_counts = Counter(
        role
        for roles in construction.anchor_roles.values()
        for role in roles
    )
    outsider_scores = [
        float(payload.get("outsider_construction_score", 0.0))
        for payload in construction.outsider_scores.values()
    ]
    return {
        "anchor_count": len(construction.anchors),
        "top_k_anchor_count": int(anchor_role_counts.get("posterior_top_k", 0)),
        "boundary_anchor_count": int(anchor_role_counts.get("boundary", 0)),
        "anchor_role_counts": dict(sorted(anchor_role_counts.items())),
        "outsider_count": len(construction.outsiders),
        "outsider_target_formula": (
            "min(available_non_anchor, max(min_outsiders, outsider_multiplier*K))"
        ),
        "k": k,
        "outsider_anchor_pair_count": len(construction.proposals),
        "scheduled_outsider_anchor_pair_count": scheduled_outsider_anchor_pairs,
        "scheduled_anchor_count": len(touched & anchor_ids),
        "scheduled_outsider_count": len(touched & outsider_ids),
        "metadata_bucket_counts": {
            "anchors": dict(
                sorted(Counter(item.metadata_bucket for item in construction.anchors).items())
            ),
            "outsiders": dict(
                sorted(Counter(item.metadata_bucket for item in construction.outsiders).items())
            ),
        },
        "outsider_score_distribution": _numeric_distribution(outsider_scores),
        "pool_reference_deltas": {
            "exact_pool": {
                "pool_item_count": len(construction.exact_pool),
                "proposal_count": len(construction.exact_proposals),
                "targeted_pool_item_delta": (
                    len(targeted_pool) - len(construction.exact_pool)
                ),
                "targeted_proposal_delta": (
                    len(construction.proposals) - len(construction.exact_proposals)
                ),
            },
            "expanded_pool": {
                "pool_item_count": len(construction.expanded_pool),
                "proposal_count": len(construction.expanded_proposals),
                "targeted_pool_item_delta": (
                    len(targeted_pool) - len(construction.expanded_pool)
                ),
                "targeted_proposal_delta": (
                    len(construction.proposals) - len(construction.expanded_proposals)
                ),
            },
        },
        "budget_tradeoffs": {
            "scheduled_pairs": len(scheduled),
            "budget": budget,
            "budget_utilization": _rate(len(scheduled), budget),
            "scheduled_unique_papers": len(touched),
            "scheduled_unique_papers_per_pair": _rate(len(touched), len(scheduled)),
            "anchor_outsider_pair_rate": _rate(
                scheduled_outsider_anchor_pairs,
                len(scheduled),
            ),
            "anchor_touch_rate": _rate(len(touched & anchor_ids), len(anchor_ids)),
            "outsider_touch_rate": _rate(
                len(touched & outsider_ids),
                len(outsider_ids),
            ),
            "targeted_pool_item_count": len(targeted_pool),
            "targeted_pool_item_rate_vs_expanded": _rate(
                len(targeted_pool),
                len(construction.expanded_pool),
            ),
        },
        "uses_future_labels_for_scheduling": False,
    }


def _prior_pairwise_degrees(
    comparisons: list[PairwiseComparison],
) -> Counter[str]:
    degrees: Counter[str] = Counter()
    for comparison in comparisons:
        degrees[comparison.left_id] += 1
        degrees[comparison.right_id] += 1
    return degrees


def _rank_scores(
    items: list[_PosteriorItem],
    *,
    key: Callable[[_PosteriorItem], tuple[float, ...]],
) -> dict[str, float]:
    if not items:
        return {}
    if len(items) == 1:
        return {items[0].paper_id: 1.0}
    ranked = sorted(items, key=key, reverse=True)
    denominator = max(1, len(ranked) - 1)
    return {
        item.paper_id: round(1.0 - (index / denominator), 8)
        for index, item in enumerate(ranked)
    }


def _numeric_distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    return {
        "count": len(values),
        "min": round(min(values), 8),
        "max": round(max(values), 8),
        "mean": round(sum(values) / len(values), 8),
    }


def _proposal_pool_profile(
    *,
    items: list[_PosteriorItem],
    pool: list[_PosteriorItem],
    scheduled: list[ScheduledPair],
    k: int,
) -> dict[str, Any]:
    touched = {
        paper_id
        for pair in scheduled
        for paper_id in (pair.left_id, pair.right_id)
    }
    by_top_k = sorted(
        items,
        key=lambda item: (item.top_k_probability, item.mean, item.paper_id),
        reverse=True,
    )
    plausible = by_top_k[: max(1, min(len(by_top_k), 2 * max(k, 1)))]
    top_k_ids = {item.paper_id for item in by_top_k[: max(k, 1)]}
    high_ucb_outsiders = [
        item
        for item in sorted(
            items,
            key=lambda item: (item.ucb, item.top_k_probability, item.paper_id),
            reverse=True,
        )
        if item.paper_id not in top_k_ids
    ][: max(k, 1)]
    exposed_high_ucb = [
        item.paper_id for item in high_ucb_outsiders if item.paper_id in touched
    ]
    plausible_touched = [
        item.paper_id for item in plausible if item.paper_id in touched
    ]
    return {
        "pool_item_count": len(pool),
        "plausible_top_k_total": len(plausible),
        "plausible_top_k_touched": len(plausible_touched),
        "plausible_top_k_touch_rate": _rate(len(plausible_touched), len(plausible)),
        "high_ucb_outsider_total": len(high_ucb_outsiders),
        "high_ucb_outsider_touched": len(exposed_high_ucb),
        "high_ucb_outsider_exposure_rate": _rate(
            len(exposed_high_ucb),
            len(high_ucb_outsiders),
        ),
        "scheduled_unique_papers": len(touched),
    }


def _evsi_score_distribution(proposals: list[_AcquisitionProposal]) -> dict[str, Any]:
    total = len(proposals)
    if total == 0:
        return {
            "proposal_count": 0,
            "zero_score_total": 0,
            "zero_score_rate": 0.0,
            "tied_score_total": 0,
            "tied_score_rate": 0.0,
        }
    zero_total = sum(1 for proposal in proposals if proposal.score <= 1e-12)
    rounded_scores = Counter(round(proposal.score, 12) for proposal in proposals)
    tied_total = sum(count for count in rounded_scores.values() if count > 1)
    return {
        "proposal_count": total,
        "zero_score_total": zero_total,
        "zero_score_rate": _rate(zero_total, total),
        "tied_score_total": tied_total,
        "tied_score_rate": _rate(tied_total, total),
    }


def _posterior_top_k_entropy(posterior: TopKPosterior) -> float:
    return round(
        sum(_binary_entropy(probability) for probability in posterior.top_k_probabilities.values()),
        8,
    )


def _posterior_top_k_set(posterior: TopKPosterior, *, k: int) -> set[str]:
    ranked = sorted(
        posterior.top_k_probabilities.items(),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    return {paper_id for paper_id, _ in ranked[: max(k, 0)]}


def _top_k_set_churn(before: set[str], after: set[str]) -> float:
    if not before and not after:
        return 0.0
    return _rate(len(before.symmetric_difference(after)), len(before | after))


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def _empty_schedule_diagnostics(
    *,
    k: int,
    budget: int,
    method: str,
) -> dict[str, Any]:
    return {
        "candidate_count": 0,
        "scheduled_total": 0,
        "pairs_considered": 0,
        "unique_pairs_considered": 0,
        "budget": budget,
        "k": k,
        "acquisition": {"method": method},
        "purpose_counts": {},
        "coverage": {
            "purpose_counts": {},
            "pair_role_counts": {},
            "incumbent_challenger_pairs": 0,
            "metadata_cross_bucket_pairs": 0,
        },
        "proposal_pool_profile": {
            "pool_item_count": 0,
            "plausible_top_k_total": 0,
            "plausible_top_k_touched": 0,
            "plausible_top_k_touch_rate": 0.0,
            "high_ucb_outsider_total": 0,
            "high_ucb_outsider_touched": 0,
            "high_ucb_outsider_exposure_rate": 0.0,
            "scheduled_unique_papers": 0,
        },
        "evsi_score_distribution": _evsi_score_distribution([]),
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


def _invert_winner(winner: str) -> str:
    if winner == "left":
        return "right"
    if winner == "right":
        return "left"
    return winner
