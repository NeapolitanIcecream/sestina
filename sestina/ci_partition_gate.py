from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from sestina.diagnostics import DiagnosticRecorder
from sestina.evsi_scheduler import (
    EVSISchedulerConfig,
    _build_evsi_context,
    _evsi_score_distribution,
    _pair_key,
    _proposal_pool_profile,
)
from sestina.models import (
    PairwiseComparison,
    PairwiseOrderMetadata,
    Paper,
    ScheduledPair,
)
from sestina.scheduler import PairSchedule, PairwiseBudget


@dataclass(frozen=True, slots=True)
class CIPartitionConfig:
    prior_strength: float = 2.0
    confidence_z: float = 1.96
    pairwise_strength: float = 2.5
    posterior_samples: int = 1200
    boundary_window: float = 4.0
    pool_multiplier: int = 2
    diverse_outsider_count: int | None = None
    per_item_cap: int | None = 6
    random_floor_fraction: float = 0.25
    min_random_floor_pairs: int = 1
    batch_size: int = 5


@dataclass(frozen=True, slots=True)
class ReliabilityAwareCIPartitionV2Config(CIPartitionConfig):
    min_cached_incident_support: int = 4
    min_effective_pairwise_n: float = 1.25
    reliable_pair_threshold: float = 0.55
    low_reliability_boundary_threshold: float = 0.55
    low_reliability_unresolved_fraction: float = 0.85
    low_reliability_random_floor_fraction: float = 0.5
    exclude_unstable_ci_decisions: bool = True


@dataclass(frozen=True, slots=True)
class CIItemInterval:
    paper_id: str
    alpha: float
    beta: float
    mean: float
    lower: float
    upper: float
    effective_pairwise_n: float
    comparisons_used: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "paper_id": self.paper_id,
            "alpha": round(self.alpha, 8),
            "beta": round(self.beta, 8),
            "mean": round(self.mean, 8),
            "lower": round(self.lower, 8),
            "upper": round(self.upper, 8),
            "effective_pairwise_n": round(self.effective_pairwise_n, 8),
            "comparisons_used": self.comparisons_used,
        }


@dataclass(frozen=True, slots=True)
class CIPartitionState:
    intervals: dict[str, CIItemInterval]
    ranked_ids: list[str]
    top_k_ids: list[str]
    unresolved_ids: list[str]
    unresolved_top_k_ids: list[str]
    unresolved_outside_ids: list[str]
    resolved_top_k_ids: list[str]
    eliminated_ids: list[str]
    kth_lower_bound: float
    best_outside_upper_bound: float

    @property
    def unresolved_count(self) -> int:
        return len(self.unresolved_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranked_ids": list(self.ranked_ids),
            "top_k_ids": list(self.top_k_ids),
            "unresolved_ids": list(self.unresolved_ids),
            "unresolved_count": self.unresolved_count,
            "unresolved_top_k_ids": list(self.unresolved_top_k_ids),
            "unresolved_outside_ids": list(self.unresolved_outside_ids),
            "resolved_top_k_ids": list(self.resolved_top_k_ids),
            "eliminated_ids": list(self.eliminated_ids),
            "kth_lower_bound": round(self.kth_lower_bound, 8),
            "best_outside_upper_bound": round(self.best_outside_upper_bound, 8),
            "intervals": {
                paper_id: interval.to_dict()
                for paper_id, interval in sorted(self.intervals.items())
            },
        }


@dataclass(frozen=True, slots=True)
class CIReplayResult:
    schedule: list[ScheduledPair]
    comparisons: list[PairwiseComparison]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def confidence_interval_partition(
    papers: Sequence[Paper],
    comparisons: Sequence[PairwiseComparison],
    *,
    k: int,
    config: CIPartitionConfig | None = None,
) -> CIPartitionState:
    """Estimate noisy top-K partition bounds from fractional pairwise evidence."""
    cfg = config or CIPartitionConfig()
    intervals = _item_intervals(papers, comparisons, config=cfg)
    ranked_ids = sorted(
        intervals,
        key=lambda paper_id: (
            intervals[paper_id].mean,
            intervals[paper_id].lower,
            paper_id,
        ),
        reverse=True,
    )
    top_k_ids = ranked_ids[: max(0, k)]
    outside_ids = ranked_ids[max(0, k) :]
    if not top_k_ids:
        kth_lower = 0.0
    else:
        kth_id = top_k_ids[-1]
        kth_lower = intervals[kth_id].lower
    best_outside_upper = (
        max((intervals[paper_id].upper for paper_id in outside_ids), default=0.0)
        if outside_ids
        else 0.0
    )
    unresolved_top = [
        paper_id
        for paper_id in top_k_ids
        if intervals[paper_id].lower <= best_outside_upper
    ]
    unresolved_outside = [
        paper_id
        for paper_id in outside_ids
        if intervals[paper_id].upper >= kth_lower
    ]
    unresolved = sorted(
        set(unresolved_top) | set(unresolved_outside),
        key=lambda paper_id: (
            abs(intervals[paper_id].mean - _boundary_mean(intervals, top_k_ids)),
            paper_id,
        ),
    )
    resolved_top = [
        paper_id
        for paper_id in top_k_ids
        if intervals[paper_id].lower > best_outside_upper
    ]
    eliminated = [
        paper_id
        for paper_id in outside_ids
        if intervals[paper_id].upper < kth_lower
    ]
    return CIPartitionState(
        intervals=intervals,
        ranked_ids=ranked_ids,
        top_k_ids=top_k_ids,
        unresolved_ids=unresolved,
        unresolved_top_k_ids=unresolved_top,
        unresolved_outside_ids=unresolved_outside,
        resolved_top_k_ids=resolved_top,
        eliminated_ids=eliminated,
        kth_lower_bound=kth_lower,
        best_outside_upper_bound=best_outside_upper,
    )


def schedule_ci_partition_pairs(
    papers: list[Paper],
    comparisons: list[PairwiseComparison],
    *,
    k: int,
    budget: PairwiseBudget,
    seed: int = 0,
    config: CIPartitionConfig | None = None,
    available_pair_keys: set[tuple[str, str]] | None = None,
    seen_pair_keys: set[tuple[str, str]] | None = None,
    diagnostics: DiagnosticRecorder | None = None,
) -> PairSchedule:
    cfg = config or CIPartitionConfig()
    acquisition_method = _ci_acquisition_method(cfg)
    recorder = diagnostics or DiagnosticRecorder()
    if budget.budget <= 0 or len(papers) < 2 or k <= 0:
        payload = _empty_schedule_diagnostics(
            k=k,
            budget=budget.budget,
            method=acquisition_method,
        )
        recorder.record(
            step="pair_scheduling",
            code="ci_partition_pair_scheduling_empty",
            message="no CI partition comparisons scheduled",
            data=payload,
        )
        return PairSchedule(pairs=[], budget=budget, diagnostics=payload)

    selected_or_seen = {
        _pair_key(comparison.left_id, comparison.right_id)
        for comparison in comparisons
    }
    selected_or_seen.update(seen_pair_keys or set())
    evsi_config = _evsi_config(cfg)
    context = _build_evsi_context(
        papers,
        comparisons=comparisons,
        k=k,
        seed=seed,
        config=evsi_config,
        seen_pairs=selected_or_seen,
        diagnostics=recorder,
    )
    all_proposals = list(context.proposals)
    if available_pair_keys is None:
        feasible = all_proposals
    else:
        feasible = [
            proposal
            for proposal in all_proposals
            if _pair_key(proposal.left_id, proposal.right_id) in available_pair_keys
        ]
    state = confidence_interval_partition(papers, comparisons, k=k, config=cfg)
    reliability = _ci_reliability_diagnostics(
        state,
        available_pair_keys=available_pair_keys,
        feasible=feasible,
        target=min(budget.budget, len(feasible)),
        config=cfg,
    )
    if not feasible:
        payload = _empty_schedule_diagnostics(
            k=k,
            budget=budget.budget,
            method=acquisition_method,
        )
        payload.update(
            _proposal_filter_payload(
                all_proposals=all_proposals,
                feasible=feasible,
                available_pair_keys=available_pair_keys,
            )
        )
        payload["ci_partition"] = state.to_dict()
        recorder.record(
            step="pair_scheduling",
            code="ci_partition_pair_scheduling_empty",
            message="no CI partition feasible cached comparisons scheduled",
            data=payload,
        )
        return PairSchedule(pairs=[], budget=budget, diagnostics=payload)

    target = min(budget.budget, len(feasible))
    floor_target = _reliability_aware_random_floor_target(
        target,
        cfg,
        reliability=reliability,
    )
    rng = random.Random(seed)
    selected: list[tuple[Any, str, float, dict[str, Any]]] = []
    selected_keys: set[tuple[str, str]] = set()
    item_counts: Counter[str] = Counter()

    random_candidates = list(feasible)
    rng.shuffle(random_candidates)
    _take_proposals(
        random_candidates,
        selected=selected,
        selected_keys=selected_keys,
        item_counts=item_counts,
        limit=floor_target,
        cap=cfg.per_item_cap,
        purpose="ci_random_coverage_floor",
        state=state,
        score_fn=lambda proposal: float(proposal.score),
        extra_diagnostics_fn=(
            lambda proposal: _ci_v2_pair_reliability_diagnostics(
                proposal,
                reliability=reliability,
                config=cfg,
            )
            if reliability is not None
            else {}
        ),
    )

    stable_feasible = (
        [
            proposal
            for proposal in feasible
            if not _ci_v2_pair_is_unstable(
                proposal,
                reliability=reliability,
                config=cfg,
            )
        ]
        if isinstance(cfg, ReliabilityAwareCIPartitionV2Config)
        else list(feasible)
    )
    scored = sorted(
        stable_feasible,
        key=lambda proposal: _ci_pair_priority_for_config(
            proposal,
            state,
            reliability=reliability,
            config=cfg,
        ),
        reverse=True,
    )
    _take_proposals(
        scored,
        selected=selected,
        selected_keys=selected_keys,
        item_counts=item_counts,
        limit=target,
        cap=cfg.per_item_cap,
        purpose=None,
        state=state,
        score_fn=lambda proposal: _ci_pair_priority_for_config(
            proposal,
            state,
            reliability=reliability,
            config=cfg,
        ),
        extra_diagnostics_fn=(
            lambda proposal: _ci_v2_pair_reliability_diagnostics(
                proposal,
                reliability=reliability,
                config=cfg,
            )
            if reliability is not None
            else {}
        ),
    )
    if len(selected) < target:
        fill_candidates = (
            random_candidates
            if isinstance(cfg, ReliabilityAwareCIPartitionV2Config)
            else scored
        )
        _take_proposals(
            fill_candidates,
            selected=selected,
            selected_keys=selected_keys,
            item_counts=item_counts,
            limit=target,
            cap=None,
            purpose=(
                "ci_v2_low_reliability_random_fallback"
                if isinstance(cfg, ReliabilityAwareCIPartitionV2Config)
                else None
            ),
            state=state,
            score_fn=lambda proposal: _ci_pair_priority_for_config(
                proposal,
                state,
                reliability=reliability,
                config=cfg,
            ),
            extra_diagnostics_fn=(
                lambda proposal: _ci_v2_pair_reliability_diagnostics(
                    proposal,
                    reliability=reliability,
                    config=cfg,
                )
                if reliability is not None
                else {}
            ),
        )

    scheduled = [
        _scheduled_pair_from_selection(
            proposal,
            purpose=purpose,
            priority=priority,
            diagnostics=proposal_diagnostics,
            seed=seed,
            index=index,
        )
        for index, (proposal, purpose, priority, proposal_diagnostics) in enumerate(
            selected[:target],
            start=1,
        )
    ]
    payload = {
        "candidate_count": len(context.pool),
        "scheduled_total": len(scheduled),
        "pairs_considered": len(all_proposals),
        "unique_pairs_considered": len(
            {_pair_key(proposal.left_id, proposal.right_id) for proposal in all_proposals}
        ),
        "budget": budget.budget,
        "k": k,
        "acquisition": {
            "method": acquisition_method,
            "source_method": "exact_evsi_feasible_pool",
            "random_seed": seed,
            "posterior_samples": cfg.posterior_samples,
            "pairwise_strength": cfg.pairwise_strength,
            "confidence_z": cfg.confidence_z,
            "prior_strength": cfg.prior_strength,
            "random_floor_fraction": cfg.random_floor_fraction,
            "min_random_floor_pairs": cfg.min_random_floor_pairs,
            "random_floor_target": floor_target,
            "per_item_cap": cfg.per_item_cap,
            "model_visible_signals": [
                "pointwise_probability_prior",
                "pairwise_soft_probability",
                "pairwise_confidence",
                "confidence_interval_overlap",
                "top_k_boundary_unresolved_status",
                "posterior_evsi_feasible_pool",
            ],
        },
        "purpose_counts": dict(sorted(Counter(pair.purpose for pair in scheduled).items())),
        "ci_partition": state.to_dict(),
        "ci_reliability": _ci_reliability_payload(reliability),
        "coverage": _ci_coverage(scheduled, state=state),
        "proposal_pool_profile": _proposal_pool_profile(
            items=context.items,
            pool=context.pool,
            scheduled=scheduled,
            k=k,
        ),
        "evsi_score_distribution": _evsi_score_distribution(all_proposals),
        **_proposal_filter_payload(
            all_proposals=all_proposals,
            feasible=feasible,
            available_pair_keys=available_pair_keys,
        ),
    }
    recorder.record(
        step="pair_scheduling",
        code="ci_partition_pair_scheduling_completed",
        message="scheduled confidence-interval top-K partition comparisons",
        data=payload,
    )
    return PairSchedule(pairs=scheduled, budget=budget, diagnostics=payload)


def replay_ci_partition_gate(
    papers: list[Paper],
    cached_comparisons: Mapping[tuple[str, str], PairwiseComparison],
    *,
    k: int,
    budget: PairwiseBudget,
    seed: int = 0,
    config: CIPartitionConfig | None = None,
) -> CIReplayResult:
    cfg = config or CIPartitionConfig()
    acquisition_method = _ci_acquisition_method(cfg)
    cached_by_key = {
        _canonical_pair_key(left_id, right_id): comparison
        for (left_id, right_id), comparison in cached_comparisons.items()
    }
    selected: list[ScheduledPair] = []
    revealed: list[PairwiseComparison] = []
    selected_keys: set[tuple[str, str]] = set()
    round_rows: list[dict[str, Any]] = []
    missing_labels = 0
    novel_pairs = 0
    initial_state = confidence_interval_partition(
        papers,
        revealed,
        k=k,
        config=cfg,
    )
    while len(selected) < budget.budget:
        remaining = budget.budget - len(selected)
        round_budget = PairwiseBudget(
            n=budget.n,
            candidate_size=budget.candidate_size,
            budget=min(max(1, cfg.batch_size), remaining),
            source=budget.source,
        )
        before = confidence_interval_partition(papers, revealed, k=k, config=cfg)
        schedule = schedule_ci_partition_pairs(
            papers,
            revealed,
            k=k,
            budget=round_budget,
            seed=seed + (1009 * (len(round_rows) + 1)),
            config=cfg,
            available_pair_keys=set(cached_by_key),
            seen_pair_keys=selected_keys,
        )
        if not schedule.pairs:
            round_rows.append(
                {
                    "round_index": len(round_rows) + 1,
                    "selected_total": 0,
                    "comparisons_before_round": len(revealed),
                    "comparisons_after_round": len(revealed),
                    "unresolved_before": before.unresolved_count,
                    "unresolved_after": before.unresolved_count,
                    "stop_reason": "no_cached_feasible_pairs",
                    "scheduler_diagnostics": schedule.diagnostics,
                }
            )
            break
        cached_in_round = 0
        for pair in schedule.pairs:
            key = _canonical_pair_key(pair.left_id, pair.right_id)
            selected.append(pair)
            selected_keys.add(key)
            cached = cached_by_key.get(key)
            if cached is None:
                missing_labels += 1
                novel_pairs += 1
                continue
            revealed.append(_orient_comparison(cached, pair))
            cached_in_round += 1
        after = confidence_interval_partition(papers, revealed, k=k, config=cfg)
        round_rows.append(
            {
                "round_index": len(round_rows) + 1,
                "selected_total": len(schedule.pairs),
                "cached_label_revealed_total": cached_in_round,
                "comparisons_before_round": len(revealed) - cached_in_round,
                "comparisons_after_round": len(revealed),
                "unresolved_before": before.unresolved_count,
                "unresolved_after": after.unresolved_count,
                "unresolved_delta": after.unresolved_count - before.unresolved_count,
                "purpose_counts": schedule.diagnostics.get("purpose_counts", {}),
                "coverage": schedule.diagnostics.get("coverage", {}),
                "ci_reliability": schedule.diagnostics.get("ci_reliability"),
                "available_label_filter": schedule.diagnostics.get(
                    "available_label_filter",
                    {},
                ),
                "stop_reason": None,
            }
        )
        if len(schedule.pairs) == 0:
            break
    final_state = confidence_interval_partition(papers, revealed, k=k, config=cfg)
    diagnostics = {
        "method": f"{acquisition_method}_cached_replay",
        "budget": budget.to_dict(),
        "scheduled_total": len(selected),
        "comparisons_revealed_total": len(revealed),
        "rounds_total": len(round_rows),
        "batch_size": cfg.batch_size,
        "initial_ci_partition": initial_state.to_dict(),
        "final_ci_partition": final_state.to_dict(),
        "confidence_bound_unresolved_count": final_state.unresolved_count,
        "round_history": round_rows,
        "purpose_counts": dict(sorted(Counter(pair.purpose for pair in selected).items())),
        "coverage": _ci_coverage(selected, state=final_state),
        "ci_reliability": _ci_replay_reliability_summary(round_rows),
        "available_label_filter": {
            "cached_pair_keys_total": len(cached_by_key),
            "scheduled_cached_pair_keys_total": len(selected_keys & set(cached_by_key)),
            "cached_pair_key_touch_rate": _rate(
                len(selected_keys & set(cached_by_key)),
                len(cached_by_key),
            ),
        },
        "label_policy": {
            "offline_cached_pairwise_labels_only": True,
            "missing_pairwise_labels": missing_labels,
            "novel_pairs_scheduled": novel_pairs,
            "future_labels_used_for_scheduling": False,
            "cached_label_values_used_before_scheduling": False,
            "cache_availability_used_for_scheduling": True,
        },
    }
    return CIReplayResult(
        schedule=selected,
        comparisons=revealed,
        diagnostics=diagnostics,
    )


def replay_reliability_aware_ci_partition_v2_gate(
    papers: list[Paper],
    cached_comparisons: Mapping[tuple[str, str], PairwiseComparison],
    *,
    k: int,
    budget: PairwiseBudget,
    seed: int = 0,
    config: ReliabilityAwareCIPartitionV2Config | None = None,
) -> CIReplayResult:
    """Replay conservative CI partition v2 using only cached label availability."""
    return replay_ci_partition_gate(
        papers,
        cached_comparisons,
        k=k,
        budget=budget,
        seed=seed,
        config=config or ReliabilityAwareCIPartitionV2Config(),
    )


def schedule_reliability_aware_ci_partition_v2_pairs(
    papers: list[Paper],
    comparisons: list[PairwiseComparison],
    *,
    k: int,
    budget: PairwiseBudget,
    seed: int = 0,
    config: ReliabilityAwareCIPartitionV2Config | None = None,
    available_pair_keys: set[tuple[str, str]] | None = None,
    seen_pair_keys: set[tuple[str, str]] | None = None,
    diagnostics: DiagnosticRecorder | None = None,
) -> PairSchedule:
    """Schedule v2 pairs with conservative reliability-gated CI decisions."""
    return schedule_ci_partition_pairs(
        papers,
        comparisons,
        k=k,
        budget=budget,
        seed=seed,
        config=config or ReliabilityAwareCIPartitionV2Config(),
        available_pair_keys=available_pair_keys,
        seen_pair_keys=seen_pair_keys,
        diagnostics=diagnostics,
    )


def schedule_cached_exact_pool_random(
    papers: list[Paper],
    comparisons: list[PairwiseComparison],
    *,
    k: int,
    budget: PairwiseBudget,
    seed: int,
    config: CIPartitionConfig | None = None,
    available_pair_keys: set[tuple[str, str]] | None = None,
) -> PairSchedule:
    cfg = config or CIPartitionConfig()
    recorder = DiagnosticRecorder()
    selected_or_seen = {
        _pair_key(comparison.left_id, comparison.right_id)
        for comparison in comparisons
    }
    context = _build_evsi_context(
        papers,
        comparisons=comparisons,
        k=k,
        seed=seed,
        config=_evsi_config(cfg),
        seen_pairs=selected_or_seen,
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
    target = min(budget.budget, len(feasible))
    rng = random.Random(seed)
    candidates = list(feasible)
    rng.shuffle(candidates)
    selected: list[tuple[Any, str, float, dict[str, Any]]] = []
    selected_keys: set[tuple[str, str]] = set()
    item_counts: Counter[str] = Counter()
    _take_proposals(
        candidates,
        selected=selected,
        selected_keys=selected_keys,
        item_counts=item_counts,
        limit=target,
        cap=cfg.per_item_cap,
        purpose="exact_pool_random_cached_replay",
        state=None,
        score_fn=lambda proposal: float(proposal.score),
    )
    if len(selected) < target:
        _take_proposals(
            candidates,
            selected=selected,
            selected_keys=selected_keys,
            item_counts=item_counts,
            limit=target,
            cap=None,
            purpose="exact_pool_random_cached_replay",
            state=None,
            score_fn=lambda proposal: float(proposal.score),
        )
    scheduled = [
        _scheduled_pair_from_selection(
            proposal,
            purpose=purpose,
            priority=priority,
            diagnostics=proposal_diagnostics,
            seed=seed,
            index=index,
        )
        for index, (proposal, purpose, priority, proposal_diagnostics) in enumerate(
            selected,
            start=1,
        )
    ]
    payload = {
        "candidate_count": len(context.pool),
        "scheduled_total": len(scheduled),
        "pairs_considered": len(all_proposals),
        "unique_pairs_considered": len(
            {_pair_key(proposal.left_id, proposal.right_id) for proposal in all_proposals}
        ),
        "budget": budget.budget,
        "k": k,
        "acquisition": {
            "method": "exact_pool_random_cached_replay",
            "source_method": "exact_evsi_feasible_pool",
            "random_seed": seed,
            "posterior_samples": cfg.posterior_samples,
            "per_item_cap": cfg.per_item_cap,
            "selection_policy": "random_within_cached_exact_evsi_feasible_pool",
        },
        "purpose_counts": dict(sorted(Counter(pair.purpose for pair in scheduled).items())),
        "coverage": {
            "random_floor_pairs": len(scheduled),
            "random_floor_rate": 1.0 if scheduled else 0.0,
            "scheduled_unique_papers": len(
                {
                    paper_id
                    for pair in scheduled
                    for paper_id in (pair.left_id, pair.right_id)
                }
            ),
        },
        "proposal_pool_profile": _proposal_pool_profile(
            items=context.items,
            pool=context.pool,
            scheduled=scheduled,
            k=k,
        ),
        "evsi_score_distribution": _evsi_score_distribution(all_proposals),
        **_proposal_filter_payload(
            all_proposals=all_proposals,
            feasible=feasible,
            available_pair_keys=available_pair_keys,
        ),
    }
    return PairSchedule(pairs=scheduled, budget=budget, diagnostics=payload)


def _item_intervals(
    papers: Sequence[Paper],
    comparisons: Sequence[PairwiseComparison],
    *,
    config: CIPartitionConfig,
) -> dict[str, CIItemInterval]:
    known_ids = {paper.paper_id for paper in papers}
    alpha = {
        paper.paper_id: 1.0
        + (config.prior_strength * paper.pointwise.good_probability)
        for paper in papers
    }
    beta = {
        paper.paper_id: 1.0
        + (config.prior_strength * (1.0 - paper.pointwise.good_probability))
        for paper in papers
    }
    effective = Counter({paper.paper_id: 0.0 for paper in papers})
    used = Counter({paper.paper_id: 0 for paper in papers})
    for comparison in comparisons:
        if (
            comparison.left_id not in known_ids
            or comparison.right_id not in known_ids
            or comparison.left_id == comparison.right_id
        ):
            continue
        p_left, weight = _comparison_probability_and_weight(comparison)
        if weight <= 0.0:
            continue
        alpha[comparison.left_id] += weight * p_left
        beta[comparison.left_id] += weight * (1.0 - p_left)
        alpha[comparison.right_id] += weight * (1.0 - p_left)
        beta[comparison.right_id] += weight * p_left
        effective[comparison.left_id] += weight
        effective[comparison.right_id] += weight
        used[comparison.left_id] += 1
        used[comparison.right_id] += 1
    return {
        paper.paper_id: _interval_from_alpha_beta(
            paper.paper_id,
            alpha=alpha[paper.paper_id],
            beta=beta[paper.paper_id],
            effective_pairwise_n=effective[paper.paper_id],
            comparisons_used=used[paper.paper_id],
            z=config.confidence_z,
        )
        for paper in papers
    }


def _interval_from_alpha_beta(
    paper_id: str,
    *,
    alpha: float,
    beta: float,
    effective_pairwise_n: float,
    comparisons_used: int,
    z: float,
) -> CIItemInterval:
    total = max(alpha + beta, 1e-9)
    mean = alpha / total
    variance = (alpha * beta) / ((total * total) * (total + 1.0))
    half_width = max(0.0, z * math.sqrt(max(variance, 0.0)))
    return CIItemInterval(
        paper_id=paper_id,
        alpha=alpha,
        beta=beta,
        mean=mean,
        lower=max(0.0, mean - half_width),
        upper=min(1.0, mean + half_width),
        effective_pairwise_n=float(effective_pairwise_n),
        comparisons_used=comparisons_used,
    )


def _comparison_probability_and_weight(
    comparison: PairwiseComparison,
) -> tuple[float, float]:
    confidence = max(0.0, min(1.0, comparison.confidence))
    if comparison.winner == "tie":
        return 0.5, 0.35 * confidence
    if comparison.winner == "uncertain":
        return 0.5, 0.15 * confidence
    soft = comparison.soft_probability
    if soft is None:
        soft = 0.75
    soft = max(0.5, min(0.999, float(soft)))
    if comparison.winner == "left":
        return soft, confidence
    return 1.0 - soft, confidence


def _ci_pair_priority(proposal: Any, state: CIPartitionState) -> float:
    left = state.intervals[proposal.left_id]
    right = state.intervals[proposal.right_id]
    unresolved = set(state.unresolved_ids)
    top_k = set(state.top_k_ids)
    left_unresolved = proposal.left_id in unresolved
    right_unresolved = proposal.right_id in unresolved
    crosses_boundary = (proposal.left_id in top_k) != (proposal.right_id in top_k)
    overlap = max(0.0, min(left.upper, right.upper) - max(left.lower, right.lower))
    interval_width = (left.upper - left.lower) + (right.upper - right.lower)
    entropy = float(proposal.diagnostics.get("head_to_head_entropy", 0.0))
    boundary_relevance = float(proposal.diagnostics.get("boundary_relevance", 0.0))
    sparse_bonus = 1.0 / (1.0 + min(left.effective_pairwise_n, right.effective_pairwise_n))
    return (
        (2.5 if crosses_boundary else 0.0)
        + (1.5 * int(left_unresolved))
        + (1.5 * int(right_unresolved))
        + overlap
        + (0.35 * interval_width)
        + (0.35 * entropy)
        + (0.25 * boundary_relevance)
        + (0.25 * sparse_bonus)
    )


def _ci_pair_priority_for_config(
    proposal: Any,
    state: CIPartitionState,
    *,
    reliability: dict[str, Any] | None,
    config: CIPartitionConfig,
) -> float:
    base = _ci_pair_priority(proposal, state)
    if not isinstance(config, ReliabilityAwareCIPartitionV2Config):
        return base
    pair_reliability = _ci_v2_pair_reliability(
        proposal,
        reliability=reliability,
    )
    boundary_bonus = 0.75 if pair_reliability >= config.reliable_pair_threshold else 0.0
    unstable_penalty = (
        1.0
        if _ci_v2_pair_is_unstable(
            proposal,
            reliability=reliability,
            config=config,
        )
        else 0.0
    )
    return max(0.0, (base * (0.35 + (0.65 * pair_reliability))) + boundary_bonus - unstable_penalty)


def _ci_reliability_diagnostics(
    state: CIPartitionState,
    *,
    available_pair_keys: set[tuple[str, str]] | None,
    feasible: Sequence[Any],
    target: int,
    config: CIPartitionConfig,
) -> dict[str, Any] | None:
    if not isinstance(config, ReliabilityAwareCIPartitionV2Config):
        return None
    cached_support = Counter()
    for left_id, right_id in available_pair_keys or set():
        cached_support[left_id] += 1
        cached_support[right_id] += 1
    feasible_support = Counter()
    for proposal in feasible:
        feasible_support[proposal.left_id] += 1
        feasible_support[proposal.right_id] += 1
    item_rows = {}
    for paper_id, interval in state.intervals.items():
        cached_factor = _bounded_ratio(
            feasible_support[paper_id],
            config.min_cached_incident_support,
        )
        observed_factor = _bounded_ratio(
            interval.effective_pairwise_n,
            config.min_effective_pairwise_n,
        )
        reliability = (0.7 * cached_factor) + (0.3 * observed_factor)
        item_rows[paper_id] = {
            "cached_incident_support": int(cached_support[paper_id]),
            "feasible_cached_incident_support": int(feasible_support[paper_id]),
            "effective_pairwise_n": round(interval.effective_pairwise_n, 8),
            "comparisons_used": interval.comparisons_used,
            "cached_support_factor": round(cached_factor, 8),
            "observed_evidence_factor": round(observed_factor, 8),
            "reliability": round(reliability, 8),
            "unresolved": paper_id in set(state.unresolved_ids),
            "top_k": paper_id in set(state.top_k_ids),
        }
    boundary_ids = sorted(set(state.unresolved_ids) | set(state.top_k_ids))
    boundary_reliability = [
        float(item_rows[paper_id]["reliability"])
        for paper_id in boundary_ids
        if paper_id in item_rows
    ]
    low_boundary = [
        paper_id
        for paper_id in boundary_ids
        if float(item_rows[paper_id]["reliability"])
        < config.reliable_pair_threshold
    ]
    unresolved_fraction = _rate(state.unresolved_count, len(state.intervals))
    mean_boundary_reliability = _mean_float(boundary_reliability)
    low_reliability_fallback = bool(
        mean_boundary_reliability < config.low_reliability_boundary_threshold
        or unresolved_fraction >= config.low_reliability_unresolved_fraction
    )
    return {
        "policy": {
            "method": "reliability_aware_ci_partition_v2",
            "min_cached_incident_support": config.min_cached_incident_support,
            "min_effective_pairwise_n": config.min_effective_pairwise_n,
            "reliable_pair_threshold": config.reliable_pair_threshold,
            "low_reliability_boundary_threshold": (
                config.low_reliability_boundary_threshold
            ),
            "low_reliability_unresolved_fraction": (
                config.low_reliability_unresolved_fraction
            ),
            "low_reliability_random_floor_fraction": (
                config.low_reliability_random_floor_fraction
            ),
            "exclude_unstable_ci_decisions": (
                config.exclude_unstable_ci_decisions
            ),
        },
        "target_pairs": target,
        "item_count": len(state.intervals),
        "boundary_item_count": len(boundary_ids),
        "unresolved_count": state.unresolved_count,
        "unresolved_fraction": unresolved_fraction,
        "mean_boundary_item_reliability": round(mean_boundary_reliability, 8),
        "low_reliability_boundary_item_count": len(low_boundary),
        "low_reliability_boundary_item_rate": _rate(
            len(low_boundary),
            len(boundary_ids),
        ),
        "low_reliability_fallback_active": low_reliability_fallback,
        "stable_ci_candidate_count": sum(
            1
            for proposal in feasible
            if not _ci_v2_pair_is_unstable(
                proposal,
                reliability={"items": item_rows},
                config=config,
            )
        ),
        "items": dict(sorted(item_rows.items())),
    }


def _ci_replay_reliability_summary(
    round_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    rows = [
        row.get("ci_reliability")
        for row in round_rows
        if isinstance(row.get("ci_reliability"), Mapping)
    ]
    if not rows:
        return None
    return {
        "round_count": len(rows),
        "mean_boundary_item_reliability": round(
            _mean_float(
                [
                    _mapping_float(
                        row,
                        "mean_boundary_item_reliability",
                        default=0.0,
                    )
                    for row in rows
                ]
            ),
            8,
        ),
        "mean_low_reliability_boundary_item_rate": round(
            _mean_float(
                [
                    _mapping_float(
                        row,
                        "low_reliability_boundary_item_rate",
                        default=0.0,
                    )
                    for row in rows
                ]
            ),
            8,
        ),
        "low_reliability_fallback_round_rate": round(
            _mean_float(
                [
                    1.0
                    if row.get("low_reliability_fallback_active")
                    else 0.0
                    for row in rows
                ]
            ),
            8,
        ),
        "mean_unresolved_fraction": round(
            _mean_float(
                [
                    _mapping_float(row, "unresolved_fraction", default=0.0)
                    for row in rows
                ]
            ),
            8,
        ),
        "mean_stable_ci_candidate_count": round(
            _mean_float(
                [
                    _mapping_float(
                        row,
                        "stable_ci_candidate_count",
                        default=0.0,
                    )
                    for row in rows
                ]
            ),
            8,
        ),
    }


def _ci_reliability_payload(
    reliability: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if reliability is None:
        return None
    return {
        key: value
        for key, value in reliability.items()
        if key != "items"
    }


def _reliability_aware_random_floor_target(
    target: int,
    config: CIPartitionConfig,
    *,
    reliability: dict[str, Any] | None,
) -> int:
    base = _random_floor_target(target, config)
    if (
        not isinstance(config, ReliabilityAwareCIPartitionV2Config)
        or not reliability
        or not reliability.get("low_reliability_fallback_active")
    ):
        return base
    fallback = math.ceil(
        target * max(0.0, config.low_reliability_random_floor_fraction)
    )
    return min(target, max(base, fallback))


def _ci_v2_pair_is_unstable(
    proposal: Any,
    *,
    reliability: dict[str, Any] | None,
    config: CIPartitionConfig,
) -> bool:
    if not isinstance(config, ReliabilityAwareCIPartitionV2Config):
        return False
    if not config.exclude_unstable_ci_decisions:
        return False
    return (
        _ci_v2_pair_reliability(proposal, reliability=reliability)
        < config.reliable_pair_threshold
    )


def _ci_v2_pair_reliability(
    proposal: Any,
    *,
    reliability: dict[str, Any] | None,
) -> float:
    if not reliability:
        return 1.0
    items = reliability.get("items")
    if not isinstance(items, Mapping):
        return 1.0
    left = items.get(proposal.left_id)
    right = items.get(proposal.right_id)
    left_value = _mapping_float(left, "reliability", default=0.0)
    right_value = _mapping_float(right, "reliability", default=0.0)
    return min(left_value, right_value)


def _ci_v2_pair_reliability_diagnostics(
    proposal: Any,
    *,
    reliability: dict[str, Any] | None,
    config: CIPartitionConfig,
) -> dict[str, Any]:
    if not isinstance(config, ReliabilityAwareCIPartitionV2Config):
        return {}
    pair_reliability = _ci_v2_pair_reliability(
        proposal,
        reliability=reliability,
    )
    items = reliability.get("items") if reliability else None
    items = items if isinstance(items, Mapping) else {}
    return {
        "ci_v2_pair_reliability": round(pair_reliability, 8),
        "ci_v2_unstable_ci_decision": _ci_v2_pair_is_unstable(
            proposal,
            reliability=reliability,
            config=config,
        ),
        "ci_v2_left_item_reliability": _mapping_float(
            items.get(proposal.left_id),
            "reliability",
            default=0.0,
        ),
        "ci_v2_right_item_reliability": _mapping_float(
            items.get(proposal.right_id),
            "reliability",
            default=0.0,
        ),
    }


def _proposal_purpose(proposal: Any, state: CIPartitionState) -> str:
    top_k = set(state.top_k_ids)
    unresolved = set(state.unresolved_ids)
    crosses_boundary = (proposal.left_id in top_k) != (proposal.right_id in top_k)
    touches_unresolved = proposal.left_id in unresolved or proposal.right_id in unresolved
    if crosses_boundary and touches_unresolved:
        return "ci_boundary_elimination"
    if touches_unresolved:
        return "ci_unresolved_local"
    return "ci_exploration"


def _take_proposals(
    proposals: Sequence[Any],
    *,
    selected: list[tuple[Any, str, float, dict[str, Any]]],
    selected_keys: set[tuple[str, str]],
    item_counts: Counter[str],
    limit: int,
    cap: int | None,
    purpose: str | None,
    state: CIPartitionState | None,
    score_fn: Any,
    extra_diagnostics_fn: Any | None = None,
) -> None:
    for proposal in proposals:
        if len(selected) >= limit:
            break
        key = _pair_key(proposal.left_id, proposal.right_id)
        if key in selected_keys:
            continue
        if cap is not None and (
            item_counts[proposal.left_id] >= cap
            or item_counts[proposal.right_id] >= cap
        ):
            continue
        selected_purpose = purpose
        if selected_purpose is None:
            if state is None:
                selected_purpose = proposal.purpose
            else:
                selected_purpose = _proposal_purpose(proposal, state)
        selected_keys.add(key)
        item_counts[proposal.left_id] += 1
        item_counts[proposal.right_id] += 1
        priority = score_fn(proposal)
        diagnostics = dict(proposal.diagnostics)
        diagnostics.update(
            {
                "source_evsi_purpose": proposal.purpose,
                "ci_priority": round(priority, 8),
            }
        )
        if state is not None:
            diagnostics.update(_ci_pair_diagnostics(proposal, state))
        if extra_diagnostics_fn is not None:
            diagnostics.update(extra_diagnostics_fn(proposal))
        selected.append((proposal, selected_purpose, priority, diagnostics))


def _ci_pair_diagnostics(proposal: Any, state: CIPartitionState) -> dict[str, Any]:
    left = state.intervals[proposal.left_id]
    right = state.intervals[proposal.right_id]
    top_k = set(state.top_k_ids)
    unresolved = set(state.unresolved_ids)
    overlap = max(0.0, min(left.upper, right.upper) - max(left.lower, right.lower))
    return {
        "ci_left_mean": round(left.mean, 8),
        "ci_right_mean": round(right.mean, 8),
        "ci_left_lower": round(left.lower, 8),
        "ci_left_upper": round(left.upper, 8),
        "ci_right_lower": round(right.lower, 8),
        "ci_right_upper": round(right.upper, 8),
        "ci_interval_overlap": round(overlap, 8),
        "ci_crosses_current_top_k_boundary": (
            proposal.left_id in top_k
        )
        != (proposal.right_id in top_k),
        "ci_touches_unresolved_item": (
            proposal.left_id in unresolved or proposal.right_id in unresolved
        ),
    }


def _scheduled_pair_from_selection(
    proposal: Any,
    *,
    purpose: str,
    priority: float,
    diagnostics: dict[str, Any],
    seed: int,
    index: int,
) -> ScheduledPair:
    rng = random.Random(seed + (104729 * index))
    if rng.random() < 0.5:
        shown_first_id = proposal.left_id
        shown_second_id = proposal.right_id
    else:
        shown_first_id = proposal.right_id
        shown_second_id = proposal.left_id
    return ScheduledPair(
        left_id=proposal.left_id,
        right_id=proposal.right_id,
        priority=round(priority, 8),
        purpose=purpose,
        order=PairwiseOrderMetadata(
            shown_first_id=shown_first_id,
            shown_second_id=shown_second_id,
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


def _ci_coverage(
    scheduled: Sequence[ScheduledPair],
    *,
    state: CIPartitionState,
) -> dict[str, Any]:
    top_k = set(state.top_k_ids)
    unresolved = set(state.unresolved_ids)
    touched = {
        paper_id
        for pair in scheduled
        for paper_id in (pair.left_id, pair.right_id)
    }
    random_floor = sum(
        1 for pair in scheduled if pair.purpose == "ci_random_coverage_floor"
    )
    return {
        "random_floor_pairs": random_floor,
        "random_floor_rate": _rate(random_floor, len(scheduled)),
        "low_reliability_random_fallback_pairs": sum(
            1
            for pair in scheduled
            if pair.purpose == "ci_v2_low_reliability_random_fallback"
        ),
        "boundary_elimination_pairs": sum(
            1 for pair in scheduled if pair.purpose == "ci_boundary_elimination"
        ),
        "unresolved_touch_pairs": sum(
            1
            for pair in scheduled
            if pair.left_id in unresolved or pair.right_id in unresolved
        ),
        "cross_current_top_k_boundary_pairs": sum(
            1
            for pair in scheduled
            if (pair.left_id in top_k) != (pair.right_id in top_k)
        ),
        "scheduled_unique_papers": len(touched),
        "scheduled_unresolved_papers": len(touched & unresolved),
        "scheduled_current_top_k_papers": len(touched & top_k),
    }


def _proposal_filter_payload(
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
            "random_floor_pairs": 0,
            "random_floor_rate": 0.0,
            "boundary_elimination_pairs": 0,
            "unresolved_touch_pairs": 0,
            "scheduled_unique_papers": 0,
        },
    }


def _random_floor_target(target: int, config: CIPartitionConfig) -> int:
    if target <= 0:
        return 0
    raw = math.floor(target * max(0.0, config.random_floor_fraction))
    floor = max(config.min_random_floor_pairs, raw)
    return min(target, floor)


def _evsi_config(config: CIPartitionConfig) -> EVSISchedulerConfig:
    return EVSISchedulerConfig(
        pairwise_strength=config.pairwise_strength,
        samples=config.posterior_samples,
        boundary_window=config.boundary_window,
        pool_multiplier=config.pool_multiplier,
        diverse_outsider_count=config.diverse_outsider_count,
        per_item_cap=config.per_item_cap,
    )


def _ci_acquisition_method(config: CIPartitionConfig) -> str:
    if isinstance(config, ReliabilityAwareCIPartitionV2Config):
        return "reliability_aware_ci_partition_v2"
    return "ci_partition_elimination"


def _orient_comparison(
    comparison: PairwiseComparison,
    pair: ScheduledPair,
) -> PairwiseComparison:
    if comparison.left_id == pair.left_id and comparison.right_id == pair.right_id:
        winner = comparison.winner
    elif comparison.left_id == pair.right_id and comparison.right_id == pair.left_id:
        winner = _invert_winner(comparison.winner)
    else:
        raise ValueError("comparison does not reference the scheduled pair")
    return PairwiseComparison(
        left_id=pair.left_id,
        right_id=pair.right_id,
        winner=winner,  # type: ignore[arg-type]
        soft_probability=comparison.soft_probability,
        confidence=comparison.confidence,
        reasons=list(comparison.reasons),
        order=pair.order,
        metadata={
            **comparison.metadata,
            "reused_original_left_id": comparison.left_id,
            "reused_original_right_id": comparison.right_id,
            "scheduled_pair_purpose": pair.purpose,
        },
    )


def _invert_winner(winner: str) -> str:
    if winner == "left":
        return "right"
    if winner == "right":
        return "left"
    return winner


def _canonical_pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    return tuple(sorted((left_id, right_id)))


def _boundary_mean(
    intervals: Mapping[str, CIItemInterval],
    top_k_ids: Sequence[str],
) -> float:
    if not top_k_ids:
        return 0.0
    return intervals[top_k_ids[-1]].mean


def _rate(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 8) if denominator else 0.0


def _bounded_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 1.0
    return max(0.0, min(1.0, float(numerator) / float(denominator)))


def _mean_float(values: Sequence[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _mapping_float(
    value: Any,
    key: str,
    *,
    default: float,
) -> float:
    if not isinstance(value, Mapping):
        return default
    raw = value.get(key)
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        return float(raw)
    return default
