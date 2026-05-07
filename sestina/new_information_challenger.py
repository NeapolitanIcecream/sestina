from __future__ import annotations

import math
import random
import re
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


DEFAULT_RUBRIC_KEYS = (
    "novelty",
    "evidence_strength",
    "practical_impact",
    "technical_depth",
    "cross_domain_interest",
)


@dataclass(frozen=True, slots=True)
class NewInformationChallengerConfig:
    pairwise_strength: float = 2.5
    posterior_samples: int = 1200
    boundary_window: float = 4.0
    pool_multiplier: int = 2
    per_item_cap: int | None = 6
    random_floor_fraction: float = 0.2
    min_random_floor_pairs: int = 1
    cached_fallback_enabled: bool = True
    cached_fallback_frontier_multiplier: int = 4
    anchor_multiplier: int = 2
    challenger_multiplier: int = 3
    min_challengers: int = 8
    max_challengers: int | None = None
    rubric_keys: tuple[str, ...] = DEFAULT_RUBRIC_KEYS
    minimum_rubric_residual: float = 0.02
    lexical_stopword_min_length: int = 4


@dataclass(frozen=True, slots=True)
class NewInformationReplayResult:
    schedule: list[ScheduledPair]
    comparisons: list[PairwiseComparison]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _NewInfoItem:
    paper: Paper
    pointwise_probability: float
    uncertainty: float
    rubric_signal: float
    rubric_residual: float
    boundary_proximity: float
    lexical_novelty: float
    metadata_diversity: float
    challenger_score: float
    metadata_bucket: str
    roles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _NewInfoProposal:
    left_id: str
    right_id: str
    score: float
    purpose: str
    diagnostics: dict[str, Any]


def schedule_new_information_challenger_pairs(
    papers: list[Paper],
    comparisons: list[PairwiseComparison],
    *,
    k: int,
    budget: PairwiseBudget,
    seed: int = 0,
    config: NewInformationChallengerConfig | None = None,
    available_pair_keys: set[tuple[str, str]] | None = None,
    seen_pair_keys: set[tuple[str, str]] | None = None,
    diagnostics: DiagnosticRecorder | None = None,
) -> PairSchedule:
    """Schedule cached false-negative challengers using model-visible pointwise fields.

    The policy deliberately avoids retrospective labels. It uses the pointwise
    scalar score as the incumbent decision surface, then looks for papers below
    or near that boundary whose rubric components, uncertainty, lexical novelty,
    or category diversity imply possible pointwise false negatives.
    """
    cfg = config or NewInformationChallengerConfig()
    recorder = diagnostics or DiagnosticRecorder()
    if budget.budget <= 0 or len(papers) < 2 or k <= 0:
        payload = _empty_schedule_diagnostics(k=k, budget=budget.budget, config=cfg)
        recorder.record(
            step="pair_scheduling",
            code="new_information_challenger_pair_scheduling_empty",
            message="no new-information challenger comparisons scheduled",
            data=payload,
        )
        return PairSchedule(pairs=[], budget=budget, diagnostics=payload)

    selected_or_seen = {
        _pair_key(comparison.left_id, comparison.right_id)
        for comparison in comparisons
    }
    selected_or_seen.update(seen_pair_keys or set())
    evsi_config = EVSISchedulerConfig(
        pairwise_strength=cfg.pairwise_strength,
        samples=cfg.posterior_samples,
        boundary_window=cfg.boundary_window,
        pool_multiplier=cfg.pool_multiplier,
        per_item_cap=cfg.per_item_cap,
    )
    context = _build_evsi_context(
        papers,
        comparisons=comparisons,
        k=k,
        seed=seed,
        config=evsi_config,
        seen_pairs=selected_or_seen,
        diagnostics=recorder,
    )
    all_items = _new_information_items(
        papers,
        context_items=context.items,
        k=k,
        config=cfg,
    )
    anchors = _anchor_items(all_items, k=k, config=cfg)
    challengers = _challenger_items(all_items, anchors=anchors, k=k, config=cfg)
    all_proposals = _new_information_proposals(
        anchors=anchors,
        challengers=challengers,
        seen_pair_keys=selected_or_seen,
    )
    feasible = (
        all_proposals
        if available_pair_keys is None
        else [
            proposal
            for proposal in all_proposals
            if _pair_key(proposal.left_id, proposal.right_id) in available_pair_keys
        ]
    )
    primary_target = min(budget.budget, len(feasible))
    primary_selected = _select_new_information_proposals(
        feasible,
        budget=primary_target,
        seed=seed,
        config=cfg,
    )
    fallback_proposals, fallback_payload = _cached_frontier_fallback_proposals(
        items=all_items,
        challengers=challengers,
        seen_pair_keys=selected_or_seen,
        primary_pair_keys={
            _pair_key(proposal.left_id, proposal.right_id)
            for proposal in all_proposals
        },
        available_pair_keys=available_pair_keys,
        k=k,
        config=cfg,
    )
    selected, fallback_cap_relaxed = _select_additional_new_information_proposals(
        primary_selected,
        fallback_proposals,
        budget=budget.budget,
        config=cfg,
    )
    fallback_selected_total = len(selected) - len(primary_selected)
    fallback_payload = {
        **fallback_payload,
        "primary_scheduled_total": len(primary_selected),
        "primary_scheduled_pairwise_shortfall": max(
            0,
            budget.budget - len(primary_selected),
        ),
        "selected_total": fallback_selected_total,
        "cap_relaxed_to_fill_shortfall": fallback_cap_relaxed,
        "remaining_shortfall": max(0, budget.budget - len(selected)),
        "budget_complete_after_fallback": len(selected) >= budget.budget,
    }
    scheduled = [
        _scheduled_pair_from_proposal(
            proposal,
            seed=seed,
            index=index,
        )
        for index, proposal in enumerate(selected, start=1)
    ]
    payload = {
        "candidate_count": len(all_items),
        "scheduled_total": len(scheduled),
        "pairs_considered": len(all_proposals),
        "unique_pairs_considered": len(
            {_pair_key(proposal.left_id, proposal.right_id) for proposal in all_proposals}
        ),
        "budget": budget.budget,
        "k": k,
        "acquisition": {
            "method": "new_information_challenger_cached_replay",
            "source_method": "pointwise_rubric_residual_false_negative_exposure",
            "random_seed": seed,
            "posterior_samples": cfg.posterior_samples,
            "pairwise_strength": cfg.pairwise_strength,
            "random_floor_fraction": cfg.random_floor_fraction,
            "min_random_floor_pairs": cfg.min_random_floor_pairs,
            "random_floor_target": _random_floor_target(primary_target, cfg),
            "per_item_cap": cfg.per_item_cap,
            "cached_fallback_enabled": cfg.cached_fallback_enabled,
            "cached_fallback_frontier_multiplier": (
                cfg.cached_fallback_frontier_multiplier
            ),
            "anchor_multiplier": cfg.anchor_multiplier,
            "challenger_multiplier": cfg.challenger_multiplier,
            "min_challengers": cfg.min_challengers,
            "max_challengers": cfg.max_challengers,
            "minimum_rubric_residual": cfg.minimum_rubric_residual,
            "rubric_keys": list(cfg.rubric_keys),
            "selection_policy": (
                "ranked rubric-residual challengers with a randomized cached "
                "coverage floor; if cached primary challenger pairs are under "
                "budget, fill from a predeclared cached frontier fallback that "
                "keeps at least one preselected challenger endpoint"
            ),
            "model_visible_signals": [
                "pointwise_good_probability",
                "pointwise_uncertainty",
                "pointwise_rubric_scores",
                "title_abstract_lexical_novelty",
                "metadata_category_diversity",
                "cached_pair_availability",
            ],
            "future_labels_used_for_scheduling": False,
        },
        "purpose_counts": dict(sorted(Counter(pair.purpose for pair in scheduled).items())),
        "coverage": _coverage(scheduled, anchors=anchors, challengers=challengers),
        "new_information_challenger": _challenger_diagnostics(
            items=all_items,
            anchors=anchors,
            challengers=challengers,
            proposals=all_proposals,
            feasible=feasible,
            scheduled=scheduled,
            budget=budget.budget,
            primary_scheduled_total=len(primary_selected),
            fallback_selected_total=fallback_selected_total,
            fallback_remaining_shortfall=fallback_payload["remaining_shortfall"],
        ),
        "cached_frontier_fallback": fallback_payload,
        "proposal_pool_profile": _proposal_pool_profile(
            items=context.items,
            pool=context.pool,
            scheduled=scheduled,
            k=k,
        ),
        "evsi_score_distribution": _evsi_score_distribution(context.proposals),
        "new_information_score_distribution": _score_distribution(all_proposals),
        **_proposal_filter_payload(
            all_proposals=all_proposals,
            feasible=feasible,
            available_pair_keys=available_pair_keys,
        ),
    }
    recorder.record(
        step="pair_scheduling",
        code="new_information_challenger_pair_scheduling_completed",
        message="scheduled new-information false-negative challenger comparisons",
        data=payload,
    )
    return PairSchedule(pairs=scheduled, budget=budget, diagnostics=payload)


def replay_new_information_challenger(
    papers: list[Paper],
    cached_comparisons: Mapping[tuple[str, str], PairwiseComparison],
    *,
    k: int,
    budget: PairwiseBudget,
    seed: int = 0,
    config: NewInformationChallengerConfig | None = None,
) -> NewInformationReplayResult:
    """Replay a one-shot no-paid challenger schedule over cached pairwise labels."""
    cfg = config or NewInformationChallengerConfig()
    cached_by_key = {
        _canonical_pair_key(left_id, right_id): comparison
        for (left_id, right_id), comparison in cached_comparisons.items()
    }
    schedule = schedule_new_information_challenger_pairs(
        papers,
        [],
        k=k,
        budget=budget,
        seed=seed,
        config=cfg,
        available_pair_keys=set(cached_by_key),
    )
    revealed: list[PairwiseComparison] = []
    missing_labels = 0
    for pair in schedule.pairs:
        cached = cached_by_key.get(_canonical_pair_key(pair.left_id, pair.right_id))
        if cached is None:
            missing_labels += 1
            continue
        revealed.append(_orient_comparison(cached, pair))
    diagnostics = {
        "method": "new_information_challenger_cached_replay",
        "budget": budget.to_dict(),
        "resolved_pairwise_budget": budget.budget,
        "scheduled_total": len(schedule.pairs),
        "comparisons_revealed_total": len(revealed),
        "missing_pairwise_labels": missing_labels,
        "budget_complete": len(schedule.pairs) >= budget.budget,
        "scheduled_pairwise_shortfall": max(0, budget.budget - len(schedule.pairs)),
        "purpose_counts": schedule.diagnostics.get("purpose_counts", {}),
        "coverage": schedule.diagnostics.get("coverage", {}),
        "new_information_challenger": schedule.diagnostics.get(
            "new_information_challenger",
            {},
        ),
        "available_label_filter": schedule.diagnostics.get(
            "available_label_filter",
            {},
        ),
        "cached_frontier_fallback": schedule.diagnostics.get(
            "cached_frontier_fallback",
            {},
        ),
        "label_policy": {
            "offline_cached_pairwise_labels_only": True,
            "missing_pairwise_labels": missing_labels,
            "novel_pairs_scheduled": missing_labels,
            "future_labels_used_for_scheduling": False,
            "future_labels_used_as_model_features": False,
            "cached_label_values_used_before_scheduling": False,
            "cache_availability_used_for_scheduling": True,
        },
    }
    return NewInformationReplayResult(
        schedule=schedule.pairs,
        comparisons=revealed,
        diagnostics=diagnostics,
    )


def _new_information_items(
    papers: list[Paper],
    *,
    context_items: Sequence[Any],
    k: int,
    config: NewInformationChallengerConfig,
) -> list[_NewInfoItem]:
    if not papers:
        return []
    ranked = sorted(
        papers,
        key=lambda paper: (
            paper.pointwise.good_probability,
            -paper.pointwise.uncertainty,
            paper.paper_id,
        ),
        reverse=True,
    )
    top_ids = {paper.paper_id for paper in ranked[: max(k, 1)]}
    boundary_probability = ranked[min(max(k - 1, 0), len(ranked) - 1)].pointwise.good_probability
    anchor_terms = _combined_terms(ranked[: max(k, 1)], config=config)
    anchor_buckets = {_metadata_bucket(paper) for paper in ranked[: max(k, 1)]}
    context_by_id = {item.paper_id: item for item in context_items}
    items: list[_NewInfoItem] = []
    for rank, paper in enumerate(ranked, start=1):
        probability = _clamp(float(paper.pointwise.good_probability))
        uncertainty = _clamp(float(paper.pointwise.uncertainty))
        rubric_signal = _rubric_signal(paper, config=config)
        rubric_residual = max(0.0, rubric_signal - probability)
        proximity = math.exp(
            -abs(probability - boundary_probability) / max(0.08, boundary_probability)
        )
        lexical = _lexical_novelty(paper, anchor_terms, config=config)
        metadata_bucket = _metadata_bucket(paper)
        diversity = 1.0 if metadata_bucket not in anchor_buckets else 0.0
        context_item = context_by_id.get(paper.paper_id)
        posterior_ucb_bonus = (
            _clamp(float(getattr(context_item, "ucb", 0.0)) / 6.0 + 0.5)
            if context_item is not None
            else probability
        )
        score = (
            (0.40 * rubric_residual)
            + (0.22 * rubric_signal)
            + (0.14 * uncertainty)
            + (0.12 * lexical)
            + (0.07 * diversity)
            + (0.05 * posterior_ucb_bonus)
        ) * (0.55 + (0.45 * proximity))
        roles = []
        if paper.paper_id in top_ids:
            roles.append("pointwise_top_k")
        if rank <= max(1, config.anchor_multiplier * max(k, 1)):
            roles.append("pointwise_anchor_band")
        if rubric_residual >= config.minimum_rubric_residual:
            roles.append("rubric_residual_false_negative")
        if probability < boundary_probability:
            roles.append("below_pointwise_boundary")
        elif probability <= boundary_probability + 0.05:
            roles.append("near_pointwise_boundary")
        if lexical >= 0.5:
            roles.append("lexically_novel")
        if diversity > 0.0:
            roles.append("metadata_diverse_from_top_k")
        items.append(
            _NewInfoItem(
                paper=paper,
                pointwise_probability=probability,
                uncertainty=uncertainty,
                rubric_signal=rubric_signal,
                rubric_residual=rubric_residual,
                boundary_proximity=proximity,
                lexical_novelty=lexical,
                metadata_diversity=diversity,
                challenger_score=score,
                metadata_bucket=metadata_bucket,
                roles=tuple(roles),
            )
        )
    return sorted(
        items,
        key=lambda item: (
            item.challenger_score,
            item.rubric_residual,
            item.rubric_signal,
            item.paper.paper_id,
        ),
        reverse=True,
    )


def _anchor_items(
    items: list[_NewInfoItem],
    *,
    k: int,
    config: NewInformationChallengerConfig,
) -> list[_NewInfoItem]:
    if not items:
        return []
    target = min(len(items), max(k, config.anchor_multiplier * max(k, 1)))
    by_probability = sorted(
        items,
        key=lambda item: (
            item.pointwise_probability,
            -item.uncertainty,
            item.paper.paper_id,
        ),
        reverse=True,
    )
    boundary_probability = by_probability[
        min(max(k - 1, 0), len(by_probability) - 1)
    ].pointwise_probability
    by_boundary = sorted(
        items,
        key=lambda item: (
            abs(item.pointwise_probability - boundary_probability),
            -item.uncertainty,
            item.paper.paper_id,
        ),
    )
    anchors = _ordered_unique_items(
        by_probability[: max(k, 1)],
        by_boundary[: max(k, 1)],
        by_probability,
        limit=target,
    )
    return anchors


def _challenger_items(
    items: list[_NewInfoItem],
    *,
    anchors: list[_NewInfoItem],
    k: int,
    config: NewInformationChallengerConfig,
) -> list[_NewInfoItem]:
    anchor_ids = {item.paper.paper_id for item in anchors}
    candidates = [
        item
        for item in items
        if item.paper.paper_id not in anchor_ids
        and (
            item.rubric_residual >= config.minimum_rubric_residual
            or item.lexical_novelty >= 0.5
            or item.metadata_diversity > 0.0
        )
    ]
    if not candidates:
        candidates = [item for item in items if item.paper.paper_id not in anchor_ids]
    target = min(
        len(candidates),
        max(config.min_challengers, config.challenger_multiplier * max(k, 1)),
    )
    if config.max_challengers is not None:
        target = min(target, config.max_challengers)
    return _diverse_prefix(candidates, limit=target)


def _new_information_proposals(
    *,
    anchors: list[_NewInfoItem],
    challengers: list[_NewInfoItem],
    seen_pair_keys: set[tuple[str, str]],
) -> list[_NewInfoProposal]:
    proposals: list[_NewInfoProposal] = []
    for challenger in challengers:
        for anchor in anchors:
            key = _pair_key(challenger.paper.paper_id, anchor.paper.paper_id)
            if key in seen_pair_keys:
                continue
            metadata_diverse = challenger.metadata_bucket != anchor.metadata_bucket
            anchor_strength = anchor.pointwise_probability
            score = challenger.challenger_score * (0.65 + (0.35 * anchor_strength))
            if metadata_diverse:
                score *= 1.05
            proposals.append(
                _NewInfoProposal(
                    left_id=anchor.paper.paper_id,
                    right_id=challenger.paper.paper_id,
                    score=score,
                    purpose="new_information_false_negative_challenge",
                    diagnostics={
                        "acquisition_score": round(score, 8),
                        "pair_role": "rubric_residual_anchor_challenger",
                        "anchor_id": anchor.paper.paper_id,
                        "challenger_id": challenger.paper.paper_id,
                        "anchor_roles": list(anchor.roles),
                        "challenger_roles": list(challenger.roles),
                        "anchor_pointwise_probability": round(
                            anchor.pointwise_probability,
                            8,
                        ),
                        "challenger_pointwise_probability": round(
                            challenger.pointwise_probability,
                            8,
                        ),
                        "challenger_rubric_signal": round(
                            challenger.rubric_signal,
                            8,
                        ),
                        "challenger_rubric_residual": round(
                            challenger.rubric_residual,
                            8,
                        ),
                        "challenger_uncertainty": round(challenger.uncertainty, 8),
                        "challenger_boundary_proximity": round(
                            challenger.boundary_proximity,
                            8,
                        ),
                        "challenger_lexical_novelty": round(
                            challenger.lexical_novelty,
                            8,
                        ),
                        "challenger_metadata_diversity": round(
                            challenger.metadata_diversity,
                            8,
                        ),
                        "anchor_metadata_bucket": anchor.metadata_bucket,
                        "challenger_metadata_bucket": challenger.metadata_bucket,
                        "metadata_diverse": metadata_diverse,
                        "anchor_title": anchor.paper.title,
                        "challenger_title": challenger.paper.title,
                    },
                )
            )
    return sorted(
        proposals,
        key=lambda proposal: (proposal.score, proposal.left_id, proposal.right_id),
        reverse=True,
    )


def _select_new_information_proposals(
    proposals: list[_NewInfoProposal],
    *,
    budget: int,
    seed: int,
    config: NewInformationChallengerConfig,
) -> list[_NewInfoProposal]:
    if budget <= 0:
        return []
    selected: list[_NewInfoProposal] = []
    selected_keys: set[tuple[str, str]] = set()
    item_counts: Counter[str] = Counter()
    cap = config.per_item_cap
    floor_target = _random_floor_target(budget, config)
    rng = random.Random(seed)
    random_candidates = list(proposals)
    rng.shuffle(random_candidates)

    def take(candidates: Sequence[_NewInfoProposal], *, limit: int, purpose: str | None) -> None:
        for proposal in candidates:
            if len(selected) >= budget or len(selected) >= limit:
                break
            key = _pair_key(proposal.left_id, proposal.right_id)
            if key in selected_keys:
                continue
            if (
                cap is not None
                and (
                    item_counts[proposal.left_id] >= cap
                    or item_counts[proposal.right_id] >= cap
                )
            ):
                continue
            selected.append(
                _replace_purpose(proposal, purpose)
                if purpose is not None
                else proposal
            )
            selected_keys.add(key)
            item_counts[proposal.left_id] += 1
            item_counts[proposal.right_id] += 1

    take(random_candidates, limit=floor_target, purpose="new_information_random_floor")
    take(proposals, limit=budget, purpose=None)
    if len(selected) < budget:
        cap = None
        take(proposals, limit=budget, purpose=None)
    return selected[:budget]


def _select_additional_new_information_proposals(
    selected: list[_NewInfoProposal],
    proposals: list[_NewInfoProposal],
    *,
    budget: int,
    config: NewInformationChallengerConfig,
) -> tuple[list[_NewInfoProposal], bool]:
    if len(selected) >= budget or not proposals:
        return list(selected[:budget]), False
    output = list(selected)
    selected_keys = {_pair_key(proposal.left_id, proposal.right_id) for proposal in output}
    item_counts: Counter[str] = Counter(
        paper_id
        for proposal in output
        for paper_id in (proposal.left_id, proposal.right_id)
    )

    def take(*, cap: int | None) -> None:
        for proposal in proposals:
            if len(output) >= budget:
                break
            key = _pair_key(proposal.left_id, proposal.right_id)
            if key in selected_keys:
                continue
            if (
                cap is not None
                and (
                    item_counts[proposal.left_id] >= cap
                    or item_counts[proposal.right_id] >= cap
                )
            ):
                continue
            output.append(proposal)
            selected_keys.add(key)
            item_counts[proposal.left_id] += 1
            item_counts[proposal.right_id] += 1

    take(cap=config.per_item_cap)
    cap_relaxed = False
    if len(output) < budget and config.per_item_cap is not None:
        cap_relaxed = True
        take(cap=None)
    return output[:budget], cap_relaxed


def _cached_frontier_fallback_proposals(
    *,
    items: list[_NewInfoItem],
    challengers: list[_NewInfoItem],
    seen_pair_keys: set[tuple[str, str]],
    primary_pair_keys: set[tuple[str, str]],
    available_pair_keys: set[tuple[str, str]] | None,
    k: int,
    config: NewInformationChallengerConfig,
) -> tuple[list[_NewInfoProposal], dict[str, Any]]:
    enabled = bool(config.cached_fallback_enabled and available_pair_keys is not None)
    payload: dict[str, Any] = {
        "enabled": enabled,
        "method": "predeclared_cached_frontier_challenger_fallback",
        "purpose": "new_information_cached_frontier_fallback",
        "selection_policy": (
            "Rank cached pairs with at least one preselected new-information "
            "challenger endpoint and one boundary-frontier comparator; exclude "
            "all primary anchor-challenger proposals and already-seen pairs."
        ),
        "frontier_multiplier": config.cached_fallback_frontier_multiplier,
        "model_visible_signals": [
            "pointwise_good_probability",
            "pointwise_uncertainty",
            "pointwise_rubric_scores",
            "title_abstract_lexical_novelty",
            "metadata_category_diversity",
            "cached_pair_availability",
        ],
        "future_labels_used_for_scheduling": False,
        "future_citation_labels_used_for_scheduling": False,
        "cached_label_values_used_before_scheduling": False,
        "cache_availability_used_for_scheduling": available_pair_keys is not None,
        "available_pair_keys_total": len(available_pair_keys or set()),
        "frontier_item_count": 0,
        "challenger_count": len(challengers),
        "frontier_pair_candidates": 0,
        "cached_feasible_proposals": 0,
    }
    if not enabled:
        payload["disabled_reason"] = (
            "cached fallback requires cached availability keys"
            if available_pair_keys is None
            else "cached fallback disabled by config"
        )
        return [], payload
    if not items or not challengers or config.cached_fallback_frontier_multiplier <= 0:
        payload["disabled_reason"] = "no frontier items or challengers available"
        return [], payload

    frontier = _fallback_frontier_items(items, k=k, config=config)
    payload["frontier_item_count"] = len(frontier)
    frontier_rank = {
        item.paper.paper_id: index
        for index, item in enumerate(frontier, start=1)
    }
    proposals_by_key: dict[tuple[str, str], _NewInfoProposal] = {}
    candidate_keys: set[tuple[str, str]] = set()
    for challenger in challengers:
        for comparator in frontier:
            if comparator.paper.paper_id == challenger.paper.paper_id:
                continue
            key = _pair_key(comparator.paper.paper_id, challenger.paper.paper_id)
            if key in seen_pair_keys or key in primary_pair_keys:
                continue
            candidate_keys.add(key)
            if key not in available_pair_keys:
                continue
            proposal = _cached_frontier_fallback_proposal(
                challenger=challenger,
                comparator=comparator,
                frontier_rank=frontier_rank[comparator.paper.paper_id],
            )
            existing = proposals_by_key.get(key)
            if existing is None or proposal.score > existing.score:
                proposals_by_key[key] = proposal
    proposals = sorted(
        proposals_by_key.values(),
        key=lambda proposal: (proposal.score, proposal.left_id, proposal.right_id),
        reverse=True,
    )
    payload["frontier_pair_candidates"] = len(candidate_keys)
    payload["cached_feasible_proposals"] = len(proposals)
    payload["top_cached_fallback_candidates"] = [
        {
            "left_id": proposal.left_id,
            "right_id": proposal.right_id,
            "score": round(proposal.score, 8),
            "challenger_id": proposal.diagnostics.get("challenger_id"),
            "frontier_comparator_id": proposal.diagnostics.get(
                "frontier_comparator_id"
            ),
            "frontier_rank": proposal.diagnostics.get("frontier_rank"),
        }
        for proposal in proposals[:10]
    ]
    return proposals, payload


def _cached_frontier_fallback_proposal(
    *,
    challenger: _NewInfoItem,
    comparator: _NewInfoItem,
    frontier_rank: int,
) -> _NewInfoProposal:
    metadata_diverse = challenger.metadata_bucket != comparator.metadata_bucket
    frontier_score = (
        0.50 * comparator.boundary_proximity
        + 0.30 * comparator.pointwise_probability
        + 0.10 * comparator.uncertainty
        + 0.10 * comparator.lexical_novelty
    )
    score = challenger.challenger_score * (0.65 + (0.35 * frontier_score))
    if metadata_diverse:
        score *= 1.03
    return _NewInfoProposal(
        left_id=comparator.paper.paper_id,
        right_id=challenger.paper.paper_id,
        score=score,
        purpose="new_information_cached_frontier_fallback",
        diagnostics={
            "acquisition_score": round(score, 8),
            "pair_role": "cached_frontier_challenger_fallback",
            "fallback_policy": "predeclared_cached_frontier_challenger_fallback",
            "challenger_id": challenger.paper.paper_id,
            "frontier_comparator_id": comparator.paper.paper_id,
            "frontier_rank": frontier_rank,
            "challenger_roles": list(challenger.roles),
            "frontier_comparator_roles": list(comparator.roles),
            "challenger_pointwise_probability": round(
                challenger.pointwise_probability,
                8,
            ),
            "frontier_pointwise_probability": round(
                comparator.pointwise_probability,
                8,
            ),
            "challenger_rubric_signal": round(challenger.rubric_signal, 8),
            "challenger_rubric_residual": round(challenger.rubric_residual, 8),
            "challenger_uncertainty": round(challenger.uncertainty, 8),
            "challenger_boundary_proximity": round(
                challenger.boundary_proximity,
                8,
            ),
            "frontier_boundary_proximity": round(
                comparator.boundary_proximity,
                8,
            ),
            "challenger_lexical_novelty": round(challenger.lexical_novelty, 8),
            "frontier_lexical_novelty": round(comparator.lexical_novelty, 8),
            "challenger_metadata_bucket": challenger.metadata_bucket,
            "frontier_metadata_bucket": comparator.metadata_bucket,
            "metadata_diverse": metadata_diverse,
            "challenger_title": challenger.paper.title,
            "frontier_comparator_title": comparator.paper.title,
            "future_labels_used_for_scheduling": False,
            "cached_label_values_used_before_scheduling": False,
        },
    )


def _fallback_frontier_items(
    items: list[_NewInfoItem],
    *,
    k: int,
    config: NewInformationChallengerConfig,
) -> list[_NewInfoItem]:
    target = min(
        len(items),
        max(1, config.cached_fallback_frontier_multiplier * max(k, 1)),
    )
    by_boundary = sorted(
        items,
        key=lambda item: (
            item.boundary_proximity,
            item.pointwise_probability,
            item.rubric_residual,
            item.paper.paper_id,
        ),
        reverse=True,
    )
    return by_boundary[:target]


def _replace_purpose(
    proposal: _NewInfoProposal,
    purpose: str,
) -> _NewInfoProposal:
    return _NewInfoProposal(
        left_id=proposal.left_id,
        right_id=proposal.right_id,
        score=proposal.score,
        purpose=purpose,
        diagnostics={
            **proposal.diagnostics,
            "source_new_information_purpose": proposal.purpose,
        },
    )


def _scheduled_pair_from_proposal(
    proposal: _NewInfoProposal,
    *,
    seed: int,
    index: int,
) -> ScheduledPair:
    return ScheduledPair(
        left_id=proposal.left_id,
        right_id=proposal.right_id,
        priority=round(float(proposal.score), 8),
        purpose=proposal.purpose,
        order=PairwiseOrderMetadata(
            shown_first_id=proposal.left_id,
            shown_second_id=proposal.right_id,
            randomized=True,
            seed=seed + index,
            extra={"scheduler": "new_information_challenger_cached_replay"},
        ),
        diagnostics=dict(proposal.diagnostics),
    )


def _coverage(
    scheduled: list[ScheduledPair],
    *,
    anchors: list[_NewInfoItem],
    challengers: list[_NewInfoItem],
) -> dict[str, Any]:
    anchor_ids = {item.paper.paper_id for item in anchors}
    challenger_ids = {item.paper.paper_id for item in challengers}
    touched = {
        paper_id
        for pair in scheduled
        for paper_id in (pair.left_id, pair.right_id)
    }
    false_negative_pairs = sum(
        1
        for pair in scheduled
        if (
            (pair.left_id in anchor_ids and pair.right_id in challenger_ids)
            or (pair.right_id in anchor_ids and pair.left_id in challenger_ids)
        )
    )
    random_floor_pairs = sum(
        1 for pair in scheduled if pair.purpose == "new_information_random_floor"
    )
    cached_frontier_fallback_pairs = sum(
        1
        for pair in scheduled
        if pair.purpose == "new_information_cached_frontier_fallback"
    )
    return {
        "random_floor_pairs": random_floor_pairs,
        "random_floor_rate": _rate(random_floor_pairs, len(scheduled)),
        "cached_frontier_fallback_pairs": cached_frontier_fallback_pairs,
        "cached_frontier_fallback_rate": _rate(
            cached_frontier_fallback_pairs,
            len(scheduled),
        ),
        "false_negative_challenge_pairs": false_negative_pairs,
        "false_negative_challenge_pair_rate": _rate(
            false_negative_pairs,
            len(scheduled),
        ),
        "anchor_count": len(anchor_ids),
        "challenger_count": len(challenger_ids),
        "scheduled_anchor_count": len(touched & anchor_ids),
        "scheduled_challenger_count": len(touched & challenger_ids),
        "scheduled_unique_papers": len(touched),
        "metadata_cross_bucket_pairs": sum(
            1 for pair in scheduled if bool(pair.diagnostics.get("metadata_diverse"))
        ),
        "average_challenger_rubric_residual": _mean(
            [
                float(pair.diagnostics.get("challenger_rubric_residual", 0.0))
                for pair in scheduled
            ]
        ),
        "average_challenger_lexical_novelty": _mean(
            [
                float(pair.diagnostics.get("challenger_lexical_novelty", 0.0))
                for pair in scheduled
            ]
        ),
    }


def _challenger_diagnostics(
    *,
    items: list[_NewInfoItem],
    anchors: list[_NewInfoItem],
    challengers: list[_NewInfoItem],
    proposals: list[_NewInfoProposal],
    feasible: list[_NewInfoProposal],
    scheduled: list[ScheduledPair],
    budget: int,
    primary_scheduled_total: int | None = None,
    fallback_selected_total: int = 0,
    fallback_remaining_shortfall: int | None = None,
) -> dict[str, Any]:
    scheduled_challenger_ids = {
        str(pair.diagnostics.get("challenger_id"))
        for pair in scheduled
        if pair.diagnostics.get("challenger_id")
    }
    return {
        "item_count": len(items),
        "anchor_count": len(anchors),
        "challenger_count": len(challengers),
        "proposal_count": len(proposals),
        "cached_feasible_proposal_count": len(feasible),
        "scheduled_total": len(scheduled),
        "budget": budget,
        "budget_complete": len(scheduled) >= budget,
        "scheduled_pairwise_shortfall": max(0, budget - len(scheduled)),
        "primary_scheduled_total": (
            len(scheduled) if primary_scheduled_total is None else primary_scheduled_total
        ),
        "primary_scheduled_pairwise_shortfall": max(
            0,
            budget
            - (
                len(scheduled)
                if primary_scheduled_total is None
                else primary_scheduled_total
            ),
        ),
        "cached_frontier_fallback_selected_pairs": fallback_selected_total,
        "remaining_shortfall_after_fallback": (
            max(0, budget - len(scheduled))
            if fallback_remaining_shortfall is None
            else fallback_remaining_shortfall
        ),
        "budget_utilization": _rate(len(scheduled), budget),
        "scheduled_challenger_count": len(scheduled_challenger_ids),
        "mean_challenger_rubric_residual": _mean(
            [item.rubric_residual for item in challengers]
        ),
        "mean_challenger_lexical_novelty": _mean(
            [item.lexical_novelty for item in challengers]
        ),
        "mean_challenger_metadata_diversity": _mean(
            [item.metadata_diversity for item in challengers]
        ),
        "challenger_role_counts": dict(
            sorted(Counter(role for item in challengers for role in item.roles).items())
        ),
        "anchor_role_counts": dict(
            sorted(Counter(role for item in anchors for role in item.roles).items())
        ),
        "metadata_bucket_counts": {
            "anchors": dict(sorted(Counter(item.metadata_bucket for item in anchors).items())),
            "challengers": dict(
                sorted(Counter(item.metadata_bucket for item in challengers).items())
            ),
        },
        "scheduled_pair_role_counts": dict(
            sorted(
                Counter(
                    str(pair.diagnostics.get("pair_role", "unknown"))
                    for pair in scheduled
                ).items()
            )
        ),
        "top_challengers": [
            {
                "paper_id": item.paper.paper_id,
                "title": item.paper.title,
                "pointwise_probability": round(item.pointwise_probability, 8),
                "rubric_signal": round(item.rubric_signal, 8),
                "rubric_residual": round(item.rubric_residual, 8),
                "lexical_novelty": round(item.lexical_novelty, 8),
                "metadata_bucket": item.metadata_bucket,
                "roles": list(item.roles),
            }
            for item in challengers[:10]
        ],
        "uses_future_labels_for_scheduling": False,
        "cached_label_values_used_before_scheduling": False,
    }


def _proposal_filter_payload(
    *,
    all_proposals: Sequence[_NewInfoProposal],
    feasible: Sequence[_NewInfoProposal],
    available_pair_keys: set[tuple[str, str]] | None,
) -> dict[str, Any]:
    return {
        "available_label_filter": {
            "available_pair_keys_total": len(available_pair_keys or set()),
            "all_candidate_proposals": len(all_proposals),
            "cached_feasible_proposals": len(feasible),
            "cache_feasible_rate": _rate(len(feasible), len(all_proposals)),
            "cache_availability_used_for_scheduling": (
                available_pair_keys is not None
            ),
        }
    }


def _rubric_signal(
    paper: Paper,
    *,
    config: NewInformationChallengerConfig,
) -> float:
    values = [
        _normalize_rubric_value(paper.pointwise.rubric_scores.get(key))
        for key in config.rubric_keys
        if key in paper.pointwise.rubric_scores
    ]
    if not values:
        return _clamp(float(paper.pointwise.good_probability))
    return _clamp(sum(values) / len(values))


def _normalize_rubric_value(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score > 1.0:
        score = score / 5.0
    return _clamp(score)


def _combined_terms(
    papers: Sequence[Paper],
    *,
    config: NewInformationChallengerConfig,
) -> set[str]:
    terms: set[str] = set()
    for paper in papers:
        terms.update(_text_terms(paper, config=config))
    return terms


def _lexical_novelty(
    paper: Paper,
    anchor_terms: set[str],
    *,
    config: NewInformationChallengerConfig,
) -> float:
    terms = _text_terms(paper, config=config)
    if not terms:
        return 0.0
    if not anchor_terms:
        return 1.0
    overlap = len(terms & anchor_terms)
    union = len(terms | anchor_terms)
    return round(1.0 - _rate(overlap, union), 8)


def _text_terms(
    paper: Paper,
    *,
    config: NewInformationChallengerConfig,
) -> set[str]:
    minimum = max(1, config.lexical_stopword_min_length)
    raw = f"{paper.title} {paper.abstract}".lower()
    return {
        token
        for token in re.findall(r"[a-z0-9]+", raw)
        if len(token) >= minimum and token not in _STOPWORDS
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


def _diverse_prefix(items: Sequence[_NewInfoItem], *, limit: int) -> list[_NewInfoItem]:
    if limit <= 0:
        return []
    selected: list[_NewInfoItem] = []
    selected_ids: set[str] = set()
    seen_buckets: set[str] = set()
    for item in items:
        if item.metadata_bucket in seen_buckets:
            continue
        selected.append(item)
        selected_ids.add(item.paper.paper_id)
        seen_buckets.add(item.metadata_bucket)
        if len(selected) >= limit:
            return selected
    for item in items:
        if item.paper.paper_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item.paper.paper_id)
        if len(selected) >= limit:
            return selected
    return selected


def _ordered_unique_items(
    *groups: Sequence[_NewInfoItem],
    limit: int,
) -> list[_NewInfoItem]:
    selected: list[_NewInfoItem] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if item.paper.paper_id in seen:
                continue
            selected.append(item)
            seen.add(item.paper.paper_id)
            if len(selected) >= limit:
                return selected
    return selected


def _score_distribution(values: Sequence[_NewInfoProposal]) -> dict[str, Any]:
    scores = [proposal.score for proposal in values]
    if not scores:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(scores),
        "min": round(min(scores), 8),
        "max": round(max(scores), 8),
        "mean": round(sum(scores) / len(scores), 8),
    }


def _empty_schedule_diagnostics(
    *,
    k: int,
    budget: int,
    config: NewInformationChallengerConfig,
) -> dict[str, Any]:
    return {
        "candidate_count": 0,
        "scheduled_total": 0,
        "pairs_considered": 0,
        "unique_pairs_considered": 0,
        "budget": budget,
        "k": k,
        "acquisition": {
            "method": "new_information_challenger_cached_replay",
            "source_method": "pointwise_rubric_residual_false_negative_exposure",
            "random_floor_fraction": config.random_floor_fraction,
            "model_visible_signals": [
                "pointwise_good_probability",
                "pointwise_uncertainty",
                "pointwise_rubric_scores",
                "title_abstract_lexical_novelty",
                "metadata_category_diversity",
            ],
            "future_labels_used_for_scheduling": False,
        },
        "purpose_counts": {},
        "coverage": {
            "random_floor_pairs": 0,
            "random_floor_rate": 0.0,
            "false_negative_challenge_pairs": 0,
        },
        "new_information_challenger": {
            "item_count": 0,
            "anchor_count": 0,
            "challenger_count": 0,
            "uses_future_labels_for_scheduling": False,
        },
    }


def _orient_comparison(
    comparison: PairwiseComparison,
    pair: ScheduledPair,
) -> PairwiseComparison:
    if comparison.left_id == pair.left_id and comparison.right_id == pair.right_id:
        winner = comparison.winner
    elif comparison.left_id == pair.right_id and comparison.right_id == pair.left_id:
        winner = _invert_winner(comparison.winner)
    else:
        raise ValueError("cached comparison does not reference scheduled pair")
    return PairwiseComparison(
        left_id=pair.left_id,
        right_id=pair.right_id,
        winner=winner,
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


def _canonical_pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    return tuple(sorted((left_id, right_id)))


def _invert_winner(winner: str) -> str:
    if winner == "left":
        return "right"
    if winner == "right":
        return "left"
    return winner


def _random_floor_target(
    target: int,
    config: NewInformationChallengerConfig,
) -> int:
    if target <= 0 or config.random_floor_fraction <= 0:
        return 0
    return min(
        target,
        max(
            config.min_random_floor_pairs,
            math.floor(target * config.random_floor_fraction),
        ),
    )


def _rate(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / denominator, 8)


def _mean(values: Sequence[int | float]) -> float:
    items = [float(value) for value in values]
    return round(sum(items) / len(items), 8) if items else 0.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


_STOPWORDS = {
    "about",
    "after",
    "also",
    "analysis",
    "approach",
    "based",
    "between",
    "data",
    "deep",
    "from",
    "have",
    "into",
    "learning",
    "method",
    "model",
    "models",
    "paper",
    "present",
    "propose",
    "results",
    "show",
    "study",
    "that",
    "their",
    "these",
    "this",
    "through",
    "using",
    "with",
}
