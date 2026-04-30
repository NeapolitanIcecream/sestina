from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sestina.candidates import CandidateSelection
from sestina.diagnostics import DiagnosticRecorder
from sestina.models import PairwiseOrderMetadata, Paper, ScheduledPair


_PURPOSE_ORDER = (
    "boundary_anchor",
    "candidate_internal",
    "sentinel_outsider",
    "audit_diversity",
)
_PURPOSE_WEIGHTS = {
    "boundary_anchor": 0.30,
    "candidate_internal": 0.30,
    "sentinel_outsider": 0.20,
    "audit_diversity": 0.20,
}


@dataclass(frozen=True, slots=True)
class PairwiseBudget:
    n: int
    candidate_size: int
    budget: int
    source: str = "default"

    def to_dict(self) -> dict[str, int | str]:
        return {
            "n": self.n,
            "candidate_size": self.candidate_size,
            "budget": self.budget,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class PairSchedule:
    pairs: list[ScheduledPair]
    budget: PairwiseBudget
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "budget": self.budget.to_dict(),
            "pairs": [pair.to_dict() for pair in self.pairs],
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class _PairProposal:
    left_id: str
    right_id: str
    priority: float
    purpose: str
    diagnostics: dict[str, Any]


def default_pairwise_budget(n: int, candidate_size: int) -> int:
    if n <= 0 or candidate_size <= 0:
        return 0
    return min(math.ceil(1.25 * candidate_size), math.ceil(0.25 * n))


def resolve_pairwise_budget(
    *,
    n: int,
    candidate_size: int,
    override: int | None = None,
    diagnostics: DiagnosticRecorder | None = None,
) -> PairwiseBudget:
    recorder = diagnostics or DiagnosticRecorder()
    if override is None:
        budget = default_pairwise_budget(n, candidate_size)
        source = "default"
    else:
        budget = max(0, int(override))
        source = "override"
    payload = PairwiseBudget(
        n=n,
        candidate_size=candidate_size,
        budget=budget,
        source=source,
    )
    recorder.record(
        step="pairwise_budget",
        code="pairwise_budget_resolved",
        message="resolved pairwise comparison budget",
        data={
            **payload.to_dict(),
            "default_formula": "min(ceil(1.25M), ceil(0.25n))",
            "nominal_cap_fraction": 0.25,
        },
    )
    if budget >= max(1, 5 * n):
        recorder.record(
            step="pairwise_budget",
            code="pairwise_budget_high",
            level="warning",
            message="pairwise budget is high relative to paper count",
            data={"n": n, "budget": budget},
        )
    return payload


def schedule_pairs(
    papers: list[Paper],
    *,
    candidate_selection: CandidateSelection,
    k: int,
    budget: PairwiseBudget,
    seed: int = 0,
    diagnostics: DiagnosticRecorder | None = None,
) -> PairSchedule:
    recorder = diagnostics or DiagnosticRecorder()
    paper_by_id = {paper.paper_id: paper for paper in papers}
    candidate_ids = [
        paper_id
        for paper_id in candidate_selection.candidate_ids
        if paper_id in paper_by_id
    ]
    all_ids = [paper.paper_id for paper in papers if paper.paper_id in paper_by_id]
    if budget.budget <= 0 or len(all_ids) < 2 or not candidate_ids:
        payload = {
            "candidate_count": len(candidate_ids),
            "scheduled_total": 0,
            "pairs_considered": 0,
            "unique_pairs_considered": 0,
            "budget": budget.budget,
            "purpose_counts": {},
            "proposal_counts_by_purpose": {},
            "purpose_targets": {},
            "coverage": _empty_coverage(),
        }
        recorder.record(
            step="pair_scheduling",
            code="pair_scheduling_empty",
            message="no pairwise comparisons scheduled",
            data=payload,
        )
        return PairSchedule(pairs=[], budget=budget, diagnostics=payload)

    ranked = sorted(
        all_ids,
        key=lambda paper_id: paper_by_id[paper_id].pointwise.good_probability,
        reverse=True,
    )
    rank_by_id = {paper_id: index + 1 for index, paper_id in enumerate(ranked)}
    boundary_rank = min(max(k, 1), len(ranked))
    proposal_pools = _build_proposal_pools(
        all_ids=all_ids,
        candidate_ids=candidate_ids,
        paper_by_id=paper_by_id,
        rank_by_id=rank_by_id,
        boundary_rank=boundary_rank,
        budget=budget.budget,
        k=k,
    )
    proposal_counts = {
        purpose: len(proposals) for purpose, proposals in proposal_pools.items()
    }
    purpose_targets = _purpose_targets(
        budget=budget.budget,
        proposal_counts=proposal_counts,
    )
    proposals = _select_proposals(
        proposal_pools,
        budget=budget.budget,
        purpose_targets=purpose_targets,
    )
    rng = random.Random(seed)
    scheduled: list[ScheduledPair] = []
    for index, proposal in enumerate(proposals):
        left_id = proposal.left_id
        right_id = proposal.right_id
        if rng.random() < 0.5:
            shown_first, shown_second = left_id, right_id
        else:
            shown_first, shown_second = right_id, left_id
        order = PairwiseOrderMetadata(
            shown_first_id=shown_first,
            shown_second_id=shown_second,
            randomized=True,
            seed=seed,
            position_bias_audit=(index % 5 == 0),
            extra={"canonical_left_id": left_id, "canonical_right_id": right_id},
        )
        scheduled.append(
            ScheduledPair(
                left_id=left_id,
                right_id=right_id,
                priority=round(proposal.priority, 6),
                purpose=proposal.purpose,
                order=order,
                diagnostics=proposal.diagnostics,
            )
        )
    unique_pairs_considered = {
        _pair_key(proposal.left_id, proposal.right_id)
        for proposals_for_purpose in proposal_pools.values()
        for proposal in proposals_for_purpose
    }
    purpose_counts = Counter(pair.purpose for pair in scheduled)
    payload = {
        "candidate_count": len(candidate_ids),
        "scheduled_total": len(scheduled),
        "pairs_considered": sum(proposal_counts.values()),
        "unique_pairs_considered": len(unique_pairs_considered),
        "budget": budget.budget,
        "boundary_rank": boundary_rank,
        "position_bias_audit_total": sum(
            1 for pair in scheduled if pair.order.position_bias_audit
        ),
        "purpose_counts": dict(sorted(purpose_counts.items())),
        "proposal_counts_by_purpose": proposal_counts,
        "purpose_targets": purpose_targets,
        "coverage": _schedule_coverage(
            scheduled,
            candidate_ids=set(candidate_ids),
            paper_by_id=paper_by_id,
            rank_by_id=rank_by_id,
            boundary_rank=boundary_rank,
            budget=budget.budget,
        ),
    }
    recorder.record(
        step="pair_scheduling",
        code="pair_scheduling_completed",
        message="scheduled pairwise comparisons without full ranking",
        data=payload,
    )
    return PairSchedule(pairs=scheduled, budget=budget, diagnostics=payload)


def _build_proposal_pools(
    *,
    all_ids: list[str],
    candidate_ids: list[str],
    paper_by_id: dict[str, Paper],
    rank_by_id: dict[str, int],
    boundary_rank: int,
    budget: int,
    k: int,
) -> dict[str, list[_PairProposal]]:
    pools: dict[str, list[_PairProposal]] = {purpose: [] for purpose in _PURPOSE_ORDER}
    candidate_set = set(candidate_ids)
    for left_id, right_id in itertools.combinations(candidate_ids, 2):
        base = _candidate_pair_proposal(
            left_id=left_id,
            right_id=right_id,
            paper_by_id=paper_by_id,
            rank_by_id=rank_by_id,
            boundary_rank=boundary_rank,
            purpose="candidate_internal",
        )
        pools["candidate_internal"].append(base)
        if _crosses_boundary(
            rank_by_id[left_id],
            rank_by_id[right_id],
            boundary_rank=boundary_rank,
        ):
            pools["boundary_anchor"].append(
                _retag_proposal(
                    base,
                    purpose="boundary_anchor",
                    priority_boost=0.20,
                    extra={"crosses_top_k_boundary": True},
                )
            )
        if _is_metadata_diverse(paper_by_id[left_id], paper_by_id[right_id]):
            pools["audit_diversity"].append(
                _retag_proposal(
                    base,
                    purpose="audit_diversity",
                    priority_boost=0.15,
                    extra={"metadata_cross_bucket": True},
                )
            )

    outsider_ids = [paper_id for paper_id in all_ids if paper_id not in candidate_set]
    if outsider_ids:
        anchor_limit = max(2, min(len(candidate_ids), max(k * 2, budget)))
        outsider_limit = max(4, min(len(outsider_ids), max(k * 4, budget * 4)))
        anchors = sorted(
            candidate_ids,
            key=lambda paper_id: (
                abs(rank_by_id[paper_id] - boundary_rank),
                -paper_by_id[paper_id].pointwise.good_probability,
                paper_id,
            ),
        )[:anchor_limit]
        outsiders = sorted(
            outsider_ids,
            key=lambda paper_id: (
                _outsider_score(
                    paper_by_id[paper_id],
                    anchors=[paper_by_id[anchor_id] for anchor_id in anchors],
                    rank=rank_by_id[paper_id],
                    boundary_rank=boundary_rank,
                ),
                paper_id,
            ),
            reverse=True,
        )[:outsider_limit]
        for anchor_id in anchors:
            for outsider_id in outsiders:
                sentinel = _sentinel_outsider_proposal(
                    anchor_id=anchor_id,
                    outsider_id=outsider_id,
                    paper_by_id=paper_by_id,
                    rank_by_id=rank_by_id,
                    boundary_rank=boundary_rank,
                )
                pools["sentinel_outsider"].append(sentinel)
                if _is_metadata_diverse(
                    paper_by_id[anchor_id],
                    paper_by_id[outsider_id],
                ):
                    pools["audit_diversity"].append(
                        _retag_proposal(
                            sentinel,
                            purpose="audit_diversity",
                            priority_boost=0.10,
                            extra={"metadata_cross_bucket": True},
                        )
                    )

    return {
        purpose: _sort_proposals(proposals)
        for purpose, proposals in pools.items()
    }


def _candidate_pair_proposal(
    *,
    left_id: str,
    right_id: str,
    paper_by_id: dict[str, Paper],
    rank_by_id: dict[str, int],
    boundary_rank: int,
    purpose: str,
) -> _PairProposal:
    priority, dominant_component, parts = _pair_priority(
        left=paper_by_id[left_id],
        right=paper_by_id[right_id],
        rank_left=rank_by_id[left_id],
        rank_right=rank_by_id[right_id],
        boundary_rank=boundary_rank,
    )
    diagnostics: dict[str, Any] = {
        **parts,
        "dominant_component": dominant_component,
        "rank_left": rank_by_id[left_id],
        "rank_right": rank_by_id[right_id],
        "metadata_cross_bucket": _is_metadata_diverse(
            paper_by_id[left_id],
            paper_by_id[right_id],
        ),
        "involves_outsider": False,
    }
    return _PairProposal(
        left_id=left_id,
        right_id=right_id,
        priority=priority,
        purpose=purpose,
        diagnostics=diagnostics,
    )


def _sentinel_outsider_proposal(
    *,
    anchor_id: str,
    outsider_id: str,
    paper_by_id: dict[str, Paper],
    rank_by_id: dict[str, int],
    boundary_rank: int,
) -> _PairProposal:
    anchor = paper_by_id[anchor_id]
    outsider = paper_by_id[outsider_id]
    anchor_boundary = 1.0 / (1.0 + abs(rank_by_id[anchor_id] - boundary_rank))
    outsider_uncertainty = outsider.pointwise.uncertainty
    outsider_quality = outsider.pointwise.good_probability
    diversity = 1.0 if _is_metadata_diverse(anchor, outsider) else 0.0
    outsider_boundary = 1.0 / (1.0 + abs(rank_by_id[outsider_id] - boundary_rank))
    priority = (
        (0.30 * anchor_boundary)
        + (0.30 * outsider_uncertainty)
        + (0.20 * outsider_quality)
        + (0.15 * diversity)
        + (0.05 * outsider_boundary)
    )
    diagnostics: dict[str, Any] = {
        "anchor_boundary": round(anchor_boundary, 6),
        "outsider_uncertainty": round(outsider_uncertainty, 6),
        "outsider_quality": round(outsider_quality, 6),
        "outsider_boundary": round(outsider_boundary, 6),
        "diversity": round(diversity, 6),
        "rank_left": rank_by_id[anchor_id],
        "rank_right": rank_by_id[outsider_id],
        "metadata_cross_bucket": bool(diversity),
        "involves_outsider": True,
    }
    return _PairProposal(
        left_id=anchor_id,
        right_id=outsider_id,
        priority=priority,
        purpose="sentinel_outsider",
        diagnostics=diagnostics,
    )


def _retag_proposal(
    proposal: _PairProposal,
    *,
    purpose: str,
    priority_boost: float,
    extra: dict[str, Any],
) -> _PairProposal:
    return _PairProposal(
        left_id=proposal.left_id,
        right_id=proposal.right_id,
        priority=proposal.priority + priority_boost,
        purpose=purpose,
        diagnostics={**proposal.diagnostics, **extra},
    )


def _purpose_targets(
    *,
    budget: int,
    proposal_counts: dict[str, int],
) -> dict[str, int]:
    possible = [
        purpose
        for purpose in _PURPOSE_ORDER
        if proposal_counts.get(purpose, 0) > 0
    ]
    if budget <= 0 or not possible:
        return {}
    targets: Counter[str] = Counter()
    required = possible if budget >= len(possible) else possible[:budget]
    for purpose in required:
        targets[purpose] = 1
    remaining = budget - sum(targets.values())
    while remaining > 0:
        choices = [
            purpose
            for purpose in possible
            if targets[purpose] < proposal_counts[purpose]
        ]
        if not choices:
            break
        purpose = max(
            choices,
            key=lambda item: (
                (budget * _PURPOSE_WEIGHTS[item]) - targets[item],
                _PURPOSE_WEIGHTS[item],
                -_PURPOSE_ORDER.index(item),
            ),
        )
        targets[purpose] += 1
        remaining -= 1
    return {purpose: targets[purpose] for purpose in _PURPOSE_ORDER if targets[purpose]}


def _select_proposals(
    proposal_pools: dict[str, list[_PairProposal]],
    *,
    budget: int,
    purpose_targets: dict[str, int],
) -> list[_PairProposal]:
    selected: list[_PairProposal] = []
    selected_by_purpose: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for purpose in _PURPOSE_ORDER:
        target = purpose_targets.get(purpose, 0)
        if target <= 0:
            continue
        for proposal in proposal_pools.get(purpose, []):
            if selected_by_purpose[purpose] >= target:
                break
            key = _pair_key(proposal.left_id, proposal.right_id)
            if key in seen:
                continue
            selected.append(proposal)
            selected_by_purpose[purpose] += 1
            seen.add(key)
            if len(selected) >= budget:
                return selected

    all_proposals = _sort_proposals(
        [
            proposal
            for purpose in _PURPOSE_ORDER
            for proposal in proposal_pools.get(purpose, [])
        ]
    )
    for proposal in all_proposals:
        if len(selected) >= budget:
            break
        key = _pair_key(proposal.left_id, proposal.right_id)
        if key in seen:
            continue
        selected.append(proposal)
        seen.add(key)
    return selected


def _sort_proposals(proposals: list[_PairProposal]) -> list[_PairProposal]:
    return sorted(
        proposals,
        key=lambda proposal: (
            proposal.priority,
            -_PURPOSE_ORDER.index(proposal.purpose),
            proposal.left_id,
            proposal.right_id,
        ),
        reverse=True,
    )


def _schedule_coverage(
    pairs: list[ScheduledPair],
    *,
    candidate_ids: set[str],
    paper_by_id: dict[str, Paper],
    rank_by_id: dict[str, int],
    boundary_rank: int,
    budget: int,
) -> dict[str, Any]:
    coverage = Counter(
        {
            "candidate_internal_pairs": 0,
            "candidate_outsider_pairs": 0,
            "outsider_outsider_pairs": 0,
            "boundary_crossing_pairs": 0,
            "metadata_cross_bucket_pairs": 0,
        }
    )
    distinct_papers: set[str] = set()
    distinct_candidates: set[str] = set()
    distinct_outsiders: set[str] = set()
    metadata_buckets: set[tuple[str, object]] = set()
    rank_midpoints: list[float] = []
    probability_gaps: list[float] = []
    for pair in pairs:
        left_id = pair.left_id
        right_id = pair.right_id
        left = paper_by_id[left_id]
        right = paper_by_id[right_id]
        left_is_candidate = left_id in candidate_ids
        right_is_candidate = right_id in candidate_ids
        if left_is_candidate and right_is_candidate:
            coverage["candidate_internal_pairs"] += 1
        elif left_is_candidate or right_is_candidate:
            coverage["candidate_outsider_pairs"] += 1
        else:
            coverage["outsider_outsider_pairs"] += 1
        if _crosses_boundary(
            rank_by_id[left_id],
            rank_by_id[right_id],
            boundary_rank=boundary_rank,
        ):
            coverage["boundary_crossing_pairs"] += 1
        if _is_metadata_diverse(left, right):
            coverage["metadata_cross_bucket_pairs"] += 1
        distinct_papers.update([left_id, right_id])
        for paper_id in (left_id, right_id):
            if paper_id in candidate_ids:
                distinct_candidates.add(paper_id)
            else:
                distinct_outsiders.add(paper_id)
        metadata_buckets.update([_metadata_bucket(left), _metadata_bucket(right)])
        rank_midpoints.append((rank_by_id[left_id] + rank_by_id[right_id]) / 2.0)
        probability_gaps.append(
            abs(left.pointwise.good_probability - right.pointwise.good_probability)
        )
    payload = _empty_coverage()
    payload.update(
        {
            **dict(coverage),
            "distinct_papers_covered": len(distinct_papers),
            "distinct_candidates_covered": len(distinct_candidates),
            "distinct_outsiders_covered": len(distinct_outsiders),
            "metadata_buckets_covered": len(metadata_buckets),
            "budget_utilization": round(len(pairs) / budget, 6) if budget > 0 else 0.0,
            "avg_rank_midpoint": round(sum(rank_midpoints) / len(rank_midpoints), 6)
            if rank_midpoints
            else None,
            "avg_probability_gap": round(
                sum(probability_gaps) / len(probability_gaps),
                6,
            )
            if probability_gaps
            else None,
        }
    )
    return payload


def _empty_coverage() -> dict[str, Any]:
    return {
        "candidate_internal_pairs": 0,
        "candidate_outsider_pairs": 0,
        "outsider_outsider_pairs": 0,
        "boundary_crossing_pairs": 0,
        "metadata_cross_bucket_pairs": 0,
        "distinct_papers_covered": 0,
        "distinct_candidates_covered": 0,
        "distinct_outsiders_covered": 0,
        "metadata_buckets_covered": 0,
        "budget_utilization": 0.0,
        "avg_rank_midpoint": None,
        "avg_probability_gap": None,
    }


def _pair_priority(
    *,
    left: Paper,
    right: Paper,
    rank_left: int,
    rank_right: int,
    boundary_rank: int,
) -> tuple[float, str, dict[str, float]]:
    q_gap = abs(left.pointwise.good_probability - right.pointwise.good_probability)
    closeness = 1.0 - min(1.0, q_gap)
    rank_mid = (rank_left + rank_right) / 2.0
    boundary_distance = abs(rank_mid - boundary_rank)
    boundary = 1.0 / (1.0 + boundary_distance)
    uncertainty = (left.pointwise.uncertainty + right.pointwise.uncertainty) / 2.0
    diversity = 1.0 if _metadata_bucket(left) != _metadata_bucket(right) else 0.0
    priority = (
        (0.35 * boundary)
        + (0.15 * closeness)
        + (0.25 * uncertainty)
        + (0.25 * diversity)
    )
    parts = {
        "boundary": round(boundary, 6),
        "closeness": round(closeness, 6),
        "uncertainty": round(uncertainty, 6),
        "diversity": round(diversity, 6),
    }
    purpose = max(parts, key=parts.get)
    return priority, purpose, parts


def _metadata_bucket(paper: Paper) -> tuple[str, object]:
    for key in ("primary_category", "category", "topic", "venue", "field"):
        value = paper.metadata.get(key)
        if value:
            return key, _metadata_value(value)
    categories = paper.metadata.get("categories")
    if isinstance(categories, (list, tuple)) and categories:
        return "category", str(categories[0])
    source = paper.metadata.get("source")
    if source:
        return "source", _metadata_value(source)
    return "unknown", "unknown"


def _metadata_value(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return value


def _is_metadata_diverse(left: Paper, right: Paper) -> bool:
    return _metadata_bucket(left) != _metadata_bucket(right)


def _crosses_boundary(
    rank_left: int,
    rank_right: int,
    *,
    boundary_rank: int,
) -> bool:
    return (rank_left <= boundary_rank < rank_right) or (
        rank_right <= boundary_rank < rank_left
    )


def _outsider_score(
    outsider: Paper,
    *,
    anchors: list[Paper],
    rank: int,
    boundary_rank: int,
) -> float:
    diversity = (
        1.0
        if any(_is_metadata_diverse(outsider, anchor) for anchor in anchors)
        else 0.0
    )
    boundary = 1.0 / (1.0 + abs(rank - boundary_rank))
    return (
        (0.45 * outsider.pointwise.uncertainty)
        + (0.25 * outsider.pointwise.good_probability)
        + (0.20 * diversity)
        + (0.10 * boundary)
    )


def _pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    return tuple(sorted((left_id, right_id)))
