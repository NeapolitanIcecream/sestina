from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field

from sestina.candidates import CandidateSelection
from sestina.diagnostics import DiagnosticRecorder
from sestina.models import PairwiseOrderMetadata, Paper, ScheduledPair


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
    payload = PairwiseBudget(n=n, candidate_size=candidate_size, budget=budget, source=source)
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
        paper_id for paper_id in candidate_selection.candidate_ids if paper_id in paper_by_id
    ]
    if budget.budget <= 0 or len(candidate_ids) < 2:
        payload = {
            "candidate_count": len(candidate_ids),
            "scheduled_total": 0,
            "pairs_considered": 0,
        }
        recorder.record(
            step="pair_scheduling",
            code="pair_scheduling_empty",
            message="no pairwise comparisons scheduled",
            data=payload,
        )
        return PairSchedule(pairs=[], budget=budget, diagnostics=payload)

    ranked = sorted(
        candidate_ids,
        key=lambda paper_id: paper_by_id[paper_id].pointwise.good_probability,
        reverse=True,
    )
    rank_by_id = {paper_id: index + 1 for index, paper_id in enumerate(ranked)}
    boundary_rank = min(max(k, 1), len(ranked))
    proposals = []
    for left_id, right_id in itertools.combinations(candidate_ids, 2):
        left = paper_by_id[left_id]
        right = paper_by_id[right_id]
        priority, purpose, parts = _pair_priority(
            left=left,
            right=right,
            rank_left=rank_by_id[left_id],
            rank_right=rank_by_id[right_id],
            boundary_rank=boundary_rank,
        )
        proposals.append((priority, left_id, right_id, purpose, parts))

    proposals.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    rng = random.Random(seed)
    scheduled: list[ScheduledPair] = []
    for index, (priority, left_id, right_id, purpose, parts) in enumerate(
        proposals[: budget.budget]
    ):
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
                priority=round(priority, 6),
                purpose=purpose,
                order=order,
                diagnostics=parts,
            )
        )
    payload = {
        "candidate_count": len(candidate_ids),
        "scheduled_total": len(scheduled),
        "pairs_considered": len(proposals),
        "budget": budget.budget,
        "boundary_rank": boundary_rank,
        "position_bias_audit_total": sum(
            1 for pair in scheduled if pair.order.position_bias_audit
        ),
    }
    recorder.record(
        step="pair_scheduling",
        code="pair_scheduling_completed",
        message="scheduled pairwise comparisons without full ranking",
        data=payload,
    )
    return PairSchedule(pairs=scheduled, budget=budget, diagnostics=payload)


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
        (0.40 * boundary)
        + (0.30 * closeness)
        + (0.20 * uncertainty)
        + (0.10 * diversity)
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
    for key in ("topic", "venue", "source", "field", "category"):
        if key in paper.metadata:
            return key, paper.metadata[key]
    return "unknown", "unknown"

