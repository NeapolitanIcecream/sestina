from __future__ import annotations

import math

import pytest

from sestina.candidates import (
    CandidateSelection,
    CandidateSelectionConfig,
    default_candidate_size,
    select_candidates,
)
from sestina.diagnostics import DiagnosticRecorder
from sestina.models import Paper, PointwiseAssessment, TargetSpec
from sestina.scheduler import (
    default_pairwise_budget,
    resolve_pairwise_budget,
    schedule_pairs,
)
from sestina.targets import resolve_target


def test_top_alpha_resolves_to_ceiling_of_paper_count(paper_set) -> None:
    papers = paper_set(10)

    resolved = resolve_target(len(papers), TargetSpec(top_alpha=0.21))

    assert resolved.k == 3
    assert resolved.mode == "top_alpha"


def test_target_requires_exactly_one_target() -> None:
    diagnostics = DiagnosticRecorder()

    try:
        resolve_target(10, TargetSpec(top_k=3, top_alpha=0.3), diagnostics=diagnostics)
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("expected invalid target to raise")

    assert diagnostics.to_dict()["events"][0]["code"] == "target_invalid"


def test_candidate_selection_uses_default_formula_and_all_groups(paper_set) -> None:
    papers = paper_set(40)

    selection = select_candidates(papers, k=5)

    assert len(selection.candidate_ids) == default_candidate_size(40, 5)
    assert len(selection.candidate_ids) == min(40, math.ceil(15 + math.sqrt(40)))
    assert set(selection.groups) == {"exploit", "boundary", "explore"}
    assert {"p1", "p2", "p3", "p4", "p5"}.issubset(selection.candidate_ids)
    assert selection.diagnostics["candidate_size"] == len(selection.candidate_ids)


def test_candidate_size_zero_override_raises_and_records_diagnostic(paper_set) -> None:
    """Regression: explicit zero previously fell through to the default size."""
    papers = paper_set(10)
    diagnostics = DiagnosticRecorder()

    message = "candidate_size must be at least resolved top K"
    with pytest.raises(ValueError, match=message):
        select_candidates(
            papers,
            k=3,
            config=CandidateSelectionConfig(candidate_size=0),
            diagnostics=diagnostics,
        )

    events = diagnostics.to_dict()["events"]
    invalid = next(
        event for event in events if event["code"] == "candidate_size_invalid"
    )
    assert invalid["level"] == "error"
    assert invalid["data"]["candidate_size_requested"] == 0
    assert invalid["data"]["minimum_candidate_size"] == 3


def test_candidate_size_below_k_raises_and_records_diagnostic(paper_set) -> None:
    papers = paper_set(10)
    diagnostics = DiagnosticRecorder()

    message = "candidate_size must be at least resolved top K"
    with pytest.raises(ValueError, match=message):
        select_candidates(
            papers,
            k=4,
            config=CandidateSelectionConfig(candidate_size=3),
            diagnostics=diagnostics,
        )

    events = diagnostics.to_dict()["events"]
    invalid = next(
        event for event in events if event["code"] == "candidate_size_invalid"
    )
    assert invalid["data"]["candidate_size_requested"] == 3
    assert invalid["data"]["minimum_candidate_size"] == 4


def test_pairwise_budget_uses_default_cap(paper_set) -> None:
    papers = paper_set(100)
    selection = select_candidates(papers, k=5)

    budget = resolve_pairwise_budget(
        n=len(papers),
        candidate_size=len(selection.candidate_ids),
    )

    assert budget.budget == default_pairwise_budget(100, len(selection.candidate_ids))
    assert budget.budget <= math.ceil(0.25 * len(papers))
    assert budget.budget < 5 * len(papers)


def test_scheduler_randomizes_order_and_stays_within_budget(paper_set) -> None:
    papers = paper_set(20)
    selection = select_candidates(papers, k=4)
    budget = resolve_pairwise_budget(
        n=len(papers),
        candidate_size=len(selection.candidate_ids),
    )

    schedule = schedule_pairs(
        papers,
        candidate_selection=selection,
        k=4,
        budget=budget,
        seed=123,
    )

    assert len(schedule.pairs) == budget.budget
    assert len(
        {frozenset([pair.left_id, pair.right_id]) for pair in schedule.pairs}
    ) == len(schedule.pairs)
    assert all(pair.order.randomized for pair in schedule.pairs)
    assert any(pair.order.position_bias_audit for pair in schedule.pairs)
    assert schedule.diagnostics["scheduled_total"] == budget.budget


def test_scheduler_allocates_purpose_coverage_when_possible() -> None:
    papers = _diverse_scheduler_papers()
    selection = CandidateSelection(
        candidate_ids=[f"p{index}" for index in range(1, 9)],
        groups={
            "exploit": ["p1", "p2", "p3", "p4"],
            "boundary": ["p3", "p4", "p5", "p6"],
            "explore": ["p7", "p8"],
        },
        scores={},
    )
    budget = resolve_pairwise_budget(
        n=len(papers),
        candidate_size=len(selection.candidate_ids),
        override=8,
    )
    diagnostics = DiagnosticRecorder()

    schedule = schedule_pairs(
        papers,
        candidate_selection=selection,
        k=4,
        budget=budget,
        seed=321,
        diagnostics=diagnostics,
    )

    unordered_pairs = {
        frozenset([pair.left_id, pair.right_id]) for pair in schedule.pairs
    }
    assert len(schedule.pairs) == 8
    assert len(unordered_pairs) == len(schedule.pairs)
    assert all(pair.order.randomized for pair in schedule.pairs)

    purpose_counts = schedule.diagnostics["purpose_counts"]
    assert purpose_counts["boundary_anchor"] > 0
    assert purpose_counts["candidate_internal"] > 0
    assert purpose_counts["sentinel_outsider"] > 0
    assert purpose_counts["audit_diversity"] > 0

    coverage = schedule.diagnostics["coverage"]
    assert coverage["candidate_internal_pairs"] > 0
    assert coverage["candidate_outsider_pairs"] > 0
    assert coverage["boundary_crossing_pairs"] > 0
    assert coverage["metadata_cross_bucket_pairs"] > 0
    assert coverage["distinct_outsiders_covered"] > 0
    assert coverage["budget_utilization"] == 1.0

    events = diagnostics.to_dict()["events"]
    completed = next(
        event for event in events if event["code"] == "pair_scheduling_completed"
    )
    assert completed["data"]["purpose_counts"] == purpose_counts
    assert completed["data"]["coverage"] == coverage
    assert completed["data"]["proposal_counts_by_purpose"]["sentinel_outsider"] > 0


def test_scheduler_accepts_structured_metadata_buckets() -> None:
    """Regression: structured metadata buckets crashed coverage set accounting."""
    papers = _diverse_scheduler_papers()
    for index, paper in enumerate(papers):
        paper.metadata["primary_category"] = {
            "id": paper.metadata["primary_category"],
            "namespace": "arxiv",
            "path": ["computer_science", str(index % 3)],
        }
    selection = CandidateSelection(
        candidate_ids=[f"p{index}" for index in range(1, 9)],
        groups={
            "exploit": ["p1", "p2", "p3", "p4"],
            "boundary": ["p3", "p4", "p5", "p6"],
            "explore": ["p7", "p8"],
        },
        scores={},
    )

    schedule = schedule_pairs(
        papers,
        candidate_selection=selection,
        k=4,
        budget=resolve_pairwise_budget(n=len(papers), candidate_size=8, override=8),
        seed=321,
    )

    assert schedule.diagnostics["coverage"]["metadata_buckets_covered"] > 0


def test_scheduler_empty_budget_emits_machine_readable_coverage(paper_set) -> None:
    papers = paper_set(5)
    selection = CandidateSelection(
        candidate_ids=["p1", "p2"],
        groups={"exploit": ["p1", "p2"], "boundary": [], "explore": []},
        scores={},
    )
    diagnostics = DiagnosticRecorder()

    schedule = schedule_pairs(
        papers,
        candidate_selection=selection,
        k=2,
        budget=resolve_pairwise_budget(n=len(papers), candidate_size=2, override=0),
        diagnostics=diagnostics,
    )

    assert schedule.pairs == []
    assert schedule.diagnostics["purpose_counts"] == {}
    assert schedule.diagnostics["coverage"]["budget_utilization"] == 0.0
    assert diagnostics.to_dict()["events"][0]["code"] == "pair_scheduling_empty"


def _diverse_scheduler_papers() -> list[Paper]:
    categories = ["cs.LG", "cs.CL", "cs.CV", "cs.AI"]
    papers = []
    for index in range(1, 13):
        papers.append(
            Paper(
                paper_id=f"p{index}",
                title=f"Paper {index}",
                pointwise=PointwiseAssessment(
                    good_probability=0.95 - (index * 0.05),
                    uncertainty=0.2 + ((index % 4) * 0.15),
                ),
                metadata={
                    "primary_category": categories[index % len(categories)],
                    "source": "arxiv",
                },
            )
        )
    return papers
