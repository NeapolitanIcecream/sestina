from __future__ import annotations

import math

import pytest

from sestina.candidates import (
    CandidateSelectionConfig,
    default_candidate_size,
    select_candidates,
)
from sestina.diagnostics import DiagnosticRecorder
from sestina.models import TargetSpec
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

    budget = resolve_pairwise_budget(n=len(papers), candidate_size=len(selection.candidate_ids))

    assert budget.budget == default_pairwise_budget(100, len(selection.candidate_ids))
    assert budget.budget <= math.ceil(0.25 * len(papers))
    assert budget.budget < 5 * len(papers)


def test_scheduler_randomizes_order_and_stays_within_budget(paper_set) -> None:
    papers = paper_set(20)
    selection = select_candidates(papers, k=4)
    budget = resolve_pairwise_budget(n=len(papers), candidate_size=len(selection.candidate_ids))

    schedule = schedule_pairs(
        papers,
        candidate_selection=selection,
        k=4,
        budget=budget,
        seed=123,
    )

    assert len(schedule.pairs) == budget.budget
    assert len({frozenset([pair.left_id, pair.right_id]) for pair in schedule.pairs}) == len(
        schedule.pairs
    )
    assert all(pair.order.randomized for pair in schedule.pairs)
    assert any(pair.order.position_bias_audit for pair in schedule.pairs)
    assert schedule.diagnostics["scheduled_total"] == budget.budget
