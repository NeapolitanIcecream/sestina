from __future__ import annotations

from sestina.models import PairwiseComparison, Paper, PointwiseAssessment
from sestina.new_information_challenger import (
    NewInformationChallengerConfig,
    replay_new_information_challenger,
    schedule_new_information_challenger_pairs,
)
from sestina.scheduler import PairwiseBudget


def test_new_information_scheduler_targets_rubric_residual_false_negatives() -> None:
    papers = [
        _paper("p1", 0.92, 0.10, rubric=0.90, topic="anchor"),
        _paper("p2", 0.84, 0.15, rubric=0.82, topic="anchor"),
        _paper("p3", 0.58, 0.20, rubric=0.56, topic="middle"),
        _paper("p4", 0.42, 0.42, rubric=0.94, topic="rescue-a"),
        _paper("p5", 0.39, 0.38, rubric=0.88, topic="rescue-b"),
        _paper("p6", 0.25, 0.20, rubric=0.30, topic="low"),
    ]
    available = {
        ("p1", "p4"),
        ("p2", "p4"),
        ("p1", "p5"),
        ("p2", "p5"),
    }

    schedule = schedule_new_information_challenger_pairs(
        papers,
        [],
        k=2,
        budget=PairwiseBudget(n=len(papers), candidate_size=6, budget=4),
        seed=13,
        config=NewInformationChallengerConfig(
            posterior_samples=120,
            anchor_multiplier=1,
            challenger_multiplier=1,
            min_challengers=2,
            random_floor_fraction=0.0,
            minimum_rubric_residual=0.05,
            per_item_cap=4,
        ),
        available_pair_keys=available,
    )

    challenger_ids = {
        pair.diagnostics["challenger_id"] for pair in schedule.pairs
    }
    assert challenger_ids == {"p4", "p5"}
    assert len(schedule.pairs) == 4
    assert schedule.diagnostics["acquisition"]["future_labels_used_for_scheduling"] is False
    assert schedule.diagnostics["new_information_challenger"][
        "cached_label_values_used_before_scheduling"
    ] is False
    assert schedule.diagnostics["available_label_filter"][
        "cache_availability_used_for_scheduling"
    ] is True
    assert all(
        pair.diagnostics["pair_role"] == "rubric_residual_anchor_challenger"
        for pair in schedule.pairs
    )


def test_new_information_replay_uses_only_cached_pairs_without_missing_labels() -> None:
    papers = [
        _paper("p1", 0.92, 0.10, rubric=0.90, topic="anchor"),
        _paper("p2", 0.84, 0.15, rubric=0.82, topic="anchor"),
        _paper("p3", 0.42, 0.42, rubric=0.94, topic="rescue"),
        _paper("p4", 0.25, 0.20, rubric=0.30, topic="low"),
    ]
    cached = {
        ("p1", "p3"): PairwiseComparison(
            left_id="p1",
            right_id="p3",
            winner="right",
            confidence=0.9,
        ),
        ("p2", "p3"): PairwiseComparison(
            left_id="p2",
            right_id="p3",
            winner="left",
            confidence=0.8,
        ),
    }

    replay = replay_new_information_challenger(
        papers,
        cached,
        k=2,
        budget=PairwiseBudget(n=len(papers), candidate_size=4, budget=3),
        seed=19,
        config=NewInformationChallengerConfig(
            posterior_samples=100,
            anchor_multiplier=1,
            challenger_multiplier=1,
            min_challengers=1,
            random_floor_fraction=0.0,
            minimum_rubric_residual=0.05,
            per_item_cap=3,
        ),
    )

    assert len(replay.schedule) == 2
    assert len(replay.comparisons) == len(replay.schedule)
    assert replay.diagnostics["missing_pairwise_labels"] == 0
    assert replay.diagnostics["label_policy"]["future_labels_used_for_scheduling"] is False
    assert replay.diagnostics["label_policy"]["cached_label_values_used_before_scheduling"] is False


def test_new_information_scheduler_fills_cached_shortfall_with_frontier_fallback() -> None:
    papers = [
        _paper("p1", 0.92, 0.10, rubric=0.90, topic="anchor"),
        _paper("p2", 0.84, 0.15, rubric=0.82, topic="anchor"),
        _paper("p3", 0.58, 0.20, rubric=0.56, topic="middle"),
        _paper("p4", 0.42, 0.42, rubric=0.94, topic="rescue-a"),
        _paper("p5", 0.39, 0.38, rubric=0.88, topic="rescue-b"),
    ]
    available = {
        ("p1", "p4"),
        ("p2", "p4"),
        ("p4", "p5"),
    }

    schedule = schedule_new_information_challenger_pairs(
        papers,
        [],
        k=2,
        budget=PairwiseBudget(n=len(papers), candidate_size=5, budget=3),
        seed=23,
        config=NewInformationChallengerConfig(
            posterior_samples=100,
            anchor_multiplier=1,
            challenger_multiplier=1,
            min_challengers=2,
            random_floor_fraction=0.0,
            minimum_rubric_residual=0.05,
            per_item_cap=3,
        ),
        available_pair_keys=available,
    )

    assert len(schedule.pairs) == 3
    assert {pair.purpose for pair in schedule.pairs} == {
        "new_information_false_negative_challenge",
        "new_information_cached_frontier_fallback",
    }
    assert any(
        {pair.left_id, pair.right_id} == {"p4", "p5"}
        and pair.purpose == "new_information_cached_frontier_fallback"
        for pair in schedule.pairs
    )
    fallback = schedule.diagnostics["cached_frontier_fallback"]
    assert fallback["enabled"] is True
    assert fallback["primary_scheduled_total"] == 2
    assert fallback["selected_total"] == 1
    assert fallback["remaining_shortfall"] == 0
    assert fallback["future_labels_used_for_scheduling"] is False
    assert fallback["cached_label_values_used_before_scheduling"] is False
    assert schedule.diagnostics["new_information_challenger"]["budget_complete"] is True


def test_new_information_replay_keeps_shortfall_when_fallback_cache_is_insufficient() -> None:
    papers = [
        _paper("p1", 0.92, 0.10, rubric=0.90, topic="anchor"),
        _paper("p2", 0.84, 0.15, rubric=0.82, topic="anchor"),
        _paper("p3", 0.58, 0.20, rubric=0.56, topic="middle"),
        _paper("p4", 0.42, 0.42, rubric=0.94, topic="rescue-a"),
        _paper("p5", 0.39, 0.38, rubric=0.88, topic="rescue-b"),
    ]
    cached = {
        ("p1", "p4"): PairwiseComparison(
            left_id="p1",
            right_id="p4",
            winner="right",
            confidence=0.9,
        ),
        ("p2", "p4"): PairwiseComparison(
            left_id="p2",
            right_id="p4",
            winner="left",
            confidence=0.8,
        ),
    }

    replay = replay_new_information_challenger(
        papers,
        cached,
        k=2,
        budget=PairwiseBudget(n=len(papers), candidate_size=5, budget=3),
        seed=29,
        config=NewInformationChallengerConfig(
            posterior_samples=100,
            anchor_multiplier=1,
            challenger_multiplier=1,
            min_challengers=2,
            random_floor_fraction=0.0,
            minimum_rubric_residual=0.05,
            per_item_cap=3,
        ),
    )

    assert len(replay.schedule) == 2
    assert replay.diagnostics["budget_complete"] is False
    assert replay.diagnostics["scheduled_pairwise_shortfall"] == 1
    assert replay.diagnostics["cached_frontier_fallback"]["remaining_shortfall"] == 1
    assert replay.diagnostics["missing_pairwise_labels"] == 0


def _paper(
    paper_id: str,
    probability: float,
    uncertainty: float,
    *,
    rubric: float,
    topic: str,
) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=f"{topic} {paper_id} method",
        abstract=f"{topic} abstract with distinctive evidence for {paper_id}",
        pointwise=PointwiseAssessment(
            good_probability=probability,
            uncertainty=uncertainty,
            rubric_scores={
                "novelty": rubric,
                "evidence_strength": rubric,
                "practical_impact": rubric,
                "technical_depth": rubric,
                "cross_domain_interest": rubric,
            },
        ),
        metadata={"topic": topic, "source": "fixture"},
    )
