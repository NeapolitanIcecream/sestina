from __future__ import annotations

from sestina.backtest import (
    Prediction,
    budget_ablation_points,
    compare_strategies,
    evaluate_predictions,
)


def test_backtest_metrics_capture_top_k_quality() -> None:
    metrics = evaluate_predictions(
        [
            Prediction("p1", 0.9),
            Prediction("p2", 0.8),
            Prediction("p3", 0.7),
            Prediction("p4", 0.1),
        ],
        relevant_ids={"p1", "p3"},
        k=2,
    )

    assert metrics.recall_at_k == 0.5
    assert metrics.precision_at_k == 0.5
    assert 0.0 < metrics.ndcg_at_k < 1.0
    assert metrics.average_precision > metrics.precision_at_k
    assert metrics.brier_score < 0.3


def test_budget_ablation_includes_zero_and_default_scale() -> None:
    points = budget_ablation_points(n=100, k=5)

    assert points[0] == 0
    assert 5 in points
    assert 15 in points
    assert 25 in points


def test_compare_strategies_returns_metrics_for_named_rankers() -> None:
    metrics = compare_strategies(
        {
            "pointwise_only": [Prediction("p1", 0.9), Prediction("p2", 0.4)],
            "active_pairwise": [Prediction("p2", 0.8), Prediction("p1", 0.7)],
        },
        relevant_ids={"p2"},
        k=1,
    )

    assert metrics["active_pairwise"].recall_at_k == 1.0
    assert metrics["pointwise_only"].recall_at_k == 0.0
