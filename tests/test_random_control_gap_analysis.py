from __future__ import annotations

from sestina.backtest import Prediction
from sestina.models import PairwiseComparison, PairwiseOrderMetadata, ScheduledPair

from scripts.analyze_random_control_gap import (
    aggregate_complete_arm_diagnostics,
    oracle_bounds,
    pair_graph_diagnostics,
    positive_exposure_diagnostics,
    top_k_error_decomposition,
)


def test_oracle_bounds_distinguish_exposure_from_observed_pairwise_wins() -> None:
    schedule = [
        _pair("positive_b", "negative_c"),
        _pair("positive_a", "negative_c"),
    ]
    comparisons = [
        PairwiseComparison(
            left_id="positive_b",
            right_id="negative_c",
            winner="right",
            soft_probability=0.8,
            confidence=1.0,
        ),
        PairwiseComparison(
            left_id="positive_a",
            right_id="negative_c",
            winner="left",
            soft_probability=0.8,
            confidence=1.0,
        ),
    ]

    bounds = oracle_bounds(
        k=2,
        relevant_ids={"positive_a", "positive_b"},
        pointwise_top_k_ids=["positive_a", "negative_x"],
        schedule=schedule,
        comparisons=comparisons,
    )

    assert bounds["pointwise_recall_at_k"] == 0.5
    assert (
        bounds["touched_positive_upper_bound"]["recall_cap"]
        == 1.0
    )
    assert (
        bounds["positive_negative_pair_label_oracle_upper_bound"]["recall_cap"]
        == 1.0
    )
    assert bounds["observed_positive_winner_upper_bound"]["recall_cap"] == 0.5
    assert bounds["label_interpretation_gap_vs_pair_label_oracle"] == 0.5


def test_positive_exposure_counts_touched_positives_and_pair_types() -> None:
    exposure = positive_exposure_diagnostics(
        [_pair("positive_a", "negative_b"), _pair("negative_c", "negative_d")],
        relevant_ids={"positive_a", "positive_e"},
        paper_count=5,
    )

    assert exposure["unique_papers_touched"] == 4
    assert exposure["unique_future_positives_touched"] == 1
    assert exposure["pairs_touching_future_positive"] == 1
    assert exposure["positive_negative_pairs"] == 1
    assert exposure["negative_negative_pairs"] == 1


def test_pair_graph_reports_degree_and_connectivity_around_positives() -> None:
    graph = pair_graph_diagnostics(
        [
            _pair("positive_a", "negative_b"),
            _pair("negative_b", "negative_c"),
            _pair("positive_d", "negative_e"),
        ],
        relevant_ids={"positive_a", "positive_d", "positive_missing"},
        posterior_top_k_ids=["positive_a", "negative_b"],
        pointwise_top_k_ids=["positive_missing", "negative_e"],
    )

    assert graph["component_count"] == 2
    assert graph["largest_component_size"] == 3
    assert graph["components_with_future_positive"] == 2
    assert graph["future_positive_degree"]["zero_degree_count"] == 1
    assert graph["posterior_top_k_degree"]["mean"] == 1.5
    assert graph["pointwise_top_k_degree"]["zero_degree_count"] == 1


def test_top_k_decomposition_marks_false_negatives_with_touch_status() -> None:
    decomposition = top_k_error_decomposition(
        predictions=[
            Prediction("negative_a", 0.9),
            Prediction("positive_a", 0.8),
            Prediction("positive_b", 0.2),
        ],
        relevant_ids={"positive_a", "positive_b"},
        k=2,
        schedule=[_pair("positive_b", "negative_a")],
        pointwise_top_k_ids=["positive_a", "negative_a"],
        labels_by_id={
            "positive_b": {
                "title": "Missed positive",
                "citation_count": 10,
                "citation_rank": 1,
                "citation_positive": True,
            }
        },
    )

    assert decomposition["selected_positive_count"] == 1
    assert decomposition["selected_false_positive_count"] == 1
    assert decomposition["false_negative_count"] == 1
    assert decomposition["false_negative_rows"][0]["paper_id"] == "positive_b"
    assert decomposition["false_negative_rows"][0]["touched_by_scheduled_pair"] is True


def test_aggregate_diagnostics_exclude_partial_label_arms() -> None:
    complete_payload = {
        "positive_exposure": {
            "scheduled_pairs_total": 1,
            "pairs_touching_future_positive": 1,
            "unique_future_positives_touched": 1,
            "unique_future_positive_touch_rate": 0.5,
            "positive_negative_pairs": 1,
            "unique_papers_touched": 2,
            "unique_paper_touch_rate": 0.5,
            "touched_future_positive_ids": ["positive_a"],
        },
        "pair_graph": {
            "component_count": 1,
            "largest_component_size": 2,
            "components_with_future_positive": 1,
            "future_positive_degree": {"mean": 1.0, "zero_degree_count": 0},
            "posterior_top_k_degree": {"mean": 1.0, "zero_degree_count": 0},
        },
        "pair_label_alignment": {
            "positive_negative_pairs_with_label": 1,
            "positive_wins": 1,
            "negative_wins": 0,
            "ties_or_uncertain": 0,
            "mean_decisive_confidence_on_positive_negative_pairs": 1.0,
            "mean_decisive_soft_probability_on_positive_negative_pairs": 0.8,
        },
        "oracle_bounds": {
            "pointwise_recall_at_k": 0.5,
            "touched_positive_upper_bound": {"recall_cap": 0.5},
            "pointwise_plus_touched_positive_upper_bound": {"recall_cap": 1.0},
            "positive_negative_pair_label_oracle_upper_bound": {"recall_cap": 1.0},
            "observed_positive_winner_upper_bound": {"recall_cap": 1.0},
            "label_interpretation_gap_vs_pair_label_oracle": 0.0,
        },
        "top_k_error_decomposition": {
            "selected_positive_count": 1,
            "selected_false_positive_count": 1,
            "false_negative_count": 1,
        },
    }
    bucket_results = [
        {
            "arms": {
                "complete_arm": complete_payload,
                "partial_arm": complete_payload,
            }
        }
    ]

    diagnostics = aggregate_complete_arm_diagnostics(
        bucket_results,
        comparison_sources={
            "complete_arm": {"aggregate_metrics_included": True},
            "partial_arm": {"aggregate_metrics_included": False},
        },
    )

    assert set(diagnostics["positive_exposure"]) == {"complete_arm"}
    assert set(diagnostics["oracle_bounds"]) == {"complete_arm"}


def _pair(left_id: str, right_id: str) -> ScheduledPair:
    return ScheduledPair(
        left_id=left_id,
        right_id=right_id,
        priority=0.0,
        purpose="test",
        order=PairwiseOrderMetadata(),
    )
