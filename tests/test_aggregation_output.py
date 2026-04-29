from __future__ import annotations

from sestina.aggregation import aggregate
from sestina.diagnostics import DiagnosticRecorder
from sestina.models import PairwiseComparison, PairwiseOrderMetadata
from sestina.output import build_recommendations
from sestina.posterior import estimate_top_k_probabilities


def test_pairwise_evidence_can_lift_lower_pointwise_paper(paper_set) -> None:
    papers = paper_set(4)
    comparisons = [
        PairwiseComparison(
            left_id="p3",
            right_id="p1",
            winner="left",
            soft_probability=0.92,
            confidence=1.0,
            order=PairwiseOrderMetadata(position_bias_audit=True),
        ),
        PairwiseComparison(
            left_id="p3",
            right_id="p2",
            winner="left",
            soft_probability=0.90,
            confidence=1.0,
        ),
    ]

    result = aggregate(papers, comparisons)

    assert result.estimates["p3"].posterior_good_probability > result.estimates[
        "p2"
    ].posterior_good_probability
    assert result.diagnostics["method"] == "map_bayesian_bradley_terry_fractional"


def test_tie_and_uncertain_comparisons_have_limited_weight(paper_set) -> None:
    papers = paper_set(3)
    comparisons = [
        PairwiseComparison(left_id="p2", right_id="p1", winner="uncertain", confidence=1.0),
        PairwiseComparison(left_id="p2", right_id="p1", winner="tie", confidence=1.0),
    ]

    result = aggregate(papers, comparisons)

    assert result.estimates["p1"].posterior_good_probability > result.estimates[
        "p2"
    ].posterior_good_probability
    assert result.diagnostics["tie_total"] == 1
    assert result.diagnostics["uncertain_total"] == 1


def test_unknown_comparison_emits_structured_diagnostic(paper_set) -> None:
    papers = paper_set(3)
    diagnostics = DiagnosticRecorder()

    aggregate(
        papers,
        [PairwiseComparison(left_id="missing", right_id="p1", winner="left")],
        diagnostics=diagnostics,
    )

    events = diagnostics.to_dict()["events"]
    assert any(event["code"] == "comparison_unknown_paper" for event in events)


def test_posterior_tiering_returns_top_k_and_near_misses(paper_set) -> None:
    papers = paper_set(8)
    aggregation = aggregate(papers, [])
    posterior = estimate_top_k_probabilities(aggregation, k=3, samples=500, seed=99)

    output = build_recommendations(papers, aggregation, posterior, k=3)

    assert len(output.recommended_good_papers) == 3
    assert output.recommended_good_papers[0].tier in {"strong_yes", "yes"}
    assert output.near_misses
    assert {item.tier for item in output.all_tiers}.issuperset({"near_miss"})

