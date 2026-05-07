from __future__ import annotations

import pytest

from scripts.run_no_paid_algorithm_sweep import (
    ARM_BORDA,
    ARM_CHALLENGER,
    ARM_CI,
    ARM_COVERAGE,
    ARM_EXACT,
    ARM_HISTORICAL,
    ARM_POSTERIOR_PRIOR,
    BLOCKED_PROTOCOL_ARTIFACT_TYPE,
    CANDIDATE_ARMS,
    SCHEMA_VERSION,
    validate_blocked_protocol_report,
    validate_no_paid_sweep_artifact_schema,
)
from sestina.models import PairwiseComparison, Paper, PointwiseAssessment
from sestina.no_paid_algorithm_sweep import (
    HybridScheduleConfig,
    paper_borda_lcb_predictions,
    paired_seed_metric_deltas,
    schedule_model_visible_hybrid_pairs,
)
from sestina.scheduler import PairwiseBudget


def test_borda_lcb_ranking_uses_revealed_pairwise_wins_without_labels() -> None:
    papers = [
        _paper("a", probability=0.72),
        _paper("b", probability=0.58),
        _paper("c", probability=0.52),
    ]
    comparisons = [
        PairwiseComparison(left_id="b", right_id="a", winner="left", confidence=1.0),
        PairwiseComparison(left_id="b", right_id="c", winner="left", confidence=1.0),
        PairwiseComparison(left_id="a", right_id="c", winner="left", confidence=0.3),
    ]

    predictions, diagnostics = paper_borda_lcb_predictions(papers, comparisons)

    assert predictions[0].paper_id == "b"
    assert diagnostics["rule_parameters"]["uses_future_labels_for_decision"] is False
    assert diagnostics["papers_with_pairwise_evidence"] == 3


def test_hybrid_schedule_preserves_random_floor_and_cache_safe_label_policy() -> None:
    papers = [_paper(f"p{i}", probability=0.8 - (i * 0.05)) for i in range(6)]
    available = {
        tuple(sorted((left.paper_id, right.paper_id)))
        for index, left in enumerate(papers)
        for right in papers[index + 1 :]
    }

    schedule, diagnostics = schedule_model_visible_hybrid_pairs(
        papers,
        k=2,
        budget=PairwiseBudget(n=6, candidate_size=6, budget=6),
        seed=17,
        available_pair_keys=available,
        config=HybridScheduleConfig(
            name="test_hybrid",
            random_floor_fraction=0.5,
            min_random_floor_pairs=2,
            per_item_cap=4,
        ),
    )

    assert len(schedule) == 6
    assert diagnostics["coverage"]["random_floor_pairs"] >= 2
    assert diagnostics["label_policy"]["future_labels_used_for_scheduling"] is False
    assert (
        diagnostics["label_policy"]["cached_label_values_used_before_scheduling"]
        is False
    )


def test_paired_seed_metric_deltas_report_seed_level_confidence_intervals() -> None:
    seed_rows = {
        "17": {
            "candidate": {
                "recall_at_k": 0.4,
                "ndcg_at_k": 0.5,
                "average_precision": 0.6,
            },
            "random": {
                "recall_at_k": 0.3,
                "ndcg_at_k": 0.4,
                "average_precision": 0.55,
            },
        },
        "101": {
            "candidate": {
                "recall_at_k": 0.2,
                "ndcg_at_k": 0.3,
                "average_precision": 0.4,
            },
            "random": {
                "recall_at_k": 0.25,
                "ndcg_at_k": 0.2,
                "average_precision": 0.35,
            },
        },
    }

    deltas = paired_seed_metric_deltas(
        seed_rows,
        comparison_arm="candidate",
        reference_arm="random",
    )

    assert deltas["metric_deltas"]["recall_at_k"]["count"] == 2
    assert deltas["metric_deltas"]["recall_at_k"]["normal_approx_95_ci"] != [
        None,
        None,
    ]
    assert deltas["seed_deltas"]["17"]["average_precision"] == 0.05


def test_sweep_schema_requires_all_nontrivial_candidate_policies() -> None:
    payload = _minimal_sweep_payload()

    validate_no_paid_sweep_artifact_schema(payload)

    payload["candidate_arms_tried"] = [
        row for row in payload["candidate_arms_tried"] if row["name"] != ARM_BORDA
    ]
    with pytest.raises(ValueError, match=ARM_BORDA):
        validate_no_paid_sweep_artifact_schema(payload)


def test_sweep_schema_blocks_active_gate_artifact_when_gate_failed() -> None:
    payload = _minimal_sweep_payload()
    payload["active_arm_gate"]["artifact_written"] = True

    with pytest.raises(ValueError, match="must not write an active-arm gate"):
        validate_no_paid_sweep_artifact_schema(payload)


def test_blocked_protocol_report_keeps_next_experiment_gate_blocked() -> None:
    from sestina.experiment_protocol import build_next_experiment_protocol

    report = {
        "artifact_type": BLOCKED_PROTOCOL_ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 0,
        "active_arm_gate_artifact_produced": False,
        "next_experiment_protocol": build_next_experiment_protocol(),
    }

    validate_blocked_protocol_report(report)

    report["active_arm_gate_artifact_produced"] = True
    with pytest.raises(ValueError, match="cannot produce an active gate"):
        validate_blocked_protocol_report(report)


def _paper(paper_id: str, *, probability: float) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        abstract="cache safe local fixture",
        pointwise=PointwiseAssessment(
            good_probability=probability,
            uncertainty=0.4,
            rubric_scores={
                "novelty": min(0.95, probability + 0.1),
                "technical_depth": probability,
            },
        ),
        metadata={"primary_category": "cs.LG" if paper_id < "p3" else "cs.CL"},
    )


def _minimal_sweep_payload() -> dict:
    return {
        "artifact_type": "sestina-no-paid-algorithm-sweep",
        "schema_version": SCHEMA_VERSION,
        "phase": "pilot",
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 0,
        "active_arm_name": ARM_CI,
        "candidate_random_control_baseline": ARM_EXACT,
        "gate_verdict": {"paid_followup_allowed": False},
        "aggregate_metrics": {},
        "paired_deltas_vs_exact_pool_random": {"seed_deltas": {"17": {}}},
        "paired_deltas_by_candidate": {},
        "seed_level_metric_intervals": {},
        "aggregate_diagnostics": {
            "confidence_bound_unresolved_count": {},
            "graph_connectivity": {},
            "oracle_caps": {},
            "unique_future_positives_touched": {},
            "weak_bucket_deltas": {},
            "weak_bucket_deltas_by_candidate": {},
            "randomized_coverage": {},
        },
        "bucket_results": [],
        "candidate_arms_tried": [
            {"name": name}
            for name in (ARM_CI, ARM_BORDA, ARM_COVERAGE, ARM_CHALLENGER)
        ],
        "control_arms": [
            {"name": name}
            for name in (ARM_EXACT, ARM_HISTORICAL, ARM_POSTERIOR_PRIOR)
        ],
        "limitations": [],
        "active_arm_gate": {
            "artifact_written": False,
            "paid_followup_allowed": False,
        },
        "protocol_outcome": {"status": "blocked_no_candidate_passed"},
    }


assert set(CANDIDATE_ARMS) == {ARM_CI, ARM_BORDA, ARM_COVERAGE, ARM_CHALLENGER}
