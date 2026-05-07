from __future__ import annotations

import pytest

from scripts.analyze_posterior_decision_shrinkage import (
    _aggregate_comparison_sources,
    _validate_aggregate_comparison_sources,
)


def test_partial_comparison_sources_are_excluded_from_aggregate_metrics() -> None:
    bucket_results = [
        {
            "arms": {
                "complete_arm": {
                    "comparison_source": {
                        "source": "followup",
                        "scheduler_kind": "complete",
                        "artifact_dir": "artifacts/complete",
                        "scheduled_pairwise_total": 20,
                        "cached_pairwise_labels_available": 20,
                        "missing_pairwise_labels": 0,
                        "partial": False,
                    }
                },
                "partial_arm": {
                    "comparison_source": {
                        "source": "followup",
                        "scheduler_kind": "legacy_partial",
                        "artifact_dir": "artifacts/partial",
                        "scheduled_pairwise_total": 20,
                        "cached_pairwise_labels_available": 8,
                        "missing_pairwise_labels": 12,
                        "partial": True,
                    }
                },
            }
        }
    ]

    summaries = _aggregate_comparison_sources(
        bucket_results,
        arm_names=["complete_arm", "partial_arm"],
    )

    assert summaries["complete_arm"]["aggregate_metrics_included"] is True
    assert summaries["partial_arm"]["aggregate_metrics_included"] is False
    assert summaries["partial_arm"]["explicit_partial_caveat"] is True
    assert summaries["partial_arm"]["aggregate_caveat"] == (
        "excluded_from_aggregate_metrics_due_to_partial_cached_pairwise_labels"
    )


def test_partial_aggregate_metrics_require_explicit_caveat() -> None:
    aggregate_metrics = {"partial_arm": {"posterior_topk": {"recall_at_k": 0.25}}}
    uncaveated_sources = {
        "partial_arm": {
            "partial": True,
            "aggregate_metrics_included": True,
        }
    }

    with pytest.raises(ValueError, match="partial pairwise labels"):
        _validate_aggregate_comparison_sources(
            aggregate_metrics,
            uncaveated_sources,
        )

    caveated_sources = {
        "partial_arm": {
            "partial": True,
            "aggregate_metrics_included": True,
            "explicit_partial_caveat": True,
            "aggregate_caveat": "included_for_diagnostic_context_only",
        }
    }

    _validate_aggregate_comparison_sources(aggregate_metrics, caveated_sources)
