from __future__ import annotations

import pytest

from sestina.diagnostics import DiagnosticRecorder
from sestina.models import PairwiseComparison
from sestina.pairwise_strength import (
    PairwiseStrengthCalibrationConfig,
    soft_strength_calibrated_comparisons,
)


def test_soft_strength_calibration_downweights_close_decisive_wins() -> None:
    comparisons = [
        PairwiseComparison(
            left_id="close",
            right_id="anchor",
            winner="left",
            soft_probability=0.54,
            confidence=1.0,
        ),
        PairwiseComparison(
            left_id="clear",
            right_id="anchor",
            winner="left",
            soft_probability=0.90,
            confidence=0.8,
        ),
        PairwiseComparison(
            left_id="tie",
            right_id="anchor",
            winner="tie",
            soft_probability=0.5,
            confidence=0.7,
        ),
    ]

    result = soft_strength_calibrated_comparisons(
        comparisons,
        config=PairwiseStrengthCalibrationConfig(
            minimum_win_multiplier=0.5,
            margin_exponent=1.0,
        ),
    )

    by_left_id = {comparison.left_id: comparison for comparison in result.comparisons}
    assert by_left_id["close"].confidence == 0.54
    assert by_left_id["clear"].confidence == pytest.approx(0.72)
    assert by_left_id["tie"].confidence == 0.7
    assert by_left_id["close"].metadata["strength_calibration"][
        "strength_multiplier"
    ] == 0.54
    assert result.diagnostics["summary"]["mean_decisive_strength_multiplier"] == 0.72
    assert result.diagnostics["rule_parameters"][
        "uses_future_labels_for_calibration"
    ] is False


def test_soft_strength_calibration_warns_when_soft_probability_missing() -> None:
    recorder = DiagnosticRecorder()

    result = soft_strength_calibrated_comparisons(
        [
            PairwiseComparison(
                left_id="missing_soft",
                right_id="anchor",
                winner="right",
                soft_probability=None,
                confidence=1.0,
            )
        ],
        diagnostics=recorder,
    )

    assert result.comparisons[0].confidence == 0.75
    assert result.diagnostics["summary"]["missing_soft_probability_count"] == 1
    events = recorder.to_dict()["events"]
    assert events[-1]["code"] == "pairwise_strength_missing_soft_probability"
    assert events[-1]["level"] == "warning"
