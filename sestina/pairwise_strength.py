from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sestina.diagnostics import DiagnosticRecorder
from sestina.models import PairwiseComparison


@dataclass(frozen=True, slots=True)
class PairwiseStrengthCalibrationConfig:
    """Parameters for soft-probability pairwise strength calibration."""

    minimum_win_multiplier: float = 0.5
    margin_exponent: float = 1.0
    default_soft_probability: float = 0.75


@dataclass(frozen=True, slots=True)
class PairwiseStrengthCalibrationResult:
    comparisons: list[PairwiseComparison]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def soft_strength_calibrated_comparisons(
    comparisons: list[PairwiseComparison],
    *,
    config: PairwiseStrengthCalibrationConfig | None = None,
    diagnostics: DiagnosticRecorder | None = None,
) -> PairwiseStrengthCalibrationResult:
    """Downweight decisive pairwise labels according to their soft win margin.

    The existing aggregation model already uses ``soft_probability`` as the
    fractional Bradley-Terry target. This calibration also uses that response
    strength as a likelihood/confidence multiplier, so a close 0.54 win does not
    carry the same curvature weight as a decisive 0.90 win. Ties and uncertain
    labels keep the aggregation module's existing limited-weight treatment.
    """
    cfg = config or PairwiseStrengthCalibrationConfig()
    recorder = diagnostics or DiagnosticRecorder()
    calibrated: list[PairwiseComparison] = []
    rows: list[dict[str, Any]] = []
    missing_soft_probability_count = 0
    clipped_soft_probability_count = 0

    for comparison in comparisons:
        row, calibrated_comparison = _calibrate_comparison(comparison, cfg)
        missing_soft_probability_count += int(row["soft_probability_missing"])
        clipped_soft_probability_count += int(row["soft_probability_clipped"])
        rows.append(row)
        calibrated.append(calibrated_comparison)

    summary = _summary_payload(
        rows,
        config=cfg,
        missing_soft_probability_count=missing_soft_probability_count,
        clipped_soft_probability_count=clipped_soft_probability_count,
    )
    payload = {
        "method": "soft_probability_strength_calibration",
        "rule_parameters": {
            "minimum_win_multiplier": cfg.minimum_win_multiplier,
            "margin_exponent": cfg.margin_exponent,
            "default_soft_probability": cfg.default_soft_probability,
            "win_multiplier_formula": (
                "minimum_win_multiplier + (1 - minimum_win_multiplier) * "
                "(abs(soft_probability - 0.5) / 0.5) ** margin_exponent"
            ),
            "tie_and_uncertain_multiplier": 1.0,
            "uses_future_labels_for_calibration": False,
        },
        "summary": summary,
        "comparison_strengths": rows,
    }
    recorder.record(
        step="pairwise_strength_calibration",
        code="pairwise_strength_calibration_completed",
        message="calibrated pairwise comparison likelihood strengths",
        data=summary,
    )
    if missing_soft_probability_count:
        recorder.record(
            step="pairwise_strength_calibration",
            code="pairwise_strength_missing_soft_probability",
            level="warning",
            message="defaulted missing pairwise soft probabilities during calibration",
            data={
                "missing_soft_probability_count": missing_soft_probability_count,
                "default_soft_probability": cfg.default_soft_probability,
            },
        )
    return PairwiseStrengthCalibrationResult(
        comparisons=calibrated,
        diagnostics=payload,
    )


def _calibrate_comparison(
    comparison: PairwiseComparison,
    config: PairwiseStrengthCalibrationConfig,
) -> tuple[dict[str, Any], PairwiseComparison]:
    original_confidence = _clamp(comparison.confidence)
    soft_probability_missing = comparison.soft_probability is None
    raw_soft_probability = (
        config.default_soft_probability
        if soft_probability_missing
        else float(comparison.soft_probability)
    )
    soft_probability = _clamp(raw_soft_probability, 0.5, 0.999)
    soft_probability_clipped = abs(soft_probability - raw_soft_probability) > 1e-12
    decisive = comparison.winner in {"left", "right"}
    if decisive:
        soft_margin = abs(soft_probability - 0.5) / 0.5
        multiplier = _clamp(
            config.minimum_win_multiplier
            + (
                (1.0 - config.minimum_win_multiplier)
                * (soft_margin ** config.margin_exponent)
            )
        )
        reason = "decisive_soft_probability_margin"
    else:
        soft_margin = 0.0
        multiplier = 1.0
        reason = "non_decisive_existing_weight"

    calibrated_confidence = _clamp(original_confidence * multiplier)
    row = {
        "left_id": comparison.left_id,
        "right_id": comparison.right_id,
        "winner": comparison.winner,
        "decisive": decisive,
        "soft_probability": round(soft_probability, 8),
        "soft_probability_missing": soft_probability_missing,
        "soft_probability_clipped": soft_probability_clipped,
        "soft_margin": round(soft_margin, 8),
        "original_confidence": round(original_confidence, 8),
        "strength_multiplier": round(multiplier, 8),
        "calibrated_confidence": round(calibrated_confidence, 8),
        "calibration_reason": reason,
    }
    calibrated = PairwiseComparison(
        left_id=comparison.left_id,
        right_id=comparison.right_id,
        winner=comparison.winner,
        soft_probability=comparison.soft_probability,
        confidence=calibrated_confidence,
        reasons=list(comparison.reasons),
        order=comparison.order,
        metadata={
            **comparison.metadata,
            "strength_calibration": {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "left_id",
                    "right_id",
                    "winner",
                }
            },
        },
    )
    return row, calibrated


def _summary_payload(
    rows: list[dict[str, Any]],
    *,
    config: PairwiseStrengthCalibrationConfig,
    missing_soft_probability_count: int,
    clipped_soft_probability_count: int,
) -> dict[str, Any]:
    winner_counts = Counter(str(row["winner"]) for row in rows)
    decisive_rows = [row for row in rows if bool(row["decisive"])]
    multipliers = [float(row["strength_multiplier"]) for row in rows]
    decisive_multipliers = [
        float(row["strength_multiplier"]) for row in decisive_rows
    ]
    original_confidences = [float(row["original_confidence"]) for row in rows]
    calibrated_confidences = [float(row["calibrated_confidence"]) for row in rows]
    soft_margins = [float(row["soft_margin"]) for row in decisive_rows]
    return {
        "comparison_count": len(rows),
        "decisive_count": len(decisive_rows),
        "tie_count": winner_counts["tie"],
        "uncertain_count": winner_counts["uncertain"],
        "winner_counts": dict(sorted(winner_counts.items())),
        "missing_soft_probability_count": missing_soft_probability_count,
        "clipped_soft_probability_count": clipped_soft_probability_count,
        "mean_original_confidence": _mean(original_confidences),
        "mean_calibrated_confidence": _mean(calibrated_confidences),
        "mean_strength_multiplier": _mean(multipliers),
        "min_strength_multiplier": round(min(multipliers), 8) if multipliers else 0.0,
        "max_strength_multiplier": round(max(multipliers), 8) if multipliers else 0.0,
        "mean_decisive_strength_multiplier": _mean(decisive_multipliers),
        "mean_decisive_soft_margin": _mean(soft_margins),
        "strength_multiplier_histogram": _histogram(
            decisive_multipliers,
            bins=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        ),
        "minimum_win_multiplier": config.minimum_win_multiplier,
        "margin_exponent": config.margin_exponent,
    }


def _histogram(values: list[float], *, bins: tuple[float, ...]) -> dict[str, int]:
    if not values:
        return {}
    counts = {f"<= {threshold:.1f}": 0 for threshold in bins}
    for value in values:
        for threshold in bins:
            if value <= threshold:
                counts[f"<= {threshold:.1f}"] += 1
                break
    return counts


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 8) if values else 0.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
