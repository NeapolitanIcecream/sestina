from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SelectionMode = Literal["content_only", "metadata_aware"]
Winner = Literal["left", "right", "tie", "uncertain"]


def _as_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True, slots=True)
class TargetSpec:
    top_k: int | None = None
    top_alpha: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TargetSpec":
        payload = data or {}
        top_k = payload.get("top_k")
        top_alpha = payload.get("top_alpha")
        return cls(
            top_k=int(top_k) if top_k is not None else None,
            top_alpha=float(top_alpha) if top_alpha is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"top_k": self.top_k, "top_alpha": self.top_alpha}


@dataclass(frozen=True, slots=True)
class PointwiseAssessment:
    good_probability: float
    uncertainty: float
    rubric_scores: dict[str, float] = field(default_factory=dict)
    summary: str = ""
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PointwiseAssessment":
        payload = data or {}
        probability = payload.get(
            "pointwise_good_probability", payload.get("good_probability", 0.5)
        )
        raw_rubric = payload.get("rubric_scores") or {}
        rubric = {
            str(key): float(value)
            for key, value in raw_rubric.items()
            if isinstance(value, int | float)
        }
        reasons = payload.get("reasons") or []
        return cls(
            good_probability=_clamp(_as_float(probability, default=0.5), 0.001, 0.999),
            uncertainty=_clamp(_as_float(payload.get("uncertainty"), default=0.5)),
            rubric_scores=rubric,
            summary=str(payload.get("summary") or ""),
            reasons=[str(reason) for reason in reasons],
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pointwise_good_probability": self.good_probability,
            "uncertainty": self.uncertainty,
            "rubric_scores": dict(self.rubric_scores),
            "summary": self.summary,
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Paper:
    paper_id: str
    title: str
    abstract: str = ""
    pointwise: PointwiseAssessment = field(
        default_factory=lambda: PointwiseAssessment(
            good_probability=0.5,
            uncertainty=0.5,
        )
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Paper":
        paper_id = data.get("paper_id", data.get("id"))
        if paper_id is None or str(paper_id).strip() == "":
            raise ValueError("paper is missing paper_id")
        pointwise = data.get("pointwise") or data.get("assessment") or {}
        return cls(
            paper_id=str(paper_id),
            title=str(data.get("title") or ""),
            abstract=str(data.get("abstract") or data.get("summary") or ""),
            pointwise=PointwiseAssessment.from_dict(pointwise),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "paper_id": self.paper_id,
            "title": self.title,
            "pointwise": self.pointwise.to_dict(),
            "metadata": dict(self.metadata),
        }
        if include_text:
            payload["abstract"] = self.abstract
        return payload


@dataclass(frozen=True, slots=True)
class PairwiseOrderMetadata:
    shown_first_id: str | None = None
    shown_second_id: str | None = None
    randomized: bool = False
    seed: int | None = None
    position_bias_audit: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PairwiseOrderMetadata":
        payload = data or {}
        return cls(
            shown_first_id=(
                str(payload["shown_first_id"])
                if payload.get("shown_first_id") is not None
                else None
            ),
            shown_second_id=(
                str(payload["shown_second_id"])
                if payload.get("shown_second_id") is not None
                else None
            ),
            randomized=bool(payload.get("randomized", False)),
            seed=int(payload["seed"]) if payload.get("seed") is not None else None,
            position_bias_audit=bool(payload.get("position_bias_audit", False)),
            extra={
                str(key): value
                for key, value in payload.items()
                if key
                not in {
                    "shown_first_id",
                    "shown_second_id",
                    "randomized",
                    "seed",
                    "position_bias_audit",
                }
            },
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "shown_first_id": self.shown_first_id,
            "shown_second_id": self.shown_second_id,
            "randomized": self.randomized,
            "seed": self.seed,
            "position_bias_audit": self.position_bias_audit,
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True, slots=True)
class PairwiseComparison:
    left_id: str
    right_id: str
    winner: Winner
    soft_probability: float | None = None
    confidence: float = 1.0
    reasons: list[str] = field(default_factory=list)
    order: PairwiseOrderMetadata = field(default_factory=PairwiseOrderMetadata)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PairwiseComparison":
        left_id = data.get("left_id", data.get("paper_a_id"))
        right_id = data.get("right_id", data.get("paper_b_id"))
        if left_id is None or right_id is None:
            raise ValueError("comparison is missing left_id or right_id")
        winner_raw = str(data.get("winner", "uncertain")).strip().lower()
        winner_label = winner_raw.replace("-", "_").replace(" ", "_")
        if winner_raw not in {"left", "right", "tie", "uncertain"}:
            if winner_raw == str(left_id).lower():
                winner_raw = "left"
            elif winner_raw == str(right_id).lower():
                winner_raw = "right"
            elif winner_label in {"a", "paper_a", "shown_a", "first", "shown_first"}:
                winner_raw = "left"
            elif winner_label in {"b", "paper_b", "shown_b", "second", "shown_second"}:
                winner_raw = "right"
            else:
                winner_raw = "uncertain"
        reasons = data.get("reasons") or []
        soft_probability = data.get("soft_probability")
        return cls(
            left_id=str(left_id),
            right_id=str(right_id),
            winner=winner_raw,  # type: ignore[arg-type]
            soft_probability=(
                _clamp(float(soft_probability), 0.001, 0.999)
                if soft_probability is not None
                else None
            ),
            confidence=_clamp(_as_float(data.get("confidence"), default=1.0)),
            reasons=[str(reason) for reason in reasons],
            order=PairwiseOrderMetadata.from_dict(data.get("order")),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "winner": self.winner,
            "soft_probability": self.soft_probability,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "order": self.order.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RunInput:
    papers: list[Paper]
    target: TargetSpec
    comparisons: list[PairwiseComparison] = field(default_factory=list)
    mode: SelectionMode = "content_only"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScheduledPair:
    left_id: str
    right_id: str
    priority: float
    purpose: str
    order: PairwiseOrderMetadata
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "priority": self.priority,
            "purpose": self.purpose,
            "order": self.order.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }
