from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from sestina.models import (
    PairwiseComparison,
    PairwiseOrderMetadata,
    Paper,
    ScheduledPair,
    Winner,
)

PaperCollection = Mapping[str, Paper] | Sequence[Paper]


class PairwiseJudge(Protocol):
    def compare(self, left: Paper, right: Paper) -> PairwiseComparison:
        """Return a pairwise comparison without mutating paper records."""


class ScheduledPairJudge(Protocol):
    def compare_scheduled(
        self,
        scheduled_pair: ScheduledPair,
        papers: PaperCollection,
    ) -> PairwiseComparison:
        """Judge a scheduled pair while preserving its randomized display order."""


@dataclass(frozen=True, slots=True)
class ScheduledPairwiseJudgeAdapter:
    judge: PairwiseJudge

    def compare_scheduled(
        self,
        scheduled_pair: ScheduledPair,
        papers: PaperCollection,
    ) -> PairwiseComparison:
        return compare_scheduled_pair(self.judge, scheduled_pair, papers)


def compare_scheduled_pair(
    judge: PairwiseJudge,
    scheduled_pair: ScheduledPair,
    papers: PaperCollection,
) -> PairwiseComparison:
    paper_by_id = _paper_mapping(papers)
    shown_first_id, shown_second_id = _scheduled_order_ids(scheduled_pair)
    missing = [
        paper_id
        for paper_id in (shown_first_id, shown_second_id)
        if paper_id not in paper_by_id
    ]
    if missing:
        raise ValueError("scheduled pair references unknown paper id")

    shown_comparison = judge.compare(
        paper_by_id[shown_first_id],
        paper_by_id[shown_second_id],
    )
    return map_scheduled_comparison_to_canonical(
        scheduled_pair,
        shown_comparison,
    )


def map_scheduled_comparison_to_canonical(
    scheduled_pair: ScheduledPair,
    shown_comparison: PairwiseComparison,
) -> PairwiseComparison:
    shown_first_id, shown_second_id = _scheduled_order_ids(scheduled_pair)
    if (
        shown_comparison.left_id != shown_first_id
        or shown_comparison.right_id != shown_second_id
    ):
        raise ValueError("scheduled pair judge response ids must match shown order")

    position_winner = _position_winner(
        shown_comparison.winner,
        shown_first_id=shown_first_id,
        shown_second_id=shown_second_id,
    )
    shown_winner_id: str | None = None
    if position_winner == "left":
        shown_winner_id = shown_first_id
        canonical_winner = (
            "left" if shown_winner_id == scheduled_pair.left_id else "right"
        )
    elif position_winner == "right":
        shown_winner_id = shown_second_id
        canonical_winner = (
            "left" if shown_winner_id == scheduled_pair.left_id else "right"
        )
    else:
        canonical_winner = position_winner

    metadata = {
        **shown_comparison.metadata,
        "scheduled_pair_priority": scheduled_pair.priority,
        "scheduled_pair_purpose": scheduled_pair.purpose,
        "judge_presented_left_id": shown_comparison.left_id,
        "judge_presented_right_id": shown_comparison.right_id,
        "raw_position_winner": str(shown_comparison.winner),
    }
    if shown_winner_id is not None:
        metadata["shown_winner_id"] = shown_winner_id

    return PairwiseComparison(
        left_id=scheduled_pair.left_id,
        right_id=scheduled_pair.right_id,
        winner=cast(Winner, canonical_winner),
        soft_probability=shown_comparison.soft_probability,
        confidence=shown_comparison.confidence,
        reasons=list(shown_comparison.reasons),
        order=scheduled_pair.order,
        metadata=metadata,
    )


@dataclass(frozen=True, slots=True)
class DeterministicPointwiseJudge:
    confidence: float = 0.75

    def compare(self, left: Paper, right: Paper) -> PairwiseComparison:
        left_q = left.pointwise.good_probability
        right_q = right.pointwise.good_probability
        if abs(left_q - right_q) < 0.03:
            winner = "tie"
            soft = 0.5
        elif left_q > right_q:
            winner = "left"
            soft = min(0.95, 0.5 + abs(left_q - right_q))
        else:
            winner = "right"
            soft = min(0.95, 0.5 + abs(left_q - right_q))
        return PairwiseComparison(
            left_id=left.paper_id,
            right_id=right.paper_id,
            winner=winner,  # type: ignore[arg-type]
            soft_probability=soft,
            confidence=self.confidence,
            reasons=["deterministic pointwise mock judge"],
            order=PairwiseOrderMetadata(
                shown_first_id=left.paper_id,
                shown_second_id=right.paper_id,
                randomized=False,
            ),
        )


@dataclass(frozen=True, slots=True)
class OpenAICompatiblePairwiseJudge:
    api_key: str
    base_url: str
    model: str = "gpt-4.1-mini"
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "OpenAICompatiblePairwiseJudge":
        api_key = os.environ.get("SESTINA_LLM_API_KEY") or os.environ.get(
            "RECOLETA_LLM_API_KEY"
        )
        base_url = os.environ.get("SESTINA_LLM_BASE_URL") or os.environ.get(
            "RECOLETA_LLM_BASE_URL"
        )
        if not api_key or not base_url:
            raise RuntimeError(
                "SESTINA_LLM_API_KEY and SESTINA_LLM_BASE_URL are required"
            )
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=(
                os.environ.get("SESTINA_LLM_MODEL")
                or os.environ.get("RECOLETA_LLM_MODEL")
                or "gpt-4.1-mini"
            ),
        )

    def compare(self, left: Paper, right: Paper) -> PairwiseComparison:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Judge which paper is more likely to be a good paper for the "
                        "target discovery set. Return strict JSON with winner, "
                        "soft_probability, confidence, and reasons. Use winner left, "
                        "right, tie, or uncertain."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "left": _paper_prompt_payload(left),
                            "right": _paper_prompt_payload(right),
                        },
                        sort_keys=True,
                    ),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"pairwise judge request failed: {exc}") from exc
        content = response_payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return PairwiseComparison.from_dict(
            {
                "left_id": left.paper_id,
                "right_id": right.paper_id,
                "winner": parsed.get("winner", "uncertain"),
                "soft_probability": parsed.get("soft_probability"),
                "confidence": parsed.get("confidence", 0.5),
                "reasons": parsed.get("reasons", []),
                "order": {
                    "shown_first_id": left.paper_id,
                    "shown_second_id": right.paper_id,
                    "randomized": False,
                },
            }
        )


def _paper_prompt_payload(paper: Paper) -> dict[str, object]:
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "pointwise_summary": paper.pointwise.summary,
        "pointwise_reasons": paper.pointwise.reasons[:4],
        "rubric_scores": paper.pointwise.rubric_scores,
    }


def _paper_mapping(papers: PaperCollection) -> Mapping[str, Paper]:
    if isinstance(papers, Mapping):
        return papers
    return {paper.paper_id: paper for paper in papers}


def _scheduled_order_ids(scheduled_pair: ScheduledPair) -> tuple[str, str]:
    shown_first_id = scheduled_pair.order.shown_first_id
    shown_second_id = scheduled_pair.order.shown_second_id
    if shown_first_id is None or shown_second_id is None:
        raise ValueError(
            "scheduled pair order must include shown_first_id and shown_second_id"
        )
    if shown_first_id == shown_second_id:
        raise ValueError("scheduled pair order must show two distinct papers")
    canonical_ids = {scheduled_pair.left_id, scheduled_pair.right_id}
    if {shown_first_id, shown_second_id} != canonical_ids:
        raise ValueError("scheduled pair order must contain canonical left/right ids")
    return shown_first_id, shown_second_id


def _position_winner(
    winner: object,
    *,
    shown_first_id: str,
    shown_second_id: str,
) -> Winner:
    raw = str(winner).strip().lower()
    normalized = raw.replace("-", "_").replace(" ", "_")
    if raw == shown_first_id.lower():
        return "left"
    if raw == shown_second_id.lower():
        return "right"
    if normalized in {"left", "a", "paper_a", "shown_a", "first", "shown_first"}:
        return "left"
    if normalized in {"right", "b", "paper_b", "shown_b", "second", "shown_second"}:
        return "right"
    if normalized == "tie":
        return "tie"
    if normalized == "uncertain":
        return "uncertain"
    raise ValueError("scheduled pair judge winner must be A, B, tie, or uncertain")
