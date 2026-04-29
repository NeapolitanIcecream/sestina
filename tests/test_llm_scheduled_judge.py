from __future__ import annotations

from dataclasses import dataclass

from sestina.llm import OpenAICompatiblePairwiseJudge, ScheduledPairwiseJudgeAdapter
from sestina.models import PairwiseComparison, PairwiseOrderMetadata, ScheduledPair


@dataclass(slots=True)
class CapturingJudge:
    winner: str = "A"
    calls: list[tuple[str, str]] | None = None

    def compare(self, left, right) -> PairwiseComparison:
        if self.calls is None:
            self.calls = []
        self.calls.append((left.paper_id, right.paper_id))
        return PairwiseComparison(
            left_id=left.paper_id,
            right_id=right.paper_id,
            winner=self.winner,  # type: ignore[arg-type]
            soft_probability=0.82,
            confidence=0.7,
            reasons=["selected the shown position"],
            metadata={"judge": "fixture"},
        )


def test_scheduled_pair_adapter_presents_randomized_order_and_maps_a_to_canonical(
    paper_set,
) -> None:
    """Regression: scheduled A/B order must not be lost before judging."""
    papers = {paper.paper_id: paper for paper in paper_set(3)}
    scheduled = ScheduledPair(
        left_id="p1",
        right_id="p2",
        priority=0.9,
        purpose="boundary",
        order=PairwiseOrderMetadata(
            shown_first_id="p2",
            shown_second_id="p1",
            randomized=True,
            seed=123,
            position_bias_audit=True,
            extra={"audit_batch": "batch-1"},
        ),
        diagnostics={"boundary": 1.0},
    )
    judge = CapturingJudge(winner="A")

    comparison = ScheduledPairwiseJudgeAdapter(judge).compare_scheduled(
        scheduled,
        papers,
    )

    assert judge.calls == [("p2", "p1")]
    assert comparison.left_id == "p1"
    assert comparison.right_id == "p2"
    assert comparison.winner == "right"
    assert comparison.soft_probability == 0.82
    assert comparison.confidence == 0.7
    assert comparison.reasons == ["selected the shown position"]
    assert comparison.order.shown_first_id == "p2"
    assert comparison.order.shown_second_id == "p1"
    assert comparison.order.randomized is True
    assert comparison.order.position_bias_audit is True
    assert comparison.order.extra["audit_batch"] == "batch-1"
    assert comparison.metadata["shown_winner_id"] == "p2"
    assert comparison.metadata["scheduled_pair_purpose"] == "boundary"


def test_scheduled_pair_adapter_maps_b_to_canonical_left(paper_set) -> None:
    papers = {paper.paper_id: paper for paper in paper_set(3)}
    scheduled = ScheduledPair(
        left_id="p1",
        right_id="p2",
        priority=0.9,
        purpose="boundary",
        order=PairwiseOrderMetadata(
            shown_first_id="p2",
            shown_second_id="p1",
            randomized=True,
            seed=123,
        ),
    )

    comparison = ScheduledPairwiseJudgeAdapter(
        CapturingJudge(winner="B")
    ).compare_scheduled(scheduled, papers)

    assert comparison.left_id == "p1"
    assert comparison.right_id == "p2"
    assert comparison.winner == "left"
    assert comparison.metadata["shown_winner_id"] == "p1"


def test_openai_compatible_judge_prefers_sestina_env(monkeypatch) -> None:
    monkeypatch.setenv("SESTINA_LLM_API_KEY", "sestina-key")
    monkeypatch.setenv("SESTINA_LLM_BASE_URL", "https://sestina.example/v1")
    monkeypatch.setenv("SESTINA_LLM_MODEL", "sestina-model")
    monkeypatch.setenv("RECOLETA_LLM_API_KEY", "legacy-key")
    monkeypatch.setenv("RECOLETA_LLM_BASE_URL", "https://legacy.example/v1")
    monkeypatch.setenv("RECOLETA_LLM_MODEL", "legacy-model")

    judge = OpenAICompatiblePairwiseJudge.from_env()

    assert judge.api_key == "sestina-key"
    assert judge.base_url == "https://sestina.example/v1"
    assert judge.model == "sestina-model"


def test_openai_compatible_judge_keeps_legacy_env_fallback(monkeypatch) -> None:
    monkeypatch.delenv("SESTINA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("SESTINA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("SESTINA_LLM_MODEL", raising=False)
    monkeypatch.setenv("RECOLETA_LLM_API_KEY", "legacy-key")
    monkeypatch.setenv("RECOLETA_LLM_BASE_URL", "https://legacy.example/v1")
    monkeypatch.setenv("RECOLETA_LLM_MODEL", "legacy-model")

    judge = OpenAICompatiblePairwiseJudge.from_env()

    assert judge.api_key == "legacy-key"
    assert judge.base_url == "https://legacy.example/v1"
    assert judge.model == "legacy-model"
