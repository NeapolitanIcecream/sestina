from __future__ import annotations

from collections.abc import Callable

import pytest

from sestina.models import Paper, PointwiseAssessment


def make_paper(
    paper_id: str,
    q: float,
    uncertainty: float = 0.3,
    *,
    topic: str = "general",
) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        abstract=f"Abstract for {paper_id}",
        pointwise=PointwiseAssessment(
            good_probability=q,
            uncertainty=uncertainty,
            rubric_scores={"novelty": q * 5},
            summary=f"{paper_id} summary",
            reasons=[f"{paper_id} reason"],
        ),
        metadata={"topic": topic, "source": "fixture"},
    )


@pytest.fixture
def paper_set() -> Callable[[int], list[Paper]]:
    return _paper_set


def _paper_set(total: int = 12) -> list[Paper]:
    papers: list[Paper] = []
    for index in range(total):
        papers.append(
            make_paper(
                f"p{index + 1}",
                q=max(0.05, 0.95 - (index * 0.06)),
                uncertainty=0.15 + ((index % 4) * 0.18),
                topic=f"bucket-{index % 3}",
            )
        )
    return papers
