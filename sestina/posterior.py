from __future__ import annotations

import random
from dataclasses import dataclass, field

from sestina.aggregation import AggregationResult
from sestina.diagnostics import DiagnosticRecorder


@dataclass(frozen=True, slots=True)
class TopKPosterior:
    top_k_probabilities: dict[str, float]
    mean_sampled_rank: dict[str, float]
    samples: int
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "top_k_probabilities": dict(self.top_k_probabilities),
            "mean_sampled_rank": dict(self.mean_sampled_rank),
            "samples": self.samples,
            "diagnostics": dict(self.diagnostics),
        }


def estimate_top_k_probabilities(
    aggregation: AggregationResult,
    *,
    k: int,
    samples: int = 2000,
    seed: int = 0,
    diagnostics: DiagnosticRecorder | None = None,
) -> TopKPosterior:
    recorder = diagnostics or DiagnosticRecorder()
    estimates = list(aggregation.estimates.values())
    if not estimates or k <= 0:
        posterior = TopKPosterior(
            top_k_probabilities={estimate.paper_id: 0.0 for estimate in estimates},
            mean_sampled_rank={estimate.paper_id: 0.0 for estimate in estimates},
            samples=0,
            diagnostics={"samples": 0, "k": k},
        )
        recorder.record(
            step="uncertainty",
            code="posterior_empty",
            message="no posterior samples needed for empty estimate set",
            data=posterior.diagnostics,
        )
        return posterior

    sample_count = max(100, int(samples))
    top_counts = {estimate.paper_id: 0 for estimate in estimates}
    rank_totals = {estimate.paper_id: 0.0 for estimate in estimates}
    rng = random.Random(seed)
    for _ in range(sample_count):
        draw = []
        for estimate in estimates:
            stddev = estimate.variance**0.5
            draw.append((rng.gauss(estimate.posterior_logit, stddev), estimate.paper_id))
        draw.sort(reverse=True)
        for rank, (_, paper_id) in enumerate(draw, start=1):
            rank_totals[paper_id] += rank
            if rank <= k:
                top_counts[paper_id] += 1

    probabilities = {
        paper_id: round(count / sample_count, 8) for paper_id, count in top_counts.items()
    }
    mean_rank = {
        paper_id: round(total / sample_count, 4)
        for paper_id, total in rank_totals.items()
    }
    payload = {
        "samples": sample_count,
        "k": k,
        "method": "independent_laplace_normal_sampling",
        "average_top_k_probability": round(sum(probabilities.values()) / len(probabilities), 8),
    }
    recorder.record(
        step="uncertainty",
        code="posterior_sampling_completed",
        message="estimated approximate top-K probabilities",
        data=payload,
    )
    return TopKPosterior(
        top_k_probabilities=probabilities,
        mean_sampled_rank=mean_rank,
        samples=sample_count,
        diagnostics=payload,
    )

