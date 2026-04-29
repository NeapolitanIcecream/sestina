from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Prediction:
    paper_id: str
    score: float


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    k: int
    recall_at_k: float
    precision_at_k: float
    ndcg_at_k: float
    average_precision: float
    brier_score: float
    near_miss_positive_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "k": self.k,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "average_precision": self.average_precision,
            "brier_score": self.brier_score,
            "near_miss_positive_rate": self.near_miss_positive_rate,
        }


def evaluate_predictions(
    predictions: list[Prediction],
    *,
    relevant_ids: set[str],
    k: int,
) -> BacktestMetrics:
    ranked = sorted(predictions, key=lambda item: item.score, reverse=True)
    top = ranked[:k]
    hits = sum(1 for item in top if item.paper_id in relevant_ids)
    relevant_total = len(relevant_ids)
    recall = hits / relevant_total if relevant_total else 0.0
    precision = hits / k if k else 0.0
    ndcg = _ndcg_at_k(ranked, relevant_ids=relevant_ids, k=k)
    average_precision = _average_precision(ranked, relevant_ids=relevant_ids)
    brier = _brier_score(predictions, relevant_ids=relevant_ids)
    near_miss = ranked[k : k + max(1, int(math.sqrt(max(1, len(ranked)))))]
    near_miss_rate = (
        sum(1 for item in near_miss if item.paper_id in relevant_ids) / len(near_miss)
        if near_miss
        else 0.0
    )
    return BacktestMetrics(
        k=k,
        recall_at_k=round(recall, 8),
        precision_at_k=round(precision, 8),
        ndcg_at_k=round(ndcg, 8),
        average_precision=round(average_precision, 8),
        brier_score=round(brier, 8),
        near_miss_positive_rate=round(near_miss_rate, 8),
    )


def compare_strategies(
    strategy_predictions: dict[str, list[Prediction]],
    *,
    relevant_ids: set[str],
    k: int,
) -> dict[str, BacktestMetrics]:
    return {
        strategy_name: evaluate_predictions(
            predictions,
            relevant_ids=relevant_ids,
            k=k,
        )
        for strategy_name, predictions in sorted(strategy_predictions.items())
    }


def budget_ablation_points(n: int, k: int) -> list[int]:
    base = max(1, k)
    candidates = {0, base, math.ceil(base + math.sqrt(max(0, n))), math.ceil(0.25 * n)}
    return sorted(value for value in candidates if value <= max(0, n))


def _ndcg_at_k(
    ranked: list[Prediction],
    *,
    relevant_ids: set[str],
    k: int,
) -> float:
    dcg = 0.0
    for index, item in enumerate(ranked[:k], start=1):
        if item.paper_id in relevant_ids:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(k, len(relevant_ids))
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def _average_precision(
    ranked: list[Prediction],
    *,
    relevant_ids: set[str],
) -> float:
    if not relevant_ids:
        return 0.0
    hit_total = 0
    precision_total = 0.0
    for index, item in enumerate(ranked, start=1):
        if item.paper_id in relevant_ids:
            hit_total += 1
            precision_total += hit_total / index
    return precision_total / len(relevant_ids)


def _brier_score(
    predictions: list[Prediction],
    *,
    relevant_ids: set[str],
) -> float:
    if not predictions:
        return 0.0
    total = 0.0
    for item in predictions:
        label = 1.0 if item.paper_id in relevant_ids else 0.0
        score = max(0.0, min(1.0, item.score))
        total += (score - label) ** 2
    return total / len(predictions)
