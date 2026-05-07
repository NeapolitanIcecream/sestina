from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from sestina.backtest import Prediction
from sestina.models import PairwiseComparison, PairwiseOrderMetadata, Paper, ScheduledPair
from sestina.scheduler import PairwiseBudget

METRICS = ("recall_at_k", "ndcg_at_k", "average_precision")


@dataclass(frozen=True, slots=True)
class BordaLCBConfig:
    prior_strength: float = 2.0
    confidence_z: float = 1.64
    pointwise_weight: float = 0.40
    lower_bound_weight: float = 0.55
    coverage_weight: float = 0.05


@dataclass(frozen=True, slots=True)
class HybridScheduleConfig:
    name: str
    random_floor_fraction: float = 0.30
    min_random_floor_pairs: int = 1
    per_item_cap: int | None = 6
    anchor_multiplier: int = 3
    challenger_multiplier: int = 4


def paper_borda_lcb_predictions(
    papers: Sequence[Paper],
    comparisons: Sequence[PairwiseComparison],
    *,
    config: BordaLCBConfig | None = None,
) -> tuple[list[Prediction], dict[str, Any]]:
    """Rank papers by a prior-smoothed paper-level win-rate lower bound.

    The rule consumes only pointwise priors and pairwise labels that were already
    revealed by the supplied schedule. It does not inspect evaluation labels.
    """
    cfg = config or BordaLCBConfig()
    paper_by_id = {paper.paper_id: paper for paper in papers}
    wins = Counter({paper.paper_id: 0.0 for paper in papers})
    trials = Counter({paper.paper_id: 0.0 for paper in papers})
    used = Counter({paper.paper_id: 0 for paper in papers})
    skipped_unknown = 0

    for comparison in comparisons:
        if (
            comparison.left_id not in paper_by_id
            or comparison.right_id not in paper_by_id
            or comparison.left_id == comparison.right_id
        ):
            skipped_unknown += 1
            continue
        left_score, right_score, weight = _comparison_fractional_scores(comparison)
        if weight <= 0.0:
            continue
        wins[comparison.left_id] += left_score * weight
        wins[comparison.right_id] += right_score * weight
        trials[comparison.left_id] += weight
        trials[comparison.right_id] += weight
        used[comparison.left_id] += 1
        used[comparison.right_id] += 1

    rows = []
    predictions = []
    for paper in papers:
        prior = float(paper.pointwise.good_probability)
        effective_trials = float(trials[paper.paper_id])
        posterior_trials = cfg.prior_strength + effective_trials
        posterior_wins = (cfg.prior_strength * prior) + float(wins[paper.paper_id])
        posterior_rate = (
            posterior_wins / posterior_trials if posterior_trials > 0.0 else prior
        )
        lower_bound = _normal_lower_bound(
            posterior_rate,
            posterior_trials,
            z=cfg.confidence_z,
        )
        coverage = (
            effective_trials / (effective_trials + cfg.prior_strength)
            if effective_trials + cfg.prior_strength > 0.0
            else 0.0
        )
        score = (
            (cfg.pointwise_weight * prior)
            + (cfg.lower_bound_weight * lower_bound)
            + (cfg.coverage_weight * coverage)
        )
        row = {
            "paper_id": paper.paper_id,
            "pointwise_good_probability": round(prior, 8),
            "comparison_count": int(used[paper.paper_id]),
            "effective_pairwise_trials": round(effective_trials, 8),
            "fractional_borda_wins": round(float(wins[paper.paper_id]), 8),
            "posterior_win_rate": round(posterior_rate, 8),
            "lower_confidence_bound": round(lower_bound, 8),
            "coverage_weight": round(coverage, 8),
            "score": round(score, 8),
        }
        rows.append(row)
        predictions.append(Prediction(paper.paper_id, round(score, 8)))

    row_by_id = {row["paper_id"]: row for row in rows}
    predictions.sort(
        key=lambda item: (
            item.score,
            row_by_id[item.paper_id]["lower_confidence_bound"],
            row_by_id[item.paper_id]["posterior_win_rate"],
            row_by_id[item.paper_id]["pointwise_good_probability"],
            item.paper_id,
        ),
        reverse=True,
    )
    ranked_rows = []
    for rank, prediction in enumerate(predictions, start=1):
        row = dict(row_by_id[prediction.paper_id])
        row["rank"] = rank
        ranked_rows.append(row)

    diagnostics = {
        "method": "paper_level_borda_win_rate_lower_confidence_bound",
        "rule_parameters": {
            "prior_strength": cfg.prior_strength,
            "confidence_z": cfg.confidence_z,
            "pointwise_weight": cfg.pointwise_weight,
            "lower_bound_weight": cfg.lower_bound_weight,
            "coverage_weight": cfg.coverage_weight,
            "uses_future_labels_for_decision": False,
        },
        "comparison_count": len(comparisons),
        "skipped_unknown_comparison_count": skipped_unknown,
        "papers_with_pairwise_evidence": sum(1 for value in used.values() if value > 0),
        "mean_effective_pairwise_trials": _mean_float(
            [float(trials[paper.paper_id]) for paper in papers]
        ),
        "ranked_rows": ranked_rows,
    }
    return predictions, diagnostics


def schedule_model_visible_hybrid_pairs(
    papers: Sequence[Paper],
    *,
    k: int,
    budget: PairwiseBudget,
    seed: int,
    available_pair_keys: set[tuple[str, str]],
    config: HybridScheduleConfig,
) -> tuple[list[ScheduledPair], dict[str, Any]]:
    """Schedule a cache-safe hybrid using pointwise, metadata, and cache availability.

    Cache availability may limit the offline replay pool, but cached label values
    are not consumed by this function.
    """
    paper_by_id = {paper.paper_id: paper for paper in papers}
    available = sorted(
        canonical_pair_key(left_id, right_id)
        for left_id, right_id in available_pair_keys
        if left_id in paper_by_id and right_id in paper_by_id and left_id != right_id
    )
    target = min(max(0, budget.budget), len(available))
    if target == 0 or k <= 0:
        return [], _empty_hybrid_diagnostics(
            config=config,
            budget=budget.budget,
            available_pair_count=len(available),
        )

    rng = random.Random(seed)
    floor_target = min(
        target,
        max(
            config.min_random_floor_pairs,
            math.ceil(target * max(0.0, config.random_floor_fraction)),
        ),
    )
    selected: list[tuple[str, str, str, float, dict[str, Any]]] = []
    selected_keys: set[tuple[str, str]] = set()
    item_counts: Counter[str] = Counter()

    shuffled_available = list(available)
    rng.shuffle(shuffled_available)
    _take_key_candidates(
        [
            (left_id, right_id, 0.0, {"selection_source": "random_floor"})
            for left_id, right_id in shuffled_available
        ],
        selected=selected,
        selected_keys=selected_keys,
        item_counts=item_counts,
        limit=floor_target,
        purpose=f"{config.name}_random_floor",
        cap=config.per_item_cap,
    )

    proposals = _model_visible_proposals(
        list(papers),
        k=k,
        available_pair_keys=set(available),
        policy_name=config.name,
        anchor_multiplier=config.anchor_multiplier,
        challenger_multiplier=config.challenger_multiplier,
    )
    _take_key_candidates(
        proposals,
        selected=selected,
        selected_keys=selected_keys,
        item_counts=item_counts,
        limit=target,
        purpose=f"{config.name}_active_probe",
        cap=config.per_item_cap,
    )
    _take_key_candidates(
        [
            (left_id, right_id, 0.0, {"selection_source": "uncapped_cached_fill"})
            for left_id, right_id in shuffled_available
        ],
        selected=selected,
        selected_keys=selected_keys,
        item_counts=item_counts,
        limit=target,
        purpose=f"{config.name}_cached_fill",
        cap=None,
    )

    scheduled = [
        _scheduled_pair(
            left_id,
            right_id,
            purpose=purpose,
            priority=priority,
            seed=seed,
            index=index,
            diagnostics=diagnostics,
            rng=rng,
        )
        for index, (left_id, right_id, purpose, priority, diagnostics) in enumerate(
            selected[:target],
            start=1,
        )
    ]
    purpose_counts = Counter(pair.purpose for pair in scheduled)
    touched = {
        paper_id
        for pair in scheduled
        for paper_id in (pair.left_id, pair.right_id)
    }
    diagnostics = {
        "method": config.name,
        "budget": budget.to_dict(),
        "scheduled_total": len(scheduled),
        "scheduled_pairwise_shortfall": max(0, budget.budget - len(scheduled)),
        "budget_complete": len(scheduled) >= budget.budget,
        "available_pair_filter": {
            "available_pair_keys_total": len(available),
            "scheduled_available_pair_keys_total": len(scheduled),
            "cache_availability_used_for_scheduling": True,
            "cached_label_values_used_before_scheduling": False,
        },
        "acquisition": {
            "method": config.name,
            "source_method": "model_visible_cache_safe_hybrid",
            "random_seed": seed,
            "random_floor_fraction": config.random_floor_fraction,
            "min_random_floor_pairs": config.min_random_floor_pairs,
            "random_floor_target": floor_target,
            "per_item_cap": config.per_item_cap,
            "model_visible_signals": [
                "pointwise_good_probability",
                "pointwise_uncertainty",
                "pointwise_rubric_scores",
                "title_abstract_text_length",
                "metadata_category",
                "cached_pair_availability",
            ],
            "future_labels_used_for_scheduling": False,
            "cached_label_values_used_before_scheduling": False,
        },
        "purpose_counts": dict(sorted(purpose_counts.items())),
        "coverage": {
            "random_floor_pairs": sum(
                1 for pair in scheduled if pair.purpose.endswith("_random_floor")
            ),
            "random_floor_rate": round(
                (
                    sum(
                        1
                        for pair in scheduled
                        if pair.purpose.endswith("_random_floor")
                    )
                    / len(scheduled)
                )
                if scheduled
                else 0.0,
                8,
            ),
            "scheduled_unique_papers": len(touched),
            "scheduled_unique_paper_rate": round(
                len(touched) / len(paper_by_id) if paper_by_id else 0.0,
                8,
            ),
        },
        "proposal_counts": {
            "model_visible_proposals": len(proposals),
            "available_pair_keys": len(available),
        },
        "label_policy": {
            "offline_cached_pairwise_labels_only": True,
            "future_labels_used_for_scheduling": False,
            "future_labels_used_as_model_features": False,
            "cached_label_values_used_before_scheduling": False,
            "cache_availability_used_for_scheduling": True,
        },
    }
    return scheduled, diagnostics


def summarize_values(values: Sequence[int | float], *, z: float = 1.96) -> dict[str, Any]:
    items = [float(value) for value in values]
    if not items:
        return {
            "count": 0,
            "mean": 0.0,
            "stddev": 0.0,
            "standard_error": 0.0,
            "min": 0.0,
            "max": 0.0,
            "normal_approx_95_ci": [None, None],
        }
    value_mean = mean(items)
    value_stddev = stdev(items) if len(items) > 1 else 0.0
    standard_error = value_stddev / math.sqrt(len(items)) if len(items) > 1 else 0.0
    return {
        "count": len(items),
        "mean": round(value_mean, 8),
        "stddev": round(value_stddev, 8),
        "standard_error": round(standard_error, 8),
        "min": round(min(items), 8),
        "max": round(max(items), 8),
        "normal_approx_95_ci": [
            round(value_mean - (z * standard_error), 8),
            round(value_mean + (z * standard_error), 8),
        ],
    }


def paired_seed_metric_deltas(
    seed_metric_rows: Mapping[str, Mapping[str, Mapping[str, float | int]]],
    *,
    comparison_arm: str,
    reference_arm: str,
    metrics: Sequence[str] = METRICS,
) -> dict[str, Any]:
    seed_deltas: dict[str, dict[str, float]] = {}
    for seed, rows_by_arm in sorted(seed_metric_rows.items(), key=lambda item: item[0]):
        comparison = rows_by_arm.get(comparison_arm, {})
        reference = rows_by_arm.get(reference_arm, {})
        seed_deltas[str(seed)] = {
            metric: round(
                float(comparison.get(metric, 0.0))
                - float(reference.get(metric, 0.0)),
                8,
            )
            for metric in metrics
        }
    return {
        "reference_arm": reference_arm,
        "comparison_arm": comparison_arm,
        "metric_deltas": {
            metric: summarize_values(
                [row[metric] for row in seed_deltas.values()]
            )
            for metric in metrics
        },
        "seed_deltas": seed_deltas,
    }


def choose_best_candidate(
    candidate_gate_summaries: Mapping[str, Mapping[str, Any]],
) -> str:
    if not candidate_gate_summaries:
        raise ValueError("no candidate gate summaries available")

    def sort_key(item: tuple[str, Mapping[str, Any]]) -> tuple[float, ...]:
        _, row = item
        recall = float(row.get("mean_recall_delta", 0.0) or 0.0)
        ndcg = float(row.get("mean_ndcg_delta", 0.0) or 0.0)
        ap = float(row.get("mean_average_precision_delta", 0.0) or 0.0)
        recall_ci = row.get("recall_delta_ci") or [None, None]
        lower = recall_ci[0] if isinstance(recall_ci, list) and recall_ci else None
        lower_value = float(lower) if isinstance(lower, int | float) else -1.0
        return (
            1.0 if row.get("paid_followup_allowed") is True else 0.0,
            recall,
            lower_value,
            ndcg,
            ap,
        )

    return max(candidate_gate_summaries.items(), key=sort_key)[0]


def canonical_pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    return tuple(sorted((str(left_id), str(right_id))))


def _comparison_fractional_scores(
    comparison: PairwiseComparison,
) -> tuple[float, float, float]:
    confidence = max(0.0, min(1.0, float(comparison.confidence)))
    if comparison.winner == "tie":
        return 0.5, 0.5, 0.35 * confidence
    if comparison.winner == "uncertain":
        return 0.5, 0.5, 0.15 * confidence
    soft = comparison.soft_probability
    if soft is None:
        soft = 0.75
    soft = max(0.5, min(0.999, float(soft)))
    if comparison.winner == "left":
        return soft, 1.0 - soft, confidence
    return 1.0 - soft, soft, confidence


def _normal_lower_bound(rate: float, trials: float, *, z: float) -> float:
    if trials <= 0.0:
        return 0.0
    variance = max(0.0, rate * (1.0 - rate) / trials)
    return max(0.0, min(1.0, rate - (z * math.sqrt(variance))))


def _model_visible_proposals(
    papers: list[Paper],
    *,
    k: int,
    available_pair_keys: set[tuple[str, str]],
    policy_name: str,
    anchor_multiplier: int,
    challenger_multiplier: int,
) -> list[tuple[str, str, float, dict[str, Any]]]:
    if len(papers) < 2:
        return []
    ranked = sorted(
        papers,
        key=lambda paper: (
            paper.pointwise.good_probability,
            paper.pointwise.uncertainty,
            paper.paper_id,
        ),
        reverse=True,
    )
    rank_by_id = {paper.paper_id: index + 1 for index, paper in enumerate(ranked)}
    boundary_index = min(max(k - 1, 0), len(ranked) - 1)
    boundary_probability = ranked[boundary_index].pointwise.good_probability
    anchor_count = min(len(ranked), max(k, k * max(1, anchor_multiplier)))
    challenger_count = min(
        len(ranked),
        max(k * 2, k * max(1, challenger_multiplier)),
    )
    anchors = ranked[:anchor_count]
    challengers = sorted(
        ranked,
        key=lambda paper: (
            _challenger_score(paper, boundary_probability=boundary_probability),
            paper.paper_id,
        ),
        reverse=True,
    )[:challenger_count]
    boundary = sorted(
        ranked,
        key=lambda paper: (
            abs(paper.pointwise.good_probability - boundary_probability),
            -paper.pointwise.uncertainty,
            paper.paper_id,
        ),
    )[:challenger_count]
    proposals = []
    seen: set[tuple[str, str]] = set()
    for left in anchors:
        for right in [*challengers, *boundary]:
            key = canonical_pair_key(left.paper_id, right.paper_id)
            if (
                left.paper_id == right.paper_id
                or key in seen
                or key not in available_pair_keys
            ):
                continue
            seen.add(key)
            priority = _pair_priority(
                left,
                right,
                boundary_probability=boundary_probability,
                rank_by_id=rank_by_id,
                policy_name=policy_name,
            )
            proposals.append(
                (
                    key[0],
                    key[1],
                    priority,
                    {
                        "selection_source": "model_visible_active_probe",
                        "policy_name": policy_name,
                        "rank_left": rank_by_id[left.paper_id],
                        "rank_right": rank_by_id[right.paper_id],
                        "pointwise_left": round(
                            left.pointwise.good_probability,
                            8,
                        ),
                        "pointwise_right": round(
                            right.pointwise.good_probability,
                            8,
                        ),
                        "metadata_cross_bucket": _metadata_bucket(left)
                        != _metadata_bucket(right),
                    },
                )
            )
    proposals.sort(key=lambda row: (row[2], row[0], row[1]), reverse=True)
    return proposals


def _pair_priority(
    left: Paper,
    right: Paper,
    *,
    boundary_probability: float,
    rank_by_id: Mapping[str, int],
    policy_name: str,
) -> float:
    left_boundary = 1.0 - abs(left.pointwise.good_probability - boundary_probability)
    right_boundary = 1.0 - abs(right.pointwise.good_probability - boundary_probability)
    uncertainty = left.pointwise.uncertainty + right.pointwise.uncertainty
    diversity = 1.0 if _metadata_bucket(left) != _metadata_bucket(right) else 0.0
    rank_gap = abs(rank_by_id[left.paper_id] - rank_by_id[right.paper_id])
    challenger = _challenger_score(left, boundary_probability=boundary_probability)
    challenger += _challenger_score(right, boundary_probability=boundary_probability)
    rank_gap_term = math.log1p(rank_gap) / math.log1p(max(rank_by_id.values()))
    if "borda" in policy_name or "lcb" in policy_name:
        return round(
            (1.5 * (left_boundary + right_boundary))
            + (0.60 * uncertainty)
            + (0.25 * diversity)
            + (0.20 * rank_gap_term),
            8,
        )
    return round(
        (0.85 * challenger)
        + (0.75 * (left_boundary + right_boundary))
        + (0.45 * diversity)
        + (0.25 * rank_gap_term),
        8,
    )


def _challenger_score(
    paper: Paper,
    *,
    boundary_probability: float,
) -> float:
    rubric_values = [
        float(value)
        for value in paper.pointwise.rubric_scores.values()
        if isinstance(value, int | float)
    ]
    rubric_mean = sum(rubric_values) / len(rubric_values) if rubric_values else 0.0
    residual = max(0.0, rubric_mean - paper.pointwise.good_probability)
    boundary = 1.0 - abs(paper.pointwise.good_probability - boundary_probability)
    lexical = min(1.0, math.log1p(len(paper.title) + len(paper.abstract)) / 9.0)
    return (
        (0.40 * paper.pointwise.uncertainty)
        + (0.30 * residual)
        + (0.20 * boundary)
        + (0.10 * lexical)
    )


def _take_key_candidates(
    candidates: Sequence[tuple[str, str, float, dict[str, Any]]],
    *,
    selected: list[tuple[str, str, str, float, dict[str, Any]]],
    selected_keys: set[tuple[str, str]],
    item_counts: Counter[str],
    limit: int,
    purpose: str,
    cap: int | None,
) -> None:
    for left_id, right_id, priority, diagnostics in candidates:
        if len(selected) >= limit:
            return
        key = canonical_pair_key(left_id, right_id)
        if key in selected_keys:
            continue
        if cap is not None and (
            item_counts[left_id] >= cap or item_counts[right_id] >= cap
        ):
            continue
        selected.append((left_id, right_id, purpose, round(priority, 8), diagnostics))
        selected_keys.add(key)
        item_counts[left_id] += 1
        item_counts[right_id] += 1


def _scheduled_pair(
    left_id: str,
    right_id: str,
    *,
    purpose: str,
    priority: float,
    seed: int,
    index: int,
    diagnostics: Mapping[str, Any],
    rng: random.Random,
) -> ScheduledPair:
    if rng.random() < 0.5:
        shown_first_id, shown_second_id = left_id, right_id
    else:
        shown_first_id, shown_second_id = right_id, left_id
    return ScheduledPair(
        left_id=left_id,
        right_id=right_id,
        priority=round(priority, 8),
        purpose=purpose,
        order=PairwiseOrderMetadata(
            shown_first_id=shown_first_id,
            shown_second_id=shown_second_id,
            randomized=True,
            seed=seed,
            position_bias_audit=(index % 5 == 0),
            extra={"canonical_left_id": left_id, "canonical_right_id": right_id},
        ),
        diagnostics=dict(diagnostics),
    )


def _empty_hybrid_diagnostics(
    *,
    config: HybridScheduleConfig,
    budget: int,
    available_pair_count: int,
) -> dict[str, Any]:
    return {
        "method": config.name,
        "budget": {"budget": budget},
        "scheduled_total": 0,
        "scheduled_pairwise_shortfall": budget,
        "budget_complete": budget == 0,
        "available_pair_filter": {
            "available_pair_keys_total": available_pair_count,
            "cache_availability_used_for_scheduling": True,
            "cached_label_values_used_before_scheduling": False,
        },
        "purpose_counts": {},
        "coverage": {
            "random_floor_pairs": 0,
            "random_floor_rate": 0.0,
            "scheduled_unique_papers": 0,
            "scheduled_unique_paper_rate": 0.0,
        },
        "label_policy": {
            "offline_cached_pairwise_labels_only": True,
            "future_labels_used_for_scheduling": False,
            "future_labels_used_as_model_features": False,
            "cached_label_values_used_before_scheduling": False,
            "cache_availability_used_for_scheduling": True,
        },
    }


def _metadata_bucket(paper: Paper) -> str:
    for key in ("primary_category", "category", "topic", "venue", "field"):
        value = paper.metadata.get(key)
        if value:
            return f"{key}:{value}"
    categories = paper.metadata.get("categories")
    if isinstance(categories, (list, tuple)) and categories:
        return f"category:{categories[0]}"
    return "unknown"


def _mean_float(values: Sequence[int | float]) -> float:
    items = [float(value) for value in values]
    return round(sum(items) / len(items), 8) if items else 0.0
