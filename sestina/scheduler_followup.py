from __future__ import annotations

import itertools
import json
import math
import os
import random
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sestina.aggregation import AggregationConfig, aggregate
from sestina.backtest import Prediction, compare_strategies
from sestina.backtest_budget import load_config
from sestina.backtest_runner import (
    BacktestBucket,
    BacktestRunnerError,
    CallEstimate,
    JsonlLedger,
    PaidRunSafetyError,
    _call_artifact,
    _call_estimate,
    _chat_json,
    _comparison_from_pairwise_response,
    _config_for_phase,
    _ledger_call_count,
    _ledger_entry,
    _ledger_stats,
    _load_ok_call_artifact,
    _normalize_rates_from_config,
    _normalize_token_assumptions_from_config,
    _pairwise_payload,
    _random_pair_schedule,
    _safe_name,
    check_model_availability,
    load_dataset_manifest,
    validate_model_names,
)
from sestina.candidates import CandidateSelection, select_candidates
from sestina.diagnostics import DiagnosticRecorder, fingerprint, write_json_artifact
from sestina.evsi_scheduler import (
    EVSISchedulerConfig,
    SequentialEVSISchedulerConfig,
    posterior_top_k_predictions,
    schedule_cache_aware_sequential_evsi,
    schedule_evsi_boundary_duels,
    schedule_exact_pool_random,
)
from sestina.models import (
    PairwiseComparison,
    PairwiseOrderMetadata,
    Paper,
    PointwiseAssessment,
    ScheduledPair,
)
from sestina.scheduler import PairwiseBudget, resolve_pairwise_budget, schedule_pairs

DEFAULT_STRENGTH_SWEEP = (0.0, 1.0, 2.5, 5.0)


class SchedulerFollowupError(BacktestRunnerError):
    """Base class for scheduler-only follow-up failures."""


class PointwiseArtifactError(SchedulerFollowupError):
    """Raised before paid calls when required pointwise artifacts are missing."""


@dataclass(frozen=True, slots=True)
class ReusablePairwiseArtifact:
    bucket: str
    pair_key: tuple[str, str]
    comparison: PairwiseComparison
    artifact_path: Path
    kind: str


@dataclass(frozen=True, slots=True)
class SchedulerOnlyBucketPlan:
    bucket: BacktestBucket
    papers: list[Paper]
    schedule: list[ScheduledPair]
    diagnostics: dict[str, Any]
    reusable_pairwise: dict[tuple[str, str], ReusablePairwiseArtifact]
    reusable_stats: dict[str, Any]
    reveal_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def novel_pairs(self) -> list[ScheduledPair]:
        return [
            pair
            for pair in self.schedule
            if canonical_pair_key(pair.left_id, pair.right_id)
            not in self.reusable_pairwise
        ]


@dataclass(slots=True)
class SchedulerOnlyRunner:
    config_path: Path
    manifest_path: Path
    source_artifact_dir: Path
    artifact_dir: Path
    ledger_path: Path
    phase: str = "pilot"
    max_usd: float = 0.50
    confirm_paid: bool = False
    seed: int = 17
    scheduler_kind: str = "quota"
    aggregation_mode: str = "score"
    timeout_seconds: float = 60.0
    urlopen: Any = urllib.request.urlopen

    def run(self) -> dict[str, Any]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        urlopen = self.urlopen

        raw_config = load_config(self.config_path)
        phase_config = _single_phase_config(raw_config, phase=self.phase)
        pairwise_model = str(phase_config["pairwise_model"])
        token_assumptions = _normalize_token_assumptions_from_config(raw_config)
        rates = _normalize_rates_from_config(raw_config)
        estimate = _call_estimate(
            "pairwise",
            pairwise_model,
            token_assumptions,
            rates,
        )
        validate_model_names([pairwise_model])

        manifest = load_dataset_manifest(self.manifest_path)
        selected_buckets = manifest.buckets_for_phase(self.phase)
        if not selected_buckets:
            raise SchedulerFollowupError(
                f"manifest has no buckets for phase {self.phase!r}"
            )

        plans = [
            build_scheduler_only_bucket_plan(
                bucket,
                source_artifact_dir=self.source_artifact_dir,
                phase=self.phase,
                seed=self.seed,
                scheduler_kind=self.scheduler_kind,
            )
            for bucket in selected_buckets
        ]
        totals = _estimate_from_plans(plans, call_estimate=estimate)
        estimate_payload = _scheduler_only_estimate_payload(
            phase=self.phase,
            max_usd=self.max_usd,
            pairwise_model=pairwise_model,
            call_estimate=estimate,
            plans=plans,
            totals=totals,
        )
        estimate_path = self.artifact_dir / f"estimate-{self.phase}.json"
        write_json_artifact(estimate_path, estimate_payload)

        ledger = JsonlLedger(self.ledger_path)
        summary: dict[str, Any] = {
            "artifact_type": "sestina-scheduler-only-followup-summary",
            "phase": self.phase,
            "dry_run": not self.confirm_paid,
            "manifest_path": str(self.manifest_path),
            "source_artifact_dir": str(self.source_artifact_dir),
            "artifact_dir": str(self.artifact_dir),
            "ledger_path": str(self.ledger_path),
            "estimate_path": str(estimate_path),
            "budget_cap_usd": self.max_usd,
            "pairwise_model": pairwise_model,
            "scheduler_kind": self.scheduler_kind,
            "aggregation_mode": self.aggregation_mode,
            "estimated_spend_usd": totals["cost_usd"],
            "paid_call_status": (
                "not_requested_no_paid_calls_made"
                if not self.confirm_paid
                else "requested_after_safety_checks"
            ),
            "model_availability": {"status": "not_checked_dry_run"},
            "bucket_results": [],
            "reuse_policy": (
                "same-bucket canonical unordered pair key from successful "
                "historical pairwise artifacts with reconstructed display order"
            ),
        }
        summary.update(totals)
        summary.update(_ledger_stats(ledger))
        blocker = _scheduler_only_blocker_payload(
            scheduler_kind=self.scheduler_kind,
            plans=plans,
            call_estimate=estimate,
        )
        if blocker:
            summary["blocker"] = blocker

        if not self.confirm_paid:
            offline_payload = _offline_bucket_results_payload(
                plans,
                phase=self.phase,
                aggregation_mode=self.aggregation_mode,
                artifact_path=self.artifact_dir / f"offline-bucket-results-{self.phase}.json",
            )
            offline_path = self.artifact_dir / f"offline-bucket-results-{self.phase}.json"
            write_json_artifact(offline_path, offline_payload)
            summary["offline_bucket_results_path"] = str(offline_path)
            summary["offline_bucket_results"] = offline_payload["bucket_results"]
            summary_path = self.artifact_dir / f"summary-{self.phase}.json"
            write_json_artifact(summary_path, summary)
            return {**summary, "summary_path": str(summary_path)}

        if self.scheduler_kind == "sequential_evsi" and totals["pairwise_novel_total"]:
            raise PaidRunSafetyError(
                "sequential_evsi paid execution is blocked because the dry-run "
                "encountered novel pairs whose labels are needed before later "
                "batches can be selected"
            )
        _validate_scheduler_only_paid_preconditions(
            max_usd=self.max_usd,
            estimate_usd=float(totals["cost_usd"]),
            artifact_dir=self.artifact_dir,
            source_artifact_dir=self.source_artifact_dir,
            ledger_path=self.ledger_path,
        )
        api_key = os.environ.get("SESTINA_LLM_API_KEY") or ""
        base_url = os.environ.get("SESTINA_LLM_BASE_URL") or ""
        model_availability = check_model_availability(
            base_url=base_url,
            api_key=api_key,
            models=[pairwise_model],
            timeout_seconds=self.timeout_seconds,
            urlopen=urlopen,
        )
        summary["model_availability"] = model_availability

        paid_calls_before = _ledger_call_count(ledger)
        for plan in plans:
            result = self._run_bucket_plan(
                plan,
                model=pairwise_model,
                estimate=estimate,
                ledger=ledger,
                api_key=api_key,
                base_url=base_url,
                urlopen=urlopen,
                aggregation_mode=self.aggregation_mode,
            )
            summary["bucket_results"].append(result)

        summary["aggregate_metrics"] = _aggregate_bucket_result_metrics(
            summary["bucket_results"]
        )
        paid_calls_after = _ledger_call_count(ledger)
        summary["paid_calls_made"] = paid_calls_after
        summary["new_ledger_entries_this_invocation"] = (
            paid_calls_after - paid_calls_before
        )
        summary.update(_ledger_stats(ledger))
        summary_path = self.artifact_dir / f"summary-{self.phase}.json"
        write_json_artifact(summary_path, summary)
        return {**summary, "summary_path": str(summary_path)}

    def _run_bucket_plan(
        self,
        plan: SchedulerOnlyBucketPlan,
        *,
        model: str,
        estimate: CallEstimate,
        ledger: JsonlLedger,
        api_key: str,
        base_url: str,
        urlopen: Any,
        aggregation_mode: str,
    ) -> dict[str, Any]:
        bucket_dir = (
            self.artifact_dir
            / self.phase
            / _safe_name(plan.bucket.name)
        )
        calls_dir = bucket_dir / "calls"
        calls_dir.mkdir(parents=True, exist_ok=True)
        paper_by_id = {paper.paper_id: paper for paper in plan.papers}
        comparisons: list[PairwiseComparison] = []
        historical_reused_count = 0
        followup_cached_count = 0
        new_count = 0

        for index, pair in enumerate(plan.schedule, start=1):
            pair_key = canonical_pair_key(pair.left_id, pair.right_id)
            artifact_path = _followup_pairwise_artifact_path(
                calls_dir,
                index=index,
                pair=pair,
            )
            cached = _load_ok_call_artifact(artifact_path)
            if cached is not None:
                comparisons.append(
                    _comparison_from_pairwise_response(pair, cached["response"])
                )
                followup_cached_count += 1
                continue

            reusable = plan.reusable_pairwise.get(pair_key)
            if reusable is not None:
                comparison = orient_comparison(reusable.comparison, pair)
                comparisons.append(comparison)
                historical_reused_count += 1
                write_json_artifact(
                    artifact_path,
                    _reused_pairwise_artifact(
                        phase=self.phase,
                        bucket=plan.bucket.name,
                        model=model,
                        estimate=estimate,
                        pair=pair,
                        source=reusable,
                        comparison=comparison,
                    ),
                )
                continue

            ledger.guard_projected_spend(
                cap_usd=self.max_usd,
                next_cost_usd=estimate.cost_usd,
            )
            try:
                response = _chat_json(
                    base_url=base_url,
                    api_key=api_key,
                    payload=_pairwise_payload(
                        model=model,
                        pair=pair,
                        papers=paper_by_id,
                    ),
                    timeout_seconds=self.timeout_seconds,
                    urlopen=urlopen,
                )
                comparison = _comparison_from_pairwise_response(pair, response)
                status = "ok"
                artifact = _call_artifact(
                    phase=self.phase,
                    bucket=plan.bucket.name,
                    model=model,
                    kind="pairwise_active",
                    estimate=estimate,
                    status=status,
                    response=response,
                    subject={"left_id": pair.left_id, "right_id": pair.right_id},
                )
            except json.JSONDecodeError as exc:
                status = "parse_error"
                artifact = _call_artifact(
                    phase=self.phase,
                    bucket=plan.bucket.name,
                    model=model,
                    kind="pairwise_active",
                    estimate=estimate,
                    status=status,
                    error=exc,
                    subject={"left_id": pair.left_id, "right_id": pair.right_id},
                )
                write_json_artifact(artifact_path, artifact)
                ledger.append(
                    _ledger_entry(
                        phase=self.phase,
                        bucket=plan.bucket.name,
                        model=model,
                        kind="pairwise_active",
                        estimate=estimate,
                        status=status,
                        artifact_path=artifact_path,
                    )
                )
                raise
            except Exception as exc:
                status = "failed"
                artifact = _call_artifact(
                    phase=self.phase,
                    bucket=plan.bucket.name,
                    model=model,
                    kind="pairwise_active",
                    estimate=estimate,
                    status=status,
                    error=exc,
                    subject={"left_id": pair.left_id, "right_id": pair.right_id},
                )
                write_json_artifact(artifact_path, artifact)
                ledger.append(
                    _ledger_entry(
                        phase=self.phase,
                        bucket=plan.bucket.name,
                        model=model,
                        kind="pairwise_active",
                        estimate=estimate,
                        status=status,
                        artifact_path=artifact_path,
                    )
                )
                raise

            write_json_artifact(artifact_path, artifact)
            ledger.append(
                _ledger_entry(
                    phase=self.phase,
                    bucket=plan.bucket.name,
                    model=model,
                    kind="pairwise_active",
                    estimate=estimate,
                    status=status,
                    artifact_path=artifact_path,
                )
            )
            comparisons.append(comparison)
            new_count += 1

        result = _bucket_metrics_payload(
            plan,
            comparisons=comparisons,
            artifact_path=bucket_dir / "bucket-result.json",
            aggregation_mode=aggregation_mode,
        )
        result["calls"] = {
            "pointwise": 0,
            "pairwise_active_total": len(plan.schedule),
            "pairwise_active_reused": plan.reusable_stats[
                "scheduled_reusable_total"
            ],
            "pairwise_active_new": plan.reusable_stats["scheduled_novel_total"],
            "pairwise_active_cached_followup": followup_cached_count,
            "pairwise_active_historical_reused_this_invocation": (
                historical_reused_count
            ),
            "pairwise_active_new_this_invocation": new_count,
        }
        write_json_artifact(bucket_dir / "bucket-result.json", result)
        return {**result, "artifact_path": str(bucket_dir / "bucket-result.json")}


def canonical_pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    return tuple(sorted((left_id, right_id)))


def orient_comparison(
    comparison: PairwiseComparison,
    pair: ScheduledPair,
) -> PairwiseComparison:
    if comparison.left_id == pair.left_id and comparison.right_id == pair.right_id:
        winner = comparison.winner
    elif comparison.left_id == pair.right_id and comparison.right_id == pair.left_id:
        winner = _invert_winner(comparison.winner)
    else:
        raise ValueError("comparison does not reference the scheduled pair")
    return PairwiseComparison(
        left_id=pair.left_id,
        right_id=pair.right_id,
        winner=winner,
        soft_probability=comparison.soft_probability,
        confidence=comparison.confidence,
        reasons=list(comparison.reasons),
        order=pair.order,
        metadata={
            **comparison.metadata,
            "reused_original_left_id": comparison.left_id,
            "reused_original_right_id": comparison.right_id,
            "scheduled_pair_purpose": pair.purpose,
        },
    )


def build_scheduler_only_bucket_plan(
    bucket: BacktestBucket,
    *,
    source_artifact_dir: Path,
    phase: str,
    seed: int,
    scheduler_kind: str = "quota",
) -> SchedulerOnlyBucketPlan:
    papers = load_pointwise_papers_from_artifacts(
        bucket,
        source_artifact_dir=source_artifact_dir,
        phase=phase,
    )
    diagnostics = DiagnosticRecorder()
    selection = select_candidates(
        papers,
        k=bucket.k,
        diagnostics=diagnostics,
    )
    budget = resolve_pairwise_budget(
        n=len(papers),
        candidate_size=len(selection.candidate_ids),
        diagnostics=diagnostics,
    )
    reveal_log: list[dict[str, Any]] = []
    reusable: dict[tuple[str, str], ReusablePairwiseArtifact] = {}
    reusable_stats: dict[str, Any] = {}
    if scheduler_kind == "quota":
        schedule = schedule_pairs(
            papers,
            candidate_selection=selection,
            k=bucket.k,
            budget=budget,
            seed=seed,
            diagnostics=diagnostics,
        )
    elif scheduler_kind == "evsi":
        schedule = schedule_evsi_boundary_duels(
            papers,
            [],
            k=bucket.k,
            budget=budget,
            seed=seed,
            diagnostics=diagnostics,
            config=EVSISchedulerConfig(samples=1200),
        )
    elif scheduler_kind == "exact_pool_random":
        schedule = schedule_exact_pool_random(
            papers,
            [],
            k=bucket.k,
            budget=budget,
            seed=seed,
            diagnostics=diagnostics,
            config=EVSISchedulerConfig(samples=1200),
        )
    elif scheduler_kind == "sequential_evsi":
        reusable, reusable_stats = load_historical_pairwise_reuse_cache(
            bucket,
            papers=papers,
            source_artifact_dir=source_artifact_dir,
            phase=phase,
            seed=seed,
        )

        def reveal_cached(pair: ScheduledPair) -> PairwiseComparison | None:
            pair_key = canonical_pair_key(pair.left_id, pair.right_id)
            reusable_artifact = reusable.get(pair_key)
            if reusable_artifact is None:
                reveal_log.append(
                    {
                        "pair_key": list(pair_key),
                        "status": "novel",
                        "revealed": False,
                    }
                )
                return None
            comparison = orient_comparison(reusable_artifact.comparison, pair)
            reveal_log.append(
                {
                    "pair_key": list(pair_key),
                    "status": "cached",
                    "revealed": True,
                    "source_kind": reusable_artifact.kind,
                    "source_artifact_path": str(reusable_artifact.artifact_path),
                }
            )
            return comparison

        schedule = schedule_cache_aware_sequential_evsi(
            papers,
            [],
            reveal_comparison=reveal_cached,
            k=bucket.k,
            budget=budget,
            seed=seed,
            diagnostics=diagnostics,
            config=SequentialEVSISchedulerConfig(
                evsi=EVSISchedulerConfig(samples=1200),
                rounds=5,
                batch_size=4,
                stop_on_novel=True,
            ),
        )
    else:
        raise ValueError(f"unknown scheduler_kind {scheduler_kind!r}")
    if not reusable_stats:
        reusable, reusable_stats = load_historical_pairwise_reuse_cache(
            bucket,
            papers=papers,
            source_artifact_dir=source_artifact_dir,
            phase=phase,
            seed=seed,
        )
    schedule_keys = {
        canonical_pair_key(pair.left_id, pair.right_id) for pair in schedule.pairs
    }
    reusable_stats = {
        **reusable_stats,
        "scheduled_pairwise_total": len(schedule.pairs),
        "scheduled_reusable_total": sum(
            1 for key in schedule_keys if key in reusable
        ),
        "scheduled_novel_total": sum(
            1 for key in schedule_keys if key not in reusable
        ),
    }
    merged_diagnostics = _merged_schedule_diagnostics(
        schedule.diagnostics,
        diagnostics.to_dict(),
    )
    merged_diagnostics["isolation"] = _scheduler_isolation_diagnostics(
        bucket,
        papers=papers,
        schedule=schedule.pairs,
        reusable_pairwise=reusable,
        scheduler_diagnostics=schedule.diagnostics,
    )
    if reveal_log:
        merged_diagnostics["cache_reveal_log"] = reveal_log
    return SchedulerOnlyBucketPlan(
        bucket=bucket,
        papers=papers,
        schedule=schedule.pairs,
        diagnostics=merged_diagnostics,
        reusable_pairwise=reusable,
        reusable_stats=reusable_stats,
        reveal_log=reveal_log,
    )


def load_pointwise_papers_from_artifacts(
    bucket: BacktestBucket,
    *,
    source_artifact_dir: Path,
    phase: str,
) -> list[Paper]:
    calls_dir = _source_calls_dir(
        source_artifact_dir,
        phase=phase,
        bucket_name=bucket.name,
    )
    papers: list[Paper] = []
    missing: list[str] = []
    for index, paper in enumerate(bucket.papers, start=1):
        path = calls_dir / (
            f"{index:04d}-pointwise-{fingerprint(paper.paper_id)}.json"
        )
        cached = _load_ok_call_artifact(path)
        if cached is None:
            missing.append(str(path))
            continue
        papers.append(
            Paper(
                paper_id=paper.paper_id,
                title=paper.title,
                abstract=paper.abstract,
                pointwise=PointwiseAssessment.from_dict(cached["response"]),
                metadata=paper.metadata,
            )
        )
    if missing:
        raise PointwiseArtifactError(
            "missing successful pointwise artifacts before scheduler-only run: "
            + ", ".join(missing[:5])
            + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else "")
        )
    return papers


def load_historical_pairwise_reuse_cache(
    bucket: BacktestBucket,
    *,
    papers: list[Paper],
    source_artifact_dir: Path,
    phase: str,
    seed: int,
) -> tuple[dict[tuple[str, str], ReusablePairwiseArtifact], dict[str, Any]]:
    calls_dir = _source_calls_dir(
        source_artifact_dir,
        phase=phase,
        bucket_name=bucket.name,
    )
    legacy_selection = legacy_select_candidates(papers, k=bucket.k)
    budget = resolve_pairwise_budget(
        n=len(papers),
        candidate_size=len(legacy_selection.candidate_ids),
    )
    active_schedule = legacy_schedule_pairs(
        papers,
        candidate_selection=legacy_selection,
        k=bucket.k,
        budget=budget,
        seed=seed,
    )
    random_schedule = _random_pair_schedule(
        legacy_selection,
        budget=budget,
        seed=seed + 7919,
    )

    reusable: dict[tuple[str, str], ReusablePairwiseArtifact] = {}
    stats = Counter(
        {
            "successful_pairwise_artifacts": 0,
            "duplicate_pair_keys": 0,
            "subject_mismatch_total": 0,
            "missing_or_failed_total": 0,
        }
    )
    by_kind: Counter[str] = Counter()
    for kind, schedule in (
        ("pairwise_active", active_schedule.pairs),
        ("pairwise_random", random_schedule),
    ):
        for index, pair in enumerate(schedule, start=1):
            path = calls_dir / (
                f"{index:04d}-{kind}-"
                f"{fingerprint(pair.left_id + ':' + pair.right_id)}.json"
            )
            cached = _load_ok_call_artifact(path)
            if cached is None:
                stats["missing_or_failed_total"] += 1
                continue
            subject = cached.get("subject") or {}
            if subject.get("left_id") != pair.left_id or (
                subject.get("right_id") != pair.right_id
            ):
                stats["subject_mismatch_total"] += 1
                continue
            key = canonical_pair_key(pair.left_id, pair.right_id)
            stats["successful_pairwise_artifacts"] += 1
            by_kind[kind] += 1
            if key in reusable:
                stats["duplicate_pair_keys"] += 1
                continue
            reusable[key] = ReusablePairwiseArtifact(
                bucket=bucket.name,
                pair_key=key,
                comparison=_comparison_from_pairwise_response(
                    pair,
                    cached["response"],
                ),
                artifact_path=path,
                kind=kind,
            )
    return reusable, {
        **dict(stats),
        "successful_pairwise_by_kind": dict(sorted(by_kind.items())),
        "unique_reusable_pair_keys": len(reusable),
        "legacy_active_schedule_diagnostics": active_schedule.diagnostics,
    }


def analyze_aggregation_variants(
    *,
    manifest_path: Path,
    source_artifact_dir: Path,
    output_path: Path,
    phase: str = "pilot",
    seed: int = 17,
    strengths: tuple[float, ...] = DEFAULT_STRENGTH_SWEEP,
) -> dict[str, Any]:
    manifest = load_dataset_manifest(manifest_path)
    bucket_rows = []
    aggregate_inputs: dict[str, dict[str, list[dict[str, float | int]]]] = {}
    coverage_totals = Counter()
    coverage_denominator = 0

    for bucket in manifest.buckets_for_phase(phase):
        papers = load_pointwise_papers_from_artifacts(
            bucket,
            source_artifact_dir=source_artifact_dir,
            phase=phase,
        )
        legacy_selection = legacy_select_candidates(papers, k=bucket.k)
        budget = resolve_pairwise_budget(
            n=len(papers),
            candidate_size=len(legacy_selection.candidate_ids),
        )
        active_schedule = legacy_schedule_pairs(
            papers,
            candidate_selection=legacy_selection,
            k=bucket.k,
            budget=budget,
            seed=seed,
        )
        random_schedule = _random_pair_schedule(
            legacy_selection,
            budget=budget,
            seed=seed + 7919,
        )
        active_comparisons = _load_historical_schedule_comparisons(
            bucket,
            schedule=active_schedule.pairs,
            source_artifact_dir=source_artifact_dir,
            phase=phase,
            kind="pairwise_active",
        )
        random_comparisons = _load_historical_schedule_comparisons(
            bucket,
            schedule=random_schedule,
            source_artifact_dir=source_artifact_dir,
            phase=phase,
            kind="pairwise_random",
        )
        revised_plan = build_scheduler_only_bucket_plan(
            bucket,
            source_artifact_dir=source_artifact_dir,
            phase=phase,
            seed=seed,
        )
        coverage = _historical_schedule_coverage(
            active_schedule.pairs,
            candidate_ids=set(legacy_selection.candidate_ids),
            relevant_ids=bucket.relevant_ids,
        )
        coverage_totals.update(
            {
                "candidate_internal_pairs": coverage["candidate_internal_pairs"],
                "candidate_outsider_pairs": coverage["candidate_outsider_pairs"],
                "distinct_positive_outsiders": coverage[
                    "distinct_positive_outsiders"
                ],
            }
        )
        coverage_denominator += coverage["scheduled_pairs_total"]

        variants = {}
        for strength in strengths:
            label = _strength_label(strength)
            metrics = _variant_metrics(
                papers,
                relevant_ids=bucket.relevant_ids,
                k=bucket.k,
                active_comparisons=active_comparisons,
                random_comparisons=random_comparisons,
                pairwise_strength=strength,
            )
            variants[label] = {
                strategy: metric.to_dict()
                for strategy, metric in metrics.items()
            }
            aggregate_inputs.setdefault(label, {})
            for strategy, metric in metrics.items():
                aggregate_inputs[label].setdefault(strategy, []).append(
                    metric.to_dict()
                )

        bucket_rows.append(
            {
                "bucket": bucket.name,
                "k": bucket.k,
                "papers_total": len(papers),
                "positive_labels_total": len(bucket.relevant_ids),
                "pointwise_artifacts_loaded": len(papers),
                "historical_pairwise_loaded": {
                    "active": len(active_comparisons),
                    "random": len(random_comparisons),
                },
                "historical_active_coverage": coverage,
                "revised_scheduler_preview": {
                    "scheduled_pairs_total": len(revised_plan.schedule),
                    "purpose_counts": _purpose_counts(revised_plan.schedule),
                    "coverage": _schedule_diagnostics_coverage(
                        revised_plan.diagnostics
                    ),
                    "reuse_stats": revised_plan.reusable_stats,
                },
                "variant_metrics": variants,
            }
        )

    aggregate_metrics = {
        label: {
            strategy: _mean_metric_rows(rows)
            for strategy, rows in sorted(strategies.items())
        }
        for label, strategies in sorted(
            aggregate_inputs.items(),
            key=lambda item: float(item[0]),
        )
    }
    aggregate_deltas = {
        label: _aggregate_deltas_for_strength(metrics)
        for label, metrics in aggregate_metrics.items()
    }
    diagnosis = _diagnose_active_underperformance(
        aggregate_metrics=aggregate_metrics,
        aggregate_deltas=aggregate_deltas,
        coverage_totals=dict(coverage_totals),
        coverage_denominator=coverage_denominator,
    )
    payload = {
        "artifact_type": "sestina-aggregation-variant-analysis",
        "manifest_path": str(manifest_path),
        "source_artifact_dir": str(source_artifact_dir),
        "phase": phase,
        "seed": seed,
        "pairwise_strength_values": list(strengths),
        "bucket_analyses": bucket_rows,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_deltas": aggregate_deltas,
        "active_underperformance_diagnosis": diagnosis,
        "conclusions": _variant_conclusions(diagnosis),
    }
    write_json_artifact(output_path, payload)
    return {**payload, "artifact_path": str(output_path)}


def legacy_select_candidates(papers: list[Paper], *, k: int) -> CandidateSelection:
    n = len(papers)
    m = min(n, math.ceil((3 * k) + math.sqrt(n)))
    if n == 0 or k == 0:
        return CandidateSelection(
            candidate_ids=[],
            groups={"exploit": [], "boundary": [], "explore": []},
            scores={},
        )
    by_quality = sorted(
        papers,
        key=lambda paper: (
            paper.pointwise.good_probability,
            -paper.pointwise.uncertainty,
            paper.paper_id,
        ),
        reverse=True,
    )
    boundary_probability = by_quality[
        min(k - 1, len(by_quality) - 1)
    ].pointwise.good_probability
    by_boundary = sorted(
        papers,
        key=lambda paper: (
            abs(paper.pointwise.good_probability - boundary_probability),
            -paper.pointwise.uncertainty,
            paper.paper_id,
        ),
    )
    by_uncertainty = sorted(
        papers,
        key=lambda paper: (
            paper.pointwise.uncertainty,
            paper.pointwise.good_probability,
            paper.paper_id,
        ),
        reverse=True,
    )
    exploit = _take_unique(by_quality, limit=max(k, math.ceil(m * 0.5)))
    boundary = _take_unique(by_boundary, limit=max(0, math.ceil(m * 0.35)))
    explore = _legacy_round_robin_by_metadata(
        by_uncertainty,
        existing=set(exploit) | set(boundary),
        limit=max(0, m - len(set(exploit) | set(boundary))),
    )
    ordered = _ordered_union(
        exploit,
        boundary,
        explore,
        [paper.paper_id for paper in by_quality],
        limit=m,
    )
    groups = {
        "exploit": [paper_id for paper_id in exploit if paper_id in ordered],
        "boundary": [paper_id for paper_id in boundary if paper_id in ordered],
        "explore": [paper_id for paper_id in explore if paper_id in ordered],
    }
    return CandidateSelection(
        candidate_ids=ordered,
        groups=groups,
        scores={},
    )


def legacy_schedule_pairs(
    papers: list[Paper],
    *,
    candidate_selection: CandidateSelection,
    k: int,
    budget: PairwiseBudget,
    seed: int,
) -> Any:
    paper_by_id = {paper.paper_id: paper for paper in papers}
    candidate_ids = [
        paper_id
        for paper_id in candidate_selection.candidate_ids
        if paper_id in paper_by_id
    ]
    if budget.budget <= 0 or len(candidate_ids) < 2:
        return _SimplePairSchedule(
            pairs=[],
            diagnostics={
                "candidate_count": len(candidate_ids),
                "scheduled_total": 0,
                "pairs_considered": 0,
            },
        )

    ranked = sorted(
        candidate_ids,
        key=lambda paper_id: paper_by_id[paper_id].pointwise.good_probability,
        reverse=True,
    )
    rank_by_id = {paper_id: index + 1 for index, paper_id in enumerate(ranked)}
    boundary_rank = min(max(k, 1), len(ranked))
    proposals = []
    for left_id, right_id in itertools.combinations(candidate_ids, 2):
        priority, purpose, parts = _legacy_pair_priority(
            left=paper_by_id[left_id],
            right=paper_by_id[right_id],
            rank_left=rank_by_id[left_id],
            rank_right=rank_by_id[right_id],
            boundary_rank=boundary_rank,
        )
        proposals.append((priority, left_id, right_id, purpose, parts))

    proposals.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    rng = random.Random(seed)
    scheduled = []
    for index, (priority, left_id, right_id, purpose, parts) in enumerate(
        proposals[: budget.budget]
    ):
        if rng.random() < 0.5:
            shown_first, shown_second = left_id, right_id
        else:
            shown_first, shown_second = right_id, left_id
        order = PairwiseOrderMetadata(
            shown_first_id=shown_first,
            shown_second_id=shown_second,
            randomized=True,
            seed=seed,
            position_bias_audit=(index % 5 == 0),
            extra={"canonical_left_id": left_id, "canonical_right_id": right_id},
        )
        scheduled.append(
            ScheduledPair(
                left_id=left_id,
                right_id=right_id,
                priority=round(priority, 6),
                purpose=purpose,
                order=order,
                diagnostics=parts,
            )
        )
    return _SimplePairSchedule(
        pairs=scheduled,
        diagnostics={
            "candidate_count": len(candidate_ids),
            "scheduled_total": len(scheduled),
            "pairs_considered": len(proposals),
            "budget": budget.budget,
            "boundary_rank": boundary_rank,
            "position_bias_audit_total": sum(
                1 for pair in scheduled if pair.order.position_bias_audit
            ),
        },
    )


@dataclass(frozen=True, slots=True)
class _SimplePairSchedule:
    pairs: list[ScheduledPair]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _single_phase_config(config: dict[str, Any], *, phase: str) -> dict[str, Any]:
    selected = _config_for_phase(config, phase=phase)
    phases = selected.get("phases", [])
    if len(phases) != 1:
        raise ValueError(f"scheduler-only runner requires one phase; got {phase}")
    return dict(phases[0])


def _validate_scheduler_only_paid_preconditions(
    *,
    max_usd: float,
    estimate_usd: float,
    artifact_dir: Path,
    source_artifact_dir: Path,
    ledger_path: Path,
) -> None:
    if max_usd <= 0:
        raise PaidRunSafetyError("--max-usd must be greater than zero")
    if max_usd > 0.50:
        raise PaidRunSafetyError("scheduler-only --max-usd must not exceed 0.50")
    if estimate_usd > max_usd:
        raise PaidRunSafetyError("scheduler-only dry-run estimate exceeds --max-usd")
    if artifact_dir.resolve() == source_artifact_dir.resolve():
        raise PaidRunSafetyError(
            "scheduler-only artifact dir must differ from source artifact dir"
        )
    if not ledger_path:
        raise PaidRunSafetyError("--ledger is required for paid runs")


def _estimate_from_plans(
    plans: list[SchedulerOnlyBucketPlan],
    *,
    call_estimate: CallEstimate,
) -> dict[str, Any]:
    scheduled = sum(len(plan.schedule) for plan in plans)
    reused = sum(
        plan.reusable_stats["scheduled_reusable_total"] for plan in plans
    )
    novel = sum(plan.reusable_stats["scheduled_novel_total"] for plan in plans)
    return {
        "pointwise_calls": 0,
        "pairwise_calls": int(novel),
        "pairwise_scheduled_total": int(scheduled),
        "pairwise_reused_total": int(reused),
        "pairwise_novel_total": int(novel),
        "audit_pairwise_calls": 0,
        "input_tokens": int(novel * call_estimate.input_tokens),
        "output_tokens": int(novel * call_estimate.output_tokens),
        "cost_usd": round(novel * call_estimate.cost_usd, 6),
    }


def _scheduler_only_estimate_payload(
    *,
    phase: str,
    max_usd: float,
    pairwise_model: str,
    call_estimate: CallEstimate,
    plans: list[SchedulerOnlyBucketPlan],
    totals: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_type": "sestina-scheduler-only-cost-estimate",
        "dry_run": True,
        "phase": phase,
        "budget_cap_usd": max_usd,
        "pairwise_model": pairwise_model,
        "pointwise_reuse": (
            "historical successful pointwise artifacts are required; "
            "no pointwise calls are estimated or executed"
        ),
        "pairwise_reuse": (
            "successful historical active/random pairwise artifacts are reused "
            "by same-bucket canonical unordered pair key"
        ),
        "per_pairwise_call_estimate": {
            "input_tokens": call_estimate.input_tokens,
            "output_tokens": call_estimate.output_tokens,
            "cost_usd": call_estimate.cost_usd,
        },
        "totals": totals,
        "buckets": [
            {
                "bucket": plan.bucket.name,
                "k": plan.bucket.k,
                "papers_total": len(plan.papers),
                "pointwise_calls": 0,
                "pairwise_scheduled_total": len(plan.schedule),
                "pairwise_reused_total": plan.reusable_stats[
                    "scheduled_reusable_total"
                ],
                "pairwise_novel_total": plan.reusable_stats[
                    "scheduled_novel_total"
                ],
                "scheduler_diagnostics": plan.diagnostics,
                "reuse_stats": plan.reusable_stats,
            }
            for plan in plans
        ],
        "model_availability": {
            "status": "not_checked_dry_run",
            "required_before_paid_run": True,
            "models_requiring_check": [pairwise_model],
        },
    }


def _scheduler_only_blocker_payload(
    *,
    scheduler_kind: str,
    plans: list[SchedulerOnlyBucketPlan],
    call_estimate: CallEstimate,
) -> dict[str, Any] | None:
    novel_total = sum(
        plan.reusable_stats["scheduled_novel_total"] for plan in plans
    )
    if novel_total <= 0:
        return None
    return {
        "status": "blocked_on_paid_pairwise_labels",
        "scheduler_kind": scheduler_kind,
        "novel_pairwise_calls_required_minimum": int(novel_total),
        "estimated_input_tokens": int(novel_total * call_estimate.input_tokens),
        "estimated_output_tokens": int(novel_total * call_estimate.output_tokens),
        "estimated_cost_usd": round(novel_total * call_estimate.cost_usd, 6),
        "paid_call_policy": (
            "dry-run records novel pairs but does not pretend labels are known"
        ),
        "recommended_action": (
            "review the dry-run novel pairs and run a guarded paid labeling pass "
            "only after supervisor approval"
        ),
    }


def _offline_bucket_results_payload(
    plans: list[SchedulerOnlyBucketPlan],
    *,
    phase: str,
    aggregation_mode: str,
    artifact_path: Path,
) -> dict[str, Any]:
    bucket_results = []
    for plan in plans:
        comparisons = _cached_schedule_comparisons(plan)
        result = _bucket_metrics_payload(
            plan,
            comparisons=comparisons,
            artifact_path=artifact_path,
            aggregation_mode=aggregation_mode,
        )
        missing = len(plan.schedule) - len(comparisons)
        result["offline_label_status"] = {
            "scheduled_pairwise_total": len(plan.schedule),
            "cached_pairwise_labels_available": len(comparisons),
            "missing_pairwise_labels": missing,
            "partial": missing > 0,
        }
        bucket_results.append(result)
    return {
        "artifact_type": "sestina-scheduler-only-offline-bucket-results",
        "phase": phase,
        "aggregation_mode": aggregation_mode,
        "bucket_results": bucket_results,
    }


def _cached_schedule_comparisons(
    plan: SchedulerOnlyBucketPlan,
) -> list[PairwiseComparison]:
    comparisons = []
    for pair in plan.schedule:
        reusable = plan.reusable_pairwise.get(
            canonical_pair_key(pair.left_id, pair.right_id)
        )
        if reusable is None:
            continue
        comparisons.append(orient_comparison(reusable.comparison, pair))
    return comparisons


def _bucket_metrics_payload(
    plan: SchedulerOnlyBucketPlan,
    *,
    comparisons: list[PairwiseComparison],
    artifact_path: Path,
    aggregation_mode: str,
) -> dict[str, Any]:
    strategies = {
        "pointwise_only": [
            Prediction(paper.paper_id, paper.pointwise.good_probability)
            for paper in plan.papers
        ],
        "sestina_active_pairwise": _aggregate_predictions_with_strength(
            plan.papers,
            comparisons,
            pairwise_strength=2.5,
        ),
    }
    if aggregation_mode in {"posterior_topk", "both"}:
        posterior_predictions, posterior = posterior_top_k_predictions(
            plan.papers,
            comparisons,
            k=plan.bucket.k,
            pairwise_strength=2.5,
            samples=2000,
            seed=17,
        )
        strategies["sestina_active_posterior_topk"] = posterior_predictions
    else:
        posterior = None
    metrics = compare_strategies(
        strategies,
        relevant_ids=plan.bucket.relevant_ids,
        k=plan.bucket.k,
    )
    return {
        "artifact_type": "sestina-scheduler-only-bucket-result",
        "phase": plan.bucket.phase,
        "bucket": plan.bucket.name,
        "k": plan.bucket.k,
        "papers_total": len(plan.papers),
        "positive_labels_total": len(plan.bucket.relevant_ids),
        "strategies": {
            name: metric.to_dict() for name, metric in metrics.items()
        },
        "aggregation_mode": aggregation_mode,
        "posterior_topk_diagnostics": (
            posterior.diagnostics if posterior is not None else None
        ),
        "scheduler_diagnostics": plan.diagnostics,
        "reuse_stats": plan.reusable_stats,
        "artifact_path": str(artifact_path),
    }


def _variant_metrics(
    papers: list[Paper],
    *,
    relevant_ids: set[str],
    k: int,
    active_comparisons: list[PairwiseComparison],
    random_comparisons: list[PairwiseComparison],
    pairwise_strength: float,
) -> Any:
    strategies = {
        "pointwise_only": [
            Prediction(paper.paper_id, paper.pointwise.good_probability)
            for paper in papers
        ],
        "pointwise_random_pairwise": _aggregate_predictions_with_strength(
            papers,
            random_comparisons,
            pairwise_strength=pairwise_strength,
        ),
        "sestina_active_pairwise": _aggregate_predictions_with_strength(
            papers,
            active_comparisons,
            pairwise_strength=pairwise_strength,
        ),
    }
    random_topk, _ = posterior_top_k_predictions(
        papers,
        random_comparisons,
        k=k,
        pairwise_strength=pairwise_strength,
        samples=2000,
        seed=17,
    )
    active_topk, _ = posterior_top_k_predictions(
        papers,
        active_comparisons,
        k=k,
        pairwise_strength=pairwise_strength,
        samples=2000,
        seed=17,
    )
    strategies["pointwise_random_pairwise_posterior_topk"] = random_topk
    strategies["sestina_active_pairwise_posterior_topk"] = active_topk
    return compare_strategies(strategies, relevant_ids=relevant_ids, k=k)


def _aggregate_predictions_with_strength(
    papers: list[Paper],
    comparisons: list[PairwiseComparison],
    *,
    pairwise_strength: float,
) -> list[Prediction]:
    result = aggregate(
        papers,
        comparisons,
        config=AggregationConfig(pairwise_strength=pairwise_strength),
    )
    return [
        Prediction(paper_id, estimate.posterior_good_probability)
        for paper_id, estimate in result.estimates.items()
    ]


def _load_historical_schedule_comparisons(
    bucket: BacktestBucket,
    *,
    schedule: list[ScheduledPair],
    source_artifact_dir: Path,
    phase: str,
    kind: str,
) -> list[PairwiseComparison]:
    calls_dir = _source_calls_dir(
        source_artifact_dir,
        phase=phase,
        bucket_name=bucket.name,
    )
    comparisons = []
    for index, pair in enumerate(schedule, start=1):
        path = calls_dir / (
            f"{index:04d}-{kind}-"
            f"{fingerprint(pair.left_id + ':' + pair.right_id)}.json"
        )
        cached = _load_ok_call_artifact(path)
        if cached is None:
            continue
        comparisons.append(
            _comparison_from_pairwise_response(pair, cached["response"])
        )
    return comparisons


def _reused_pairwise_artifact(
    *,
    phase: str,
    bucket: str,
    model: str,
    estimate: CallEstimate,
    pair: ScheduledPair,
    source: ReusablePairwiseArtifact,
    comparison: PairwiseComparison,
) -> dict[str, Any]:
    payload = _call_artifact(
        phase=phase,
        bucket=bucket,
        model=model,
        kind="pairwise_active",
        estimate=estimate,
        status="reused",
        response=comparison.to_dict(),
        subject={"left_id": pair.left_id, "right_id": pair.right_id},
    )
    payload["source_artifact_path"] = str(source.artifact_path)
    payload["source_kind"] = source.kind
    payload["reuse_key"] = list(source.pair_key)
    return payload


def _followup_pairwise_artifact_path(
    calls_dir: Path,
    *,
    index: int,
    pair: ScheduledPair,
) -> Path:
    return calls_dir / (
        f"{index:04d}-pairwise_active-"
        f"{fingerprint(pair.left_id + ':' + pair.right_id)}.json"
    )


def _source_calls_dir(
    source_artifact_dir: Path,
    *,
    phase: str,
    bucket_name: str,
) -> Path:
    return source_artifact_dir / phase / _safe_name(bucket_name) / "calls"


def _invert_winner(winner: str) -> str:
    if winner == "left":
        return "right"
    if winner == "right":
        return "left"
    return winner


def _take_unique(papers: list[Paper], *, limit: int) -> list[str]:
    seen: set[str] = set()
    selected = []
    for paper in papers:
        if paper.paper_id in seen:
            continue
        seen.add(paper.paper_id)
        selected.append(paper.paper_id)
        if len(selected) >= limit:
            break
    return selected


def _legacy_round_robin_by_metadata(
    papers: list[Paper],
    *,
    existing: set[str],
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []
    buckets: dict[str, list[Paper]] = {}
    for paper in papers:
        if paper.paper_id in existing:
            continue
        buckets.setdefault(_legacy_bucket_key(paper), []).append(paper)
    selected: list[str] = []
    keys = sorted(buckets)
    while keys and len(selected) < limit:
        next_keys = []
        for key in keys:
            bucket = buckets[key]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop(0).paper_id)
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected


def _legacy_bucket_key(paper: Paper) -> str:
    for key in ("topic", "venue", "source", "field", "category"):
        value = paper.metadata.get(key)
        if value:
            return f"{key}:{value}"
    return "unknown"


def _ordered_union(*groups: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    selected = []
    for group in groups:
        for paper_id in group:
            if paper_id in seen:
                continue
            seen.add(paper_id)
            selected.append(paper_id)
            if len(selected) >= limit:
                return selected
    return selected


def _legacy_pair_priority(
    *,
    left: Paper,
    right: Paper,
    rank_left: int,
    rank_right: int,
    boundary_rank: int,
) -> tuple[float, str, dict[str, float]]:
    q_gap = abs(left.pointwise.good_probability - right.pointwise.good_probability)
    closeness = 1.0 - min(1.0, q_gap)
    rank_mid = (rank_left + rank_right) / 2.0
    boundary = 1.0 / (1.0 + abs(rank_mid - boundary_rank))
    uncertainty = (left.pointwise.uncertainty + right.pointwise.uncertainty) / 2.0
    diversity = (
        1.0 if _legacy_metadata_bucket(left) != _legacy_metadata_bucket(right) else 0.0
    )
    priority = (
        (0.40 * boundary)
        + (0.30 * closeness)
        + (0.20 * uncertainty)
        + (0.10 * diversity)
    )
    parts = {
        "boundary": round(boundary, 6),
        "closeness": round(closeness, 6),
        "uncertainty": round(uncertainty, 6),
        "diversity": round(diversity, 6),
    }
    purpose = max(parts, key=parts.get)
    return priority, purpose, parts


def _legacy_metadata_bucket(paper: Paper) -> tuple[str, object]:
    for key in ("topic", "venue", "source", "field", "category"):
        if key in paper.metadata:
            return key, paper.metadata[key]
    return "unknown", "unknown"


def _historical_schedule_coverage(
    schedule: list[ScheduledPair],
    *,
    candidate_ids: set[str],
    relevant_ids: set[str],
) -> dict[str, Any]:
    candidate_internal = 0
    candidate_outsider = 0
    outsider_outsider = 0
    distinct_papers = set()
    distinct_outsiders = set()
    distinct_positive_outsiders = set()
    purpose_counts = Counter(pair.purpose for pair in schedule)
    dominant_counts = Counter(
        str(pair.diagnostics.get("dominant_component") or pair.purpose)
        for pair in schedule
    )
    for pair in schedule:
        left_is_candidate = pair.left_id in candidate_ids
        right_is_candidate = pair.right_id in candidate_ids
        if left_is_candidate and right_is_candidate:
            candidate_internal += 1
        elif left_is_candidate or right_is_candidate:
            candidate_outsider += 1
        else:
            outsider_outsider += 1
        distinct_papers.update([pair.left_id, pair.right_id])
        for paper_id in (pair.left_id, pair.right_id):
            if paper_id not in candidate_ids:
                distinct_outsiders.add(paper_id)
                if paper_id in relevant_ids:
                    distinct_positive_outsiders.add(paper_id)
    return {
        "scheduled_pairs_total": len(schedule),
        "candidate_internal_pairs": candidate_internal,
        "candidate_outsider_pairs": candidate_outsider,
        "outsider_outsider_pairs": outsider_outsider,
        "distinct_papers_compared": len(distinct_papers),
        "distinct_outsiders_compared": len(distinct_outsiders),
        "distinct_positive_outsiders": len(distinct_positive_outsiders),
        "purpose_counts": dict(sorted(purpose_counts.items())),
        "dominant_component_counts": dict(sorted(dominant_counts.items())),
    }


def _purpose_counts(schedule: list[ScheduledPair]) -> dict[str, int]:
    return dict(sorted(Counter(pair.purpose for pair in schedule).items()))


def _scheduler_isolation_diagnostics(
    bucket: BacktestBucket,
    *,
    papers: list[Paper],
    schedule: list[ScheduledPair],
    reusable_pairwise: dict[tuple[str, str], ReusablePairwiseArtifact],
    scheduler_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    touched = {
        paper_id
        for pair in schedule
        for paper_id in (pair.left_id, pair.right_id)
    }
    degrees = Counter(
        paper_id
        for pair in schedule
        for paper_id in (pair.left_id, pair.right_id)
    )
    ranked = sorted(
        papers,
        key=lambda paper: (
            paper.pointwise.good_probability,
            -paper.pointwise.uncertainty,
            paper.paper_id,
        ),
        reverse=True,
    )
    plausible_ids = [
        paper.paper_id for paper in ranked[: max(1, min(len(ranked), 2 * bucket.k))]
    ]
    anchor_ids = [paper.paper_id for paper in ranked[: max(bucket.k, 0)]]
    components = _schedule_components(schedule)
    component_by_id = {
        paper_id: index
        for index, component in enumerate(components)
        for paper_id in component
    }
    anchor_component_ids = {
        component_by_id[paper_id]
        for paper_id in anchor_ids
        if paper_id in component_by_id
    }
    pool_profile = scheduler_diagnostics.get("proposal_pool_profile") or {}
    return {
        "unique_papers_touched": len(touched),
        "unique_papers_touched_rate": _safe_rate(len(touched), len(papers)),
        "plausible_top_k_papers": len(plausible_ids),
        "plausible_top_k_papers_touched": sum(
            1 for paper_id in plausible_ids if paper_id in touched
        ),
        "plausible_top_k_degree_distribution": _degree_distribution(
            [degrees[paper_id] for paper_id in plausible_ids]
        ),
        "connected_components": {
            "component_count": len(components),
            "largest_component_size": max((len(component) for component in components), default=0),
            "anchor_papers": len(anchor_ids),
            "anchor_papers_touched": sum(
                1 for paper_id in anchor_ids if paper_id in touched
            ),
            "components_with_anchor": len(anchor_component_ids),
        },
        "high_ucb_outsider_exposure": {
            "high_ucb_outsider_total": int(
                pool_profile.get("high_ucb_outsider_total") or 0
            ),
            "high_ucb_outsider_touched": int(
                pool_profile.get("high_ucb_outsider_touched") or 0
            ),
            "high_ucb_outsider_exposure_rate": float(
                pool_profile.get("high_ucb_outsider_exposure_rate") or 0.0
            ),
        },
        "per_batch_top_k": scheduler_diagnostics.get("batch_history", []),
        "evsi_score_distribution": scheduler_diagnostics.get(
            "evsi_score_distribution",
            {},
        ),
        "retrospective_future_positive_exposure": (
            _future_positive_exposure(schedule, relevant_ids=bucket.relevant_ids)
        ),
        "positive_vs_negative_pairwise_win_rate": (
            _positive_negative_win_rate(
                schedule,
                relevant_ids=bucket.relevant_ids,
                reusable_pairwise=reusable_pairwise,
            )
        ),
    }


def _schedule_components(schedule: list[ScheduledPair]) -> list[set[str]]:
    graph: dict[str, set[str]] = {}
    for pair in schedule:
        graph.setdefault(pair.left_id, set()).add(pair.right_id)
        graph.setdefault(pair.right_id, set()).add(pair.left_id)
    components: list[set[str]] = []
    seen: set[str] = set()
    for paper_id in sorted(graph):
        if paper_id in seen:
            continue
        stack = [paper_id]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(sorted(graph.get(current, set()) - seen))
        components.append(component)
    return components


def _degree_distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "zero_degree_count": 0,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 8),
        "zero_degree_count": sum(1 for value in values if value == 0),
    }


def _future_positive_exposure(
    schedule: list[ScheduledPair],
    *,
    relevant_ids: set[str],
) -> dict[str, Any]:
    if not schedule:
        return {
            "scheduled_pairs_total": 0,
            "pairs_touching_future_positive": 0,
            "pair_exposure_rate": 0.0,
            "unique_future_positives_touched": 0,
        }
    positive_pairs = 0
    positives_touched: set[str] = set()
    for pair in schedule:
        pair_ids = {pair.left_id, pair.right_id}
        exposed = pair_ids & relevant_ids
        if exposed:
            positive_pairs += 1
            positives_touched.update(exposed)
    return {
        "scheduled_pairs_total": len(schedule),
        "pairs_touching_future_positive": positive_pairs,
        "pair_exposure_rate": _safe_rate(positive_pairs, len(schedule)),
        "unique_future_positives_touched": len(positives_touched),
        "unique_future_positive_touch_rate": _safe_rate(
            len(positives_touched),
            len(relevant_ids),
        ),
    }


def _positive_negative_win_rate(
    schedule: list[ScheduledPair],
    *,
    relevant_ids: set[str],
    reusable_pairwise: dict[tuple[str, str], ReusablePairwiseArtifact],
) -> dict[str, Any]:
    eligible = 0
    positive_wins = 0
    negative_wins = 0
    ties_or_uncertain = 0
    pairwise_labels_available = 0
    for pair in schedule:
        reusable = reusable_pairwise.get(canonical_pair_key(pair.left_id, pair.right_id))
        if reusable is None:
            continue
        pairwise_labels_available += 1
        left_positive = pair.left_id in relevant_ids
        right_positive = pair.right_id in relevant_ids
        if left_positive == right_positive:
            continue
        eligible += 1
        comparison = orient_comparison(reusable.comparison, pair)
        if comparison.winner == "tie" or comparison.winner == "uncertain":
            ties_or_uncertain += 1
        elif (comparison.winner == "left" and left_positive) or (
            comparison.winner == "right" and right_positive
        ):
            positive_wins += 1
        else:
            negative_wins += 1
    return {
        "pairwise_labels_available": pairwise_labels_available,
        "positive_negative_pairs_with_label": eligible,
        "positive_wins": positive_wins,
        "negative_wins": negative_wins,
        "ties_or_uncertain": ties_or_uncertain,
        "positive_win_rate": _safe_rate(positive_wins, eligible),
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def _schedule_diagnostics_coverage(diagnostics: dict[str, Any]) -> dict[str, Any]:
    coverage = diagnostics.get("coverage")
    if isinstance(coverage, dict):
        return coverage
    for event in diagnostics.get("events", []):
        if event.get("code") == "pair_scheduling_completed":
            data = event.get("data") or {}
            coverage = data.get("coverage")
            if isinstance(coverage, dict):
                return coverage
    return {}


def _merged_schedule_diagnostics(
    schedule_diagnostics: dict[str, Any],
    recorder_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    events = recorder_diagnostics.get("events", [])
    if not isinstance(events, list):
        events = []
    return {
        **schedule_diagnostics,
        "events": events,
    }


def _mean_metric_rows(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    keys = (
        "recall_at_k",
        "precision_at_k",
        "ndcg_at_k",
        "average_precision",
        "brier_score",
        "near_miss_positive_rate",
    )
    if not rows:
        return {}
    averaged = {
        key: round(sum(float(row[key]) for row in rows) / len(rows), 8)
        for key in keys
        if key in rows[0]
    }
    averaged["k"] = int(rows[0].get("k", 0))
    averaged["bucket_count"] = len(rows)
    return averaged


def _aggregate_bucket_result_metrics(
    bucket_results: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    rows_by_strategy: dict[str, list[dict[str, float | int]]] = {}
    for result in bucket_results:
        for strategy, metrics in (result.get("strategies") or {}).items():
            rows_by_strategy.setdefault(str(strategy), []).append(metrics)
    return {
        strategy: _mean_metric_rows(rows)
        for strategy, rows in sorted(rows_by_strategy.items())
    }


def _aggregate_deltas_for_strength(
    metrics: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float]]:
    active = metrics.get("sestina_active_pairwise", {})
    random_pairwise = metrics.get("pointwise_random_pairwise", {})
    pointwise = metrics.get("pointwise_only", {})
    return {
        "active_minus_pointwise": _metric_delta(active, pointwise),
        "active_minus_random_pairwise": _metric_delta(active, random_pairwise),
        "random_pairwise_minus_pointwise": _metric_delta(random_pairwise, pointwise),
    }


def _metric_delta(
    left: dict[str, float | int],
    right: dict[str, float | int],
) -> dict[str, float]:
    keys = ("recall_at_k", "precision_at_k", "ndcg_at_k", "average_precision")
    return {
        key: round(float(left.get(key, 0.0)) - float(right.get(key, 0.0)), 8)
        for key in keys
    }


def _diagnose_active_underperformance(
    *,
    aggregate_metrics: dict[str, dict[str, dict[str, float | int]]],
    aggregate_deltas: dict[str, dict[str, dict[str, float]]],
    coverage_totals: dict[str, int],
    coverage_denominator: int,
) -> dict[str, Any]:
    default = aggregate_deltas.get("2.5", {})
    active_vs_random = default.get("active_minus_random_pairwise", {})
    best_active = _best_metric(
        aggregate_metrics,
        strategy="sestina_active_pairwise",
        metric="ndcg_at_k",
    )
    best_random = _best_metric(
        aggregate_metrics,
        strategy="pointwise_random_pairwise",
        metric="ndcg_at_k",
    )
    candidate_outsider_rate = (
        coverage_totals.get("candidate_outsider_pairs", 0) / coverage_denominator
        if coverage_denominator
        else 0.0
    )
    primary_cause = "aggregation_weighting"
    if best_active["value"] <= best_random["value"] and candidate_outsider_rate == 0:
        primary_cause = "historical_schedule_coverage"
    elif best_active["value"] <= best_random["value"]:
        primary_cause = "schedule_or_judgment_mix"
    return {
        "default_strength_active_minus_random": active_vs_random,
        "best_active_ndcg": best_active,
        "best_random_ndcg": best_random,
        "historical_active_candidate_outsider_rate": round(
            candidate_outsider_rate,
            8,
        ),
        "historical_active_coverage_totals": coverage_totals,
        "primary_cause": primary_cause,
        "interpretation": (
            "The strength sweep does not remove the random-pairwise edge while "
            "the historical active schedule has no outsider coverage."
            if primary_cause == "historical_schedule_coverage"
            else "Pairwise weighting materially changes outcomes; inspect schedule "
            "coverage and judgment mix together."
        ),
    }


def _best_metric(
    aggregate_metrics: dict[str, dict[str, dict[str, float | int]]],
    *,
    strategy: str,
    metric: str,
) -> dict[str, float | str]:
    best_label = ""
    best_value = -1.0
    for label, metrics in aggregate_metrics.items():
        value = float(metrics.get(strategy, {}).get(metric, -1.0))
        if value > best_value:
            best_label = label
            best_value = value
    return {"strength": best_label, "value": round(best_value, 8)}


def _variant_conclusions(diagnosis: dict[str, Any]) -> list[str]:
    default_delta = diagnosis["default_strength_active_minus_random"]
    primary = diagnosis["primary_cause"]
    return [
        (
            "At pairwise_strength=2.5, historical active minus random pairwise "
            f"delta is recall {default_delta.get('recall_at_k', 0.0):+.6f}, "
            f"nDCG {default_delta.get('ndcg_at_k', 0.0):+.6f}, and AP "
            f"{default_delta.get('average_precision', 0.0):+.6f}."
        ),
        (
            "The strength sweep points to schedule coverage as the main issue."
            if primary == "historical_schedule_coverage"
            else "The strength sweep leaves aggregation weighting as a live factor."
        ),
        (
            "The revised scheduler preview should be validated with fresh "
            "scheduler-only pairwise calls because the old active artifacts do "
            "not cover sentinel/outsider pairs."
        ),
    ]


def _strength_label(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)
