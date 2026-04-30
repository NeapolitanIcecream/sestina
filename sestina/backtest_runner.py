from __future__ import annotations

import itertools
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from sestina.aggregation import aggregate
from sestina.backtest import Prediction, compare_strategies
from sestina.backtest_budget import estimate_from_config, load_config, write_report
from sestina.candidates import CandidateSelection, select_candidates
from sestina.diagnostics import DiagnosticRecorder, fingerprint, write_json_artifact
from sestina.models import (
    PairwiseComparison,
    PairwiseOrderMetadata,
    Paper,
    PointwiseAssessment,
    ScheduledPair,
)
from sestina.scheduler import PairwiseBudget, resolve_pairwise_budget, schedule_pairs

PhaseName = Literal["smoke", "pilot", "main", "audit", "all"]
CallKind = Literal["pointwise", "pairwise_active", "pairwise_random", "audit_pairwise"]

PROMPT_VERSION = "sestina-backtest-v1"
PAID_STATUS_VALUES = frozenset({"ok", "failed", "parse_error"})
PHASE_CHOICES = ("smoke", "pilot", "main", "audit", "all")


class BacktestRunnerError(RuntimeError):
    """Base class for guarded backtest runner failures."""


class PaidRunSafetyError(BacktestRunnerError):
    """Raised when a paid run is missing an explicit safety precondition."""


class ModelAvailabilityError(BacktestRunnerError):
    """Raised when a configured model cannot be verified against the endpoint."""


class DatasetManifestError(BacktestRunnerError):
    """Raised when the dataset manifest is missing required labeled buckets."""


@dataclass(frozen=True, slots=True)
class BudgetLimitExceeded(BacktestRunnerError):
    cap_usd: float
    projected_usd: float

    def __str__(self) -> str:
        return (
            f"projected paid backtest spend ${self.projected_usd:.6f} exceeds "
            f"budget cap ${self.cap_usd:.6f}"
        )


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    phase: str
    bucket: str | None
    model: str
    kind: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    status: str
    artifact_path: str
    created_at_unix: float
    prompt_version: str = PROMPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "bucket": self.bucket,
            "model": self.model,
            "kind": self.kind,
            "estimated_tokens": {
                "input": self.estimated_input_tokens,
                "output": self.estimated_output_tokens,
            },
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "status": self.status,
            "artifact_path": self.artifact_path,
            "prompt_version": self.prompt_version,
            "created_at_unix": round(self.created_at_unix, 3),
        }


@dataclass(slots=True)
class JsonlLedger:
    path: Path

    def existing_spend_usd(self) -> float:
        if not self.path.exists():
            return 0.0
        total = 0.0
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if str(entry.get("status")) in PAID_STATUS_VALUES:
                total += float(entry.get("estimated_cost_usd") or 0.0)
        return round(total, 6)

    def append(self, entry: LedgerEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")

    def guard_projected_spend(self, *, cap_usd: float, next_cost_usd: float) -> None:
        projected = self.existing_spend_usd() + next_cost_usd
        if projected > cap_usd:
            raise BudgetLimitExceeded(cap_usd=cap_usd, projected_usd=projected)


@dataclass(frozen=True, slots=True)
class CallEstimate:
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass(frozen=True, slots=True)
class BacktestBucket:
    name: str
    phase: str
    k: int
    papers: list[Paper]
    relevant_ids: set[str]
    baseline_scores: dict[str, float]
    source: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    path: Path
    payload: dict[str, Any]
    buckets: list[BacktestBucket]

    def buckets_for_phase(self, phase: str) -> list[BacktestBucket]:
        return [bucket for bucket in self.buckets if bucket.phase == phase]


UrlOpen = Callable[..., Any]


def validate_model_names(models: Iterable[str]) -> None:
    for model in sorted(set(models)):
        if "/" not in model:
            raise ModelAvailabilityError(
                "model names must include a provider prefix; "
                f"OpenAI-routed models should look like openai/{model}"
            )


def check_model_availability(
    *,
    base_url: str,
    api_key: str,
    models: Iterable[str],
    timeout_seconds: float = 30.0,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> dict[str, Any]:
    requested = sorted(set(models))
    validate_model_names(requested)
    if not api_key or not base_url:
        raise ModelAvailabilityError(
            "SESTINA_LLM_API_KEY and SESTINA_LLM_BASE_URL are required"
        )

    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ModelAvailabilityError(
            f"model availability check failed: {type(exc).__name__}"
        ) from exc

    available = _extract_model_ids(payload)
    missing = [model for model in requested if model not in available]
    result = {
        "status": "available" if not missing else "missing",
        "requested_models": requested,
        "missing_models": missing,
        "available_requested_models": [model for model in requested if model in available],
    }
    if missing:
        raise ModelAvailabilityError(
            "configured models are not available from endpoint: "
            + ", ".join(missing)
        )
    return result


def load_dataset_manifest(path: Path) -> DatasetManifest:
    payload = json.loads(path.read_text())
    buckets = [_bucket_from_manifest(raw) for raw in payload.get("buckets", [])]
    if not buckets:
        raise DatasetManifestError("dataset manifest does not contain buckets")
    for bucket in buckets:
        if not bucket.relevant_ids:
            raise DatasetManifestError(
                f"bucket {bucket.name} has no positive good_paper labels"
            )
        if len(bucket.papers) < bucket.k:
            raise DatasetManifestError(
                f"bucket {bucket.name} has fewer papers than target K"
            )
        if len(bucket.relevant_ids) < bucket.k:
            raise DatasetManifestError(
                f"bucket {bucket.name} has fewer than K positive labels"
            )
    return DatasetManifest(path=path, payload=payload, buckets=buckets)


@dataclass(slots=True)
class BacktestRunner:
    config_path: Path
    phase: PhaseName
    max_usd: float
    artifact_dir: Path
    ledger_path: Path
    manifest_path: Path | None = None
    confirm_paid: bool = False
    seed: int = 17
    timeout_seconds: float = 60.0
    urlopen: UrlOpen = urllib.request.urlopen

    def run(self) -> dict[str, Any]:
        if self.phase not in PHASE_CHOICES:
            raise ValueError(f"unknown backtest phase {self.phase!r}")
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        ledger = JsonlLedger(self.ledger_path)

        raw_config = load_config(self.config_path)
        selected_config = _config_for_phase(raw_config, phase=self.phase)
        estimate = estimate_from_config(
            selected_config,
            max_usd=self.max_usd,
            validate_budget=True,
        )
        estimate_path = self.artifact_dir / f"estimate-{self.phase}.json"
        write_report(estimate_path, estimate)

        summary: dict[str, Any] = {
            "artifact_type": "sestina-backtest-run-summary",
            "phase": self.phase,
            "dry_run": not self.confirm_paid,
            "budget_cap_usd": self.max_usd,
            "estimate_path": str(estimate_path),
            "ledger_path": str(self.ledger_path),
            "artifact_dir": str(self.artifact_dir),
            "estimated_spend_usd": estimate["totals"]["cost_usd"],
            "actual_ledger_spend_usd": ledger.existing_spend_usd(),
            "paid_calls_made": _ledger_call_count(ledger),
            "new_ledger_entries_this_invocation": 0,
            "model_availability": {"status": "not_checked_dry_run"},
            "bucket_results": [],
        }
        summary.update(_ledger_stats(ledger))

        if not self.confirm_paid:
            summary_path = self.artifact_dir / f"summary-{self.phase}.json"
            write_json_artifact(summary_path, summary)
            return {**summary, "summary_path": str(summary_path)}

        self._validate_paid_preconditions(selected_config, estimate=estimate)
        manifest = load_dataset_manifest(_required_path(self.manifest_path, "manifest"))
        selected_phases = _selected_phase_names(selected_config)
        if not any(manifest.buckets_for_phase(phase) for phase in selected_phases):
            raise DatasetManifestError(
                "dataset manifest has no labeled buckets for selected phase(s): "
                + ", ".join(selected_phases)
            )

        api_key = os.environ.get("SESTINA_LLM_API_KEY")
        base_url = os.environ.get("SESTINA_LLM_BASE_URL")
        models = _configured_models(selected_config)
        model_availability = check_model_availability(
            base_url=base_url or "",
            api_key=api_key or "",
            models=models,
            timeout_seconds=self.timeout_seconds,
            urlopen=self.urlopen,
        )
        summary["model_availability"] = model_availability
        summary["manifest_path"] = str(manifest.path)

        paid_calls_before = _ledger_call_count(ledger)
        for phase_config in selected_config.get("phases", []):
            phase_name = str(phase_config["name"])
            if phase_name == "audit":
                # Audit prompts require sampled completed results. The initial
                # executor is intentionally smoke-first and does not synthesize
                # audit calls from an empty sample.
                continue
            for bucket in manifest.buckets_for_phase(phase_name):
                existing_result = self._load_existing_bucket_result(
                    bucket,
                    phase=phase_name,
                )
                if existing_result is not None:
                    summary["bucket_results"].append(existing_result)
                    continue
                result = self._run_bucket(
                    bucket,
                    phase_config=phase_config,
                    ledger=ledger,
                    api_key=api_key or "",
                    base_url=base_url or "",
                )
                summary["bucket_results"].append(result)

        paid_calls_after = _ledger_call_count(ledger)
        summary["paid_calls_made"] = paid_calls_after
        summary["new_ledger_entries_this_invocation"] = (
            paid_calls_after - paid_calls_before
        )
        summary.update(_ledger_stats(ledger))
        summary_path = self.artifact_dir / f"summary-{self.phase}.json"
        write_json_artifact(summary_path, summary)
        return {**summary, "summary_path": str(summary_path)}

    def _load_existing_bucket_result(
        self,
        bucket: BacktestBucket,
        *,
        phase: str,
    ) -> dict[str, Any] | None:
        result_path = (
            self.artifact_dir
            / phase
            / _safe_name(bucket.name)
            / "bucket-result.json"
        )
        if not result_path.exists():
            return None
        payload = json.loads(result_path.read_text())
        return {**payload, "artifact_path": str(result_path), "reused_artifact": True}

    def _validate_paid_preconditions(
        self,
        config: dict[str, Any],
        *,
        estimate: dict[str, Any],
    ) -> None:
        if self.max_usd <= 0:
            raise PaidRunSafetyError("--max-usd must be greater than zero")
        if self.max_usd > 100:
            raise PaidRunSafetyError("--max-usd must not exceed the USD 100 hard cap")
        if not self.ledger_path:
            raise PaidRunSafetyError("--ledger is required for paid runs")
        if not self.artifact_dir:
            raise PaidRunSafetyError("--artifact-dir is required for paid runs")
        if self.manifest_path is None:
            raise PaidRunSafetyError("--manifest is required for paid runs")
        if estimate["totals"]["cost_usd"] > self.max_usd:
            raise PaidRunSafetyError("dry-run estimate exceeds --max-usd")
        for phase in estimate.get("phases", []):
            if phase.get("within_phase_allocation") is False:
                raise PaidRunSafetyError(
                    f"phase {phase['name']} estimate exceeds its allocation"
                )
        if any(
            ablation.get("extra_pairwise_calls", 0) != 0
            for phase in estimate.get("phases", [])
            for bucket in phase.get("buckets", [])
            for ablation in [bucket.get("pairwise_budget_ablation") or {}]
        ):
            raise PaidRunSafetyError("pairwise budget ablation must use prefix reuse")
        validate_model_names(_configured_models(config))

    def _run_bucket(
        self,
        bucket: BacktestBucket,
        *,
        phase_config: dict[str, Any],
        ledger: JsonlLedger,
        api_key: str,
        base_url: str,
    ) -> dict[str, Any]:
        phase = str(phase_config["name"])
        pointwise_model = str(phase_config["pointwise_model"])
        pairwise_model = str(phase_config["pairwise_model"])
        token_assumptions = _normalize_token_assumptions_from_config(
            load_config(self.config_path)
        )
        rates = _normalize_rates_from_config(load_config(self.config_path))
        bucket_dir = self.artifact_dir / phase / _safe_name(bucket.name)
        bucket_dir.mkdir(parents=True, exist_ok=True)

        pointwise_papers = []
        for index, paper in enumerate(bucket.papers, start=1):
            assessment = self._judge_pointwise(
                paper,
                phase=phase,
                bucket=bucket.name,
                index=index,
                model=pointwise_model,
                token_assumptions=token_assumptions,
                rates=rates,
                ledger=ledger,
                bucket_dir=bucket_dir,
                api_key=api_key,
                base_url=base_url,
            )
            pointwise_papers.append(
                Paper(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    abstract=paper.abstract,
                    pointwise=assessment,
                    metadata=paper.metadata,
                )
            )

        diagnostics = DiagnosticRecorder()
        selection = select_candidates(
            pointwise_papers,
            k=bucket.k,
            diagnostics=diagnostics,
        )
        budget = resolve_pairwise_budget(
            n=len(pointwise_papers),
            candidate_size=len(selection.candidate_ids),
            diagnostics=diagnostics,
        )
        active_schedule = schedule_pairs(
            pointwise_papers,
            candidate_selection=selection,
            k=bucket.k,
            budget=budget,
            seed=self.seed,
            diagnostics=diagnostics,
        )
        random_schedule = _random_pair_schedule(
            selection,
            budget=budget,
            seed=self.seed + 7919,
        )
        active_comparisons = self._judge_schedule(
            active_schedule.pairs,
            pointwise_papers,
            phase=phase,
            bucket=bucket.name,
            strategy="active",
            model=pairwise_model,
            token_assumptions=token_assumptions,
            rates=rates,
            ledger=ledger,
            bucket_dir=bucket_dir,
            api_key=api_key,
            base_url=base_url,
        )
        random_comparisons = self._judge_schedule(
            random_schedule,
            pointwise_papers,
            phase=phase,
            bucket=bucket.name,
            strategy="random",
            model=pairwise_model,
            token_assumptions=token_assumptions,
            rates=rates,
            ledger=ledger,
            bucket_dir=bucket_dir,
            api_key=api_key,
            base_url=base_url,
        )

        strategy_predictions = {
            "random": _random_predictions(pointwise_papers, seed=self.seed),
            "semantic_baseline": [
                Prediction(paper_id, bucket.baseline_scores.get(paper_id, 0.0))
                for paper_id in bucket.baseline_scores
            ],
            "pointwise_only": [
                Prediction(paper.paper_id, paper.pointwise.good_probability)
                for paper in pointwise_papers
            ],
            "pointwise_random_pairwise": _aggregate_predictions(
                pointwise_papers,
                random_comparisons,
            ),
            "sestina_active_pairwise": _aggregate_predictions(
                pointwise_papers,
                active_comparisons,
            ),
        }
        metrics = compare_strategies(
            strategy_predictions,
            relevant_ids=bucket.relevant_ids,
            k=bucket.k,
        )
        ablation = _pairwise_ablation_metrics(
            pointwise_papers,
            relevant_ids=bucket.relevant_ids,
            k=bucket.k,
            pairwise_budget=budget.budget,
            active_comparisons=active_comparisons,
            random_comparisons=random_comparisons,
            configured_points=_ablation_points(load_config(self.config_path)),
        )
        result_payload = {
            "artifact_type": "sestina-backtest-bucket-result",
            "phase": phase,
            "bucket": bucket.name,
            "k": bucket.k,
            "papers_total": len(pointwise_papers),
            "positive_labels_total": len(bucket.relevant_ids),
            "strategies": {
                name: metric.to_dict() for name, metric in metrics.items()
            },
            "pairwise_budget": budget.to_dict(),
            "pairwise_budget_ablation": ablation,
            "calls": {
                "pointwise": len(pointwise_papers),
                "pairwise_active": len(active_comparisons),
                "pairwise_random": len(random_comparisons),
            },
            "diagnostics": diagnostics.to_dict(),
        }
        result_path = bucket_dir / "bucket-result.json"
        write_json_artifact(result_path, result_payload)
        return {**result_payload, "artifact_path": str(result_path)}

    def _judge_pointwise(
        self,
        paper: Paper,
        *,
        phase: str,
        bucket: str,
        index: int,
        model: str,
        token_assumptions: dict[str, dict[str, int]],
        rates: dict[str, dict[str, float]],
        ledger: JsonlLedger,
        bucket_dir: Path,
        api_key: str,
        base_url: str,
    ) -> PointwiseAssessment:
        estimate = _call_estimate("pointwise", model, token_assumptions, rates)
        artifact_path = bucket_dir / "calls" / f"{index:04d}-pointwise-{fingerprint(paper.paper_id)}.json"
        cached = _load_ok_call_artifact(artifact_path)
        if cached is not None:
            return PointwiseAssessment.from_dict(cached["response"])
        ledger.guard_projected_spend(cap_usd=self.max_usd, next_cost_usd=estimate.cost_usd)
        payload = _pointwise_payload(model=model, paper=paper)
        try:
            response = _chat_json(
                base_url=base_url,
                api_key=api_key,
                payload=payload,
                timeout_seconds=self.timeout_seconds,
                urlopen=self.urlopen,
            )
            assessment = PointwiseAssessment.from_dict(response)
            status = "ok"
            artifact = _call_artifact(
                phase=phase,
                bucket=bucket,
                model=model,
                kind="pointwise",
                estimate=estimate,
                status=status,
                response=response,
                subject={"paper_id": paper.paper_id},
            )
        except json.JSONDecodeError as exc:
            status = "parse_error"
            artifact = _call_artifact(
                phase=phase,
                bucket=bucket,
                model=model,
                kind="pointwise",
                estimate=estimate,
                status=status,
                error=exc,
                subject={"paper_id": paper.paper_id},
            )
            write_json_artifact(artifact_path, artifact)
            ledger.append(
                _ledger_entry(
                    phase=phase,
                    bucket=bucket,
                    model=model,
                    kind="pointwise",
                    estimate=estimate,
                    status=status,
                    artifact_path=artifact_path,
                )
            )
            raise
        except Exception as exc:
            status = "failed"
            artifact = _call_artifact(
                phase=phase,
                bucket=bucket,
                model=model,
                kind="pointwise",
                estimate=estimate,
                status=status,
                error=exc,
                subject={"paper_id": paper.paper_id},
            )
            write_json_artifact(artifact_path, artifact)
            ledger.append(
                _ledger_entry(
                    phase=phase,
                    bucket=bucket,
                    model=model,
                    kind="pointwise",
                    estimate=estimate,
                    status=status,
                    artifact_path=artifact_path,
                )
            )
            raise

        write_json_artifact(artifact_path, artifact)
        ledger.append(
            _ledger_entry(
                phase=phase,
                bucket=bucket,
                model=model,
                kind="pointwise",
                estimate=estimate,
                status=status,
                artifact_path=artifact_path,
            )
        )
        return assessment

    def _judge_schedule(
        self,
        schedule: list[ScheduledPair],
        papers: list[Paper],
        *,
        phase: str,
        bucket: str,
        strategy: Literal["active", "random"],
        model: str,
        token_assumptions: dict[str, dict[str, int]],
        rates: dict[str, dict[str, float]],
        ledger: JsonlLedger,
        bucket_dir: Path,
        api_key: str,
        base_url: str,
    ) -> list[PairwiseComparison]:
        paper_by_id = {paper.paper_id: paper for paper in papers}
        comparisons: list[PairwiseComparison] = []
        kind: CallKind = (
            "pairwise_active" if strategy == "active" else "pairwise_random"
        )
        for index, pair in enumerate(schedule, start=1):
            estimate = _call_estimate("pairwise", model, token_assumptions, rates)
            artifact_path = (
                bucket_dir
                / "calls"
                / f"{index:04d}-{kind}-{fingerprint(pair.left_id + ':' + pair.right_id)}.json"
            )
            cached = _load_ok_call_artifact(artifact_path)
            if cached is not None:
                comparisons.append(
                    _comparison_from_pairwise_response(pair, cached["response"])
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
                    urlopen=self.urlopen,
                )
                comparison = _comparison_from_pairwise_response(pair, response)
                status = "ok"
                artifact = _call_artifact(
                    phase=phase,
                    bucket=bucket,
                    model=model,
                    kind=kind,
                    estimate=estimate,
                    status=status,
                    response=response,
                    subject={"left_id": pair.left_id, "right_id": pair.right_id},
                )
            except json.JSONDecodeError as exc:
                status = "parse_error"
                artifact = _call_artifact(
                    phase=phase,
                    bucket=bucket,
                    model=model,
                    kind=kind,
                    estimate=estimate,
                    status=status,
                    error=exc,
                    subject={"left_id": pair.left_id, "right_id": pair.right_id},
                )
                write_json_artifact(artifact_path, artifact)
                ledger.append(
                    _ledger_entry(
                        phase=phase,
                        bucket=bucket,
                        model=model,
                        kind=kind,
                        estimate=estimate,
                        status=status,
                        artifact_path=artifact_path,
                    )
                )
                raise
            except Exception as exc:
                status = "failed"
                artifact = _call_artifact(
                    phase=phase,
                    bucket=bucket,
                    model=model,
                    kind=kind,
                    estimate=estimate,
                    status=status,
                    error=exc,
                    subject={"left_id": pair.left_id, "right_id": pair.right_id},
                )
                write_json_artifact(artifact_path, artifact)
                ledger.append(
                    _ledger_entry(
                        phase=phase,
                        bucket=bucket,
                        model=model,
                        kind=kind,
                        estimate=estimate,
                        status=status,
                        artifact_path=artifact_path,
                    )
                )
                raise

            write_json_artifact(artifact_path, artifact)
            ledger.append(
                _ledger_entry(
                    phase=phase,
                    bucket=bucket,
                    model=model,
                    kind=kind,
                    estimate=estimate,
                    status=status,
                    artifact_path=artifact_path,
                )
            )
            comparisons.append(comparison)
        return comparisons


def _bucket_from_manifest(raw: dict[str, Any]) -> BacktestBucket:
    papers: list[Paper] = []
    relevant_ids: set[str] = set()
    baseline_scores: dict[str, float] = {}
    for item in raw.get("papers", []):
        paper_id = str(item.get("paper_id") or item.get("id") or "")
        if not paper_id:
            raise DatasetManifestError("paper is missing paper_id")
        baseline_score = float(item.get("baseline_score", 0.5))
        labels = item.get("labels") or {}
        good_paper = bool(item.get("good_paper", labels.get("good_paper", False)))
        if good_paper:
            relevant_ids.add(paper_id)
        baseline_scores[paper_id] = baseline_score
        papers.append(
            Paper(
                paper_id=paper_id,
                title=str(item.get("title") or ""),
                abstract=str(item.get("abstract") or item.get("summary") or ""),
                pointwise=PointwiseAssessment(
                    good_probability=max(0.001, min(0.999, baseline_score)),
                    uncertainty=float(item.get("uncertainty", 0.5)),
                    summary=str(item.get("baseline_summary") or ""),
                    reasons=[str(reason) for reason in item.get("baseline_reasons", [])],
                    rubric_scores={
                        str(key): float(value)
                        for key, value in (item.get("rubric_scores") or {}).items()
                        if isinstance(value, int | float)
                    },
                ),
                metadata=dict(item.get("metadata") or {}),
            )
        )
    return BacktestBucket(
        name=str(raw["name"]),
        phase=str(raw.get("phase", "smoke")),
        k=int(raw["k"]),
        papers=papers,
        relevant_ids=relevant_ids,
        baseline_scores=baseline_scores,
        source=dict(raw.get("source") or {}),
    )


def _chat_json(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    urlopen: UrlOpen,
) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {type(exc).__name__}") from exc
    content = response_payload["choices"][0]["message"]["content"]
    return json.loads(content)


def _pointwise_payload(*, model: str, paper: Paper) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Assess whether a paper is a good candidate for a curated "
                    "research discovery bucket. Return strict JSON with "
                    "good_probability, uncertainty, summary, reasons, and "
                    "rubric_scores. Use only the supplied title, abstract, "
                    "summary, and metadata."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "abstract_or_summary": paper.abstract[:6000],
                        "metadata": _bounded_metadata(paper.metadata),
                    },
                    sort_keys=True,
                ),
            },
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }


def _pairwise_payload(
    *,
    model: str,
    pair: ScheduledPair,
    papers: dict[str, Paper],
) -> dict[str, Any]:
    shown_first_id = pair.order.shown_first_id or pair.left_id
    shown_second_id = pair.order.shown_second_id or pair.right_id
    left = papers[shown_first_id]
    right = papers[shown_second_id]
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Compare two candidate papers for a curated research "
                    "discovery bucket. Return strict JSON with winner, "
                    "soft_probability, confidence, and reasons. The winner "
                    "must be left, right, tie, or uncertain."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "left": _paper_pair_payload(left),
                        "right": _paper_pair_payload(right),
                        "pair_purpose": pair.purpose,
                    },
                    sort_keys=True,
                ),
            },
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }


def _paper_pair_payload(paper: Paper) -> dict[str, Any]:
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "pointwise_good_probability": paper.pointwise.good_probability,
        "pointwise_summary": paper.pointwise.summary,
        "pointwise_reasons": paper.pointwise.reasons[:4],
        "rubric_scores": paper.pointwise.rubric_scores,
        "metadata": _bounded_metadata(paper.metadata),
    }


def _comparison_from_pairwise_response(
    pair: ScheduledPair,
    response: dict[str, Any],
) -> PairwiseComparison:
    shown_first_id = pair.order.shown_first_id or pair.left_id
    shown_second_id = pair.order.shown_second_id or pair.right_id
    shown = PairwiseComparison.from_dict(
        {
            "left_id": shown_first_id,
            "right_id": shown_second_id,
            "winner": response.get("winner", "uncertain"),
            "soft_probability": response.get("soft_probability"),
            "confidence": response.get("confidence", 0.5),
            "reasons": response.get("reasons", []),
            "order": pair.order.to_dict(),
            "metadata": {
                "scheduled_pair_priority": pair.priority,
                "scheduled_pair_purpose": pair.purpose,
            },
        }
    )
    if shown.winner == "left":
        canonical_winner = "left" if shown_first_id == pair.left_id else "right"
    elif shown.winner == "right":
        canonical_winner = "left" if shown_second_id == pair.left_id else "right"
    else:
        canonical_winner = shown.winner
    return PairwiseComparison(
        left_id=pair.left_id,
        right_id=pair.right_id,
        winner=canonical_winner,  # type: ignore[arg-type]
        soft_probability=shown.soft_probability,
        confidence=shown.confidence,
        reasons=shown.reasons,
        order=pair.order,
        metadata={
            **shown.metadata,
            "judge_presented_left_id": shown_first_id,
            "judge_presented_right_id": shown_second_id,
            "raw_position_winner": shown.winner,
        },
    )


def _extract_model_ids(payload: dict[str, Any]) -> set[str]:
    data = payload.get("data")
    if not isinstance(data, list):
        return set()
    ids = set()
    for item in data:
        if isinstance(item, dict) and item.get("id") is not None:
            ids.add(str(item["id"]))
    return ids


def _config_for_phase(config: dict[str, Any], *, phase: str) -> dict[str, Any]:
    selected = dict(config)
    phases = list(config.get("phases", []))
    if phase != "all":
        phases = [item for item in phases if str(item.get("name")) == phase]
    else:
        phases = [item for item in phases if str(item.get("name")) != "reserve"]
    if not phases:
        raise ValueError(f"config has no phase named {phase!r}")
    selected["phases"] = phases
    return selected


def _selected_phase_names(config: dict[str, Any]) -> list[str]:
    return [str(phase["name"]) for phase in config.get("phases", [])]


def _configured_models(config: dict[str, Any]) -> list[str]:
    models = set()
    for phase in config.get("phases", []):
        for key in ("pointwise_model", "pairwise_model", "audit_model"):
            if phase.get(key):
                models.add(str(phase[key]))
    return sorted(models)


def _required_path(path: Path | None, name: str) -> Path:
    if path is None:
        raise PaidRunSafetyError(f"{name} path is required")
    return path


def _normalize_token_assumptions_from_config(
    config: dict[str, Any],
) -> dict[str, dict[str, int]]:
    defaults = {
        "pointwise": {"input_tokens_per_call": 900, "output_tokens_per_call": 220},
        "pairwise": {"input_tokens_per_call": 1500, "output_tokens_per_call": 180},
        "audit_pairwise": {
            "input_tokens_per_call": 1500,
            "output_tokens_per_call": 220,
        },
    }
    for key, value in (config.get("token_assumptions") or {}).items():
        defaults[str(key)] = {
            "input_tokens_per_call": int(value["input_tokens_per_call"]),
            "output_tokens_per_call": int(value["output_tokens_per_call"]),
        }
    return defaults


def _normalize_rates_from_config(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    rates = {}
    for model, raw in (config.get("rate_card") or {}).items():
        rates[str(model)] = {
            "input_usd_per_1m_tokens": float(raw["input_usd_per_1m_tokens"]),
            "output_usd_per_1m_tokens": float(raw["output_usd_per_1m_tokens"]),
            "discount_multiplier": float(raw.get("discount_multiplier", 1.0)),
        }
    return rates


def _call_estimate(
    kind: Literal["pointwise", "pairwise", "audit_pairwise"],
    model: str,
    token_assumptions: dict[str, dict[str, int]],
    rates: dict[str, dict[str, float]],
) -> CallEstimate:
    assumption = token_assumptions[kind]
    rate = rates[model]
    input_tokens = assumption["input_tokens_per_call"]
    output_tokens = assumption["output_tokens_per_call"]
    cost = (
        ((input_tokens / 1_000_000) * rate["input_usd_per_1m_tokens"])
        + ((output_tokens / 1_000_000) * rate["output_usd_per_1m_tokens"])
    ) * rate["discount_multiplier"]
    return CallEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost, 6),
    )


def _ledger_entry(
    *,
    phase: str,
    bucket: str,
    model: str,
    kind: str,
    estimate: CallEstimate,
    status: str,
    artifact_path: Path,
) -> LedgerEntry:
    return LedgerEntry(
        phase=phase,
        bucket=bucket,
        model=model,
        kind=kind,
        estimated_input_tokens=estimate.input_tokens,
        estimated_output_tokens=estimate.output_tokens,
        estimated_cost_usd=estimate.cost_usd,
        status=status,
        artifact_path=str(artifact_path),
        created_at_unix=time.time(),
    )


def _call_artifact(
    *,
    phase: str,
    bucket: str,
    model: str,
    kind: str,
    estimate: CallEstimate,
    status: str,
    subject: dict[str, Any],
    response: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    artifact = {
        "artifact_type": "sestina-backtest-call",
        "phase": phase,
        "bucket": bucket,
        "model": model,
        "kind": kind,
        "prompt_version": PROMPT_VERSION,
        "estimated_tokens": {
            "input": estimate.input_tokens,
            "output": estimate.output_tokens,
        },
        "estimated_cost_usd": estimate.cost_usd,
        "status": status,
        "subject": subject,
    }
    if response is not None:
        artifact["response"] = response
    if error is not None:
        artifact["error"] = {
            "type": type(error).__name__,
            "message": str(error)[:220],
        }
    return artifact


def _random_pair_schedule(
    selection: CandidateSelection,
    *,
    budget: PairwiseBudget,
    seed: int,
) -> list[ScheduledPair]:
    candidates = list(selection.candidate_ids)
    pairs = list(itertools.combinations(candidates, 2))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    scheduled = []
    for index, (left_id, right_id) in enumerate(pairs[: budget.budget]):
        if rng.random() < 0.5:
            shown_first_id, shown_second_id = left_id, right_id
        else:
            shown_first_id, shown_second_id = right_id, left_id
        scheduled.append(
            ScheduledPair(
                left_id=left_id,
                right_id=right_id,
                priority=0.0,
                purpose="random_pairwise_control",
                order=PairwiseOrderMetadata(
                    shown_first_id=shown_first_id,
                    shown_second_id=shown_second_id,
                    randomized=True,
                    seed=seed,
                    position_bias_audit=(index % 5 == 0),
                ),
            )
        )
    return scheduled


def _aggregate_predictions(
    papers: list[Paper],
    comparisons: list[PairwiseComparison],
) -> list[Prediction]:
    aggregation = aggregate(papers, comparisons)
    return [
        Prediction(paper_id, estimate.posterior_good_probability)
        for paper_id, estimate in aggregation.estimates.items()
    ]


def _random_predictions(papers: list[Paper], *, seed: int) -> list[Prediction]:
    rng = random.Random(seed)
    return [Prediction(paper.paper_id, rng.random()) for paper in papers]


def _pairwise_ablation_metrics(
    papers: list[Paper],
    *,
    relevant_ids: set[str],
    k: int,
    pairwise_budget: int,
    active_comparisons: list[PairwiseComparison],
    random_comparisons: list[PairwiseComparison],
    configured_points: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for label in configured_points:
        prefix = min(pairwise_budget, max(0, _resolve_ablation_point(label, len(papers), k, pairwise_budget)))
        strategies = {
            "pointwise_random_pairwise": _aggregate_predictions(
                papers,
                random_comparisons[:prefix],
            ),
            "sestina_active_pairwise": _aggregate_predictions(
                papers,
                active_comparisons[:prefix],
            ),
        }
        metrics = compare_strategies(strategies, relevant_ids=relevant_ids, k=k)
        for strategy, metric in metrics.items():
            rows.append(
                {
                    "label": label,
                    "resolved_pairwise_calls_per_strategy": prefix,
                    "strategy": strategy,
                    **metric.to_dict(),
                }
            )
    return rows


def _resolve_ablation_point(label: str, n: int, k: int, pairwise_budget: int) -> int:
    expression = label.replace(" ", "").lower()
    if expression == "0":
        return 0
    if expression == "k":
        return k
    if expression == "k+sqrt(n)":
        return math.ceil(k + math.sqrt(n))
    if expression in {"b_pair", "bpair"}:
        return pairwise_budget
    return int(expression)


def _ablation_points(config: dict[str, Any]) -> list[str]:
    raw = config.get("pairwise_budget_ablation") or {}
    if not raw.get("enabled", False):
        return ["B_pair"]
    return [str(item) for item in raw.get("points", ["B_pair"])]


def _bounded_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = {}
    for key, value in metadata.items():
        if key in {"api_key", "token", "secret", "full_text"}:
            continue
        if isinstance(value, str):
            allowed[key] = value[:500]
        elif isinstance(value, int | float | bool) or value is None:
            allowed[key] = value
        elif isinstance(value, list):
            allowed[key] = value[:20]
        elif isinstance(value, dict):
            allowed[key] = {str(k): str(v)[:200] for k, v in list(value.items())[:20]}
    return allowed


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
    return safe[:120].strip("-") or "bucket"


def _ledger_call_count(ledger: JsonlLedger) -> int:
    if not ledger.path.exists():
        return 0
    return sum(1 for line in ledger.path.read_text().splitlines() if line.strip())


def _ledger_stats(ledger: JsonlLedger) -> dict[str, Any]:
    rows = []
    if ledger.path.exists():
        rows = [
            json.loads(line)
            for line in ledger.path.read_text().splitlines()
            if line.strip()
        ]
    by_kind: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for row in rows:
        kind = str(row.get("kind") or "unknown")
        status = str(row.get("status") or "unknown")
        by_kind[kind] = by_kind.get(kind, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "actual_ledger_spend_usd": ledger.existing_spend_usd(),
        "ledger_entries_total": len(rows),
        "ledger_entries_by_kind": by_kind,
        "ledger_entries_by_status": by_status,
    }


def _load_ok_call_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if payload.get("status") != "ok" or not isinstance(payload.get("response"), dict):
        return None
    return payload
