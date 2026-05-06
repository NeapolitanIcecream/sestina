#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_posterior_decision_shrinkage import _mean_metric_rows  # noqa: E402
from scripts.analyze_random_control_gap import (  # noqa: E402
    oracle_bounds,
    pair_graph_diagnostics,
    positive_exposure_diagnostics,
    top_k_error_decomposition,
)
from scripts.analyze_random_variance_replication import (  # noqa: E402
    DEFAULT_SEEDS,
    KNOWN_PAID_SPEND_BEFORE_WORKFLOW_USD,
    POSTERIOR_STRATEGY,
    summarize_values,
)
from scripts.run_ci_partition_gate import (  # noqa: E402
    _manifest_label_lookup,
    _orient_cached_comparison,
    _pair_key,
    _pairwise_cache_dirs,
    load_cached_pairwise_labels,
)
from sestina.backtest import Prediction, compare_strategies  # noqa: E402
from sestina.backtest_budget import load_config  # noqa: E402
from sestina.backtest_runner import (  # noqa: E402
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
    _normalize_rates_from_config,
    _normalize_token_assumptions_from_config,
    _pairwise_payload,
    _random_pair_schedule,
    _safe_name,
    check_model_availability,
    load_dataset_manifest,
    validate_model_names,
)
from sestina.diagnostics import DiagnosticRecorder, fingerprint, write_json_artifact  # noqa: E402
from sestina.evsi_scheduler import (  # noqa: E402
    EVSISchedulerConfig,
    posterior_top_k_predictions,
    schedule_exact_pool_random,
)
from sestina.models import PairwiseComparison, PairwiseOrderMetadata, ScheduledPair  # noqa: E402
from sestina.scheduler import resolve_pairwise_budget  # noqa: E402
from sestina.scheduler_followup import (  # noqa: E402
    legacy_select_candidates,
    load_pointwise_papers_from_artifacts,
)

ARTIFACT_TYPE = "sestina-full-random-variance-completion"
PLAN_ARTIFACT_TYPE = "sestina-full-random-variance-missing-label-plan"
SUMMARY_ARTIFACT_TYPE = "sestina-full-random-variance-labeling-summary"
ARM_HISTORICAL_FULL = "historical_random_full_schedule"
ARM_EXACT_FULL = "exact_pool_random_full_schedule"
PAIRWISE_COMPLETION_KIND = "pairwise_full_random_variance"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts" / "backtest-arxiv-full-random-variance-completion"
)
REQUIRED_PLAN_KEYS = {
    "artifact_type",
    "schema_version",
    "phase",
    "analysis_parameters",
    "guardrails",
    "totals",
    "buckets",
    "missing_pairs_by_bucket",
}
REQUIRED_FINAL_KEYS = {
    "artifact_type",
    "schema_version",
    "phase",
    "paid_calls_made",
    "paid_spend_usd",
    "planning_artifact_path",
    "ledger_reconciliation",
    "aggregate_metrics",
    "paired_deltas",
    "seed_results",
    "recommendation",
    "limitations",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plan and optionally complete full-schedule historical-random and "
            "exact-pool-random variance labels. The default mode is a no-paid "
            "missing-label plan. Paid mode is pairwise-only and guarded by a "
            "provider-prefixed model check, JSONL ledger, separate artifact dir, "
            "and --max-usd <= 5."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "experiments" / "arxiv_historical_pilot_budget_config.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-datasets"
            / "arxiv-historical-pilot-manifest.json"
        ),
    )
    parser.add_argument(
        "--source-artifact-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "backtest-arxiv-pilot-live",
    )
    parser.add_argument(
        "--prior-random-variance-artifact",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-random-variance-replication"
            / "random-variance-replication.json"
        ),
    )
    parser.add_argument("--phase", default="pilot")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="separate directory for plan, paid pairwise call artifacts, and final output",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "ledger.jsonl",
        help="JSONL ledger path for paid pairwise calls",
    )
    parser.add_argument(
        "--max-usd",
        type=float,
        default=5.00,
        help="per-workflow hard cap; paid runs require <= 5.00",
    )
    parser.add_argument(
        "--confirm-paid",
        action="store_true",
        help="after writing the no-paid plan, allow guarded paid pairwise calls",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="comma-separated replay seeds",
    )
    parser.add_argument("--scheduler-samples", type=int, default=1200)
    parser.add_argument("--posterior-samples", type=int, default=1200)
    parser.add_argument("--pairwise-strength", type=float, default=2.5)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument(
        "--pairwise-cache-artifact-dir",
        action="append",
        type=Path,
        default=None,
        help=(
            "Additional artifact directories to scan for cached pairwise labels. "
            "Defaults to all artifacts/backtest-arxiv-*-live directories; the "
            "completion artifact dir is always included for resumability."
        ),
    )
    args = parser.parse_args(argv)

    try:
        payload = run_full_random_variance_completion(
            config_path=args.config,
            manifest_path=args.manifest,
            source_artifact_dir=args.source_artifact_dir,
            prior_random_variance_artifact_path=args.prior_random_variance_artifact,
            artifact_dir=args.artifact_dir,
            ledger_path=args.ledger,
            phase=args.phase,
            max_usd=args.max_usd,
            confirm_paid=args.confirm_paid,
            seeds=_parse_seeds(args.seeds),
            scheduler_samples=args.scheduler_samples,
            posterior_samples=args.posterior_samples,
            pairwise_strength=args.pairwise_strength,
            bootstrap_samples=args.bootstrap_samples,
            pairwise_cache_artifact_dirs=args.pairwise_cache_artifact_dir,
        )
    except Exception as exc:  # noqa: BLE001
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        error_path = args.artifact_dir / f"error-{args.phase}.json"
        write_json_artifact(
            error_path,
            {
                "artifact_type": "sestina-full-random-variance-completion-error",
                "phase": args.phase,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "paid_run_requested": args.confirm_paid,
            },
        )
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        sys.stderr.write(f"error_artifact={error_path}\n")
        return 2

    sys.stdout.write(json.dumps(_stdout_summary(payload), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def run_full_random_variance_completion(
    *,
    config_path: Path,
    manifest_path: Path,
    source_artifact_dir: Path,
    prior_random_variance_artifact_path: Path,
    artifact_dir: Path,
    ledger_path: Path,
    phase: str = "pilot",
    max_usd: float = 5.00,
    confirm_paid: bool = False,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    scheduler_samples: int = 1200,
    posterior_samples: int = 1200,
    pairwise_strength: float = 2.5,
    bootstrap_samples: int = 4000,
    pairwise_cache_artifact_dirs: Sequence[Path] | None = None,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    initial_plan_path = artifact_dir / "initial-missing-label-plan.json"
    plan_path = artifact_dir / "missing-label-plan.json"
    final_path = artifact_dir / "full-random-variance-completion.json"
    summary_path = artifact_dir / f"labeling-summary-{phase}.json"

    initial_plan = build_missing_label_plan(
        config_path=config_path,
        manifest_path=manifest_path,
        source_artifact_dir=source_artifact_dir,
        artifact_dir=artifact_dir,
        ledger_path=ledger_path,
        output_path=initial_plan_path,
        phase=phase,
        max_usd=max_usd,
        seeds=seeds,
        scheduler_samples=scheduler_samples,
        posterior_samples=posterior_samples,
        pairwise_strength=pairwise_strength,
        bootstrap_samples=bootstrap_samples,
        pairwise_cache_artifact_dirs=pairwise_cache_artifact_dirs,
        include_completion_cache=False,
        existing_ledger_spend_override=0.0,
    )
    plan = build_missing_label_plan(
        config_path=config_path,
        manifest_path=manifest_path,
        source_artifact_dir=source_artifact_dir,
        artifact_dir=artifact_dir,
        ledger_path=ledger_path,
        output_path=plan_path,
        phase=phase,
        max_usd=max_usd,
        seeds=seeds,
        scheduler_samples=scheduler_samples,
        posterior_samples=posterior_samples,
        pairwise_strength=pairwise_strength,
        bootstrap_samples=bootstrap_samples,
        pairwise_cache_artifact_dirs=pairwise_cache_artifact_dirs,
        include_completion_cache=True,
    )
    if not confirm_paid:
        return {
            "artifact_type": PLAN_ARTIFACT_TYPE,
            "plan_path": str(plan_path),
            "initial_plan_path": str(initial_plan_path),
            "paid_calls_made": 0,
            "paid_spend_usd": 0.0,
            "plan": plan,
            "initial_plan": initial_plan,
        }

    if not plan["guardrails"]["paid_run_allowed_after_plan"]:
        return {
            "artifact_type": PLAN_ARTIFACT_TYPE,
            "plan_path": str(plan_path),
            "initial_plan_path": str(initial_plan_path),
            "paid_calls_made": 0,
            "paid_spend_usd": 0.0,
            "blocked": True,
            "blocking_reasons": plan["guardrails"]["blocking_reasons"],
            "plan": plan,
        }

    if (
        int((plan.get("totals") or {}).get("pairwise_calls", 0)) == 0
        and summary_path.exists()
    ):
        labeling_summary = json.loads(summary_path.read_text())
    else:
        labeling_summary = run_guarded_pairwise_label_completion(
            plan,
            config_path=config_path,
            manifest_path=manifest_path,
            source_artifact_dir=source_artifact_dir,
            artifact_dir=artifact_dir,
            ledger_path=ledger_path,
            summary_path=summary_path,
            phase=phase,
            max_usd=max_usd,
        )
    final = analyze_completed_full_random_variance(
        config_path=config_path,
        manifest_path=manifest_path,
        source_artifact_dir=source_artifact_dir,
        prior_random_variance_artifact_path=prior_random_variance_artifact_path,
        artifact_dir=artifact_dir,
        ledger_path=ledger_path,
        initial_plan_path=initial_plan_path,
        latest_plan_path=plan_path,
        output_path=final_path,
        phase=phase,
        max_usd=max_usd,
        seeds=seeds,
        scheduler_samples=scheduler_samples,
        posterior_samples=posterior_samples,
        pairwise_strength=pairwise_strength,
        bootstrap_samples=bootstrap_samples,
        pairwise_cache_artifact_dirs=pairwise_cache_artifact_dirs,
        labeling_summary=labeling_summary,
    )
    return {**final, "artifact_path": str(final_path)}


def build_missing_label_plan(
    *,
    config_path: Path,
    manifest_path: Path,
    source_artifact_dir: Path,
    artifact_dir: Path,
    ledger_path: Path,
    output_path: Path,
    phase: str,
    max_usd: float,
    seeds: Sequence[int],
    scheduler_samples: int,
    posterior_samples: int,
    pairwise_strength: float,
    bootstrap_samples: int,
    pairwise_cache_artifact_dirs: Sequence[Path] | None,
    include_completion_cache: bool = True,
    existing_ledger_spend_override: float | None = None,
) -> dict[str, Any]:
    raw_config = load_config(config_path)
    phase_config = _config_for_phase(raw_config, phase=phase)["phases"][0]
    pairwise_model = str(phase_config["pairwise_model"])
    validate_model_names([pairwise_model])
    token_assumptions = _normalize_token_assumptions_from_config(raw_config)
    rates = _normalize_rates_from_config(raw_config)
    pairwise_estimate = _call_estimate(
        "pairwise",
        pairwise_model,
        token_assumptions,
        rates,
    )
    manifest = load_dataset_manifest(manifest_path)
    buckets = manifest.buckets_for_phase(phase)
    if not buckets:
        raise ValueError(f"manifest has no buckets for phase {phase!r}")
    cache_dirs = _completion_cache_dirs(
        source_artifact_dir=source_artifact_dir,
        artifact_dir=artifact_dir,
        phase=phase,
        explicit_dirs=pairwise_cache_artifact_dirs,
        include_completion_cache=include_completion_cache,
    )
    ledger = JsonlLedger(ledger_path)

    bucket_rows = []
    missing_pairs_by_bucket: dict[str, list[dict[str, Any]]] = {}
    full_schedule_rows: dict[str, list[dict[str, Any]]] = {
        ARM_HISTORICAL_FULL: [],
        ARM_EXACT_FULL: [],
    }
    scheduled_total = 0
    cached_total = 0
    missing_schedule_total = 0
    unique_missing_total = 0

    for bucket in buckets:
        papers = load_pointwise_papers_from_artifacts(
            bucket,
            source_artifact_dir=source_artifact_dir,
            phase=phase,
        )
        selection = legacy_select_candidates(papers, k=bucket.k)
        budget = resolve_pairwise_budget(
            n=len(papers),
            candidate_size=len(selection.candidate_ids),
        )
        cached, cache_stats = load_cached_pairwise_labels(
            bucket.name,
            artifact_dirs=cache_dirs,
            phase=phase,
        )
        missing_pairs: dict[tuple[str, str], dict[str, Any]] = {}
        bucket_scheduled = 0
        bucket_cached = 0
        bucket_missing_scheduled = 0

        for seed in seeds:
            schedules = _full_random_schedules(
                papers,
                selection=selection,
                k=bucket.k,
                budget=budget,
                seed=int(seed),
                scheduler_samples=scheduler_samples,
                pairwise_strength=pairwise_strength,
            )
            for arm, schedule in schedules.items():
                row = _schedule_label_plan_row(
                    arm=arm,
                    bucket=bucket.name,
                    seed=int(seed),
                    schedule=schedule,
                    cached_keys=set(cached),
                )
                full_schedule_rows[arm].append(row)
                bucket_scheduled += int(row["scheduled_pairwise_total"])
                bucket_cached += int(row["cached_pairwise_labels_available"])
                bucket_missing_scheduled += int(row["missing_pairwise_labels"])
                for pair in schedule:
                    key = _pair_key(pair.left_id, pair.right_id)
                    if key in cached:
                        continue
                    item = missing_pairs.get(key)
                    if item is None:
                        label_index = len(missing_pairs) + 1
                        item = {
                            "label_index_within_bucket": label_index,
                            "bucket": bucket.name,
                            "pair_key": list(key),
                            "representative_scheduled_pair": pair.to_dict(),
                            "required_by": [],
                            "artifact_path": str(
                                _completion_pairwise_artifact_path(
                                    artifact_dir,
                                    phase=phase,
                                    bucket=bucket.name,
                                    label_index=label_index,
                                    pair=pair,
                                )
                            ),
                        }
                        missing_pairs[key] = item
                    item["required_by"].append(
                        {
                            "arm": arm,
                            "seed": int(seed),
                            "purpose": pair.purpose,
                            "shown_first_id": pair.order.shown_first_id,
                            "shown_second_id": pair.order.shown_second_id,
                        }
                    )

        missing_items = list(missing_pairs.values())
        missing_pairs_by_bucket[bucket.name] = missing_items
        bucket_unique_missing = len(missing_items)
        scheduled_total += bucket_scheduled
        cached_total += bucket_cached
        missing_schedule_total += bucket_missing_scheduled
        unique_missing_total += bucket_unique_missing
        bucket_rows.append(
            {
                "bucket": bucket.name,
                "k": bucket.k,
                "papers_total": len(papers),
                "positive_labels_total": len(bucket.relevant_ids),
                "pointwise_calls": 0,
                "pairwise_budget": budget.to_dict(),
                "pointwise_artifacts_loaded": len(papers),
                "pairwise_cache": cache_stats,
                "scheduled_pairwise_total": bucket_scheduled,
                "cached_pairwise_labels_available": bucket_cached,
                "missing_pairwise_labels_scheduled_occurrences": (
                    bucket_missing_scheduled
                ),
                "unique_missing_pairwise_labels": bucket_unique_missing,
                "estimated_completion_cost_usd": round(
                    bucket_unique_missing * pairwise_estimate.cost_usd,
                    6,
                ),
            }
        )

    estimated_spend = round(unique_missing_total * pairwise_estimate.cost_usd, 6)
    existing_ledger_spend = (
        round(float(existing_ledger_spend_override), 6)
        if existing_ledger_spend_override is not None
        else ledger.existing_spend_usd()
    )
    guardrails = _plan_guardrails(
        pairwise_model=pairwise_model,
        artifact_dir=artifact_dir,
        source_artifact_dir=source_artifact_dir,
        ledger_path=ledger_path,
        max_usd=max_usd,
        estimated_spend_usd=estimated_spend,
        existing_ledger_spend_usd=existing_ledger_spend,
    )
    payload = {
        "artifact_type": PLAN_ARTIFACT_TYPE,
        "schema_version": 1,
        "phase": phase,
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "source_artifact_dir": str(source_artifact_dir),
        "artifact_dir": str(artifact_dir),
        "ledger_path": str(ledger_path),
        "output_path": str(output_path),
        "dry_run": True,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "completion_cache_included": include_completion_cache,
        "ledger_spend_override_note": (
            "existing ledger spend overridden to reconstruct the initial "
            "pre-paid missing-label plan"
            if existing_ledger_spend_override is not None
            else None
        ),
        "known_paid_spend_before_workflow_usd": (
            KNOWN_PAID_SPEND_BEFORE_WORKFLOW_USD
        ),
        "pairwise_model": pairwise_model,
        "analysis_parameters": {
            "seeds": [int(seed) for seed in seeds],
            "seed_count": len(seeds),
            "scheduler_samples": scheduler_samples,
            "posterior_samples": posterior_samples,
            "pairwise_strength": pairwise_strength,
            "bootstrap_samples": bootstrap_samples,
            "confidence_level": 0.95,
            "arms": [ARM_HISTORICAL_FULL, ARM_EXACT_FULL],
        },
        "per_pairwise_call_estimate": {
            "input_tokens": pairwise_estimate.input_tokens,
            "output_tokens": pairwise_estimate.output_tokens,
            "cost_usd": pairwise_estimate.cost_usd,
        },
        "pairwise_cache_artifact_dirs": [str(path) for path in cache_dirs],
        "guardrails": guardrails,
        "totals": {
            "pointwise_calls": 0,
            "pairwise_calls": unique_missing_total,
            "pairwise_scheduled_occurrences": scheduled_total,
            "pairwise_cached_occurrences": cached_total,
            "pairwise_missing_scheduled_occurrences": missing_schedule_total,
            "unique_missing_pairwise_labels": unique_missing_total,
            "input_tokens": unique_missing_total * pairwise_estimate.input_tokens,
            "output_tokens": unique_missing_total * pairwise_estimate.output_tokens,
            "estimated_additional_spend_usd": estimated_spend,
            "existing_ledger_spend_usd": existing_ledger_spend,
            "projected_workflow_ledger_spend_usd": round(
                existing_ledger_spend + estimated_spend,
                6,
            ),
        },
        "full_schedule_cache_probe": {
            "arms": {
                arm: _aggregate_plan_rows(rows)
                for arm, rows in full_schedule_rows.items()
            },
            "seed_bucket_rows": {
                arm: _compact_plan_rows(rows)
                for arm, rows in full_schedule_rows.items()
            },
        },
        "buckets": bucket_rows,
        "missing_pairs_by_bucket": missing_pairs_by_bucket,
        "label_policy": {
            "pointwise_paid_calls_allowed": False,
            "pointwise_calls": 0,
            "pairwise_paid_calls_allowed": True,
            "pairwise_completion_kind": PAIRWISE_COMPLETION_KIND,
            "dedupe_key": "same-bucket canonical unordered pair key",
            "future_labels_used_for_scheduling": False,
            "future_labels_used_for_retrospective_metrics_only": True,
        },
    }
    validate_missing_label_plan_schema(payload)
    write_json_artifact(output_path, payload)
    return payload


def run_guarded_pairwise_label_completion(
    plan: dict[str, Any],
    *,
    config_path: Path,
    manifest_path: Path,
    source_artifact_dir: Path,
    artifact_dir: Path,
    ledger_path: Path,
    summary_path: Path,
    phase: str,
    max_usd: float,
    urlopen: Any = urllib.request.urlopen,
) -> dict[str, Any]:
    _validate_paid_plan_preconditions(
        plan,
        source_artifact_dir=source_artifact_dir,
        artifact_dir=artifact_dir,
        ledger_path=ledger_path,
        max_usd=max_usd,
    )
    raw_config = load_config(config_path)
    phase_config = _config_for_phase(raw_config, phase=phase)["phases"][0]
    model = str(phase_config["pairwise_model"])
    validate_model_names([model])
    estimate = _call_estimate(
        "pairwise",
        model,
        _normalize_token_assumptions_from_config(raw_config),
        _normalize_rates_from_config(raw_config),
    )
    api_key = os.environ.get("SESTINA_LLM_API_KEY") or ""
    base_url = os.environ.get("SESTINA_LLM_BASE_URL") or ""
    model_availability = check_model_availability(
        base_url=base_url,
        api_key=api_key,
        models=[model],
        urlopen=urlopen,
    )
    manifest = load_dataset_manifest(manifest_path)
    paper_lookup = {
        bucket.name: {
            paper.paper_id: paper
            for paper in load_pointwise_papers_from_artifacts(
                bucket,
                source_artifact_dir=source_artifact_dir,
                phase=phase,
            )
        }
        for bucket in manifest.buckets_for_phase(phase)
    }
    ledger = JsonlLedger(ledger_path)
    paid_calls_before = _ledger_call_count(ledger)
    new_ok = 0
    cached_ok = 0
    by_bucket: dict[str, dict[str, int]] = {}

    for bucket, items in sorted((plan.get("missing_pairs_by_bucket") or {}).items()):
        papers = paper_lookup[bucket]
        bucket_counts = by_bucket.setdefault(
            str(bucket),
            {"planned": 0, "cached_ok": 0, "new_ok": 0},
        )
        for item in items:
            bucket_counts["planned"] += 1
            pair = _scheduled_pair_from_dict(item["representative_scheduled_pair"])
            label_index = int(item["label_index_within_bucket"])
            artifact_path = _completion_pairwise_artifact_path(
                artifact_dir,
                phase=phase,
                bucket=str(bucket),
                label_index=label_index,
                pair=pair,
            )
            cached = _load_ok_json_artifact(artifact_path)
            if cached is not None:
                cached_ok += 1
                bucket_counts["cached_ok"] += 1
                continue
            artifact_path = _next_pairwise_attempt_path(artifact_path)
            ledger.guard_projected_spend(
                cap_usd=max_usd,
                next_cost_usd=estimate.cost_usd,
            )
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                response = _chat_json(
                    base_url=base_url,
                    api_key=api_key,
                    payload=_pairwise_payload(
                        model=model,
                        pair=pair,
                        papers=papers,
                    ),
                    timeout_seconds=60.0,
                    urlopen=urlopen,
                )
                comparison = _comparison_from_pairwise_response(pair, response)
                status = "ok"
                artifact = _call_artifact(
                    phase=phase,
                    bucket=str(bucket),
                    model=model,
                    kind=PAIRWISE_COMPLETION_KIND,
                    estimate=estimate,
                    status=status,
                    response=response,
                    subject={"left_id": pair.left_id, "right_id": pair.right_id},
                )
                artifact.update(
                    {
                        "workflow": "sestina-full-random-variance-completion",
                        "comparison": comparison.to_dict(),
                        "scheduled_pair": pair.to_dict(),
                        "completion_plan": {
                            "label_index_within_bucket": label_index,
                            "pair_key": item["pair_key"],
                            "required_by": item["required_by"],
                        },
                    }
                )
            except json.JSONDecodeError as exc:
                status = "parse_error"
                artifact = _call_artifact(
                    phase=phase,
                    bucket=str(bucket),
                    model=model,
                    kind=PAIRWISE_COMPLETION_KIND,
                    estimate=estimate,
                    status=status,
                    error=exc,
                    subject={"left_id": pair.left_id, "right_id": pair.right_id},
                )
                write_json_artifact(artifact_path, artifact)
                ledger.append(
                    _ledger_entry(
                        phase=phase,
                        bucket=str(bucket),
                        model=model,
                        kind=PAIRWISE_COMPLETION_KIND,
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
                    bucket=str(bucket),
                    model=model,
                    kind=PAIRWISE_COMPLETION_KIND,
                    estimate=estimate,
                    status=status,
                    error=exc,
                    subject={"left_id": pair.left_id, "right_id": pair.right_id},
                )
                write_json_artifact(artifact_path, artifact)
                ledger.append(
                    _ledger_entry(
                        phase=phase,
                        bucket=str(bucket),
                        model=model,
                        kind=PAIRWISE_COMPLETION_KIND,
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
                    bucket=str(bucket),
                    model=model,
                    kind=PAIRWISE_COMPLETION_KIND,
                    estimate=estimate,
                    status=status,
                    artifact_path=artifact_path,
                )
            )
            new_ok += 1
            bucket_counts["new_ok"] += 1

    paid_calls_after = _ledger_call_count(ledger)
    call_artifacts = _completion_call_artifact_stats(
        artifact_dir,
        phase=phase,
    )
    summary = {
        "artifact_type": SUMMARY_ARTIFACT_TYPE,
        "schema_version": 1,
        "phase": phase,
        "dry_run": False,
        "artifact_dir": str(artifact_dir),
        "ledger_path": str(ledger_path),
        "summary_path": str(summary_path),
        "pairwise_model": model,
        "model_availability": model_availability,
        "budget_cap_usd": max_usd,
        "pointwise_calls": 0,
        "pairwise_completion_kind": PAIRWISE_COMPLETION_KIND,
        "planned_pairwise_calls": plan["totals"]["pairwise_calls"],
        "cached_ok_call_artifacts_reused_this_invocation": cached_ok,
        "new_ok_call_artifacts_this_invocation": new_ok,
        "new_ledger_entries_this_invocation": paid_calls_after - paid_calls_before,
        "bucket_results": [
            {"bucket": bucket, **counts} for bucket, counts in sorted(by_bucket.items())
        ],
        "call_artifact_stats": call_artifacts,
        **_ledger_stats(ledger),
    }
    write_json_artifact(summary_path, summary)
    return summary


def analyze_completed_full_random_variance(
    *,
    config_path: Path,
    manifest_path: Path,
    source_artifact_dir: Path,
    prior_random_variance_artifact_path: Path,
    artifact_dir: Path,
    ledger_path: Path,
    initial_plan_path: Path,
    latest_plan_path: Path,
    output_path: Path,
    phase: str,
    max_usd: float,
    seeds: Sequence[int],
    scheduler_samples: int,
    posterior_samples: int,
    pairwise_strength: float,
    bootstrap_samples: int,
    pairwise_cache_artifact_dirs: Sequence[Path] | None,
    labeling_summary: dict[str, Any],
) -> dict[str, Any]:
    raw_config = load_config(config_path)
    phase_config = _config_for_phase(raw_config, phase=phase)["phases"][0]
    pairwise_model = str(phase_config["pairwise_model"])
    validate_model_names([pairwise_model])
    manifest = load_dataset_manifest(manifest_path)
    buckets = manifest.buckets_for_phase(phase)
    labels_by_bucket = _manifest_label_lookup(manifest.payload)
    cache_dirs = _completion_cache_dirs(
        source_artifact_dir=source_artifact_dir,
        artifact_dir=artifact_dir,
        phase=phase,
        explicit_dirs=pairwise_cache_artifact_dirs,
        include_completion_cache=True,
    )

    seed_results = []
    completion_rows = {ARM_HISTORICAL_FULL: [], ARM_EXACT_FULL: []}
    cache_stats_by_bucket = {}
    for seed in seeds:
        seed_payload = {"seed": int(seed), "buckets": []}
        for bucket in buckets:
            papers = load_pointwise_papers_from_artifacts(
                bucket,
                source_artifact_dir=source_artifact_dir,
                phase=phase,
            )
            selection = legacy_select_candidates(papers, k=bucket.k)
            budget = resolve_pairwise_budget(
                n=len(papers),
                candidate_size=len(selection.candidate_ids),
            )
            cached, cache_stats = load_cached_pairwise_labels(
                bucket.name,
                artifact_dirs=cache_dirs,
                phase=phase,
            )
            cache_stats_by_bucket.setdefault(bucket.name, cache_stats)
            schedules = _full_random_schedules(
                papers,
                selection=selection,
                k=bucket.k,
                budget=budget,
                seed=int(seed),
                scheduler_samples=scheduler_samples,
                pairwise_strength=pairwise_strength,
            )
            pointwise_predictions = [
                Prediction(paper.paper_id, paper.pointwise.good_probability)
                for paper in papers
            ]
            pointwise_top_k_ids = _top_k_ids(pointwise_predictions, k=bucket.k)
            arms = {}
            for arm, schedule in schedules.items():
                row = _schedule_label_plan_row(
                    arm=arm,
                    bucket=bucket.name,
                    seed=int(seed),
                    schedule=schedule,
                    cached_keys=set(cached),
                )
                completion_rows[arm].append(row)
                comparisons = _cached_comparisons_for_schedule(schedule, cached=cached)
                if len(comparisons) != len(schedule):
                    raise RuntimeError(
                        f"{bucket.name} seed {seed} arm {arm} is still missing "
                        f"{len(schedule) - len(comparisons)} labels"
                    )
                arms[arm] = _arm_payload(
                    papers,
                    relevant_ids=bucket.relevant_ids,
                    k=bucket.k,
                    schedule=schedule,
                    comparisons=comparisons,
                    pointwise_predictions=pointwise_predictions,
                    pointwise_top_k_ids=pointwise_top_k_ids,
                    labels_by_id=labels_by_bucket.get(bucket.name, {}),
                    posterior_samples=posterior_samples,
                    pairwise_strength=pairwise_strength,
                    seed=int(seed),
                    comparison_source={
                        "source": "complete_full_schedule_cached_and_paid_labels",
                        "scheduled_pairwise_total": len(schedule),
                        "cached_pairwise_labels_available": len(comparisons),
                        "missing_pairwise_labels": 0,
                        "partial": False,
                    },
                    scheduler_diagnostics=_schedule_diagnostics(
                        arm=arm,
                        schedule=schedule,
                        budget=budget.budget,
                    ),
                )
            seed_payload["buckets"].append(
                {
                    "bucket": bucket.name,
                    "seed": int(seed),
                    "k": bucket.k,
                    "papers_total": len(papers),
                    "positive_labels_total": len(bucket.relevant_ids),
                    "budget": budget.to_dict(),
                    "pairwise_cache": cache_stats,
                    "pointwise_metrics": compare_strategies(
                        {"pointwise_only": pointwise_predictions},
                        relevant_ids=bucket.relevant_ids,
                        k=bucket.k,
                    )["pointwise_only"].to_dict(),
                    "arms": arms,
                }
            )
        seed_results.append(seed_payload)

    aggregate_metrics = _aggregate_metrics(
        seed_results,
        arms=(ARM_HISTORICAL_FULL, ARM_EXACT_FULL),
        bootstrap_samples=bootstrap_samples,
    )
    paired_deltas = _paired_deltas(
        seed_results,
        reference_arm=ARM_EXACT_FULL,
        comparison_arm=ARM_HISTORICAL_FULL,
        bootstrap_samples=bootstrap_samples,
    )
    ledger = JsonlLedger(ledger_path)
    ledger_reconciliation = {
        **_ledger_stats(ledger),
        "call_artifact_stats": _completion_call_artifact_stats(
            artifact_dir,
            phase=phase,
        ),
        "labeling_summary_path": str(labeling_summary.get("summary_path", "")),
        "initial_estimated_plan_spend_usd": _load_plan_estimated_spend(
            initial_plan_path
        ),
        "latest_estimated_remaining_plan_spend_usd": _load_plan_estimated_spend(
            latest_plan_path
        ),
    }
    prior_context = _prior_random_control_context(
        prior_random_variance_artifact_path=prior_random_variance_artifact_path
    )
    completion_status = {
        "arms": {
            arm: _aggregate_plan_rows(rows) for arm, rows in completion_rows.items()
        },
        "all_seed_bucket_rows_complete": all(
            row["complete"] for rows in completion_rows.values() for row in rows
        ),
    }
    recommendation = _completion_recommendation(
        aggregate_metrics=aggregate_metrics,
        paired_deltas=paired_deltas,
        prior_context=prior_context,
    )
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": 1,
        "phase": phase,
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "source_artifact_dir": str(source_artifact_dir),
        "artifact_dir": str(artifact_dir),
        "output_path": str(output_path),
        "planning_artifact_path": str(initial_plan_path),
        "latest_planning_artifact_path": str(latest_plan_path),
        "paid_calls_made": _ledger_call_count(ledger),
        "paid_spend_usd": ledger.existing_spend_usd(),
        "known_paid_spend_before_workflow_usd": (
            KNOWN_PAID_SPEND_BEFORE_WORKFLOW_USD
        ),
        "known_paid_spend_after_workflow_usd": round(
            KNOWN_PAID_SPEND_BEFORE_WORKFLOW_USD + ledger.existing_spend_usd(),
            6,
        ),
        "budget_cap_usd": max_usd,
        "pairwise_model": pairwise_model,
        "analysis_parameters": {
            "seeds": [int(seed) for seed in seeds],
            "seed_count": len(seeds),
            "scheduler_samples": scheduler_samples,
            "posterior_samples": posterior_samples,
            "pairwise_strength": pairwise_strength,
            "bootstrap_samples": bootstrap_samples,
            "confidence_level": 0.95,
            "interval_unit": (
                "seed-level means across the 8 buckets; bucket rows are not "
                "treated as independent for headline intervals"
            ),
        },
        "label_policy": {
            "pointwise_paid_calls_made": 0,
            "pairwise_paid_calls_made": _ledger_call_count(ledger),
            "pairwise_completion_kind": PAIRWISE_COMPLETION_KIND,
            "future_labels_used_for_scheduling": False,
            "future_labels_used_for_retrospective_metrics_only": True,
        },
        "pairwise_cache_artifact_dirs": [str(path) for path in cache_dirs],
        "pairwise_cache_stats_by_bucket": cache_stats_by_bucket,
        "full_schedule_completion_status": completion_status,
        "ledger_reconciliation": ledger_reconciliation,
        "aggregate_metrics": aggregate_metrics,
        "paired_deltas": paired_deltas,
        "per_seed_metrics": _per_seed_metrics(seed_results),
        "per_bucket_metrics": _per_bucket_metrics(seed_results),
        "seed_results": seed_results,
        "prior_random_control_context": prior_context,
        "recommendation": recommendation,
        "limitations": [
            "Only the two predeclared full-schedule random arms were completed: historical random and exact-pool random.",
            "Expanded-pool random and targeted-outsider random remain single-seed prior controls and are not methodologically comparable as full-schedule interval estimates.",
            "Future citation labels are used only for retrospective metrics and diagnostics, not for scheduling or posterior scoring.",
            "The experiment has 20 random seeds over 8 historical buckets; one selected positive changes a seed-level mean Recall@K by 0.025.",
            "Spend is estimated from configured token assumptions and reconciled through the workflow JSONL ledger; it is not provider invoice data.",
        ],
    }
    validate_final_artifact_schema(payload)
    write_json_artifact(output_path, payload)
    return payload


def validate_missing_label_plan_schema(payload: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_PLAN_KEYS - set(payload))
    if missing:
        raise ValueError("missing-label plan missing top-level keys: " + ", ".join(missing))
    if payload.get("artifact_type") != PLAN_ARTIFACT_TYPE:
        raise ValueError("missing-label plan has unexpected artifact_type")
    totals = payload.get("totals")
    if not isinstance(totals, dict):
        raise ValueError("missing-label plan totals must be an object")
    if int(totals.get("pointwise_calls", -1)) != 0:
        raise ValueError("missing-label plan must not estimate pointwise calls")


def validate_final_artifact_schema(payload: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_FINAL_KEYS - set(payload))
    if missing:
        raise ValueError("final artifact missing top-level keys: " + ", ".join(missing))
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("final artifact has unexpected artifact_type")
    label_policy = payload.get("label_policy") or {}
    if int(label_policy.get("pointwise_paid_calls_made", -1)) != 0:
        raise ValueError("final artifact must report zero pointwise paid calls")


def _validate_paid_plan_preconditions(
    plan: dict[str, Any],
    *,
    source_artifact_dir: Path,
    artifact_dir: Path,
    ledger_path: Path,
    max_usd: float,
) -> None:
    if plan.get("artifact_type") != PLAN_ARTIFACT_TYPE:
        raise PaidRunSafetyError("paid completion requires a missing-label plan")
    if max_usd <= 0:
        raise PaidRunSafetyError("--max-usd must be greater than zero")
    if max_usd > 5.00:
        raise PaidRunSafetyError("full-random completion --max-usd must not exceed 5.00")
    if artifact_dir.resolve() == source_artifact_dir.resolve():
        raise PaidRunSafetyError("artifact dir must differ from source artifact dir")
    if not ledger_path:
        raise PaidRunSafetyError("--ledger is required for paid runs")
    if int((plan.get("totals") or {}).get("pointwise_calls", -1)) != 0:
        raise PaidRunSafetyError("pointwise calls are forbidden for this workflow")
    if not bool((plan.get("guardrails") or {}).get("paid_run_allowed_after_plan")):
        raise PaidRunSafetyError("missing-label plan did not satisfy paid guardrails")
    projected = float((plan.get("totals") or {}).get("projected_workflow_ledger_spend_usd") or 0.0)
    if projected > max_usd:
        raise PaidRunSafetyError("projected workflow ledger spend exceeds --max-usd")


def _plan_guardrails(
    *,
    pairwise_model: str,
    artifact_dir: Path,
    source_artifact_dir: Path,
    ledger_path: Path,
    max_usd: float,
    estimated_spend_usd: float,
    existing_ledger_spend_usd: float,
) -> dict[str, Any]:
    checks = {
        "provider_prefixed_model_name": "/" in pairwise_model,
        "model_availability_check_required_before_paid_calls": True,
        "jsonl_ledger_path_configured": bool(ledger_path)
        and ledger_path.suffix == ".jsonl",
        "separate_artifact_directory": artifact_dir.resolve()
        != source_artifact_dir.resolve(),
        "pointwise_calls_forbidden": True,
        "planned_pointwise_calls_zero": True,
        "pairwise_only_completion_kind": PAIRWISE_COMPLETION_KIND,
        "workflow_hard_cap_lte_usd_5": 0.0 < max_usd <= 5.0,
        "estimated_additional_spend_lte_cap": estimated_spend_usd <= max_usd,
        "projected_workflow_ledger_spend_lte_cap": (
            existing_ledger_spend_usd + estimated_spend_usd
        )
        <= max_usd,
    }
    blocking = [
        key
        for key, value in checks.items()
        if isinstance(value, bool) and not value
    ]
    return {
        **checks,
        "model_availability": {
            "status": "not_checked_dry_run",
            "required_before_paid_calls": True,
            "models_requiring_check": [pairwise_model],
        },
        "paid_run_allowed_after_plan": not blocking,
        "blocking_reasons": blocking,
    }


def _full_random_schedules(
    papers: list[Any],
    *,
    selection: Any,
    k: int,
    budget: Any,
    seed: int,
    scheduler_samples: int,
    pairwise_strength: float,
) -> dict[str, list[ScheduledPair]]:
    historical = _random_pair_schedule(
        selection,
        budget=budget,
        seed=seed + 7919,
    )
    exact = schedule_exact_pool_random(
        papers,
        [],
        k=k,
        budget=budget,
        seed=seed,
        config=EVSISchedulerConfig(
            samples=scheduler_samples,
            pairwise_strength=pairwise_strength,
        ),
        diagnostics=DiagnosticRecorder(),
    )
    return {
        ARM_HISTORICAL_FULL: historical,
        ARM_EXACT_FULL: exact.pairs,
    }


def _schedule_label_plan_row(
    *,
    arm: str,
    bucket: str,
    seed: int,
    schedule: list[ScheduledPair],
    cached_keys: set[tuple[str, str]],
) -> dict[str, Any]:
    scheduled_keys = [_pair_key(pair.left_id, pair.right_id) for pair in schedule]
    missing = [key for key in scheduled_keys if key not in cached_keys]
    return {
        "arm": arm,
        "bucket": bucket,
        "seed": int(seed),
        "scheduled_pairwise_total": len(schedule),
        "cached_pairwise_labels_available": len(schedule) - len(missing),
        "missing_pairwise_labels": len(missing),
        "cache_reuse_rate": _rate(len(schedule) - len(missing), len(schedule)),
        "complete": len(missing) == 0,
        "missing_pair_keys": [list(key) for key in missing],
        "sample_missing_pair_keys": [list(key) for key in missing[:10]],
    }


def _aggregate_plan_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scheduled = sum(int(row["scheduled_pairwise_total"]) for row in rows)
    cached = sum(int(row["cached_pairwise_labels_available"]) for row in rows)
    missing = sum(int(row["missing_pairwise_labels"]) for row in rows)
    complete_rows = [row for row in rows if bool(row["complete"])]
    return {
        "seed_bucket_row_count": len(rows),
        "complete_seed_bucket_row_count": len(complete_rows),
        "complete_seed_bucket_row_rate": _rate(len(complete_rows), len(rows)),
        "scheduled_pairwise_total": scheduled,
        "cached_pairwise_labels_available": cached,
        "missing_pairwise_labels": missing,
        "cache_reuse_rate": _rate(cached, scheduled),
    }


def _compact_plan_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "missing_pair_keys"}
        for row in rows
    ]


def _arm_payload(
    papers: list[Any],
    *,
    relevant_ids: set[str],
    k: int,
    schedule: list[ScheduledPair],
    comparisons: list[PairwiseComparison],
    pointwise_predictions: list[Prediction],
    pointwise_top_k_ids: list[str],
    labels_by_id: dict[str, dict[str, Any]],
    posterior_samples: int,
    pairwise_strength: float,
    seed: int,
    comparison_source: dict[str, Any],
    scheduler_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    posterior_predictions, posterior = posterior_top_k_predictions(
        papers,
        comparisons,
        k=k,
        pairwise_strength=pairwise_strength,
        samples=posterior_samples,
        seed=seed,
    )
    metrics = compare_strategies(
        {
            "pointwise_only": pointwise_predictions,
            POSTERIOR_STRATEGY: posterior_predictions,
        },
        relevant_ids=relevant_ids,
        k=k,
    )
    posterior_top_k_ids = _top_k_ids(posterior_predictions, k=k)
    return {
        "comparison_source": comparison_source,
        "metrics": {name: metric.to_dict() for name, metric in metrics.items()},
        "positive_exposure": positive_exposure_diagnostics(
            schedule,
            relevant_ids=relevant_ids,
            paper_count=len(papers),
        ),
        "pair_graph": pair_graph_diagnostics(
            schedule,
            relevant_ids=relevant_ids,
            posterior_top_k_ids=posterior_top_k_ids,
            pointwise_top_k_ids=pointwise_top_k_ids,
        ),
        "oracle_bounds": oracle_bounds(
            k=k,
            relevant_ids=relevant_ids,
            pointwise_top_k_ids=pointwise_top_k_ids,
            schedule=schedule,
            comparisons=comparisons,
        ),
        "top_k_error_decomposition": top_k_error_decomposition(
            predictions=posterior_predictions,
            relevant_ids=relevant_ids,
            k=k,
            schedule=schedule,
            pointwise_top_k_ids=pointwise_top_k_ids,
            labels_by_id=labels_by_id,
        ),
        "posterior_topk_diagnostics": posterior.diagnostics,
        "scheduler_diagnostics": scheduler_diagnostics,
    }


def _cached_comparisons_for_schedule(
    schedule: list[ScheduledPair],
    *,
    cached: dict[tuple[str, str], Any],
) -> list[PairwiseComparison]:
    comparisons = []
    for pair in schedule:
        label = cached.get(_pair_key(pair.left_id, pair.right_id))
        if label is None:
            continue
        comparisons.append(_orient_cached_comparison(label, pair))
    return comparisons


def _aggregate_metrics(
    seed_results: list[dict[str, Any]],
    *,
    arms: Sequence[str],
    bootstrap_samples: int,
) -> dict[str, Any]:
    output = {}
    for arm in arms:
        seed_rows: dict[int, dict[str, float | int]] = {}
        bucket_rows: dict[str, list[dict[str, float | int]]] = {}
        all_rows = []
        for seed_payload in seed_results:
            seed = int(seed_payload["seed"])
            rows = []
            for bucket in seed_payload["buckets"]:
                row = bucket["arms"][arm]["metrics"][POSTERIOR_STRATEGY]
                rows.append(row)
                all_rows.append(row)
                bucket_rows.setdefault(str(bucket["bucket"]), []).append(row)
            seed_rows[seed] = _mean_metric_rows(rows)
        output[arm] = {
            "bucket_seed_row_mean": _mean_metric_rows(all_rows),
            "seed_metric_rows": {
                str(seed): row for seed, row in sorted(seed_rows.items())
            },
            "seed_level_intervals": _metric_intervals(
                list(seed_rows.values()),
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=7001,
            ),
            "bucket_level_intervals": {
                bucket: _metric_intervals(
                    rows,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=8001 + index,
                )
                for index, (bucket, rows) in enumerate(sorted(bucket_rows.items()))
            },
        }
    return output


def _paired_deltas(
    seed_results: list[dict[str, Any]],
    *,
    reference_arm: str,
    comparison_arm: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    metrics = (
        "recall_at_k",
        "precision_at_k",
        "ndcg_at_k",
        "average_precision",
        "brier_score",
        "near_miss_positive_rate",
    )
    seed_deltas = {}
    bucket_deltas = []
    for seed_payload in seed_results:
        seed = int(seed_payload["seed"])
        rows_by_metric = {metric: [] for metric in metrics}
        selected_positive_deltas = []
        for bucket in seed_payload["buckets"]:
            reference = bucket["arms"][reference_arm]
            comparison = bucket["arms"][comparison_arm]
            reference_metrics = reference["metrics"][POSTERIOR_STRATEGY]
            comparison_metrics = comparison["metrics"][POSTERIOR_STRATEGY]
            row = {"seed": seed, "bucket": bucket["bucket"]}
            for metric in metrics:
                delta = round(
                    float(comparison_metrics[metric])
                    - float(reference_metrics[metric]),
                    8,
                )
                rows_by_metric[metric].append(delta)
                row[f"{metric}_delta"] = delta
            comparison_hits = int(
                comparison["top_k_error_decomposition"]["selected_positive_count"]
            )
            reference_hits = int(
                reference["top_k_error_decomposition"]["selected_positive_count"]
            )
            hit_delta = comparison_hits - reference_hits
            selected_positive_deltas.append(hit_delta)
            row["selected_positive_delta"] = hit_delta
            bucket_deltas.append(row)
        seed_deltas[seed] = {
            metric: round(sum(values) / len(values), 8)
            for metric, values in rows_by_metric.items()
        }
        seed_deltas[seed]["selected_positive_delta_total"] = sum(
            selected_positive_deltas
        )
    return {
        "reference_arm": reference_arm,
        "comparison_arm": comparison_arm,
        "interpretation": (
            "Positive deltas mean the comparison arm exceeded the reference arm "
            "on the same seed."
        ),
        "seed_deltas": {str(seed): row for seed, row in sorted(seed_deltas.items())},
        "bucket_deltas": bucket_deltas,
        "metric_delta_intervals": {
            metric: summarize_values(
                [float(row[metric]) for row in seed_deltas.values()],
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=9001 + index,
            )
            for index, metric in enumerate(metrics)
        },
        "selected_positive_total_delta": sum(
            int(row["selected_positive_delta"]) for row in bucket_deltas
        ),
    }


def _metric_intervals(
    rows: list[dict[str, float | int]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not rows:
        return {}
    keys = [
        key
        for key in (
            "recall_at_k",
            "precision_at_k",
            "ndcg_at_k",
            "average_precision",
            "brier_score",
            "near_miss_positive_rate",
        )
        if key in rows[0]
    ]
    return {
        key: summarize_values(
            [float(row[key]) for row in rows],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index,
        )
        for index, key in enumerate(keys)
    }


def _per_seed_metrics(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for seed_payload in seed_results:
        seed = int(seed_payload["seed"])
        rows[str(seed)] = {}
        for arm in (ARM_HISTORICAL_FULL, ARM_EXACT_FULL):
            metric_rows = [
                bucket["arms"][arm]["metrics"][POSTERIOR_STRATEGY]
                for bucket in seed_payload["buckets"]
            ]
            rows[str(seed)][arm] = _mean_metric_rows(metric_rows)
    return rows


def _per_bucket_metrics(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_bucket: dict[str, dict[str, list[dict[str, float | int]]]] = {}
    for seed_payload in seed_results:
        for bucket in seed_payload["buckets"]:
            bucket_name = str(bucket["bucket"])
            for arm in (ARM_HISTORICAL_FULL, ARM_EXACT_FULL):
                by_bucket.setdefault(bucket_name, {}).setdefault(arm, []).append(
                    bucket["arms"][arm]["metrics"][POSTERIOR_STRATEGY]
                )
    return {
        bucket: {
            arm: _mean_metric_rows(rows)
            for arm, rows in sorted(arms.items())
        }
        for bucket, arms in sorted(by_bucket.items())
    }


def _completion_recommendation(
    *,
    aggregate_metrics: dict[str, Any],
    paired_deltas: dict[str, Any],
    prior_context: dict[str, Any],
) -> dict[str, Any]:
    historical_recall = aggregate_metrics[ARM_HISTORICAL_FULL][
        "seed_level_intervals"
    ]["recall_at_k"]
    exact_recall = aggregate_metrics[ARM_EXACT_FULL]["seed_level_intervals"][
        "recall_at_k"
    ]
    recall_delta = paired_deltas["metric_delta_intervals"]["recall_at_k"]
    ndcg_delta = paired_deltas["metric_delta_intervals"]["ndcg_at_k"]
    ap_delta = paired_deltas["metric_delta_intervals"]["average_precision"]
    historical_mean = float(historical_recall["mean"])
    exact_mean = float(exact_recall["mean"])
    random_baseline_supported = historical_mean >= 0.30 and exact_mean >= 0.30
    return {
        "complete_random_baseline_robustness_supported": random_baseline_supported,
        "headline": (
            "Complete full-schedule random baselines remain mandatory, but the "
            "single seed-17 random reference should not be treated as a stable "
            "point estimate."
        ),
        "evidence": {
            ARM_HISTORICAL_FULL: historical_recall,
            ARM_EXACT_FULL: exact_recall,
            "historical_minus_exact_recall_delta": recall_delta,
            "historical_minus_exact_ndcg_delta": ndcg_delta,
            "historical_minus_exact_ap_delta": ap_delta,
        },
        "use_in_next_comparison": (
            "Compare any future active arm against paired full-schedule random "
            "seeds and report seed-unit intervals for active-minus-random deltas."
        ),
        "claim_threshold": (
            "Do not claim an active arm beats random unless the paired "
            "active-minus-random Recall@K interval is positive, or the mean "
            "Recall@K gain is at least 0.025 with nonnegative nDCG/AP deltas "
            "and no missing-label caveat."
        ),
        "prior_random_controls_included_as_context_only": prior_context.get(
            "context_only_controls",
            [],
        ),
        "recommended_next_action": (
            "Stop random-baseline spending here. Use this completed artifact as "
            "the variance reference for future no-paid simulator gates or a "
            "predeclared active-arm comparison."
        ),
    }


def _prior_random_control_context(
    *,
    prior_random_variance_artifact_path: Path,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "prior_random_variance_artifact_path": str(prior_random_variance_artifact_path),
        "context_only_controls": [],
    }
    if prior_random_variance_artifact_path.exists():
        prior = json.loads(prior_random_variance_artifact_path.read_text())
        context["prior_cached_replay_summary"] = {
            "paid_calls_made": prior.get("paid_calls_made"),
            "paid_spend_usd": prior.get("paid_spend_usd"),
            "cached_replay_seed_level_intervals": {
                arm: {
                    metric: payload["seed_level_intervals"][metric]
                    for metric in ("recall_at_k", "ndcg_at_k", "average_precision")
                    if metric in payload.get("seed_level_intervals", {})
                }
                for arm, payload in (
                    (prior.get("cached_replay") or {})
                    .get("aggregate_metrics", {})
                    .items()
                )
            },
        }
    random_control_path = (
        REPO_ROOT
        / "artifacts"
        / "backtest-arxiv-random-control-diagnosis"
        / "random-control-gap-analysis.json"
    )
    if random_control_path.exists():
        random_control = json.loads(random_control_path.read_text())
        aggregate = random_control.get("aggregate_metrics") or {}
        for arm in ("expanded_pool_random", "targeted_outsider_random", "cctd_gf"):
            if arm not in aggregate:
                continue
            context["context_only_controls"].append(
                {
                    "arm": arm,
                    "source_artifact_path": str(random_control_path),
                    "posterior_topk_metrics": aggregate[arm].get("posterior_topk"),
                    "included_in_headline_intervals": False,
                    "reason": (
                        "prior single-seed candidate-construction/control arm; "
                        "not a completed 20-seed full-schedule random variance arm"
                    ),
                }
            )
    return context


def _completion_cache_dirs(
    *,
    source_artifact_dir: Path,
    artifact_dir: Path,
    phase: str,
    explicit_dirs: Sequence[Path] | None,
    include_completion_cache: bool = True,
) -> list[Path]:
    dirs = _pairwise_cache_dirs(
        source_artifact_dir,
        phase=phase,
        explicit_dirs=explicit_dirs,
    )
    if include_completion_cache and artifact_dir not in dirs:
        dirs.append(artifact_dir)
    deduped = []
    seen = set()
    for path in dirs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _completion_pairwise_artifact_path(
    artifact_dir: Path,
    *,
    phase: str,
    bucket: str,
    label_index: int,
    pair: ScheduledPair,
) -> Path:
    return (
        artifact_dir
        / phase
        / _safe_name(bucket)
        / "calls"
        / (
            f"{label_index:04d}-{PAIRWISE_COMPLETION_KIND}-"
            f"{fingerprint(pair.left_id + ':' + pair.right_id)}.json"
        )
    )


def _next_pairwise_attempt_path(path: Path) -> Path:
    if not path.exists():
        return path
    for attempt in range(2, 100):
        candidate = path.with_name(f"{path.stem}-attempt-{attempt}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate retry artifact path for {path}")


def _completion_call_artifact_stats(
    artifact_dir: Path,
    *,
    phase: str,
) -> dict[str, Any]:
    rows = []
    for path in sorted((artifact_dir / phase).glob("*/calls/*pairwise*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            rows.append({"status": "malformed", "kind": "unknown", "cost": 0.0})
            continue
        rows.append(
            {
                "status": str(payload.get("status") or "unknown"),
                "kind": str(payload.get("kind") or "unknown"),
                "cost": float(payload.get("estimated_cost_usd") or 0.0),
            }
        )
    by_status = Counter(row["status"] for row in rows)
    by_kind = Counter(row["kind"] for row in rows)
    return {
        "artifact_count": len(rows),
        "artifact_count_by_status": dict(sorted(by_status.items())),
        "artifact_count_by_kind": dict(sorted(by_kind.items())),
        "estimated_cost_usd_by_artifact": round(
            sum(float(row["cost"]) for row in rows),
            6,
        ),
    }


def _schedule_diagnostics(
    *,
    arm: str,
    schedule: list[ScheduledPair],
    budget: int,
) -> dict[str, Any]:
    return {
        "method": arm,
        "scheduled_total": len(schedule),
        "budget": budget,
        "purpose_counts": dict(sorted(Counter(pair.purpose for pair in schedule).items())),
        "position_bias_audit_total": sum(
            1 for pair in schedule if pair.order.position_bias_audit
        ),
    }


def _load_plan_estimated_spend(plan_path: Path) -> float | None:
    if not plan_path.exists():
        return None
    payload = json.loads(plan_path.read_text())
    return float(
        (payload.get("totals") or {}).get("estimated_additional_spend_usd") or 0.0
    )


def _scheduled_pair_from_dict(payload: dict[str, Any]) -> ScheduledPair:
    return ScheduledPair(
        left_id=str(payload["left_id"]),
        right_id=str(payload["right_id"]),
        priority=float(payload.get("priority", 0.0)),
        purpose=str(payload.get("purpose") or PAIRWISE_COMPLETION_KIND),
        order=PairwiseOrderMetadata.from_dict(payload.get("order")),
        diagnostics=dict(payload.get("diagnostics") or {}),
    )


def _load_ok_json_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if payload.get("status") != "ok":
        return None
    return payload


def _top_k_ids(predictions: Sequence[Prediction], *, k: int) -> list[str]:
    return [
        prediction.paper_id
        for prediction in sorted(
            predictions,
            key=lambda item: (item.score, item.paper_id),
            reverse=True,
        )[:k]
    ]


def _rate(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 8) if denominator else 0.0


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _stdout_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("artifact_type") == PLAN_ARTIFACT_TYPE:
        plan = payload.get("plan") or payload
        return {
            "artifact_type": payload["artifact_type"],
            "plan_path": payload.get("plan_path", plan.get("output_path")),
            "paid_calls_made": payload.get("paid_calls_made", 0),
            "paid_spend_usd": payload.get("paid_spend_usd", 0.0),
            "blocked": payload.get("blocked", False),
            "totals": plan.get("totals"),
            "guardrails": plan.get("guardrails"),
        }
    aggregate = payload["aggregate_metrics"]
    return {
        "artifact_type": payload["artifact_type"],
        "artifact_path": payload.get("artifact_path", payload.get("output_path")),
        "paid_calls_made": payload["paid_calls_made"],
        "paid_spend_usd": payload["paid_spend_usd"],
        "known_paid_spend_after_workflow_usd": payload[
            "known_paid_spend_after_workflow_usd"
        ],
        "seed_level_intervals": {
            arm: {
                metric: aggregate[arm]["seed_level_intervals"][metric]
                for metric in ("recall_at_k", "ndcg_at_k", "average_precision")
            }
            for arm in (ARM_HISTORICAL_FULL, ARM_EXACT_FULL)
        },
        "historical_minus_exact_paired_deltas": {
            metric: payload["paired_deltas"]["metric_delta_intervals"][metric]
            for metric in ("recall_at_k", "ndcg_at_k", "average_precision")
        },
        "recommendation": payload["recommendation"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
