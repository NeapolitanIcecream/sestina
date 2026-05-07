#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ci_partition_gate import (  # noqa: E402
    DEFAULT_SEEDS,
    _pair_key,
    _pairwise_cache_dirs,
    _parse_seeds,
    load_cached_pairwise_labels,
)
from sestina.active_arm_gate import (  # noqa: E402
    CURRENT_KNOWN_SPEND_USD,
    DEFAULT_PAID_CAP_USD,
)
from sestina.backtest_budget import load_config  # noqa: E402
from sestina.backtest_runner import (  # noqa: E402
    JsonlLedger,
    _call_estimate,
    _config_for_phase,
    _normalize_rates_from_config,
    _normalize_token_assumptions_from_config,
    load_dataset_manifest,
    validate_model_names,
)
from sestina.candidates import select_candidates  # noqa: E402
from sestina.diagnostics import write_json_artifact  # noqa: E402
from sestina.new_information_challenger import (  # noqa: E402
    NewInformationChallengerConfig,
    schedule_new_information_challenger_pairs,
)
from sestina.scheduler import resolve_pairwise_budget  # noqa: E402
from sestina.scheduler_followup import load_pointwise_papers_from_artifacts  # noqa: E402


ARTIFACT_TYPE = "sestina-new-information-paid-dry-run"
SCHEMA_VERSION = 1
ARM_NEW_INFO = "new_information_challenger_cached_replay"
ARM_EXACT = "exact_pool_random_cached_replay"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts" / "backtest-arxiv-new-information-paid-dry-run"
)
DEFAULT_BUDGET_FILL_ARTIFACT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-new-information-budget-fill-gate"
    / "new-information-budget-fill-gate.json"
)
DEFAULT_ACTIVE_GATE_ARTIFACT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-new-information-budget-fill-gate"
    / "active-arm-gate.json"
)
DEFAULT_RANDOM_VARIANCE_ARTIFACT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-full-random-variance-completion"
    / "full-random-variance-completion.json"
)
REQUIRED_KEYS = {
    "artifact_type",
    "schema_version",
    "dry_run",
    "paid_calls_made",
    "paid_spend_usd",
    "pointwise_calls_made",
    "input_artifacts",
    "frozen_inputs",
    "planned_execution",
    "totals",
    "guardrails",
    "caveats",
    "go_no_go",
    "planned_pair_occurrences_path",
    "planned_unique_pair_labels_by_bucket",
    "validation_commands",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the budget-filled new-information challenger arm inputs and "
            "emit a no-paid pairwise-only paid-workflow dry-run/go-no-go artifact."
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
        "--budget-fill-artifact",
        type=Path,
        default=DEFAULT_BUDGET_FILL_ARTIFACT,
    )
    parser.add_argument(
        "--active-gate-artifact",
        type=Path,
        default=DEFAULT_ACTIVE_GATE_ARTIFACT,
    )
    parser.add_argument(
        "--random-variance-artifact",
        type=Path,
        default=DEFAULT_RANDOM_VARIANCE_ARTIFACT,
    )
    parser.add_argument("--phase", default="pilot")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="separate directory for the dry-run manifest and later ledger path",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "ledger.jsonl",
        help="planned JSONL ledger path for any later paid pairwise-only run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "paid-dry-run-go-no-go.json",
    )
    parser.add_argument(
        "--planned-pairs-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "planned-pair-occurrences.jsonl",
    )
    parser.add_argument(
        "--max-usd",
        type=float,
        default=2.0,
        help="candidate hard cap for a later guarded paid pairwise workflow",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="comma-separated seed set to freeze",
    )
    parser.add_argument("--scheduler-samples", type=int, default=800)
    parser.add_argument("--posterior-samples", type=int, default=1200)
    parser.add_argument("--pairwise-strength", type=float, default=2.5)
    parser.add_argument("--random-floor-fraction", type=float, default=0.2)
    parser.add_argument("--anchor-multiplier", type=int, default=2)
    parser.add_argument("--challenger-multiplier", type=int, default=3)
    parser.add_argument("--min-challengers", type=int, default=8)
    parser.add_argument("--minimum-rubric-residual", type=float, default=0.02)
    parser.add_argument("--per-item-cap", type=int, default=6)
    parser.add_argument(
        "--pairwise-cache-artifact-dir",
        action="append",
        type=Path,
        default=None,
        help=(
            "Override cache dirs to scan for pairwise reuse. By default, the "
            "exact cache dirs recorded in the budget-fill artifact are used; "
            "if absent, all artifacts/backtest-arxiv-*-live dirs are scanned."
        ),
    )
    args = parser.parse_args(argv)

    payload = build_new_information_paid_dry_run(
        config_path=args.config,
        manifest_path=args.manifest,
        source_artifact_dir=args.source_artifact_dir,
        budget_fill_artifact_path=args.budget_fill_artifact,
        active_gate_artifact_path=args.active_gate_artifact,
        random_variance_artifact_path=args.random_variance_artifact,
        artifact_dir=args.artifact_dir,
        ledger_path=args.ledger,
        output_path=args.output,
        planned_pairs_output_path=args.planned_pairs_output,
        phase=args.phase,
        max_usd=args.max_usd,
        seeds=_parse_seeds(args.seeds),
        scheduler_samples=args.scheduler_samples,
        posterior_samples=args.posterior_samples,
        pairwise_strength=args.pairwise_strength,
        random_floor_fraction=args.random_floor_fraction,
        anchor_multiplier=args.anchor_multiplier,
        challenger_multiplier=args.challenger_multiplier,
        min_challengers=args.min_challengers,
        minimum_rubric_residual=args.minimum_rubric_residual,
        per_item_cap=args.per_item_cap,
        pairwise_cache_artifact_dirs=args.pairwise_cache_artifact_dir,
    )
    sys.stdout.write(json.dumps(_stdout_summary(payload), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def build_new_information_paid_dry_run(
    *,
    config_path: Path,
    manifest_path: Path,
    source_artifact_dir: Path,
    budget_fill_artifact_path: Path,
    active_gate_artifact_path: Path,
    random_variance_artifact_path: Path,
    artifact_dir: Path,
    ledger_path: Path,
    output_path: Path,
    planned_pairs_output_path: Path,
    phase: str,
    max_usd: float,
    seeds: Sequence[int],
    scheduler_samples: int,
    posterior_samples: int,
    pairwise_strength: float,
    random_floor_fraction: float,
    anchor_multiplier: int,
    challenger_multiplier: int,
    min_challengers: int,
    minimum_rubric_residual: float,
    per_item_cap: int | None,
    pairwise_cache_artifact_dirs: Sequence[Path] | None,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    budget_fill_artifact = _read_json(budget_fill_artifact_path)
    active_gate_artifact = _read_json(active_gate_artifact_path)
    random_variance_artifact = _read_json(random_variance_artifact_path)
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
    cache_dirs = _freeze_cache_dirs(
        budget_fill_artifact,
        source_artifact_dir=source_artifact_dir,
        phase=phase,
        explicit_dirs=pairwise_cache_artifact_dirs,
    )
    new_info_config = NewInformationChallengerConfig(
        pairwise_strength=pairwise_strength,
        posterior_samples=scheduler_samples,
        random_floor_fraction=random_floor_fraction,
        anchor_multiplier=anchor_multiplier,
        challenger_multiplier=challenger_multiplier,
        min_challengers=min_challengers,
        minimum_rubric_residual=minimum_rubric_residual,
        per_item_cap=per_item_cap,
    )

    occurrence_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    unique_labels_by_bucket: dict[str, dict[str, dict[str, Any]]] = {}
    cache_source_counts: Counter[str] = Counter()
    cache_kind_counts: Counter[str] = Counter()
    scheduled_total = 0
    cached_occurrences = 0
    missing_occurrences = 0
    active_shortfall = 0

    for bucket in buckets:
        papers = load_pointwise_papers_from_artifacts(
            bucket,
            source_artifact_dir=source_artifact_dir,
            phase=phase,
        )
        selection = select_candidates(papers, k=bucket.k)
        budget = resolve_pairwise_budget(
            n=len(papers),
            candidate_size=len(selection.candidate_ids),
        )
        cached, cache_stats = load_cached_pairwise_labels(
            bucket.name,
            artifact_dirs=cache_dirs,
            phase=phase,
        )
        bucket_unique: dict[str, dict[str, Any]] = {}
        bucket_scheduled = 0
        bucket_cached = 0
        bucket_missing = 0
        bucket_shortfall = 0
        purpose_counts: Counter[str] = Counter()

        for seed in seeds:
            schedule = schedule_new_information_challenger_pairs(
                papers,
                [],
                k=bucket.k,
                budget=budget,
                seed=int(seed),
                config=new_info_config,
                available_pair_keys=set(cached),
            )
            row_shortfall = max(0, budget.budget - len(schedule.pairs))
            bucket_shortfall += row_shortfall
            for index, pair in enumerate(schedule.pairs, start=1):
                key = _pair_key(pair.left_id, pair.right_id)
                key_text = "::".join(key)
                cached_label = cached.get(key)
                cache_status = (
                    "cached_reuse" if cached_label is not None else "missing_label"
                )
                if cached_label is not None:
                    bucket_cached += 1
                    cache_source_counts[str(cached_label.artifact_dir)] += 1
                    cache_kind_counts[cached_label.kind] += 1
                else:
                    bucket_missing += 1
                purpose_counts[pair.purpose] += 1
                bucket_scheduled += 1
                occurrence = {
                    "row_id": f"{int(seed)}:{bucket.name}",
                    "seed": int(seed),
                    "bucket": bucket.name,
                    "k": bucket.k,
                    "pair_index": index,
                    "pair_key": list(key),
                    "left_id": pair.left_id,
                    "right_id": pair.right_id,
                    "purpose": pair.purpose,
                    "priority": pair.priority,
                    "order": pair.order.to_dict(),
                    "cache_status": cache_status,
                    "cached_artifact_path": (
                        str(cached_label.artifact_path)
                        if cached_label is not None
                        else None
                    ),
                    "cached_artifact_kind": (
                        cached_label.kind if cached_label is not None else None
                    ),
                    "source_new_information_purpose": pair.diagnostics.get(
                        "source_new_information_purpose",
                        pair.purpose,
                    ),
                    "challenger_id": pair.diagnostics.get("challenger_id"),
                    "frontier_comparator_id": pair.diagnostics.get(
                        "frontier_comparator_id"
                    ),
                    "anchor_id": pair.diagnostics.get("anchor_id"),
                    "future_labels_used_for_scheduling": False,
                    "cached_label_values_used_before_scheduling": False,
                }
                occurrence_rows.append(occurrence)
                unique = bucket_unique.get(key_text)
                if unique is None:
                    unique = {
                        "bucket": bucket.name,
                        "pair_key": list(key),
                        "left_id": pair.left_id,
                        "right_id": pair.right_id,
                        "cache_status": cache_status,
                        "cached_artifact_path": occurrence["cached_artifact_path"],
                        "cached_artifact_kind": occurrence["cached_artifact_kind"],
                        "required_by": [],
                    }
                    bucket_unique[key_text] = unique
                unique["required_by"].append(
                    {
                        "seed": int(seed),
                        "pair_index": index,
                        "purpose": pair.purpose,
                    }
                )

        unique_labels_by_bucket[bucket.name] = bucket_unique
        bucket_unique_labels = len(bucket_unique)
        bucket_unique_missing = sum(
            1 for row in bucket_unique.values() if row["cache_status"] != "cached_reuse"
        )
        scheduled_total += bucket_scheduled
        cached_occurrences += bucket_cached
        missing_occurrences += bucket_missing
        active_shortfall += bucket_shortfall
        bucket_rows.append(
            {
                "bucket": bucket.name,
                "k": bucket.k,
                "papers_total": len(papers),
                "positive_labels_total": len(bucket.relevant_ids),
                "seed_count": len(seeds),
                "pairwise_budget_per_seed": budget.to_dict(),
                "pointwise_artifacts_loaded": len(papers),
                "pointwise_calls": 0,
                "scheduled_pairwise_occurrences": bucket_scheduled,
                "cached_pairwise_occurrences": bucket_cached,
                "missing_pairwise_occurrences": bucket_missing,
                "active_budget_shortfall": bucket_shortfall,
                "unique_planned_pair_labels": bucket_unique_labels,
                "unique_missing_pairwise_labels": bucket_unique_missing,
                "purpose_counts": dict(sorted(purpose_counts.items())),
                "pairwise_cache": cache_stats,
            }
        )

    unique_missing_total = sum(
        1
        for bucket_unique in unique_labels_by_bucket.values()
        for row in bucket_unique.values()
        if row["cache_status"] != "cached_reuse"
    )
    unique_planned_total = sum(len(rows) for rows in unique_labels_by_bucket.values())
    estimated_spend = round(unique_missing_total * pairwise_estimate.cost_usd, 6)
    existing_ledger_spend = JsonlLedger(ledger_path).existing_spend_usd()
    projected_workflow_spend = round(existing_ledger_spend + estimated_spend, 6)
    guardrails = _guardrails(
        pairwise_model=pairwise_model,
        artifact_dir=artifact_dir,
        source_artifact_dir=source_artifact_dir,
        ledger_path=ledger_path,
        max_usd=max_usd,
        estimated_spend_usd=estimated_spend,
        projected_workflow_spend_usd=projected_workflow_spend,
        active_shortfall=active_shortfall,
        active_gate_artifact=active_gate_artifact,
        budget_fill_artifact=budget_fill_artifact,
        random_variance_artifact=random_variance_artifact,
    )
    caveats = _caveats(
        budget_fill_artifact=budget_fill_artifact,
        active_gate_artifact=active_gate_artifact,
        planned_active_shortfall=active_shortfall,
        planned_unique_missing_labels=unique_missing_total,
    )
    go_no_go = _go_no_go(
        guardrails=guardrails,
        caveats=caveats,
        estimated_spend_usd=estimated_spend,
        max_usd=max_usd,
    )
    planned_unique = {
        bucket: list(rows.values())
        for bucket, rows in sorted(unique_labels_by_bucket.items())
    }
    _write_jsonl(planned_pairs_output_path, occurrence_rows)
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "dry_run": True,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 0,
        "known_paid_spend_before_workflow_usd": CURRENT_KNOWN_SPEND_USD,
        "paid_cap_usd": DEFAULT_PAID_CAP_USD,
        "input_artifacts": {
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "source_artifact_dir": str(source_artifact_dir),
            "budget_fill_artifact_path": str(budget_fill_artifact_path),
            "budget_fill_artifact_sha256": _sha256(budget_fill_artifact_path),
            "active_gate_artifact_path": str(active_gate_artifact_path),
            "active_gate_artifact_sha256": _sha256(active_gate_artifact_path),
            "random_variance_artifact_path": str(random_variance_artifact_path),
            "random_variance_artifact_sha256": _sha256(
                random_variance_artifact_path
            ),
            "scheduler_code_path": str(
                REPO_ROOT / "sestina" / "new_information_challenger.py"
            ),
            "scheduler_code_sha256": _sha256(
                REPO_ROOT / "sestina" / "new_information_challenger.py"
            ),
            "guarded_runner_code_path": str(
                REPO_ROOT / "sestina" / "scheduler_followup.py"
            ),
            "guarded_runner_code_sha256": _sha256(
                REPO_ROOT / "sestina" / "scheduler_followup.py"
            ),
        },
        "frozen_inputs": {
            "active_arm_name": ARM_NEW_INFO,
            "random_control_comparator": ARM_EXACT,
            "phase": phase,
            "seeds": [int(seed) for seed in seeds],
            "seed_count": len(seeds),
            "bucket_rows": [
                {
                    "bucket": row["bucket"],
                    "k": row["k"],
                    "papers_total": row["papers_total"],
                    "pairwise_budget_per_seed": row["pairwise_budget_per_seed"],
                }
                for row in bucket_rows
            ],
            "candidate_construction_policy": {
                "method": "new_information_challenger_cached_replay",
                "source_method": (
                    "pointwise_rubric_residual_false_negative_exposure"
                ),
                "pairwise_strength": pairwise_strength,
                "scheduler_samples": scheduler_samples,
                "posterior_samples_for_scoring": posterior_samples,
                "random_floor_fraction": random_floor_fraction,
                "anchor_multiplier": anchor_multiplier,
                "challenger_multiplier": challenger_multiplier,
                "min_challengers": min_challengers,
                "minimum_rubric_residual": minimum_rubric_residual,
                "per_item_cap": per_item_cap,
                "model_visible_signals": [
                    "pointwise_good_probability",
                    "pointwise_uncertainty",
                    "pointwise_rubric_scores",
                    "title_abstract_lexical_novelty",
                    "metadata_category_diversity",
                    "cached_pair_availability",
                ],
                "future_labels_used_for_scheduling": False,
                "cached_label_values_used_before_scheduling": False,
            },
            "fallback_policy": _fallback_policy_from_artifact(
                budget_fill_artifact
            ),
            "pairwise_cache_artifact_dirs": [str(path) for path in cache_dirs],
            "model_provider": pairwise_model.split("/", 1)[0],
            "pairwise_model": pairwise_model,
            "artifact_dir": str(artifact_dir),
            "ledger_path": str(ledger_path),
            "scoring_and_evaluation_commands": [
                (
                    "uv run python scripts/run_new_information_challenger_simulator.py "
                    "--output artifacts/backtest-arxiv-new-information-budget-fill-gate/"
                    "new-information-budget-fill-gate.json "
                    "--active-gate-output artifacts/backtest-arxiv-new-information-"
                    "budget-fill-gate/active-arm-gate.json"
                ),
                (
                    "uv run python scripts/run_active_arm_gate.py "
                    "--active-artifact artifacts/backtest-arxiv-new-information-"
                    "budget-fill-gate/new-information-budget-fill-gate.json "
                    "--random-variance-artifact artifacts/backtest-arxiv-full-random-"
                    "variance-completion/full-random-variance-completion.json "
                    "--output artifacts/backtest-arxiv-new-information-budget-fill-gate/"
                    "active-arm-gate.json "
                    "--active-arm new_information_challenger_cached_replay "
                    "--random-control-arm exact_pool_random_cached_replay"
                ),
            ],
        },
        "planned_execution": {
            "mode": "no_paid_dry_run_only",
            "later_paid_workflow_allowed_by_this_artifact": (
                go_no_go["decision"] == "go"
            ),
            "planned_pairwise_label_kind": "pairwise_active",
            "random_control_paid_labels_planned": 0,
            "pointwise_calls_planned": 0,
            "model_availability": {
                "status": "not_checked_dry_run",
                "required_before_paid_calls": True,
                "models_requiring_check": [pairwise_model],
            },
            "guarded_runner": {
                "required": True,
                "status": guardrails["checks"][
                    "guarded_pairwise_runner_ready_for_new_information"
                ],
                "notes": [
                    (
                        "Current dry-run freezes the exact pair rows but does "
                        "not execute a paid runner."
                    ),
                    (
                        "A later paid execution must use a reviewed guarded "
                        "pairwise-only runner that loads historical pointwise "
                        "artifacts and forbids pointwise calls."
                    ),
                ],
            },
            "recommended_max_usd": go_no_go["recommended_max_usd"],
            "stop_rule": go_no_go["stop_rule"],
        },
        "per_pairwise_call_estimate": {
            "input_tokens": pairwise_estimate.input_tokens,
            "output_tokens": pairwise_estimate.output_tokens,
            "cost_usd": pairwise_estimate.cost_usd,
        },
        "totals": {
            "pointwise_calls": 0,
            "pairwise_scheduled_occurrences": scheduled_total,
            "pairwise_cached_occurrences": cached_occurrences,
            "pairwise_missing_occurrences": missing_occurrences,
            "unique_planned_pair_labels": unique_planned_total,
            "unique_missing_pairwise_labels": unique_missing_total,
            "pairwise_calls_to_buy": unique_missing_total,
            "input_tokens": unique_missing_total * pairwise_estimate.input_tokens,
            "output_tokens": unique_missing_total * pairwise_estimate.output_tokens,
            "estimated_additional_spend_usd": estimated_spend,
            "known_paid_spend_before_workflow_usd": CURRENT_KNOWN_SPEND_USD,
            "projected_known_paid_spend_after_workflow_usd": round(
                CURRENT_KNOWN_SPEND_USD + estimated_spend,
                6,
            ),
            "existing_planned_ledger_spend_usd": existing_ledger_spend,
            "projected_planned_ledger_spend_usd": projected_workflow_spend,
            "active_budget_shortfall": active_shortfall,
            "random_control_budget_shortfall": 0,
            "cache_reuse_by_artifact_dir": dict(sorted(cache_source_counts.items())),
            "cache_reuse_by_kind": dict(sorted(cache_kind_counts.items())),
        },
        "buckets": bucket_rows,
        "guardrails": guardrails,
        "caveats": caveats,
        "go_no_go": go_no_go,
        "planned_pair_occurrences_path": str(planned_pairs_output_path),
        "planned_pair_occurrence_count": len(occurrence_rows),
        "planned_unique_pair_labels_by_bucket": planned_unique,
        "validation_commands": [
            (
                "uv run python scripts/run_new_information_paid_dry_run.py "
                "--output artifacts/backtest-arxiv-new-information-paid-dry-run/"
                "paid-dry-run-go-no-go.json"
            ),
            "uv run pytest tests/test_new_information_paid_dry_run.py",
            "git diff --check",
            "uv run pytest -p no:cacheprovider",
        ],
    }
    validate_paid_dry_run_artifact_schema(payload)
    write_json_artifact(output_path, payload)
    return {**payload, "artifact_path": str(output_path)}


def validate_paid_dry_run_artifact_schema(payload: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_KEYS - set(payload))
    if missing:
        raise ValueError(
            "new-information paid dry-run artifact missing top-level keys: "
            + ", ".join(missing)
        )
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("new-information paid dry-run has unexpected artifact_type")
    if payload.get("dry_run") is not True:
        raise ValueError("new-information paid dry-run must be a dry-run artifact")
    if payload.get("paid_calls_made") != 0 or payload.get("paid_spend_usd") != 0.0:
        raise ValueError("new-information paid dry-run must make zero paid calls")
    if payload.get("pointwise_calls_made") != 0:
        raise ValueError("new-information paid dry-run must make zero pointwise calls")
    totals = payload.get("totals")
    if not isinstance(totals, Mapping):
        raise ValueError("new-information paid dry-run totals must be an object")
    if int(totals.get("pointwise_calls", -1)) != 0:
        raise ValueError("new-information paid dry-run totals must be pointwise-zero")
    guardrails = payload.get("guardrails")
    if not isinstance(guardrails, Mapping):
        raise ValueError("new-information paid dry-run guardrails must be an object")
    caveats = payload.get("caveats")
    if not isinstance(caveats, Mapping):
        raise ValueError("new-information paid dry-run caveats must be an object")
    go_no_go = payload.get("go_no_go")
    if not isinstance(go_no_go, Mapping):
        raise ValueError("new-information paid dry-run go_no_go must be an object")
    decision = go_no_go.get("decision")
    if decision not in {"go", "no_go"}:
        raise ValueError("new-information paid dry-run decision must be go/no_go")
    if decision == "go" and go_no_go.get("recommended_max_usd", 0.0) <= 0.0:
        raise ValueError("go decision requires a positive max-usd recommendation")
    if decision == "go" and caveats.get("unresolved_blocking_caveats"):
        raise ValueError("go decision cannot have unresolved blocking caveats")


def _guardrails(
    *,
    pairwise_model: str,
    artifact_dir: Path,
    source_artifact_dir: Path,
    ledger_path: Path,
    max_usd: float,
    estimated_spend_usd: float,
    projected_workflow_spend_usd: float,
    active_shortfall: int,
    active_gate_artifact: Mapping[str, Any],
    budget_fill_artifact: Mapping[str, Any],
    random_variance_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    active_gate_verdict = active_gate_artifact.get("gate_verdict") or {}
    active_gate_caveats = active_gate_artifact.get("caveats") or {}
    budget_caveat = active_gate_caveats.get("budget_completeness_caveat") or {}
    missing_caveat = active_gate_caveats.get("missing_label_caveat") or {}
    checks = {
        "source_budget_fill_zero_paid": budget_fill_artifact.get("paid_calls_made")
        == 0
        and float(budget_fill_artifact.get("paid_spend_usd") or 0.0) == 0.0,
        "source_budget_fill_zero_pointwise": budget_fill_artifact.get(
            "pointwise_calls_made"
        )
        == 0,
        "reviewed_active_arm_gate_paid_followup_allowed": active_gate_artifact.get(
            "paid_followup_allowed"
        )
        is True,
        "reviewed_active_arm_gate_no_blocking_reasons": not bool(
            active_gate_verdict.get("blocking_reasons") or []
        ),
        "reviewed_active_gate_missing_label_caveat_false": not bool(
            missing_caveat.get("present")
        ),
        "reviewed_active_gate_budget_completeness_caveat_false": not bool(
            budget_caveat.get("present")
        ),
        "frozen_plan_active_shortfall_zero": active_shortfall == 0,
        "provider_prefixed_model_name": "/" in pairwise_model,
        "model_availability_check_required_before_paid_calls": True,
        "jsonl_ledger_path_configured": bool(ledger_path)
        and ledger_path.suffix == ".jsonl",
        "separate_artifact_directory": artifact_dir.resolve()
        != source_artifact_dir.resolve(),
        "pointwise_calls_forbidden": True,
        "planned_pointwise_calls_zero": True,
        "pairwise_only_workflow": True,
        "workflow_hard_cap_lte_usd_2": 0.0 < max_usd <= 2.0,
        "estimated_additional_spend_lte_cap": estimated_spend_usd <= max_usd,
        "projected_workflow_ledger_spend_lte_cap": (
            projected_workflow_spend_usd <= max_usd
        ),
        "full_random_variance_reference_available": random_variance_artifact.get(
            "artifact_type"
        )
        == "sestina-full-random-variance-completion",
        "guarded_pairwise_runner_ready_for_new_information": False,
    }
    blocking = [
        key
        for key, value in checks.items()
        if isinstance(value, bool) and not value
    ]
    return {
        "checks": checks,
        "blocking_reasons": blocking,
        "paid_guardrails_clear": not blocking,
        "model_availability": {
            "status": "not_checked_dry_run",
            "required_before_paid_calls": True,
            "models_requiring_check": [pairwise_model],
        },
        "guarded_runner_note": (
            "The repo has guarded pairwise-only runners, but no paid runner path "
            "is currently wired to execute the 20-seed frozen new-information "
            "challenger manifest. That must be implemented or explicitly "
            "reviewed before any paid execution."
        ),
    }


def _caveats(
    *,
    budget_fill_artifact: Mapping[str, Any],
    active_gate_artifact: Mapping[str, Any],
    planned_active_shortfall: int,
    planned_unique_missing_labels: int,
) -> dict[str, Any]:
    replay_gate = budget_fill_artifact.get("new_information_replay_gate_verdict")
    weak_bucket = (
        (budget_fill_artifact.get("aggregate_diagnostics") or {}).get(
            "weak_bucket_deltas"
        )
        or {}
    )
    active_gate_caveats = active_gate_artifact.get("caveats") or {}
    unresolved: list[str] = []
    if isinstance(replay_gate, Mapping) and replay_gate.get(
        "weak_oracle_headroom_preserved"
    ) is False:
        unresolved.append("replay_local_weak_bucket_oracle_headroom_fell")
    if planned_active_shortfall:
        unresolved.append("planned_active_budget_shortfall")
    return {
        "reviewed_active_gate_caveats": {
            "budget_completeness": active_gate_caveats.get(
                "budget_completeness_caveat"
            ),
            "missing_label": active_gate_caveats.get("missing_label_caveat"),
            "offline_replay": active_gate_caveats.get("offline_replay_caveat"),
            "cache_reuse": active_gate_caveats.get("cache_reuse_caveat"),
        },
        "replay_local_false_negative_diagnostic": {
            "present": isinstance(replay_gate, Mapping),
            "blocking": bool(
                isinstance(replay_gate, Mapping)
                and replay_gate.get("paid_followup_allowed") is False
            ),
            "paid_followup_allowed": (
                replay_gate.get("paid_followup_allowed")
                if isinstance(replay_gate, Mapping)
                else None
            ),
            "blocking_reasons": (
                replay_gate.get("blocking_reasons")
                if isinstance(replay_gate, Mapping)
                else None
            ),
            "weak_oracle_headroom_preserved": (
                replay_gate.get("weak_oracle_headroom_preserved")
                if isinstance(replay_gate, Mapping)
                else None
            ),
            "mean_pointwise_plus_touched_recall_cap_delta": weak_bucket.get(
                "mean_pointwise_plus_touched_recall_cap_delta"
            ),
            "mean_positive_negative_pair_recall_cap_delta": weak_bucket.get(
                "mean_positive_negative_pair_recall_cap_delta"
            ),
            "unique_future_positives_touched_delta_total": weak_bucket.get(
                "unique_future_positives_touched_delta_total"
            ),
            "judgment": (
                "unresolved_blocking_caveat"
                if "replay_local_weak_bucket_oracle_headroom_fell" in unresolved
                else "cleared"
            ),
        },
        "planned_paid_label_caveat": {
            "unique_missing_pairwise_labels": planned_unique_missing_labels,
            "message": (
                "The frozen budget-filled replay schedule is cache-complete "
                "under the reviewed cache dirs."
                if planned_unique_missing_labels == 0
                else "The frozen schedule still needs paid pairwise labels."
            ),
        },
        "unresolved_blocking_caveats": unresolved,
    }


def _go_no_go(
    *,
    guardrails: Mapping[str, Any],
    caveats: Mapping[str, Any],
    estimated_spend_usd: float,
    max_usd: float,
) -> dict[str, Any]:
    guardrail_blocks = list(guardrails.get("blocking_reasons") or [])
    caveat_blocks = list(caveats.get("unresolved_blocking_caveats") or [])
    decision = "no_go" if guardrail_blocks or caveat_blocks else "go"
    if decision == "go":
        recommended_max = round(min(max_usd, max(0.01, estimated_spend_usd + 0.02)), 6)
        recommendation = (
            "Go only for a later reviewed pairwise-only paid workflow using the "
            "frozen manifest, JSONL ledger, provider-prefixed model availability "
            "check, hard cap, and immediate stop at the planned missing-label "
            "count or cap."
        )
    else:
        recommended_max = 0.0
        recommendation = (
            "No-go for paid execution. Resolve the listed guardrail failures and "
            "explicitly resolve or reviewer-accept the replay-local weak-bucket "
            "oracle-headroom caveat before buying labels."
        )
    return {
        "decision": decision,
        "guardrail_blocking_reasons": guardrail_blocks,
        "caveat_blocking_reasons": caveat_blocks,
        "estimated_additional_spend_usd": estimated_spend_usd,
        "requested_max_usd": max_usd,
        "recommended_max_usd": recommended_max,
        "stop_rule": (
            "Do not run paid calls in this workflow. If a later reviewer approves "
            "a go artifact, stop the paid run at the frozen unique missing-label "
            "count, the JSONL ledger cap, the first pointwise-call attempt, the "
            "first model-availability failure, or any parse/error retry requiring "
            "manual review."
        ),
        "recommendation": recommendation,
    }


def _freeze_cache_dirs(
    budget_fill_artifact: Mapping[str, Any],
    *,
    source_artifact_dir: Path,
    phase: str,
    explicit_dirs: Sequence[Path] | None,
) -> list[Path]:
    if explicit_dirs:
        return _pairwise_cache_dirs(
            source_artifact_dir,
            phase=phase,
            explicit_dirs=explicit_dirs,
        )
    recorded = (
        ((budget_fill_artifact.get("budget_fill") or {}).get("inputs") or {}).get(
            "pairwise_cache_artifact_dirs"
        )
        or budget_fill_artifact.get("pairwise_cache_artifact_dirs")
        or []
    )
    if recorded:
        dirs = [Path(str(path)) for path in recorded]
        if source_artifact_dir not in dirs:
            dirs.insert(0, source_artifact_dir)
        return _dedupe_paths(dirs)
    return _pairwise_cache_dirs(source_artifact_dir, phase=phase, explicit_dirs=None)


def _fallback_policy_from_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    fallback = ((payload.get("budget_fill") or {}).get("fallback_policy") or {})
    if isinstance(fallback, Mapping) and fallback:
        return dict(fallback)
    return {
        "name": "predeclared_cached_frontier_challenger_fallback",
        "purpose": "new_information_cached_frontier_fallback",
        "enabled": True,
        "frontier_multiplier": 4,
        "future_labels_used_for_scheduling": False,
        "cached_label_values_used_before_scheduling": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    output = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        output.append(path)
    return output


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stdout_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_path": payload.get("artifact_path"),
        "artifact_type": payload["artifact_type"],
        "dry_run": payload["dry_run"],
        "paid_calls_made": payload["paid_calls_made"],
        "paid_spend_usd": payload["paid_spend_usd"],
        "pointwise_calls_made": payload["pointwise_calls_made"],
        "active_arm_name": payload["frozen_inputs"]["active_arm_name"],
        "random_control_comparator": payload["frozen_inputs"][
            "random_control_comparator"
        ],
        "totals": payload["totals"],
        "go_no_go": payload["go_no_go"],
        "planned_pair_occurrences_path": payload["planned_pair_occurrences_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
