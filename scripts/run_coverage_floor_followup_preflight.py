#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ci_partition_gate import DEFAULT_SEEDS, _parse_seeds  # noqa: E402
from sestina.active_arm_gate import (  # noqa: E402
    CURRENT_KNOWN_SPEND_USD,
    DEFAULT_PAID_CAP_USD,
    validate_active_arm_gate_artifact_schema,
)
from sestina.backtest_budget import load_config  # noqa: E402
from sestina.backtest_runner import (  # noqa: E402
    JsonlLedger,
    ModelAvailabilityError,
    _call_artifact,
    _call_estimate,
    _chat_json_with_usage,
    _comparison_from_pairwise_response,
    _config_for_phase,
    _ledger_entry,
    _normalize_rates_from_config,
    _normalize_token_assumptions_from_config,
    _pairwise_payload,
    check_model_availability,
    load_dataset_manifest,
    usage_cost_payload,
    validate_model_names,
)
from sestina.candidates import select_candidates  # noqa: E402
from sestina.ci_partition_gate import (  # noqa: E402
    CIPartitionConfig,
    schedule_cached_exact_pool_random,
)
from sestina.diagnostics import fingerprint, write_json_artifact  # noqa: E402
from sestina.experiment_protocol import (  # noqa: E402
    build_next_experiment_protocol,
    validate_next_experiment_protocol,
)
from sestina.models import (  # noqa: E402
    PairwiseComparison,
    PairwiseOrderMetadata,
    Paper,
    PointwiseAssessment,
    ScheduledPair,
)
from sestina.no_paid_algorithm_sweep import (  # noqa: E402
    HybridScheduleConfig,
    canonical_pair_key,
    schedule_model_visible_hybrid_pairs,
)
from sestina.scheduler import resolve_pairwise_budget  # noqa: E402
from sestina.scheduler_followup import (  # noqa: E402
    PointwiseArtifactError,
    load_pointwise_papers_from_artifacts,
)


ARTIFACT_TYPE = "sestina-coverage-floor-followup-preflight"
SCHEMA_VERSION = 1
ARM_COVERAGE = "randomized_coverage_floor_hybrid_cached_replay"
ARM_EXACT = "exact_pool_random_cached_replay"
PAIRWISE_CALL_KIND = "pairwise_active"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts" / "backtest-arxiv-coverage-floor-followup-preflight"
)
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "arxiv_historical_pilot_budget_config.json"
DEFAULT_NO_PAID_SWEEP_ARTIFACT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-no-paid-algorithm-sweep"
    / "no-paid-algorithm-sweep.json"
)
DEFAULT_ACTIVE_GATE_ARTIFACT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-no-paid-algorithm-sweep"
    / "active-arm-gate.json"
)
DEFAULT_FRESH_HOLDOUT_MANIFEST = (
    REPO_ROOT
    / "artifacts"
    / "backtest-datasets"
    / "arxiv-historical-coverage-floor-fresh-holdout-manifest.json"
)
DEFAULT_SOURCE_ARTIFACT_DIR = (
    REPO_ROOT / "artifacts" / "backtest-arxiv-coverage-floor-fresh-holdout-pointwise"
)
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "coverage-floor-followup-preflight.json"
DEFAULT_LEDGER = DEFAULT_OUTPUT_DIR / "coverage-floor-followup-ledger.jsonl"
DEFAULT_PLANNED_PAIRS = DEFAULT_OUTPUT_DIR / "planned-pair-occurrences.jsonl"
DEFAULT_MAX_USD = round(DEFAULT_PAID_CAP_USD - CURRENT_KNOWN_SPEND_USD, 6)
REQUIRED_TOP_LEVEL_KEYS = {
    "artifact_type",
    "schema_version",
    "mode",
    "dry_run",
    "paid_calls_made",
    "paid_spend_usd",
    "pointwise_calls_made",
    "method",
    "input_artifacts",
    "frozen_no_paid_sweep",
    "fresh_holdout",
    "provider_model_availability",
    "planned_execution",
    "ledger",
    "max_usd_cap",
    "totals",
    "guardrails",
    "final_go_no_go",
    "validation_commands",
}
RunnerMode = Literal["planning", "execute"]
UrlOpen = Any


class CoverageFloorPreflightError(RuntimeError):
    """Base class for coverage-floor preflight failures."""


class PointwiseCallForbiddenError(CoverageFloorPreflightError):
    """Raised before any pointwise-like call can be attempted."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the pairwise-only dry-run/preflight artifact for the "
            "coverage-floor no-paid sweep winner. This never makes pointwise "
            "calls and only executes pairwise labels in --mode execute after all "
            "guardrails pass."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--no-paid-sweep-artifact",
        type=Path,
        default=DEFAULT_NO_PAID_SWEEP_ARTIFACT,
    )
    parser.add_argument(
        "--active-gate-artifact",
        type=Path,
        default=DEFAULT_ACTIVE_GATE_ARTIFACT,
    )
    parser.add_argument(
        "--fresh-holdout-manifest",
        type=Path,
        default=DEFAULT_FRESH_HOLDOUT_MANIFEST,
    )
    parser.add_argument(
        "--source-artifact-dir",
        type=Path,
        default=DEFAULT_SOURCE_ARTIFACT_DIR,
        help=(
            "reviewed pointwise artifact directory for the fresh holdout; this "
            "script will not create pointwise labels"
        ),
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--planned-pairs-output", type=Path, default=DEFAULT_PLANNED_PAIRS)
    parser.add_argument("--phase", default=None)
    parser.add_argument(
        "--seeds",
        default=None,
        help="comma-separated seed set; defaults to the frozen no-paid sweep seeds",
    )
    parser.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--mode", choices=("planning", "execute"), default="planning")
    parser.add_argument(
        "--confirm-guarded-pairwise-only-execution",
        action="store_true",
        help="required with --mode execute before any pairwise network calls",
    )
    args = parser.parse_args(argv)

    payload = build_coverage_floor_followup_preflight(
        config_path=args.config,
        no_paid_sweep_artifact_path=args.no_paid_sweep_artifact,
        active_gate_artifact_path=args.active_gate_artifact,
        fresh_holdout_manifest_path=args.fresh_holdout_manifest,
        source_artifact_dir=args.source_artifact_dir,
        artifact_dir=args.artifact_dir,
        ledger_path=args.ledger,
        output_path=args.output,
        planned_pairs_output_path=args.planned_pairs_output,
        phase=args.phase,
        seeds=_parse_seeds(args.seeds) if args.seeds else None,
        max_usd=args.max_usd,
        timeout_seconds=args.timeout_seconds,
        mode=args.mode,
        confirm_guarded_pairwise_only_execution=(
            args.confirm_guarded_pairwise_only_execution
        ),
    )
    sys.stdout.write(json.dumps(_stdout_summary(payload), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    if args.mode == "execute" and payload["final_go_no_go"]["decision"] != "go":
        return 2
    return 0


def build_coverage_floor_followup_preflight(
    *,
    config_path: Path,
    no_paid_sweep_artifact_path: Path,
    active_gate_artifact_path: Path,
    fresh_holdout_manifest_path: Path,
    source_artifact_dir: Path,
    artifact_dir: Path,
    ledger_path: Path,
    output_path: Path,
    planned_pairs_output_path: Path,
    phase: str | None = None,
    seeds: Sequence[int] | None = None,
    max_usd: float = DEFAULT_MAX_USD,
    timeout_seconds: float = 30.0,
    mode: RunnerMode = "planning",
    confirm_guarded_pairwise_only_execution: bool = False,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    no_paid_sweep = _read_json(no_paid_sweep_artifact_path)
    active_gate = _read_json(active_gate_artifact_path)
    validate_active_arm_gate_artifact_schema(active_gate)
    raw_config = load_config(config_path)
    frozen = _freeze_no_paid_winner(
        no_paid_sweep=no_paid_sweep,
        active_gate=active_gate,
        no_paid_sweep_artifact_path=no_paid_sweep_artifact_path,
        active_gate_artifact_path=active_gate_artifact_path,
        requested_phase=phase,
        requested_seeds=seeds,
    )
    phase_name = str(frozen["phase"])
    seed_set = [int(seed) for seed in frozen["seeds"]]
    phase_config = _config_for_phase(raw_config, phase=phase_name)["phases"][0]
    pairwise_model = str(phase_config["pairwise_model"])
    token_assumptions = _normalize_token_assumptions_from_config(raw_config)
    rates = _normalize_rates_from_config(raw_config)
    pairwise_estimate = _call_estimate(
        "pairwise",
        pairwise_model,
        token_assumptions,
        rates,
    )
    provider = _provider_model_availability(
        pairwise_model=pairwise_model,
        timeout_seconds=timeout_seconds,
        urlopen=urlopen,
    )
    fresh_holdout, manifest = _fresh_holdout_state(
        manifest_path=fresh_holdout_manifest_path,
        no_paid_sweep=no_paid_sweep,
        phase=phase_name,
    )
    planned_rows: list[dict[str, Any]] = []
    bucket_plans: list[dict[str, Any]] = []
    planning_errors: list[str] = []
    if manifest is not None and fresh_holdout["usable_for_planning"] is True:
        try:
            planned_rows, bucket_plans = _plan_pairwise_rows(
                manifest=manifest,
                source_artifact_dir=source_artifact_dir,
                phase=phase_name,
                seeds=seed_set,
                frozen=frozen,
            )
        except PointwiseArtifactError as exc:
            planning_errors.append("fresh_pointwise_artifacts_missing")
            fresh_holdout["pointwise_artifacts"] = {
                "status": "missing",
                "source_artifact_dir": str(source_artifact_dir),
                "pointwise_calls_made": 0,
                "error": str(exc),
            }
        except Exception as exc:
            planning_errors.append("pairwise_planning_failed")
            fresh_holdout["planning_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        else:
            fresh_holdout["pointwise_artifacts"] = {
                "status": "available",
                "source_artifact_dir": str(source_artifact_dir),
                "pointwise_calls_made": 0,
                "policy": "reviewed pointwise artifacts loaded; no new pointwise calls",
            }
    else:
        fresh_holdout["pointwise_artifacts"] = {
            "status": "not_checked_manifest_unavailable",
            "source_artifact_dir": str(source_artifact_dir),
            "pointwise_calls_made": 0,
        }

    existing_pairwise_artifacts, existing_pairwise_artifact_report = (
        _existing_pairwise_artifacts(artifact_dir=artifact_dir, phase=phase_name)
    )
    if planned_rows:
        planned_rows = _mark_cached_pairwise_rows(
            planned_rows,
            existing_pairwise_artifacts=existing_pairwise_artifacts,
        )
    planned_stats = _planned_pair_stats(planned_rows)
    estimated_additional_spend = round(
        planned_stats["unique_missing_pairwise_labels"] * pairwise_estimate.cost_usd,
        6,
    )
    existing_ledger_spend = JsonlLedger(ledger_path).existing_spend_usd()
    projected_ledger_spend = round(
        existing_ledger_spend + estimated_additional_spend,
        6,
    )
    resume_state = _pairwise_resume_state(
        ledger_path=ledger_path,
        existing_pairwise_artifact_paths=set(existing_pairwise_artifacts.values()),
        existing_pairwise_artifact_report=existing_pairwise_artifact_report,
    )
    if planned_rows:
        _write_jsonl(planned_pairs_output_path, planned_rows)
        planned_pairs_path: str | None = str(planned_pairs_output_path)
        planned_pairs_sha = _sha256(planned_pairs_output_path)
    else:
        planned_pairs_path = None
        planned_pairs_sha = None

    protocol = build_next_experiment_protocol(
        no_paid_gate_artifact=active_gate,
        priority_direction="no_paid_replay_gate_randomized_coverage_floor",
        fresh_holdout_request={
            "requested": True,
            "dry_run": True,
            "provider_availability_check": True,
            "ledger_path": str(ledger_path),
            "separate_artifact_directory": artifact_dir.resolve()
            != source_artifact_dir.resolve(),
            "max_usd": max_usd,
            "known_spend_before_validation_usd": CURRENT_KNOWN_SPEND_USD,
            "pointwise_calls_planned": 0,
            "explicit_pointwise_approval": False,
            "standing_campaign_authorization": True,
            "fresh_holdout_pointwise_artifacts_authorized": True,
            "historical_paid_artifacts_immutable": True,
        },
    )
    validate_next_experiment_protocol(protocol)
    guarded_execution = _guarded_execution_state(
        mode=mode,
        confirm_guarded_pairwise_only_execution=(
            confirm_guarded_pairwise_only_execution
        ),
        planned_stats=planned_stats,
        provider_model_availability=provider,
        max_usd=max_usd,
        estimated_additional_spend=estimated_additional_spend,
        projected_ledger_spend=projected_ledger_spend,
        artifact_dir=artifact_dir,
        ledger_path=ledger_path,
        output_path=output_path,
    )
    guardrails = _guardrails(
        no_paid_sweep=no_paid_sweep,
        active_gate=active_gate,
        frozen=frozen,
        fresh_holdout=fresh_holdout,
        provider_model_availability=provider,
        planned_stats=planned_stats,
        planning_errors=planning_errors,
        max_usd=max_usd,
        artifact_dir=artifact_dir,
        source_artifact_dir=source_artifact_dir,
        ledger_path=ledger_path,
        existing_ledger_spend=existing_ledger_spend,
        estimated_additional_spend=estimated_additional_spend,
        projected_ledger_spend=projected_ledger_spend,
        resume_state=resume_state,
        protocol=protocol,
        guarded_execution=guarded_execution,
    )
    final_go_no_go = _final_go_no_go(
        guardrails=guardrails,
        provider_model_availability=provider,
        guarded_execution=guarded_execution,
        estimated_additional_spend=estimated_additional_spend,
        mode=mode,
    )
    execution_summary = {
        "attempted": False,
        "status": "not_attempted",
        "paid_pairwise_calls_attempted": 0,
        "paid_pairwise_calls_succeeded": 0,
        "new_ledger_entries": 0,
    }
    paid_calls_made = 0
    paid_spend_usd = 0.0
    if mode == "execute" and final_go_no_go["decision"] == "go":
        ledger = JsonlLedger(ledger_path)
        spend_before = ledger.existing_spend_usd()
        execution_summary = _execute_pairwise_only(
            planned_rows=planned_rows,
            source_artifact_dir=source_artifact_dir,
            phase=phase_name,
            artifact_dir=artifact_dir,
            ledger=ledger,
            max_usd=max_usd,
            pairwise_model=pairwise_model,
            pairwise_estimate=pairwise_estimate,
            rates=rates,
            timeout_seconds=timeout_seconds,
            urlopen=urlopen,
        )
        paid_calls_made = int(execution_summary["paid_pairwise_calls_succeeded"])
        paid_spend_usd = round(ledger.existing_spend_usd() - spend_before, 6)

    totals = {
        **planned_stats,
        "pairwise_calls_to_buy": planned_stats["unique_missing_pairwise_labels"],
        "estimated_additional_spend_usd": estimated_additional_spend,
        "known_paid_spend_before_workflow_usd": CURRENT_KNOWN_SPEND_USD,
        "paid_cap_usd": DEFAULT_PAID_CAP_USD,
        "projected_known_paid_spend_after_workflow_usd": round(
            CURRENT_KNOWN_SPEND_USD + estimated_additional_spend,
            6,
        ),
        "existing_planned_ledger_spend_usd": existing_ledger_spend,
        "projected_planned_ledger_spend_usd": projected_ledger_spend,
    }
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "dry_run": mode == "planning",
        "paid_calls_made": paid_calls_made,
        "paid_spend_usd": paid_spend_usd,
        "pointwise_calls_made": 0,
        "method": {
            "summary": (
                "Pairwise-only fresh-holdout dry-run/preflight for the "
                "coverage-floor no-paid sweep winner."
            ),
            "paid_labeling_invoked": mode == "execute" and paid_calls_made > 0,
            "pointwise_runner_invoked": False,
            "pointwise_calls_forbidden": True,
            "label_generation_calls_made": paid_calls_made,
            "model_availability_check": "GET /models",
            "secrets_printed_or_stored": False,
        },
        "input_artifacts": {
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "no_paid_sweep_artifact_path": str(no_paid_sweep_artifact_path),
            "no_paid_sweep_artifact_sha256": _sha256(no_paid_sweep_artifact_path),
            "active_gate_artifact_path": str(active_gate_artifact_path),
            "active_gate_artifact_sha256": _sha256(active_gate_artifact_path),
            "fresh_holdout_manifest_path": str(fresh_holdout_manifest_path),
            "fresh_holdout_manifest_sha256": (
                _sha256(fresh_holdout_manifest_path)
                if fresh_holdout_manifest_path.exists()
                else None
            ),
            "source_artifact_dir": str(source_artifact_dir),
            "planned_pairs_path": planned_pairs_path,
            "planned_pairs_sha256": planned_pairs_sha,
            "runner_code_path": str(
                REPO_ROOT / "scripts" / "run_coverage_floor_followup_preflight.py"
            ),
        },
        "frozen_no_paid_sweep": frozen,
        "fresh_holdout": fresh_holdout,
        "provider_model_availability": provider,
        "planned_execution": {
            "mode": mode,
            "active_arm_name": ARM_COVERAGE,
            "random_control_variant": ARM_EXACT,
            "pairwise_model": pairwise_model,
            "model_provider": pairwise_model.split("/", 1)[0],
            "planned_pair_occurrences_path": planned_pairs_path,
            "planned_pair_occurrences_written": bool(planned_pairs_path),
            "planned_call_kind": PAIRWISE_CALL_KIND,
            "pointwise_calls_planned": 0,
            "pairwise_call_roles": ["coverage_floor_active", "exact_pool_random_control"],
            "per_pairwise_call_estimate": {
                "input_tokens": pairwise_estimate.input_tokens,
                "output_tokens": pairwise_estimate.output_tokens,
                "cost_usd": pairwise_estimate.cost_usd,
            },
            "bucket_plans": bucket_plans,
            "no_paid_label_purchase_authorized_by_preflight": True,
        },
        "ledger": {
            "path": str(ledger_path),
            "format": "jsonl",
            "line_count": _ledger_line_count(ledger_path),
            "existing_spend_usd_before_workflow": existing_ledger_spend,
            "spend_usd_after_workflow": JsonlLedger(ledger_path).existing_spend_usd(),
            "historical_ledgers_rewritten": False,
        },
        "resume_state": resume_state,
        "max_usd_cap": {
            "requested_max_usd": max_usd,
            "cap_policy_max_usd": DEFAULT_MAX_USD,
            "paid_cap_usd": DEFAULT_PAID_CAP_USD,
            "known_paid_spend_before_workflow_usd": CURRENT_KNOWN_SPEND_USD,
            "estimated_additional_spend_usd": estimated_additional_spend,
            "projected_planned_ledger_spend_usd": projected_ledger_spend,
        },
        "totals": totals,
        "guardrails": guardrails,
        "guarded_execution": guarded_execution,
        "execution_summary": execution_summary,
        "final_go_no_go": final_go_no_go,
        "next_experiment_protocol": protocol,
        "validation_commands": [
            "uv run python scripts/run_coverage_floor_followup_preflight.py",
            "uv run python -m json.tool artifacts/backtest-arxiv-coverage-floor-followup-preflight/coverage-floor-followup-preflight.json >/dev/null",
            "uv run pytest tests/test_coverage_floor_followup_preflight.py",
            "git diff --check",
        ],
        "output_path": str(output_path),
    }
    validate_coverage_floor_preflight_artifact_schema(payload)
    write_json_artifact(output_path, payload)
    return {**payload, "artifact_path": str(output_path)}


def validate_coverage_floor_preflight_artifact_schema(
    payload: Mapping[str, Any],
) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(payload))
    if missing:
        raise ValueError(
            "coverage-floor preflight artifact missing top-level keys: "
            + ", ".join(missing)
        )
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("coverage-floor preflight has unexpected artifact_type")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("coverage-floor preflight has unexpected schema_version")
    if payload.get("pointwise_calls_made") != 0:
        raise ValueError("coverage-floor preflight must make zero pointwise calls")
    if payload.get("mode") == "planning":
        if payload.get("dry_run") is not True:
            raise ValueError("planning preflight must be marked dry_run")
        if payload.get("paid_calls_made") != 0:
            raise ValueError("planning preflight must make zero paid calls")
        if float(payload.get("paid_spend_usd") or 0.0) != 0.0:
            raise ValueError("planning preflight must spend zero USD")
    cap = (payload.get("max_usd_cap") or {}).get("requested_max_usd")
    if not isinstance(cap, int | float) or not (0.0 < float(cap) <= DEFAULT_MAX_USD):
        raise ValueError(
            "coverage-floor preflight max-usd cap must be within campaign remaining"
        )
    final = payload.get("final_go_no_go")
    if not isinstance(final, Mapping):
        raise ValueError("coverage-floor preflight final_go_no_go must be an object")
    if final.get("decision") not in {"go", "no_go"}:
        raise ValueError("coverage-floor preflight decision must be go/no_go")
    totals = payload.get("totals")
    if not isinstance(totals, Mapping):
        raise ValueError("coverage-floor preflight totals must be an object")
    if int(totals.get("pointwise_calls") or 0) != 0:
        raise ValueError("coverage-floor preflight totals must be pointwise-zero")
    serialized = json.dumps(payload, sort_keys=True)
    if "Authorization" in serialized or "Bearer " in serialized:
        raise ValueError("coverage-floor preflight artifact must not include secrets")


def _freeze_no_paid_winner(
    *,
    no_paid_sweep: Mapping[str, Any],
    active_gate: Mapping[str, Any],
    no_paid_sweep_artifact_path: Path,
    active_gate_artifact_path: Path,
    requested_phase: str | None,
    requested_seeds: Sequence[int] | None,
) -> dict[str, Any]:
    analysis = _mapping(no_paid_sweep.get("analysis_parameters"))
    active_arm_name = str(active_gate.get("active_arm_name") or "")
    random_control = str(active_gate.get("candidate_random_control_baseline") or "")
    if active_arm_name != ARM_COVERAGE:
        raise ValueError(f"active gate winner must be {ARM_COVERAGE}")
    if random_control != ARM_EXACT:
        raise ValueError(f"active gate random control must be {ARM_EXACT}")
    frozen_seeds = [int(seed) for seed in (analysis.get("seeds") or DEFAULT_SEEDS)]
    if requested_seeds is not None and [int(seed) for seed in requested_seeds] != frozen_seeds:
        raise ValueError("requested seeds must match the frozen no-paid sweep seed set")
    phase = requested_phase or str(no_paid_sweep.get("phase") or "pilot")
    if phase != str(no_paid_sweep.get("phase") or "pilot"):
        raise ValueError("requested phase must match the frozen no-paid sweep phase")
    aggregate_metrics = _mapping(no_paid_sweep.get("aggregate_metrics")).get(
        ARM_COVERAGE,
        {},
    )
    paired_deltas = no_paid_sweep.get("paired_deltas_vs_exact_pool_random") or {}
    label_policy = _mapping(no_paid_sweep.get("label_policy"))
    gate_verdict = _mapping(active_gate.get("gate_verdict"))
    return {
        "active_arm_name": active_arm_name,
        "random_control_variant": random_control,
        "candidate_arms_tried": no_paid_sweep.get("candidate_arms_tried") or [],
        "control_arms": no_paid_sweep.get("control_arms") or [],
        "phase": phase,
        "seeds": frozen_seeds,
        "seed_count": len(frozen_seeds),
        "metrics": {
            "primary": "recall_at_k",
            "secondary": ["ndcg_at_k", "average_precision"],
            "aggregate_metrics": aggregate_metrics,
            "paired_active_minus_random_deltas": paired_deltas.get("metric_deltas")
            or {},
        },
        "active_policy": {
            "method": ARM_COVERAGE,
            "selected_strategy": "posterior_topk",
            "source_method": "model_visible_cache_safe_hybrid",
            "random_floor_fraction": 0.35,
            "min_random_floor_pairs": 1,
            "per_item_cap": 6,
            "anchor_multiplier": 2,
            "challenger_multiplier": 5,
            "scheduler_samples": int(analysis.get("scheduler_samples") or 800),
            "posterior_samples": int(analysis.get("posterior_samples") or 900),
            "pairwise_strength": float(analysis.get("pairwise_strength") or 2.5),
            "confidence_z": float(analysis.get("confidence_z") or 1.96),
        },
        "proposal_pool": {
            "replay_policy": "cached labels were only revealed after scheduling",
            "fresh_holdout_policy": (
                "use all same-bucket candidate pair keys available before "
                "labeling; do not use cache label values"
            ),
            "model_visible_signals": [
                "pointwise_good_probability",
                "pointwise_uncertainty",
                "pointwise_rubric_scores",
                "title_abstract_text_length",
                "metadata_category",
            ],
            "cached_pair_availability_used_in_no_paid_replay": bool(
                label_policy.get("cache_availability_used_for_scheduling")
            ),
            "cached_label_values_used_before_scheduling": False,
        },
        "stopping_rule": {
            "dry_run_preflight_stop_rule": (
                "Stop before paid labels if the fresh holdout manifest or "
                "reviewed pointwise artifacts are absent, provider/model "
                "availability is unavailable, planned rows are not pairwise-only, "
                "the JSONL ledger is invalid, the hard cap would be exceeded, "
                "or leakage checks fail."
            ),
            "no_paid_gate_rule": gate_verdict.get("recommended_next_action"),
        },
        "no_leakage_constraints": {
            "future_labels_used_for_scheduling": False,
            "future_labels_used_as_model_features": False,
            "good_paper_used_for_scheduling": False,
            "matched_title_or_work_id_used_for_scheduling": False,
            "cached_label_values_used_before_scheduling": False,
            "future_labels_for_retrospective_evaluation_only": True,
            "source_policy": active_gate.get("label_leakage") or {},
        },
        "input_artifacts": {
            "no_paid_sweep_artifact_path": str(no_paid_sweep_artifact_path),
            "no_paid_sweep_artifact_sha256": _sha256(no_paid_sweep_artifact_path),
            "active_gate_artifact_path": str(active_gate_artifact_path),
            "active_gate_artifact_sha256": _sha256(active_gate_artifact_path),
            "development_replay_manifest_path": no_paid_sweep.get("manifest_path"),
            "development_replay_set": no_paid_sweep.get("development_replay_set"),
        },
    }


def _fresh_holdout_state(
    *,
    manifest_path: Path,
    no_paid_sweep: Mapping[str, Any],
    phase: str,
) -> tuple[dict[str, Any], Any | None]:
    builder_command = (
        "uv run python scripts/build_arxiv_historical_manifest.py "
        "--bucket CATEGORY:YYYY-MM --bucket CATEGORY:YYYY-MM "
        "--limit 80 --k 5 --phase pilot --metadata-provider auto "
        "--unmatched-policy drop --output "
        "artifacts/backtest-datasets/arxiv-historical-coverage-floor-"
        "fresh-holdout-manifest.json"
    )
    base = {
        "manifest_path": str(manifest_path),
        "requested_phase": phase,
        "builder": {
            "existing_tool": "scripts/build_arxiv_historical_manifest.py",
            "can_build_after_predeclared_bucket_specs": True,
            "not_run_by_preflight_reason": (
                "manifest construction is handled by the autonomous fresh "
                "holdout design/build step before this pairwise-only preflight"
            ),
            "reproduction_command_template": builder_command,
        },
    }
    if not manifest_path.exists():
        return {
            **base,
            "present": False,
            "status": "blocked_missing_fresh_holdout_manifest",
            "usable_for_planning": False,
            "blocking_reasons": ["fresh_holdout_manifest_missing"],
            "paid_label_generation_needed_to_create_manifest": False,
            "fresh_validation_claimed": False,
        }, None
    manifest = load_dataset_manifest(manifest_path)
    buckets = manifest.buckets_for_phase(phase)
    development_path = no_paid_sweep.get("manifest_path")
    development_names = _development_bucket_names(no_paid_sweep)
    bucket_names = [bucket.name for bucket in buckets]
    same_manifest = _same_path(manifest_path, development_path)
    overlaps_development = bool(set(bucket_names) & development_names)
    blockers = []
    if not buckets:
        blockers.append("fresh_holdout_manifest_has_no_requested_phase_buckets")
    if same_manifest or overlaps_development:
        blockers.append("fresh_holdout_manifest_is_development_replay")
    return {
        **base,
        "present": True,
        "status": "loaded" if not blockers else "blocked_manifest_not_fresh",
        "usable_for_planning": not blockers,
        "blocking_reasons": blockers,
        "bucket_count": len(buckets),
        "bucket_names": bucket_names,
        "same_as_development_replay_manifest": same_manifest,
        "overlaps_development_replay_buckets": overlaps_development,
        "label_policy": {
            "future_citation_labels_allowed_only_for_later_evaluation": True,
            "labels_loaded_into_scheduler": False,
            "good_paper_used_for_scheduling": False,
        },
        "fresh_validation_claimed": False,
    }, manifest


def _plan_pairwise_rows(
    *,
    manifest: Any,
    source_artifact_dir: Path,
    phase: str,
    seeds: Sequence[int],
    frozen: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policy = _mapping(frozen.get("active_policy"))
    active_config = HybridScheduleConfig(
        name=ARM_COVERAGE,
        random_floor_fraction=float(policy.get("random_floor_fraction") or 0.35),
        min_random_floor_pairs=int(policy.get("min_random_floor_pairs") or 1),
        per_item_cap=int(policy.get("per_item_cap") or 6),
        anchor_multiplier=int(policy.get("anchor_multiplier") or 2),
        challenger_multiplier=int(policy.get("challenger_multiplier") or 5),
    )
    random_config = CIPartitionConfig(
        pairwise_strength=float(policy.get("pairwise_strength") or 2.5),
        posterior_samples=int(policy.get("scheduler_samples") or 800),
        confidence_z=float(policy.get("confidence_z") or 1.96),
        random_floor_fraction=float(policy.get("random_floor_fraction") or 0.35),
    )
    rows: list[dict[str, Any]] = []
    bucket_plans: list[dict[str, Any]] = []
    for bucket in manifest.buckets_for_phase(phase):
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
        all_pair_keys = _all_unordered_pair_keys(papers)
        bucket_rows_before = len(rows)
        for seed in seeds:
            active_schedule, active_diagnostics = schedule_model_visible_hybrid_pairs(
                papers,
                k=bucket.k,
                budget=budget,
                seed=int(seed),
                available_pair_keys=all_pair_keys,
                config=active_config,
            )
            random_schedule = schedule_cached_exact_pool_random(
                papers,
                [],
                k=bucket.k,
                budget=budget,
                seed=int(seed),
                config=random_config,
                available_pair_keys=None,
            )
            rows.extend(
                _rows_for_schedule(
                    schedule=active_schedule,
                    arm_name=ARM_COVERAGE,
                    role="coverage_floor_active",
                    bucket_name=bucket.name,
                    k=bucket.k,
                    seed=int(seed),
                )
            )
            rows.extend(
                _rows_for_schedule(
                    schedule=random_schedule.pairs,
                    arm_name=ARM_EXACT,
                    role="exact_pool_random_control",
                    bucket_name=bucket.name,
                    k=bucket.k,
                    seed=int(seed),
                )
            )
            if int(seed) == int(seeds[0]):
                bucket_plans.append(
                    {
                        "bucket": bucket.name,
                        "k": bucket.k,
                        "papers_total": len(papers),
                        "candidate_size": len(selection.candidate_ids),
                        "pairwise_budget_per_seed": budget.to_dict(),
                        "first_seed": int(seed),
                        "active_schedule_diagnostics": active_diagnostics,
                        "random_control_schedule_diagnostics": random_schedule.diagnostics,
                    }
                )
        bucket_plans[-1]["planned_occurrences_all_seeds"] = (
            len(rows) - bucket_rows_before
        )
    return rows, bucket_plans


def _rows_for_schedule(
    *,
    schedule: Sequence[ScheduledPair],
    arm_name: str,
    role: str,
    bucket_name: str,
    k: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    for index, pair in enumerate(schedule, start=1):
        key = canonical_pair_key(pair.left_id, pair.right_id)
        output.append(
            {
                "row_id": f"{seed}:{bucket_name}:{arm_name}:{index}",
                "seed": seed,
                "bucket": bucket_name,
                "k": k,
                "arm_name": arm_name,
                "row_role": role,
                "pair_index": index,
                "pair_key": list(key),
                "left_id": pair.left_id,
                "right_id": pair.right_id,
                "purpose": pair.purpose,
                "priority": pair.priority,
                "order": pair.order.to_dict(),
                "planned_call_kind": PAIRWISE_CALL_KIND,
                "cache_status": "missing_label",
                "cached_artifact_path": None,
                "cached_artifact_kind": None,
                "future_labels_used_for_scheduling": False,
                "cached_label_values_used_before_scheduling": False,
                "good_paper_used_for_scheduling": False,
                "matched_title_or_work_id_used_for_scheduling": False,
            }
        )
    return output


def _guardrails(
    *,
    no_paid_sweep: Mapping[str, Any],
    active_gate: Mapping[str, Any],
    frozen: Mapping[str, Any],
    fresh_holdout: Mapping[str, Any],
    provider_model_availability: Mapping[str, Any],
    planned_stats: Mapping[str, Any],
    planning_errors: Sequence[str],
    max_usd: float,
    artifact_dir: Path,
    source_artifact_dir: Path,
    ledger_path: Path,
    existing_ledger_spend: float,
    estimated_additional_spend: float,
    projected_ledger_spend: float,
    resume_state: Mapping[str, Any],
    protocol: Mapping[str, Any],
    guarded_execution: Mapping[str, Any],
) -> dict[str, Any]:
    gate_verdict = _mapping(active_gate.get("gate_verdict"))
    holdout_protocol = _mapping(protocol.get("fresh_holdout_validation_protocol"))
    checks = {
        "no_paid_sweep_zero_paid": no_paid_sweep.get("paid_calls_made") == 0
        and float(no_paid_sweep.get("paid_spend_usd") or 0.0) == 0.0,
        "no_paid_sweep_zero_pointwise": no_paid_sweep.get("pointwise_calls_made") == 0,
        "active_gate_winner_matches_coverage_floor": frozen.get("active_arm_name")
        == ARM_COVERAGE,
        "random_control_variant_frozen": frozen.get("random_control_variant")
        == ARM_EXACT,
        "active_gate_paid_followup_allowed": active_gate.get("paid_followup_allowed")
        is True
        and gate_verdict.get("paid_followup_allowed") is True,
        "active_gate_no_blocking_reasons": not bool(
            gate_verdict.get("blocking_reasons") or []
        ),
        "active_gate_no_leakage": gate_verdict.get(
            "no_future_label_or_cached_label_leakage"
        )
        is True,
        "fresh_holdout_manifest_present": fresh_holdout.get("present") is True,
        "fresh_holdout_manifest_usable": fresh_holdout.get("usable_for_planning")
        is True,
        "fresh_holdout_pointwise_artifacts_available": (
            fresh_holdout.get("pointwise_artifacts") or {}
        ).get("status")
        == "available",
        "no_pairwise_planning_errors": not bool(planning_errors),
        "provider_model_available": provider_model_availability.get("status")
        == "available",
        "jsonl_ledger_path_configured": ledger_path.suffix == ".jsonl",
        "ledger_under_separate_artifact_directory": _is_relative_to(
            ledger_path.resolve(),
            artifact_dir.resolve(),
        ),
        "separate_artifact_directory": artifact_dir.resolve()
        != source_artifact_dir.resolve(),
        "historical_ledgers_not_rewritten": True,
        "hard_max_usd_cap_within_campaign_remaining": (
            0.0 < max_usd <= DEFAULT_MAX_USD
        ),
        "hard_max_usd_cap_lte_paid_cap": max_usd <= DEFAULT_PAID_CAP_USD,
        "estimated_additional_spend_lte_cap": estimated_additional_spend <= max_usd,
        "projected_ledger_spend_lte_cap": projected_ledger_spend <= max_usd,
        "projected_campaign_total_lte_paid_cap": (
            CURRENT_KNOWN_SPEND_USD + projected_ledger_spend <= DEFAULT_PAID_CAP_USD
        ),
        "existing_planned_ledger_resumable": existing_ledger_spend == 0.0
        or resume_state.get("resumable") is True,
        "planned_rows_pairwise_only": planned_stats["pointwise_like_planned_rows"]
        == 0
        and planned_stats["non_pairwise_call_rows"] == 0,
        "planned_pointwise_calls_zero": planned_stats["pointwise_calls"] == 0,
        "future_labels_not_used_for_scheduling": planned_stats[
            "future_label_scheduling_rows"
        ]
        == 0,
        "cached_label_values_not_used_before_scheduling": planned_stats[
            "cached_label_value_scheduling_rows"
        ]
        == 0,
        "next_experiment_protocol_allows_dry_run_preflight": (
            holdout_protocol.get("allowed_to_begin") is True
        ),
        "guarded_pairwise_only_execution_supported": guarded_execution.get(
            "supported"
        )
        is True,
        "execute_mode_confirmed_if_requested": guarded_execution.get(
            "execute_mode_confirmed_if_requested"
        )
        is True,
    }
    blocking = _false_keys(checks)
    return {
        "checks": checks,
        "blocking_reasons": sorted(set(blocking) | set(planning_errors)),
        "policy": {
            "pointwise_calls_forbidden": True,
            "allowed_call_kind": PAIRWISE_CALL_KIND,
            "abort_on_pointwise_call_attempt": True,
            "hard_max_usd_cap": max_usd,
            "separate_artifact_directory_required": True,
            "historical_paid_artifacts_immutable": True,
        },
    }


def _guarded_execution_state(
    *,
    mode: RunnerMode,
    confirm_guarded_pairwise_only_execution: bool,
    planned_stats: Mapping[str, Any],
    provider_model_availability: Mapping[str, Any],
    max_usd: float,
    estimated_additional_spend: float,
    projected_ledger_spend: float,
    artifact_dir: Path,
    ledger_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    supported = True
    execute_confirmed = (
        confirm_guarded_pairwise_only_execution if mode == "execute" else True
    )
    return {
        "supported": supported,
        "mode": mode,
        "execute_mode_confirmed_if_requested": execute_confirmed,
        "runner_path": str(REPO_ROOT / "scripts" / "run_coverage_floor_followup_preflight.py"),
        "artifact_dir": str(artifact_dir),
        "ledger_path": str(ledger_path),
        "output_path": str(output_path),
        "allowed_call_kind": PAIRWISE_CALL_KIND,
        "pointwise_call_trap_enabled": True,
        "planned_pairwise_calls_to_buy": planned_stats[
            "unique_missing_pairwise_labels"
        ],
        "provider_model_availability_status": provider_model_availability.get(
            "status"
        ),
        "estimated_additional_spend_usd": estimated_additional_spend,
        "projected_ledger_spend_usd": projected_ledger_spend,
        "max_usd": max_usd,
    }


def _final_go_no_go(
    *,
    guardrails: Mapping[str, Any],
    provider_model_availability: Mapping[str, Any],
    guarded_execution: Mapping[str, Any],
    estimated_additional_spend: float,
    mode: RunnerMode,
) -> dict[str, Any]:
    blockers = sorted(set(guardrails.get("blocking_reasons") or []))
    decision = "no_go" if blockers else "go"
    return {
        "decision": decision,
        "blocking_reasons": blockers,
        "provider_model_availability_status": provider_model_availability.get(
            "status"
        ),
        "guarded_pairwise_only_execution_supported": guarded_execution.get(
            "supported"
        ),
        "paid_validation_may_run_now": decision == "go" and mode == "execute",
        "paid_label_purchase_authorized_by_this_artifact": decision == "go"
        and mode == "execute",
        "expected_execution_mode": "guarded_pairwise_only",
        "estimated_additional_spend_usd": estimated_additional_spend,
        "expected_pointwise_calls": 0,
        "recommendation": (
            "Proceed only in guarded execute mode for this exact frozen plan."
            if decision == "go"
            else (
                "Blocked. Do not make paid labels; satisfy the listed "
                "preflight prerequisites first."
            )
        ),
        "stop_rule": (
            "Stop before any label-generation call on a pointwise-like call "
            "attempt, unavailable provider/model, missing fresh holdout data, "
            "missing reviewed pointwise artifacts, future-label or cached-label "
            "leakage risk, invalid JSONL ledger, hard cap breach, or manifest "
            "identity mismatch."
        ),
    }


def _execute_pairwise_only(
    *,
    planned_rows: Sequence[Mapping[str, Any]],
    source_artifact_dir: Path,
    phase: str,
    artifact_dir: Path,
    ledger: JsonlLedger,
    max_usd: float,
    pairwise_model: str,
    pairwise_estimate: Any,
    rates: dict[str, dict[str, float]],
    timeout_seconds: float,
    urlopen: UrlOpen,
) -> dict[str, Any]:
    _ensure_jsonl_ledger_exists(ledger.path)
    _assert_pairwise_only_call_kind(PAIRWISE_CALL_KIND)
    missing_rows = _unique_missing_rows(planned_rows)
    papers_by_bucket = _load_pointwise_papers_for_rows(
        missing_rows,
        source_artifact_dir=source_artifact_dir,
        phase=phase,
    )
    calls_before = _ledger_line_count(ledger.path)
    succeeded = 0
    for index, row in enumerate(missing_rows, start=1):
        _assert_pairwise_only_call_kind(str(row.get("planned_call_kind") or ""))
        bucket = str(row["bucket"])
        left_id = str(row["left_id"])
        right_id = str(row["right_id"])
        pair = ScheduledPair(
            left_id=left_id,
            right_id=right_id,
            priority=float(row.get("priority") or 0.0),
            purpose=str(row.get("purpose") or "coverage_floor_pairwise"),
            order=PairwiseOrderMetadata.from_dict(row.get("order") or {}),
        )
        ledger.guard_projected_spend(
            cap_usd=max_usd,
            next_cost_usd=pairwise_estimate.cost_usd,
        )
        response = _chat_json_with_usage(
            base_url=os.environ.get("SESTINA_LLM_BASE_URL") or "",
            api_key=os.environ.get("SESTINA_LLM_API_KEY") or "",
            payload=_pairwise_payload(
                model=pairwise_model,
                pair=pair,
                papers=papers_by_bucket[bucket],
            ),
            timeout_seconds=timeout_seconds,
            urlopen=urlopen,
        )
        comparison = _comparison_from_pairwise_response(pair, response.content)
        cost_payload = usage_cost_payload(
            model=pairwise_model,
            estimate=pairwise_estimate,
            rates=rates,
            response=response,
        )
        artifact_path = (
            artifact_dir
            / phase
            / _safe_name(bucket)
            / "calls"
            / (
                f"{index:04d}-{PAIRWISE_CALL_KIND}-"
                f"{fingerprint(bucket + ':' + left_id + ':' + right_id)}.json"
            )
        )
        _write_pairwise_call_artifact(
            artifact_path,
            phase=phase,
            bucket=bucket,
            model=pairwise_model,
            estimate=pairwise_estimate,
            status="ok",
            response=response.content,
            comparison=comparison,
            ledger=ledger,
            cost_payload=cost_payload,
            subject={
                "left_id": left_id,
                "right_id": right_id,
                "row_roles": row.get("row_roles") or [row.get("row_role")],
            },
        )
        succeeded += 1
    return {
        "attempted": True,
        "status": "completed_pairwise_only",
        "paid_pairwise_calls_attempted": len(missing_rows),
        "paid_pairwise_calls_succeeded": succeeded,
        "new_ledger_entries": _ledger_line_count(ledger.path) - calls_before,
    }


def _write_pairwise_call_artifact(
    artifact_path: Path,
    *,
    phase: str,
    bucket: str,
    model: str,
    estimate: Any,
    status: str,
    ledger: JsonlLedger,
    response: dict[str, Any] | None = None,
    comparison: PairwiseComparison | None = None,
    subject: dict[str, Any] | None = None,
    cost_payload: Mapping[str, Any] | None = None,
) -> None:
    _assert_pairwise_only_call_kind(PAIRWISE_CALL_KIND)
    artifact = _call_artifact(
        phase=phase,
        bucket=bucket,
        model=model,
        kind=PAIRWISE_CALL_KIND,
        estimate=estimate,
        status=status,
        response=response,
        subject=subject
        or {
            "left_id": comparison.left_id if comparison is not None else None,
            "right_id": comparison.right_id if comparison is not None else None,
        },
        cost_payload=cost_payload,
    )
    if comparison is not None:
        artifact["comparison"] = comparison.to_dict()
    write_json_artifact(artifact_path, artifact)
    ledger.append(
        _ledger_entry(
            phase=phase,
            bucket=bucket,
            model=model,
            kind=PAIRWISE_CALL_KIND,
            estimate=estimate,
            status=status,
            artifact_path=artifact_path,
            cost_payload=cost_payload,
        )
    )


def _provider_model_availability(
    *,
    pairwise_model: str,
    timeout_seconds: float,
    urlopen: UrlOpen,
) -> dict[str, Any]:
    try:
        validate_model_names([pairwise_model])
    except ModelAvailabilityError as exc:
        return {
            "status": "invalid_model_name",
            "required_before_any_paid_label_generation": True,
            "check_method": "GET /models",
            "requested_models": [pairwise_model],
            "missing_models": [pairwise_model],
            "label_generation_calls_made": 0,
            "chat_completions_calls_made": 0,
            "api_key_env_present": bool(os.environ.get("SESTINA_LLM_API_KEY")),
            "base_url_env_present": bool(os.environ.get("SESTINA_LLM_BASE_URL")),
            "secrets_printed_or_stored": False,
            "error": str(exc),
        }
    try:
        result = check_model_availability(
            base_url=os.environ.get("SESTINA_LLM_BASE_URL") or "",
            api_key=os.environ.get("SESTINA_LLM_API_KEY") or "",
            models=[pairwise_model],
            timeout_seconds=timeout_seconds,
            urlopen=urlopen,
        )
    except ModelAvailabilityError as exc:
        return {
            "status": "unavailable",
            "required_before_any_paid_label_generation": True,
            "check_method": "GET /models",
            "requested_models": [pairwise_model],
            "missing_models": [pairwise_model],
            "label_generation_calls_made": 0,
            "chat_completions_calls_made": 0,
            "api_key_env_present": bool(os.environ.get("SESTINA_LLM_API_KEY")),
            "base_url_env_present": bool(os.environ.get("SESTINA_LLM_BASE_URL")),
            "secrets_printed_or_stored": False,
            "error": str(exc),
        }
    return {
        **result,
        "required_before_any_paid_label_generation": True,
        "check_method": "GET /models",
        "label_generation_calls_made": 0,
        "chat_completions_calls_made": 0,
        "api_key_env_present": bool(os.environ.get("SESTINA_LLM_API_KEY")),
        "base_url_env_present": bool(os.environ.get("SESTINA_LLM_BASE_URL")),
        "secrets_printed_or_stored": False,
    }


def _existing_pairwise_artifacts(
    *,
    artifact_dir: Path,
    phase: str,
) -> tuple[dict[tuple[str, tuple[str, str]], Path], dict[str, Any]]:
    artifacts: dict[tuple[str, tuple[str, str]], Path] = {}
    invalid_json = 0
    invalid_payload = 0
    duplicate_keys = 0
    calls_root = artifact_dir / phase
    for path in sorted(calls_root.glob(f"*/calls/*-{PAIRWISE_CALL_KIND}-*.json")):
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            invalid_json += 1
            continue
        if payload.get("kind") != PAIRWISE_CALL_KIND or payload.get("status") != "ok":
            invalid_payload += 1
            continue
        bucket = str(payload.get("bucket") or "")
        subject = _mapping(payload.get("subject"))
        comparison = _mapping(payload.get("comparison"))
        left_id = str(subject.get("left_id") or comparison.get("left_id") or "")
        right_id = str(subject.get("right_id") or comparison.get("right_id") or "")
        if not bucket or not left_id or not right_id:
            invalid_payload += 1
            continue
        key = (bucket, canonical_pair_key(left_id, right_id))
        if key in artifacts:
            duplicate_keys += 1
            continue
        artifacts[key] = path
    return artifacts, {
        "artifact_count": len(artifacts),
        "invalid_json_files": invalid_json,
        "invalid_payload_files": invalid_payload,
        "duplicate_pair_keys": duplicate_keys,
    }


def _mark_cached_pairwise_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    existing_pairwise_artifacts: Mapping[tuple[str, tuple[str, str]], Path],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        marked = dict(row)
        pair_key = row.get("pair_key")
        if _valid_pair_key(pair_key):
            key = (
                str(row.get("bucket") or ""),
                canonical_pair_key(str(pair_key[0]), str(pair_key[1])),  # type: ignore[index]
            )
            artifact_path = existing_pairwise_artifacts.get(key)
            if artifact_path is not None:
                marked["cache_status"] = "cached_reuse"
                marked["cached_artifact_path"] = str(artifact_path)
                marked["cached_artifact_kind"] = PAIRWISE_CALL_KIND
        output.append(marked)
    return output


def _pairwise_resume_state(
    *,
    ledger_path: Path,
    existing_pairwise_artifact_paths: set[Path],
    existing_pairwise_artifact_report: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_artifacts = {_safe_resolve(path) for path in existing_pairwise_artifact_paths}
    ledger_entries = 0
    invalid_json_lines = 0
    non_pairwise_entries = 0
    pointwise_like_entries = 0
    non_ok_entries = 0
    missing_artifact_entries = 0
    artifact_not_indexed_entries = 0
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            if not line.strip():
                continue
            ledger_entries += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_json_lines += 1
                continue
            kind = str(row.get("kind") or "")
            if kind != PAIRWISE_CALL_KIND:
                non_pairwise_entries += 1
            if "pointwise" in kind.lower():
                pointwise_like_entries += 1
            if row.get("status") != "ok":
                non_ok_entries += 1
            artifact_path = Path(str(row.get("artifact_path") or ""))
            if not artifact_path.is_file():
                missing_artifact_entries += 1
            elif _safe_resolve(artifact_path) not in resolved_artifacts:
                artifact_not_indexed_entries += 1
    resumable = (
        invalid_json_lines == 0
        and non_pairwise_entries == 0
        and pointwise_like_entries == 0
        and non_ok_entries == 0
        and missing_artifact_entries == 0
        and artifact_not_indexed_entries == 0
        and existing_pairwise_artifact_report.get("invalid_json_files") == 0
        and existing_pairwise_artifact_report.get("invalid_payload_files") == 0
    )
    return {
        "resumable": resumable,
        "ledger_path": str(ledger_path),
        "ledger_entries": ledger_entries,
        "existing_pairwise_artifacts": dict(existing_pairwise_artifact_report),
        "invalid_json_ledger_lines": invalid_json_lines,
        "non_pairwise_ledger_entries": non_pairwise_entries,
        "pointwise_like_ledger_entries": pointwise_like_entries,
        "non_ok_ledger_entries": non_ok_entries,
        "missing_artifact_ledger_entries": missing_artifact_entries,
        "artifact_not_indexed_ledger_entries": artifact_not_indexed_entries,
        "resume_policy": (
            "existing ledger entries may be resumed only when every ledger row "
            "is an ok pairwise_active call with a matching artifact under the "
            "current artifact directory"
        ),
    }


def _planned_pair_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    arm_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    unique: dict[tuple[str, tuple[str, str]], dict[str, Any]] = {}
    invalid_pair_key_rows = 0
    invalid_cache_status_rows = 0
    pointwise_like_rows = 0
    non_pairwise_rows = 0
    future_label_rows = 0
    cached_label_value_rows = 0
    for row in rows:
        status = str(row.get("cache_status") or "")
        status_counts[status] += 1
        role_counts[str(row.get("row_role") or "")] += 1
        arm_counts[str(row.get("arm_name") or "")] += 1
        bucket = str(row.get("bucket") or "")
        if bucket:
            bucket_counts[bucket] += 1
        pair_key = row.get("pair_key")
        if not _valid_pair_key(pair_key):
            invalid_pair_key_rows += 1
            continue
        if status not in {"cached_reuse", "missing_label"}:
            invalid_cache_status_rows += 1
        if _row_is_pointwise_like(row):
            pointwise_like_rows += 1
        if _row_call_kind(row) not in {"", PAIRWISE_CALL_KIND}:
            non_pairwise_rows += 1
        if row.get("future_labels_used_for_scheduling") is not False:
            future_label_rows += 1
        if row.get("cached_label_values_used_before_scheduling") is not False:
            cached_label_value_rows += 1
        key_tuple = tuple(sorted(str(item) for item in pair_key))
        unique_key = (bucket, key_tuple)  # type: ignore[arg-type]
        existing = unique.get(unique_key)
        if existing is None:
            unique[unique_key] = {"cache_status": status}
        elif existing["cache_status"] == "cached_reuse" and status != "cached_reuse":
            existing["cache_status"] = status
    unique_missing = sum(
        1 for row in unique.values() if row["cache_status"] != "cached_reuse"
    )
    return {
        "pointwise_calls": 0,
        "pairwise_scheduled_occurrences": len(rows),
        "pairwise_cached_occurrences": status_counts.get("cached_reuse", 0),
        "pairwise_missing_occurrences": status_counts.get("missing_label", 0),
        "unique_planned_pair_labels": len(unique),
        "unique_missing_pairwise_labels": unique_missing,
        "unique_cached_pairwise_labels": len(unique) - unique_missing,
        "invalid_pair_key_rows": invalid_pair_key_rows,
        "invalid_cache_status_rows": invalid_cache_status_rows,
        "pointwise_like_planned_rows": pointwise_like_rows,
        "non_pairwise_call_rows": non_pairwise_rows,
        "future_label_scheduling_rows": future_label_rows,
        "cached_label_value_scheduling_rows": cached_label_value_rows,
        "cache_status_counts": dict(sorted(status_counts.items())),
        "row_role_counts": dict(sorted(role_counts.items())),
        "arm_counts": dict(sorted(arm_counts.items())),
        "bucket_occurrence_counts": dict(sorted(bucket_counts.items())),
    }


def _unique_missing_rows(
    planned_rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    output = []
    seen: set[tuple[str, tuple[str, str]]] = set()
    for row in planned_rows:
        if row.get("cache_status") == "cached_reuse":
            continue
        pair_key = row.get("pair_key")
        if not _valid_pair_key(pair_key):
            continue
        key = (str(row.get("bucket") or ""), tuple(sorted(str(item) for item in pair_key)))
        if key in seen:
            continue
        seen.add(key)  # type: ignore[arg-type]
        output.append(row)
    return output


def _load_pointwise_papers_for_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_artifact_dir: Path,
    phase: str,
) -> dict[str, dict[str, Paper]]:
    needed: dict[str, set[str]] = {}
    for row in rows:
        bucket = str(row.get("bucket") or "")
        if not bucket:
            continue
        needed.setdefault(bucket, set()).update(
            {str(row.get("left_id") or ""), str(row.get("right_id") or "")}
        )
    loaded: dict[str, dict[str, Paper]] = {}
    for bucket, paper_ids in needed.items():
        paper_ids.discard("")
        calls_dir = source_artifact_dir / phase / bucket / "calls"
        papers: dict[str, Paper] = {}
        for path in sorted(calls_dir.glob("*-pointwise-*.json")):
            payload = _read_json(path)
            if payload.get("kind") != "pointwise" or payload.get("status") != "ok":
                continue
            subject = payload.get("subject") or {}
            paper_id = str(subject.get("paper_id") or "")
            if paper_id not in paper_ids:
                continue
            papers[paper_id] = Paper(
                paper_id=paper_id,
                title=str(subject.get("title") or paper_id),
                abstract=str(subject.get("abstract") or ""),
                pointwise=PointwiseAssessment.from_dict(payload["response"]),
                metadata=dict(subject.get("metadata") or {}),
            )
        missing = sorted(paper_ids - set(papers))
        if missing:
            raise CoverageFloorPreflightError(
                "missing reviewed pointwise artifacts for guarded pairwise "
                f"execution in bucket {bucket}: {', '.join(missing)}"
            )
        loaded[bucket] = papers
    return loaded


def _all_unordered_pair_keys(papers: Sequence[Paper]) -> set[tuple[str, str]]:
    return {
        canonical_pair_key(left.paper_id, right.paper_id)
        for left, right in combinations(papers, 2)
    }


def _assert_pairwise_only_call_kind(kind: str) -> None:
    if kind != PAIRWISE_CALL_KIND:
        raise PointwiseCallForbiddenError(
            f"coverage-floor runner only permits {PAIRWISE_CALL_KIND!r}; "
            f"attempted {kind!r}"
        )
    if "pointwise" in kind.lower():
        raise PointwiseCallForbiddenError(
            "coverage-floor runner aborts on pointwise-call attempts"
        )


def _development_bucket_names(no_paid_sweep: Mapping[str, Any]) -> set[str]:
    names = set()
    for seed_row in no_paid_sweep.get("bucket_results") or []:
        for row in seed_row.get("buckets") or []:
            if row.get("bucket"):
                names.add(str(row["bucket"]))
    return names


def _valid_pair_key(value: object) -> bool:
    return (
        isinstance(value, list | tuple)
        and len(value) == 2
        and all(isinstance(item, str) and item for item in value)
    )


def _row_call_kind(row: Mapping[str, Any]) -> str:
    for key in ("planned_call_kind", "call_kind", "kind"):
        if row.get(key):
            return str(row[key])
    if row.get("cache_status") == "missing_label":
        return PAIRWISE_CALL_KIND
    return ""


def _row_is_pointwise_like(row: Mapping[str, Any]) -> bool:
    for key in ("planned_call_kind", "call_kind", "kind", "cached_artifact_kind"):
        value = row.get(key)
        if value is not None and "pointwise" in str(value).lower():
            return True
    return False


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


def _ensure_jsonl_ledger_exists(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same_path(path: Path, other: object) -> bool:
    if other is None:
        return False
    try:
        return path.resolve() == Path(str(other)).resolve()
    except OSError:
        return False


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _ledger_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _false_keys(checks: Mapping[str, bool]) -> list[str]:
    return sorted(key for key, value in checks.items() if value is False)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _stdout_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_path": payload.get("artifact_path") or payload.get("output_path"),
        "artifact_type": payload["artifact_type"],
        "mode": payload["mode"],
        "dry_run": payload["dry_run"],
        "paid_calls_made": payload["paid_calls_made"],
        "paid_spend_usd": payload["paid_spend_usd"],
        "pointwise_calls_made": payload["pointwise_calls_made"],
        "active_arm_name": payload["frozen_no_paid_sweep"]["active_arm_name"],
        "provider_model_availability_status": payload[
            "provider_model_availability"
        ]["status"],
        "fresh_holdout_status": payload["fresh_holdout"]["status"],
        "decision": payload["final_go_no_go"]["decision"],
        "blocking_reasons": payload["final_go_no_go"]["blocking_reasons"],
        "pairwise_calls_to_buy": payload["totals"]["pairwise_calls_to_buy"],
        "estimated_additional_spend_usd": payload["totals"][
            "estimated_additional_spend_usd"
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
