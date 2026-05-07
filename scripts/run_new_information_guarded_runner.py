#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sestina.active_arm_gate import (  # noqa: E402
    CURRENT_KNOWN_SPEND_USD,
    DEFAULT_PAID_CAP_USD,
)
from sestina.backtest_budget import load_config  # noqa: E402
from sestina.backtest_runner import (  # noqa: E402
    JsonlLedger,
    ModelAvailabilityError,
    _call_artifact,
    _call_estimate,
    _chat_json,
    _comparison_from_pairwise_response,
    _config_for_phase,
    _ledger_entry,
    _normalize_rates_from_config,
    _normalize_token_assumptions_from_config,
    _pairwise_payload,
    check_model_availability,
    validate_model_names,
)
from sestina.diagnostics import fingerprint, write_json_artifact  # noqa: E402
from sestina.models import (  # noqa: E402
    PairwiseComparison,
    PairwiseOrderMetadata,
    Paper,
    PointwiseAssessment,
    ScheduledPair,
)


ARTIFACT_TYPE = "sestina-new-information-guarded-runner-go-no-go"
SCHEMA_VERSION = 1
ARM_NEW_INFO = "new_information_challenger_cached_replay"
ARM_EXACT = "exact_pool_random_cached_replay"
PAIRWISE_CALL_KIND = "pairwise_active"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts" / "backtest-arxiv-new-information-guarded-runner"
)
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "arxiv_historical_pilot_budget_config.json"
DEFAULT_MANIFEST = (
    REPO_ROOT / "artifacts" / "backtest-datasets" / "arxiv-historical-pilot-manifest.json"
)
DEFAULT_SOURCE_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "backtest-arxiv-pilot-live"
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
DEFAULT_DRY_RUN_ARTIFACT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-new-information-paid-dry-run"
    / "paid-dry-run-go-no-go.json"
)
DEFAULT_PLANNED_PAIRS = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-new-information-paid-dry-run"
    / "planned-pair-occurrences.jsonl"
)
DEFAULT_CAVEAT_ADJUDICATION = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-new-information-caveat-adjudication"
    / "caveat-adjudication.json"
)
DEFAULT_LEDGER = DEFAULT_OUTPUT_DIR / "guarded-runner-ledger.jsonl"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "guarded-runner-go-no-go.json"
REQUIRED_TOP_LEVEL_KEYS = {
    "artifact_type",
    "schema_version",
    "mode",
    "dry_run",
    "paid_calls_made",
    "paid_spend_usd",
    "pointwise_calls_made",
    "input_artifacts",
    "frozen_manifest_validation",
    "caveat_scope",
    "guardrails",
    "model_availability",
    "ledger",
    "planned_execution",
    "totals",
    "go_no_go",
    "validation_commands",
}
REQUIRED_CAVEAT_TOPICS = {
    "active_gate_not_weakened": (
        "do not weaken",
        "active-arm gate",
    ),
    "future_labels_retrospective_only": (
        "future labels",
        "retrospective diagnostics",
    ),
    "zero_pointwise_calls": ("zero pointwise calls",),
    "no_historical_rewrites": (
        "do not rewrite historical paid ledgers",
    ),
    "scoped_to_frozen_manifest": (
        "scoped only to the frozen budget-filled",
    ),
    "zero_missing_scope": ("zero unique missing labels",),
    "guarded_pairwise_runner_required": (
        "guarded pairwise-only runner",
        "provider model availability checks",
        "jsonl ledger",
        "hard max-usd cap",
        "separate artifact directory",
    ),
    "pointwise_abort_required": ("abort on any pointwise-call attempt",),
}

RunnerMode = Literal["planning", "execute"]
UrlOpen = Any


class GuardedRunnerError(RuntimeError):
    """Base class for guarded new-information runner failures."""


class PointwiseCallForbiddenError(GuardedRunnerError):
    """Raised when the guarded runner is asked to make a pointwise-like call."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build or execute the reviewed guarded pairwise-only runner path for "
            "the frozen budget-filled new-information challenger manifest."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--source-artifact-dir",
        type=Path,
        default=DEFAULT_SOURCE_ARTIFACT_DIR,
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
        "--dry-run-artifact",
        type=Path,
        default=DEFAULT_DRY_RUN_ARTIFACT,
    )
    parser.add_argument("--planned-pairs", type=Path, default=DEFAULT_PLANNED_PAIRS)
    parser.add_argument(
        "--caveat-adjudication",
        type=Path,
        default=DEFAULT_CAVEAT_ADJUDICATION,
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("planning", "execute"), default="planning")
    parser.add_argument(
        "--max-usd",
        type=float,
        default=0.01,
        help=(
            "Hard cap for this guarded runner ledger. The current frozen "
            "zero-missing manifest should remain cache-only/zero-spend."
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-parse-retries", type=int, default=0)
    parser.add_argument(
        "--perform-model-availability-check",
        action="store_true",
        help="Optional planning-mode model availability check; execute mode always checks.",
    )
    parser.add_argument(
        "--confirm-guarded-pairwise-only-execution",
        action="store_true",
        help="Required with --mode execute before any pairwise network calls.",
    )
    parser.add_argument(
        "--allow-nonzero-missing-labels",
        action="store_true",
        help=(
            "Allow execution planning for a future nonzero-missing manifest. "
            "This is not valid for the current caveat-accepted frozen manifest."
        ),
    )
    args = parser.parse_args(argv)

    payload = build_new_information_guarded_runner_go_no_go(
        config_path=args.config,
        manifest_path=args.manifest,
        source_artifact_dir=args.source_artifact_dir,
        budget_fill_artifact_path=args.budget_fill_artifact,
        active_gate_artifact_path=args.active_gate_artifact,
        dry_run_artifact_path=args.dry_run_artifact,
        planned_pairs_path=args.planned_pairs,
        caveat_adjudication_path=args.caveat_adjudication,
        artifact_dir=args.artifact_dir,
        ledger_path=args.ledger,
        output_path=args.output,
        mode=args.mode,
        max_usd=args.max_usd,
        timeout_seconds=args.timeout_seconds,
        max_parse_retries=args.max_parse_retries,
        perform_model_availability_check=args.perform_model_availability_check,
        confirm_guarded_pairwise_only_execution=(
            args.confirm_guarded_pairwise_only_execution
        ),
        require_zero_missing_labels=not args.allow_nonzero_missing_labels,
    )
    sys.stdout.write(json.dumps(_stdout_summary(payload), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    if args.mode == "execute" and payload["go_no_go"]["decision"] != "go":
        return 2
    return 0


def build_new_information_guarded_runner_go_no_go(
    *,
    config_path: Path,
    manifest_path: Path,
    source_artifact_dir: Path,
    budget_fill_artifact_path: Path,
    active_gate_artifact_path: Path,
    dry_run_artifact_path: Path,
    planned_pairs_path: Path,
    caveat_adjudication_path: Path,
    artifact_dir: Path,
    ledger_path: Path,
    output_path: Path,
    mode: RunnerMode = "planning",
    max_usd: float = 0.01,
    timeout_seconds: float = 60.0,
    max_parse_retries: int = 0,
    perform_model_availability_check: bool = False,
    confirm_guarded_pairwise_only_execution: bool = False,
    require_zero_missing_labels: bool = True,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _ensure_jsonl_ledger_exists(ledger_path)

    budget_fill_artifact = _read_json(budget_fill_artifact_path)
    active_gate_artifact = _read_json(active_gate_artifact_path)
    dry_run_artifact = _read_json(dry_run_artifact_path)
    caveat_adjudication = _read_json(caveat_adjudication_path)
    planned_rows = _read_jsonl(planned_pairs_path)
    raw_config = load_config(config_path)
    phase = str((dry_run_artifact.get("frozen_inputs") or {}).get("phase") or "pilot")
    phase_config = _config_for_phase(raw_config, phase=phase)["phases"][0]
    pairwise_model = str(phase_config["pairwise_model"])
    token_assumptions = _normalize_token_assumptions_from_config(raw_config)
    rates = _normalize_rates_from_config(raw_config)
    pairwise_estimate = _call_estimate(
        "pairwise",
        pairwise_model,
        token_assumptions,
        rates,
    )
    planned_stats = _planned_pair_stats(planned_rows)
    paid_calls_made = 0
    paid_spend_usd = 0.0
    parse_retry_count = 0

    frozen_validation = _frozen_manifest_validation(
        config_path=config_path,
        manifest_path=manifest_path,
        source_artifact_dir=source_artifact_dir,
        budget_fill_artifact_path=budget_fill_artifact_path,
        active_gate_artifact_path=active_gate_artifact_path,
        dry_run_artifact_path=dry_run_artifact_path,
        planned_pairs_path=planned_pairs_path,
        caveat_adjudication_path=caveat_adjudication_path,
        budget_fill_artifact=budget_fill_artifact,
        active_gate_artifact=active_gate_artifact,
        dry_run_artifact=dry_run_artifact,
        caveat_adjudication=caveat_adjudication,
        planned_stats=planned_stats,
        pairwise_model=pairwise_model,
        require_zero_missing_labels=require_zero_missing_labels,
    )
    caveat_scope = _caveat_scope(
        caveat_adjudication,
        dry_run_artifact=dry_run_artifact,
        planned_stats=planned_stats,
        require_zero_missing_labels=require_zero_missing_labels,
    )
    ledger = JsonlLedger(ledger_path)
    existing_ledger_spend = ledger.existing_spend_usd()
    estimated_additional_spend = round(
        planned_stats["unique_missing_pairwise_labels"] * pairwise_estimate.cost_usd,
        6,
    )
    model_availability = _model_availability_summary(
        mode=mode,
        perform_model_availability_check=perform_model_availability_check,
        pairwise_model=pairwise_model,
        timeout_seconds=timeout_seconds,
        urlopen=urlopen,
    )
    guardrails = _guardrails(
        mode=mode,
        max_usd=max_usd,
        artifact_dir=artifact_dir,
        source_artifact_dir=source_artifact_dir,
        dry_run_artifact=dry_run_artifact,
        active_gate_artifact=active_gate_artifact,
        planned_stats=planned_stats,
        frozen_validation=frozen_validation,
        caveat_scope=caveat_scope,
        model_availability=model_availability,
        ledger_path=ledger_path,
        existing_ledger_spend=existing_ledger_spend,
        estimated_additional_spend=estimated_additional_spend,
        pairwise_model=pairwise_model,
        confirm_guarded_pairwise_only_execution=(
            confirm_guarded_pairwise_only_execution
        ),
        require_zero_missing_labels=require_zero_missing_labels,
    )
    execution_summary: dict[str, Any] = {
        "attempted": False,
        "paid_pairwise_calls_attempted": 0,
        "paid_pairwise_calls_succeeded": 0,
        "parse_retry_count": 0,
        "new_ledger_entries": 0,
        "status": "not_attempted_planning_mode",
    }

    go_no_go = _go_no_go(
        mode=mode,
        guardrails=guardrails,
        frozen_validation=frozen_validation,
        caveat_scope=caveat_scope,
        model_availability=model_availability,
        planned_stats=planned_stats,
        max_usd=max_usd,
        estimated_additional_spend=estimated_additional_spend,
    )

    if mode == "execute" and go_no_go["decision"] == "go":
        execution_summary = _execute_guarded_pairwise_only(
            planned_rows=planned_rows,
            planned_stats=planned_stats,
            config_path=config_path,
            source_artifact_dir=source_artifact_dir,
            phase=phase,
            artifact_dir=artifact_dir,
            ledger=ledger,
            max_usd=max_usd,
            pairwise_model=pairwise_model,
            pairwise_estimate=pairwise_estimate,
            timeout_seconds=timeout_seconds,
            max_parse_retries=max_parse_retries,
            urlopen=urlopen,
        )
        paid_calls_made = int(execution_summary["paid_pairwise_calls_succeeded"])
        parse_retry_count = int(execution_summary["parse_retry_count"])
        paid_spend_usd = round(
            ledger.existing_spend_usd() - existing_ledger_spend,
            6,
        )

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
        "parse_retry_count": parse_retry_count,
        "max_parse_retries": max_parse_retries,
    }
    planned_execution = _planned_execution(
        mode=mode,
        artifact_dir=artifact_dir,
        ledger_path=ledger_path,
        output_path=output_path,
        pairwise_model=pairwise_model,
        pairwise_estimate=pairwise_estimate,
        max_usd=max_usd,
        planned_stats=planned_stats,
        require_zero_missing_labels=require_zero_missing_labels,
    )
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "dry_run": mode == "planning",
        "paid_calls_made": paid_calls_made,
        "paid_spend_usd": paid_spend_usd,
        "pointwise_calls_made": 0,
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
            "dry_run_artifact_path": str(dry_run_artifact_path),
            "dry_run_artifact_sha256": _sha256(dry_run_artifact_path),
            "planned_pairs_path": str(planned_pairs_path),
            "planned_pairs_sha256": _sha256(planned_pairs_path),
            "caveat_adjudication_path": str(caveat_adjudication_path),
            "caveat_adjudication_sha256": _sha256(caveat_adjudication_path),
            "runner_code_path": str(REPO_ROOT / "scripts" / "run_new_information_guarded_runner.py"),
            "runner_code_sha256": _sha256(
                REPO_ROOT / "scripts" / "run_new_information_guarded_runner.py"
            ),
        },
        "frozen_manifest_validation": frozen_validation,
        "caveat_scope": caveat_scope,
        "guardrails": guardrails,
        "model_availability": model_availability,
        "ledger": {
            "path": str(ledger_path),
            "format": "jsonl",
            "separate_artifact_directory": guardrails["checks"][
                "separate_artifact_directory"
            ],
            "existing_spend_usd_before_workflow": existing_ledger_spend,
            "spend_usd_after_workflow": ledger.existing_spend_usd(),
            "new_entries_this_invocation": execution_summary["new_ledger_entries"],
            "historical_ledgers_rewritten": False,
        },
        "planned_execution": planned_execution,
        "execution_summary": execution_summary,
        "totals": totals,
        "go_no_go": go_no_go,
        "validation_commands": [
            "uv run python scripts/run_new_information_guarded_runner.py",
            "uv run python -m json.tool artifacts/backtest-arxiv-new-information-guarded-runner/guarded-runner-go-no-go.json >/dev/null",
            "uv run pytest tests/test_new_information_guarded_runner.py",
            "git diff --check",
            "uv run pytest -p no:cacheprovider",
        ],
        "output_path": str(output_path),
    }
    validate_guarded_runner_artifact_schema(payload)
    write_json_artifact(output_path, payload)
    return {**payload, "artifact_path": str(output_path)}


def validate_guarded_runner_artifact_schema(payload: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(payload))
    if missing:
        raise ValueError(
            "guarded runner artifact missing top-level keys: " + ", ".join(missing)
        )
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("guarded runner artifact has unexpected artifact_type")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("guarded runner artifact has unexpected schema_version")
    if payload.get("paid_calls_made", 1) < 0:
        raise ValueError("guarded runner paid_calls_made cannot be negative")
    if payload.get("pointwise_calls_made") != 0:
        raise ValueError("guarded runner must make zero pointwise calls")
    if payload.get("paid_spend_usd", 0.0) < 0.0:
        raise ValueError("guarded runner paid_spend_usd cannot be negative")
    go_no_go = payload.get("go_no_go")
    if not isinstance(go_no_go, Mapping):
        raise ValueError("guarded runner go_no_go must be an object")
    if go_no_go.get("decision") not in {"go", "no_go"}:
        raise ValueError("guarded runner decision must be go/no_go")
    if go_no_go.get("decision") == "go" and not go_no_go.get(
        "runner_ready_for_later_execution"
    ):
        raise ValueError("go decision requires runner readiness")
    totals = payload.get("totals")
    if not isinstance(totals, Mapping):
        raise ValueError("guarded runner totals must be an object")
    if int(totals.get("pointwise_like_planned_rows", -1)) != 0 and go_no_go.get(
        "decision"
    ) == "go":
        raise ValueError("go decision cannot include pointwise-like planned rows")


def _frozen_manifest_validation(
    *,
    config_path: Path,
    manifest_path: Path,
    source_artifact_dir: Path,
    budget_fill_artifact_path: Path,
    active_gate_artifact_path: Path,
    dry_run_artifact_path: Path,
    planned_pairs_path: Path,
    caveat_adjudication_path: Path,
    budget_fill_artifact: Mapping[str, Any],
    active_gate_artifact: Mapping[str, Any],
    dry_run_artifact: Mapping[str, Any],
    caveat_adjudication: Mapping[str, Any],
    planned_stats: Mapping[str, Any],
    pairwise_model: str,
    require_zero_missing_labels: bool,
) -> dict[str, Any]:
    dry_input = dry_run_artifact.get("input_artifacts") or {}
    dry_frozen = dry_run_artifact.get("frozen_inputs") or {}
    dry_planned_execution = dry_run_artifact.get("planned_execution") or {}
    dry_totals = dry_run_artifact.get("totals") or {}
    caveat_inputs = caveat_adjudication.get("input_artifacts") or {}
    checks = {
        "dry_run_artifact_type": (
            dry_run_artifact.get("artifact_type")
            == "sestina-new-information-paid-dry-run"
        ),
        "dry_run_is_no_paid_planning_artifact": dry_run_artifact.get("dry_run")
        is True,
        "dry_run_zero_paid": dry_run_artifact.get("paid_calls_made") == 0
        and float(dry_run_artifact.get("paid_spend_usd") or 0.0) == 0.0,
        "dry_run_zero_pointwise": dry_run_artifact.get("pointwise_calls_made") == 0
        and int(dry_totals.get("pointwise_calls") or 0) == 0,
        "budget_fill_zero_paid": budget_fill_artifact.get("paid_calls_made") == 0
        and float(budget_fill_artifact.get("paid_spend_usd") or 0.0) == 0.0,
        "budget_fill_zero_pointwise": budget_fill_artifact.get(
            "pointwise_calls_made"
        )
        == 0,
        "active_arm_name_matches": dry_frozen.get("active_arm_name") == ARM_NEW_INFO,
        "random_control_matches": dry_frozen.get("random_control_comparator")
        == ARM_EXACT,
        "planned_pairwise_label_kind_pairwise_active": dry_planned_execution.get(
            "planned_pairwise_label_kind"
        )
        == PAIRWISE_CALL_KIND,
        "planned_pointwise_calls_zero": dry_planned_execution.get(
            "pointwise_calls_planned"
        )
        == 0,
        "random_control_paid_labels_zero": dry_planned_execution.get(
            "random_control_paid_labels_planned"
        )
        == 0,
        "provider_prefixed_pairwise_model": "/" in pairwise_model
        and dry_frozen.get("pairwise_model") == pairwise_model,
        "config_sha_matches_dry_run": _sha_matches(
            config_path,
            dry_input.get("config_sha256"),
        ),
        "manifest_sha_matches_dry_run": _sha_matches(
            manifest_path,
            dry_input.get("manifest_sha256"),
        ),
        "budget_fill_sha_matches_dry_run": _sha_matches(
            budget_fill_artifact_path,
            dry_input.get("budget_fill_artifact_sha256"),
        ),
        "active_gate_sha_matches_dry_run": _sha_matches(
            active_gate_artifact_path,
            dry_input.get("active_gate_artifact_sha256"),
        ),
        "dry_run_sha_matches_caveat": _sha_matches(
            dry_run_artifact_path,
            caveat_inputs.get("dry_run_artifact_sha256"),
        ),
        "planned_pairs_sha_matches_caveat": _sha_matches(
            planned_pairs_path,
            caveat_inputs.get("planned_pairs_sha256"),
        ),
        "budget_fill_sha_matches_caveat": _sha_matches(
            budget_fill_artifact_path,
            caveat_inputs.get("budget_fill_artifact_sha256"),
        ),
        "active_gate_sha_matches_caveat": _sha_matches(
            active_gate_artifact_path,
            caveat_inputs.get("active_gate_artifact_sha256"),
        ),
        "planned_pairs_path_matches_dry_run": _same_path(
            planned_pairs_path,
            dry_run_artifact.get("planned_pair_occurrences_path"),
        ),
        "planned_pair_occurrence_count_matches": int(
            dry_run_artifact.get("planned_pair_occurrence_count") or -1
        )
        == planned_stats["pairwise_scheduled_occurrences"],
        "planned_unique_pair_labels_matches": _int_field(
            dry_totals,
            "unique_planned_pair_labels",
            default=-1,
        )
        == planned_stats["unique_planned_pair_labels"],
        "unique_missing_pairwise_labels_matches": _int_field(
            dry_totals,
            "unique_missing_pairwise_labels",
            default=-1,
        )
        == planned_stats["unique_missing_pairwise_labels"],
        "pairwise_missing_occurrences_matches": _int_field(
            dry_totals,
            "pairwise_missing_occurrences",
            default=-1,
        )
        == planned_stats["pairwise_missing_occurrences"],
        "active_budget_shortfall_zero": int(
            dry_totals.get("active_budget_shortfall") or 0
        )
        == 0,
        "planned_rows_have_pair_keys": planned_stats["invalid_pair_key_rows"] == 0,
        "planned_rows_pairwise_only": planned_stats["pointwise_like_planned_rows"]
        == 0,
        "planned_rows_do_not_use_future_labels_for_scheduling": planned_stats[
            "future_label_scheduling_rows"
        ]
        == 0
        and ((dry_frozen.get("candidate_construction_policy") or {}).get(
            "future_labels_used_for_scheduling"
        )
        is False),
        "planned_rows_do_not_use_cached_label_values_before_scheduling": (
            planned_stats["cached_label_value_scheduling_rows"] == 0
            and ((dry_frozen.get("candidate_construction_policy") or {}).get(
                "cached_label_values_used_before_scheduling"
            )
            is False)
        ),
        "cache_status_values_valid": planned_stats["invalid_cache_status_rows"] == 0,
        "source_artifact_dir_matches_dry_run": _same_path(
            source_artifact_dir,
            dry_input.get("source_artifact_dir"),
        ),
        "runner_go_no_go_artifact_is_separate": not _same_path(
            caveat_adjudication_path,
            dry_run_artifact_path,
        ),
    }
    checks["frozen_zero_missing_label_manifest"] = (
        planned_stats["unique_missing_pairwise_labels"] == 0
        if require_zero_missing_labels
        else True
    )
    blocking = _false_keys(checks)
    return {
        "checks": checks,
        "blocking_reasons": blocking,
        "valid": not blocking,
        "identity": {
            "active_arm_name": dry_frozen.get("active_arm_name"),
            "random_control_comparator": dry_frozen.get(
                "random_control_comparator"
            ),
            "phase": dry_frozen.get("phase"),
            "seed_count": dry_frozen.get("seed_count"),
            "seeds": dry_frozen.get("seeds"),
            "pairwise_model": pairwise_model,
            "planned_pairs_path": str(planned_pairs_path),
        },
        "shape": {
            "planned_pair_occurrences": planned_stats[
                "pairwise_scheduled_occurrences"
            ],
            "unique_planned_pair_labels": planned_stats[
                "unique_planned_pair_labels"
            ],
            "unique_missing_pairwise_labels": planned_stats[
                "unique_missing_pairwise_labels"
            ],
        },
    }


def _caveat_scope(
    caveat_adjudication: Mapping[str, Any],
    *,
    dry_run_artifact: Mapping[str, Any],
    planned_stats: Mapping[str, Any],
    require_zero_missing_labels: bool,
) -> dict[str, Any]:
    constraints = [str(item) for item in caveat_adjudication.get("constraints") or []]
    joined = " ".join(constraints).lower()
    topic_checks = {
        name: all(fragment in joined for fragment in fragments)
        for name, fragments in REQUIRED_CAVEAT_TOPICS.items()
    }
    dry_totals = dry_run_artifact.get("totals") or {}
    checks = {
        "caveat_adjudication_accepted_with_constraints": caveat_adjudication.get(
            "decision"
        )
        == "caveat_accepted_with_constraints",
        "caveat_adjudication_zero_paid": caveat_adjudication.get(
            "paid_calls_made"
        )
        == 0
        and float(caveat_adjudication.get("paid_spend_usd") or 0.0) == 0.0,
        "caveat_adjudication_zero_pointwise": caveat_adjudication.get(
            "pointwise_calls_made"
        )
        == 0,
        "accepted_scope_matches_current_zero_missing_manifest": (
            planned_stats["unique_missing_pairwise_labels"] == 0
            and _int_field(dry_totals, "unique_missing_pairwise_labels") == 0
        )
        if require_zero_missing_labels
        else True,
        **topic_checks,
    }
    blocking = _false_keys(checks)
    return {
        "decision": caveat_adjudication.get("decision"),
        "constraints": constraints,
        "checks": checks,
        "blocking_reasons": blocking,
        "accepted_for_current_manifest": not blocking,
        "scope_note": (
            "Acceptance is scoped only to the current frozen zero-missing-label "
            "budget-filled manifest and reviewed artifacts."
        ),
    }


def _guardrails(
    *,
    mode: RunnerMode,
    max_usd: float,
    artifact_dir: Path,
    source_artifact_dir: Path,
    dry_run_artifact: Mapping[str, Any],
    active_gate_artifact: Mapping[str, Any],
    planned_stats: Mapping[str, Any],
    frozen_validation: Mapping[str, Any],
    caveat_scope: Mapping[str, Any],
    model_availability: Mapping[str, Any],
    ledger_path: Path,
    existing_ledger_spend: float,
    estimated_additional_spend: float,
    pairwise_model: str,
    confirm_guarded_pairwise_only_execution: bool,
    require_zero_missing_labels: bool,
) -> dict[str, Any]:
    active_gate_verdict = active_gate_artifact.get("gate_verdict") or {}
    active_gate_caveats = active_gate_artifact.get("caveats") or {}
    budget_caveat = active_gate_caveats.get("budget_completeness_caveat") or {}
    missing_caveat = active_gate_caveats.get("missing_label_caveat") or {}
    dry_requested_max = (
        (dry_run_artifact.get("go_no_go") or {}).get("requested_max_usd") or 2.0
    )
    checks = {
        "frozen_manifest_validation_clear": frozen_validation.get("valid") is True,
        "caveat_accepted_for_current_manifest": caveat_scope.get(
            "accepted_for_current_manifest"
        )
        is True,
        "reviewed_active_arm_gate_paid_followup_allowed": active_gate_artifact.get(
            "paid_followup_allowed"
        )
        is True
        and active_gate_verdict.get("paid_followup_allowed", True) is True,
        "reviewed_active_arm_gate_no_blocking_reasons": not bool(
            active_gate_verdict.get("blocking_reasons") or []
        ),
        "reviewed_active_gate_missing_label_caveat_false": not bool(
            missing_caveat.get("present")
        ),
        "reviewed_active_gate_budget_completeness_caveat_false": not bool(
            budget_caveat.get("present")
        ),
        "provider_prefixed_model_name": "/" in pairwise_model,
        "model_availability_checked_or_required": model_availability.get("status")
        in {
            "available",
            "required_later_not_checked_planning",
            "not_required_zero_missing_planning",
        },
        "model_availability_available_for_execute": (
            model_availability.get("status") == "available"
            if mode == "execute"
            else True
        ),
        "jsonl_ledger_path_configured": ledger_path.suffix == ".jsonl",
        "ledger_under_separate_artifact_directory": _is_relative_to(
            ledger_path.resolve(),
            artifact_dir.resolve(),
        ),
        "separate_artifact_directory": artifact_dir.resolve()
        != source_artifact_dir.resolve()
        and not _same_path(
            artifact_dir,
            (dry_run_artifact.get("frozen_inputs") or {}).get("artifact_dir"),
        ),
        "historical_ledgers_not_rewritten": _is_relative_to(
            ledger_path.resolve(),
            artifact_dir.resolve(),
        ),
        "pointwise_call_trap_enabled": True,
        "pointwise_calls_forbidden": True,
        "planned_pointwise_like_rows_zero": planned_stats[
            "pointwise_like_planned_rows"
        ]
        == 0,
        "planned_pointwise_calls_zero": planned_stats["pointwise_calls"] == 0,
        "pairwise_only_workflow": planned_stats["non_pairwise_call_rows"] == 0,
        "random_control_paid_labels_zero": (
            (dry_run_artifact.get("planned_execution") or {}).get(
                "random_control_paid_labels_planned"
            )
            == 0
        ),
        "frozen_zero_missing_label_manifest": (
            planned_stats["unique_missing_pairwise_labels"] == 0
            if require_zero_missing_labels
            else True
        ),
        "cache_only_current_manifest": planned_stats["unique_missing_pairwise_labels"]
        == 0,
        "hard_max_usd_cap_positive": max_usd > 0.0,
        "hard_max_usd_cap_lte_paid_cap": max_usd <= DEFAULT_PAID_CAP_USD,
        "hard_max_usd_cap_lte_dry_run_request": max_usd <= float(dry_requested_max),
        "estimated_additional_spend_lte_cap": estimated_additional_spend <= max_usd,
        "projected_workflow_ledger_spend_lte_cap": (
            existing_ledger_spend + estimated_additional_spend <= max_usd
        ),
        "projected_known_spend_lte_paid_cap": (
            CURRENT_KNOWN_SPEND_USD + estimated_additional_spend
            <= DEFAULT_PAID_CAP_USD
        ),
        "existing_runner_ledger_spend_zero": existing_ledger_spend == 0.0,
        "parse_retries_recorded": True,
        "execute_mode_explicitly_confirmed": (
            confirm_guarded_pairwise_only_execution if mode == "execute" else True
        ),
    }
    blocking = _false_keys(checks)
    return {
        "checks": checks,
        "blocking_reasons": blocking,
        "non_paid_guardrails_satisfied": not blocking,
        "guarded_runner_review_state": "implemented_for_review",
        "guardrail_policy": {
            "do_not_weaken_reviewed_active_gate": True,
            "future_labels_for_retrospective_diagnostics_only": True,
            "zero_pointwise_calls": True,
            "allowed_network_call_kind": PAIRWISE_CALL_KIND,
            "abort_on_pointwise_call_attempt": True,
            "ledger_format": "jsonl",
            "hard_max_usd_cap": max_usd,
        },
    }


def _model_availability_summary(
    *,
    mode: RunnerMode,
    perform_model_availability_check: bool,
    pairwise_model: str,
    timeout_seconds: float,
    urlopen: UrlOpen,
) -> dict[str, Any]:
    try:
        validate_model_names([pairwise_model])
    except ModelAvailabilityError as exc:
        return {
            "status": "invalid_model_name",
            "required_before_guarded_execution": True,
            "requested_models": [pairwise_model],
            "missing_models": [pairwise_model],
            "error": str(exc),
        }
    if mode != "execute" and not perform_model_availability_check:
        return {
            "status": "required_later_not_checked_planning",
            "required_before_guarded_execution": True,
            "requested_models": [pairwise_model],
            "missing_models": [],
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
            "required_before_guarded_execution": True,
            "requested_models": [pairwise_model],
            "missing_models": [pairwise_model],
            "error": str(exc),
        }
    return {
        **result,
        "required_before_guarded_execution": True,
    }


def _go_no_go(
    *,
    mode: RunnerMode,
    guardrails: Mapping[str, Any],
    frozen_validation: Mapping[str, Any],
    caveat_scope: Mapping[str, Any],
    model_availability: Mapping[str, Any],
    planned_stats: Mapping[str, Any],
    max_usd: float,
    estimated_additional_spend: float,
) -> dict[str, Any]:
    blockers = sorted(
        set(guardrails.get("blocking_reasons") or [])
        | set(frozen_validation.get("blocking_reasons") or [])
        | set(caveat_scope.get("blocking_reasons") or [])
    )
    decision = "no_go" if blockers else "go"
    cache_only = planned_stats["unique_missing_pairwise_labels"] == 0
    ready = decision == "go"
    recommended_max = round(min(max_usd, max(0.01, estimated_additional_spend)), 6)
    if ready and cache_only:
        recommendation = (
            "Runner infrastructure is ready for later reviewed execution of the "
            "current frozen manifest. The manifest is cache-only with zero "
            "unique missing labels, so later execution should make zero paid "
            "calls and buy no labels unless a new manifest is produced and "
            "separately reviewed."
        )
    elif ready:
        recommendation = (
            "Runner infrastructure is ready only under guarded pairwise-active "
            "execution with model availability checked, JSONL ledger writes, "
            "and hard cap enforcement."
        )
    else:
        recommendation = (
            "No-go for later execution until the listed frozen-manifest, caveat, "
            "or runner guardrail blockers are resolved."
        )
    return {
        "decision": decision,
        "runner_ready_for_later_execution": ready,
        "runner_ready_for_later_execution_mode": (
            "cache_only_zero_spend" if ready and cache_only else mode
        ),
        "blocking_reasons": blockers,
        "guardrail_blocking_reasons": list(guardrails.get("blocking_reasons") or []),
        "frozen_manifest_blocking_reasons": list(
            frozen_validation.get("blocking_reasons") or []
        ),
        "caveat_scope_blocking_reasons": list(
            caveat_scope.get("blocking_reasons") or []
        ),
        "model_availability_status": model_availability.get("status"),
        "estimated_additional_spend_usd": estimated_additional_spend,
        "recommended_max_usd": recommended_max if ready else 0.0,
        "cache_only_zero_spend_unless_manifest_changes": cache_only,
        "paid_label_purchase_authorized_by_this_artifact": False,
        "stop_rule": (
            "Do not run paid calls in this workflow. In a later reviewed run, "
            "stop at the first pointwise-call attempt, model-availability "
            "failure, JSON parse/error retry requiring manual review, hard-cap "
            "breach, ledger write failure, or any mismatch from the frozen "
            "manifest identity."
        ),
        "recommendation": recommendation,
    }


def _planned_execution(
    *,
    mode: RunnerMode,
    artifact_dir: Path,
    ledger_path: Path,
    output_path: Path,
    pairwise_model: str,
    pairwise_estimate: Any,
    max_usd: float,
    planned_stats: Mapping[str, Any],
    require_zero_missing_labels: bool,
) -> dict[str, Any]:
    cache_only = planned_stats["unique_missing_pairwise_labels"] == 0
    return {
        "mode": mode,
        "expected_execution_mode": (
            "cache_only_zero_spend" if cache_only else "guarded_pairwise_active"
        ),
        "artifact_dir": str(artifact_dir),
        "ledger_path": str(ledger_path),
        "output_path": str(output_path),
        "allowed_call_kind": PAIRWISE_CALL_KIND,
        "pointwise_calls_planned": 0,
        "random_control_paid_labels_planned": 0,
        "pairwise_model": pairwise_model,
        "model_provider": pairwise_model.split("/", 1)[0],
        "model_availability_required_before_execute": True,
        "hard_max_usd_cap": max_usd,
        "require_zero_missing_labels_for_current_caveat_scope": (
            require_zero_missing_labels
        ),
        "per_pairwise_call_estimate": {
            "input_tokens": pairwise_estimate.input_tokens,
            "output_tokens": pairwise_estimate.output_tokens,
            "cost_usd": pairwise_estimate.cost_usd,
        },
        "planned_pairwise_calls_to_buy": planned_stats[
            "unique_missing_pairwise_labels"
        ],
        "expected_additional_spend_usd": round(
            planned_stats["unique_missing_pairwise_labels"]
            * pairwise_estimate.cost_usd,
            6,
        ),
        "cache_only_zero_spend_unless_manifest_changes": cache_only,
    }


def _execute_guarded_pairwise_only(
    *,
    planned_rows: Sequence[Mapping[str, Any]],
    planned_stats: Mapping[str, Any],
    config_path: Path,
    source_artifact_dir: Path,
    phase: str,
    artifact_dir: Path,
    ledger: JsonlLedger,
    max_usd: float,
    pairwise_model: str,
    pairwise_estimate: Any,
    timeout_seconds: float,
    max_parse_retries: int,
    urlopen: UrlOpen,
) -> dict[str, Any]:
    _assert_pairwise_only_call_kind(PAIRWISE_CALL_KIND)
    missing_rows = _unique_missing_rows(planned_rows)
    if not missing_rows:
        return {
            "attempted": True,
            "paid_pairwise_calls_attempted": 0,
            "paid_pairwise_calls_succeeded": 0,
            "parse_retry_count": 0,
            "new_ledger_entries": 0,
            "status": "cache_only_zero_missing_labels",
        }

    papers_by_bucket = _load_pointwise_papers_for_missing_rows(
        missing_rows,
        source_artifact_dir=source_artifact_dir,
        phase=phase,
    )
    calls_before = _ledger_line_count(ledger.path)
    parse_retry_count = 0
    succeeded = 0
    for index, row in enumerate(missing_rows, start=1):
        _assert_pairwise_only_call_kind(PAIRWISE_CALL_KIND)
        bucket = str(row["bucket"])
        left_id = str(row["left_id"])
        right_id = str(row["right_id"])
        paper_by_id = papers_by_bucket[bucket]
        pair = ScheduledPair(
            left_id=left_id,
            right_id=right_id,
            priority=float(row.get("priority") or 0.0),
            purpose=str(row.get("purpose") or "new_information_guarded_pairwise"),
            order=PairwiseOrderMetadata.from_dict(row.get("order") or {}),
        )
        attempts = max_parse_retries + 1
        for attempt in range(1, attempts + 1):
            _assert_pairwise_only_call_kind(PAIRWISE_CALL_KIND)
            ledger.guard_projected_spend(
                cap_usd=max_usd,
                next_cost_usd=pairwise_estimate.cost_usd,
            )
            artifact_path = (
                artifact_dir
                / phase
                / _safe_name(bucket)
                / "calls"
                / (
                    f"{index:04d}-{PAIRWISE_CALL_KIND}-"
                    f"{fingerprint(bucket + ':' + left_id + ':' + right_id)}"
                    f"-attempt{attempt:02d}.json"
                )
            )
            try:
                response = _chat_json(
                    base_url=os.environ.get("SESTINA_LLM_BASE_URL") or "",
                    api_key=os.environ.get("SESTINA_LLM_API_KEY") or "",
                    payload=_pairwise_payload(
                        model=pairwise_model,
                        pair=pair,
                        papers=paper_by_id,
                    ),
                    timeout_seconds=timeout_seconds,
                    urlopen=urlopen,
                )
                comparison = _comparison_from_pairwise_response(pair, response)
                _write_pairwise_call_artifact(
                    artifact_path,
                    phase=phase,
                    bucket=bucket,
                    model=pairwise_model,
                    estimate=pairwise_estimate,
                    status="ok",
                    comparison=comparison,
                    response=response,
                    ledger=ledger,
                )
                succeeded += 1
                break
            except json.JSONDecodeError as exc:
                parse_retry_count += 1
                _write_pairwise_call_artifact(
                    artifact_path,
                    phase=phase,
                    bucket=bucket,
                    model=pairwise_model,
                    estimate=pairwise_estimate,
                    status="parse_error",
                    error=exc,
                    ledger=ledger,
                    subject={"left_id": left_id, "right_id": right_id},
                )
                if attempt >= attempts:
                    raise
            except Exception as exc:
                _write_pairwise_call_artifact(
                    artifact_path,
                    phase=phase,
                    bucket=bucket,
                    model=pairwise_model,
                    estimate=pairwise_estimate,
                    status="failed",
                    error=exc,
                    ledger=ledger,
                    subject={"left_id": left_id, "right_id": right_id},
                )
                raise

    return {
        "attempted": True,
        "paid_pairwise_calls_attempted": len(missing_rows),
        "paid_pairwise_calls_succeeded": succeeded,
        "parse_retry_count": parse_retry_count,
        "new_ledger_entries": _ledger_line_count(ledger.path) - calls_before,
        "status": "completed_pairwise_only",
        "planned_unique_missing_pairwise_labels": planned_stats[
            "unique_missing_pairwise_labels"
        ],
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
    error: BaseException | None = None,
    subject: dict[str, Any] | None = None,
) -> None:
    _assert_pairwise_only_call_kind(PAIRWISE_CALL_KIND)
    if comparison is not None:
        subject = {"left_id": comparison.left_id, "right_id": comparison.right_id}
    artifact = _call_artifact(
        phase=phase,
        bucket=bucket,
        model=model,
        kind=PAIRWISE_CALL_KIND,
        estimate=estimate,
        status=status,
        response=response,
        error=error,
        subject=subject or {},
    )
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
        )
    )


def _planned_pair_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    purpose_counts: Counter[str] = Counter()
    source_purpose_counts: Counter[str] = Counter()
    cache_kind_counts: Counter[str] = Counter()
    bucket_occurrence_counts: Counter[str] = Counter()
    unique: dict[tuple[str, tuple[str, str]], dict[str, Any]] = {}
    invalid_pair_key_rows = 0
    invalid_cache_status_rows = 0
    pointwise_like_rows = 0
    non_pairwise_rows = 0
    future_label_rows = 0
    cached_label_value_rows = 0

    for row in rows:
        bucket = str(row.get("bucket") or "")
        status = str(row.get("cache_status") or "")
        status_counts[status] += 1
        purpose_counts[str(row.get("purpose") or "")] += 1
        source_purpose_counts[str(row.get("source_new_information_purpose") or "")] += 1
        if row.get("cached_artifact_kind") is not None:
            cache_kind_counts[str(row.get("cached_artifact_kind"))] += 1
        if bucket:
            bucket_occurrence_counts[bucket] += 1
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
            unique[unique_key] = {
                "bucket": bucket,
                "pair_key": list(key_tuple),
                "cache_status": status,
            }
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
        "purpose_counts": dict(sorted(purpose_counts.items())),
        "source_new_information_purpose_counts": dict(
            sorted(
                (key, value)
                for key, value in source_purpose_counts.items()
                if key
            )
        ),
        "cache_reuse_by_kind": dict(sorted(cache_kind_counts.items())),
        "bucket_occurrence_counts": dict(sorted(bucket_occurrence_counts.items())),
    }


def _assert_pairwise_only_call_kind(kind: str) -> None:
    if kind != PAIRWISE_CALL_KIND:
        raise PointwiseCallForbiddenError(
            "guarded new-information runner only permits "
            f"{PAIRWISE_CALL_KIND!r} calls; attempted {kind!r}"
        )
    if "pointwise" in kind.lower():
        raise PointwiseCallForbiddenError(
            "guarded new-information runner aborts on pointwise-call attempts"
        )


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


def _load_pointwise_papers_for_missing_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_artifact_dir: Path,
    phase: str,
) -> dict[str, dict[str, Paper]]:
    needed_by_bucket: dict[str, set[str]] = {}
    for row in rows:
        bucket = str(row["bucket"])
        needed_by_bucket.setdefault(bucket, set()).update(
            {str(row["left_id"]), str(row["right_id"])}
        )
    loaded: dict[str, dict[str, Paper]] = {}
    for bucket, needed_ids in needed_by_bucket.items():
        calls_dir = source_artifact_dir / phase / bucket / "calls"
        bucket_papers: dict[str, Paper] = {}
        for path in sorted(calls_dir.glob("*-pointwise-*.json")):
            payload = _read_json(path)
            if payload.get("kind") != "pointwise" or payload.get("status") != "ok":
                continue
            subject = payload.get("subject") or {}
            paper_id = str(subject.get("paper_id") or "")
            if paper_id not in needed_ids:
                continue
            response = payload.get("response") or {}
            bucket_papers[paper_id] = Paper(
                paper_id=paper_id,
                title=str(subject.get("title") or paper_id),
                abstract=str(subject.get("abstract") or ""),
                pointwise=PointwiseAssessment.from_dict(response),
                metadata=dict(subject.get("metadata") or {}),
            )
        missing = sorted(needed_ids - set(bucket_papers))
        if missing:
            raise GuardedRunnerError(
                "missing source pointwise artifacts for pairwise-only payload "
                f"construction in bucket {bucket}: {', '.join(missing)}"
            )
        loaded[bucket] = bucket_papers
    return loaded


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object row in {path}")
        rows.append(row)
    return rows


def _ensure_jsonl_ledger_exists(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_matches(path: Path, expected: object) -> bool:
    return isinstance(expected, str) and _sha256(path) == expected


def _same_path(path: Path, other: object) -> bool:
    if other is None:
        return False
    try:
        return path.resolve() == Path(str(other)).resolve()
    except OSError:
        return False


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _false_keys(checks: Mapping[str, bool]) -> list[str]:
    return sorted(key for key, value in checks.items() if value is False)


def _int_field(payload: Mapping[str, Any], key: str, *, default: int = 0) -> int:
    value = payload.get(key)
    if value is None:
        return default
    return int(value)


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


def _ledger_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _stdout_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_path": payload.get("artifact_path") or payload.get("output_path"),
        "artifact_type": payload["artifact_type"],
        "mode": payload["mode"],
        "dry_run": payload["dry_run"],
        "paid_calls_made": payload["paid_calls_made"],
        "paid_spend_usd": payload["paid_spend_usd"],
        "pointwise_calls_made": payload["pointwise_calls_made"],
        "runner_ready_for_later_execution": payload["go_no_go"][
            "runner_ready_for_later_execution"
        ],
        "decision": payload["go_no_go"]["decision"],
        "blocking_reasons": payload["go_no_go"]["blocking_reasons"],
        "expected_execution_mode": payload["planned_execution"][
            "expected_execution_mode"
        ],
        "unique_missing_pairwise_labels": payload["totals"][
            "unique_missing_pairwise_labels"
        ],
        "pairwise_calls_to_buy": payload["totals"]["pairwise_calls_to_buy"],
        "ledger_path": payload["ledger"]["path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
