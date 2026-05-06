#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_new_information_guarded_runner import (  # noqa: E402
    ARM_EXACT,
    ARM_NEW_INFO,
    DEFAULT_ACTIVE_GATE_ARTIFACT,
    DEFAULT_BUDGET_FILL_ARTIFACT,
    DEFAULT_CAVEAT_ADJUDICATION,
    DEFAULT_CONFIG,
    DEFAULT_DRY_RUN_ARTIFACT,
    DEFAULT_MANIFEST,
    DEFAULT_PLANNED_PAIRS,
    DEFAULT_SOURCE_ARTIFACT_DIR,
    PAIRWISE_CALL_KIND,
    validate_guarded_runner_artifact_schema,
    _false_keys,
    _int_field,
    _planned_pair_stats,
    _read_json,
    _read_jsonl,
    _same_path,
    _sha256,
)
from sestina.active_arm_gate import (  # noqa: E402
    CURRENT_KNOWN_SPEND_USD,
    DEFAULT_PAID_CAP_USD,
)
from sestina.backtest_budget import load_config  # noqa: E402
from sestina.backtest_runner import (  # noqa: E402
    JsonlLedger,
    ModelAvailabilityError,
    _config_for_phase,
    check_model_availability,
    validate_model_names,
)
from sestina.diagnostics import write_json_artifact  # noqa: E402


ARTIFACT_TYPE = "sestina-new-information-execution-preflight"
SCHEMA_VERSION = 1
DEFAULT_GUARDED_RUNNER_ARTIFACT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-new-information-guarded-runner"
    / "guarded-runner-go-no-go.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts" / "backtest-arxiv-new-information-execution-preflight"
)
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "execution-preflight-go-no-go.json"
DEFAULT_GUARDED_RUNNER_SCRIPT = (
    REPO_ROOT / "scripts" / "run_new_information_guarded_runner.py"
)
DEFAULT_CHALLENGER_CODE = REPO_ROOT / "sestina" / "new_information_challenger.py"
DEFAULT_DECISION_MEMO = (
    REPO_ROOT / "docs" / "internal" / "sestina-experiment-decision-memo.md"
)
DEFAULT_HISTORICAL_RESULTS_DOC = (
    REPO_ROOT / "docs" / "internal" / "historical-arxiv-pilot-results.md"
)
REQUIRED_TOP_LEVEL_KEYS = {
    "artifact_type",
    "schema_version",
    "dry_run",
    "paid_calls_made",
    "paid_spend_usd",
    "pointwise_calls_made",
    "method",
    "input_artifacts",
    "provider_model_availability",
    "frozen_manifest_validation",
    "guarded_runner_state",
    "guardrail_checks",
    "ledger",
    "max_usd_cap",
    "totals",
    "final_go_no_go",
    "caveats",
    "validation_commands",
}
UrlOpen = Any


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final no-paid execution preflight for the frozen "
            "budget-filled new-information challenger manifest."
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
    parser.add_argument(
        "--guarded-runner-artifact",
        type=Path,
        default=DEFAULT_GUARDED_RUNNER_ARTIFACT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-usd", type=float, default=0.01)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    payload = build_new_information_execution_preflight(
        config_path=args.config,
        manifest_path=args.manifest,
        source_artifact_dir=args.source_artifact_dir,
        budget_fill_artifact_path=args.budget_fill_artifact,
        active_gate_artifact_path=args.active_gate_artifact,
        dry_run_artifact_path=args.dry_run_artifact,
        planned_pairs_path=args.planned_pairs,
        caveat_adjudication_path=args.caveat_adjudication,
        guarded_runner_artifact_path=args.guarded_runner_artifact,
        output_path=args.output,
        max_usd=args.max_usd,
        timeout_seconds=args.timeout_seconds,
    )
    sys.stdout.write(json.dumps(_stdout_summary(payload), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def build_new_information_execution_preflight(
    *,
    config_path: Path,
    manifest_path: Path,
    source_artifact_dir: Path,
    budget_fill_artifact_path: Path,
    active_gate_artifact_path: Path,
    dry_run_artifact_path: Path,
    planned_pairs_path: Path,
    caveat_adjudication_path: Path,
    guarded_runner_artifact_path: Path,
    output_path: Path,
    max_usd: float = 0.01,
    timeout_seconds: float = 30.0,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> dict[str, Any]:
    raw_config = load_config(config_path)
    budget_fill_artifact = _read_json(budget_fill_artifact_path)
    active_gate_artifact = _read_json(active_gate_artifact_path)
    dry_run_artifact = _read_json(dry_run_artifact_path)
    caveat_adjudication = _read_json(caveat_adjudication_path)
    guarded_runner = _read_json(guarded_runner_artifact_path)
    validate_guarded_runner_artifact_schema(guarded_runner)
    planned_rows = _read_jsonl(planned_pairs_path)
    planned_stats = _planned_pair_stats(planned_rows)

    phase = str(
        (guarded_runner.get("frozen_manifest_validation") or {})
        .get("identity", {})
        .get("phase")
        or (dry_run_artifact.get("frozen_inputs") or {}).get("phase")
        or "pilot"
    )
    phase_config = _config_for_phase(raw_config, phase=phase)["phases"][0]
    pairwise_model = str(
        (guarded_runner.get("planned_execution") or {}).get("pairwise_model")
        or phase_config["pairwise_model"]
    )
    provider_model_availability = _provider_model_availability(
        pairwise_model=pairwise_model,
        timeout_seconds=timeout_seconds,
        urlopen=urlopen,
    )
    frozen_manifest_validation = _frozen_manifest_validation(
        config_path=config_path,
        manifest_path=manifest_path,
        source_artifact_dir=source_artifact_dir,
        budget_fill_artifact_path=budget_fill_artifact_path,
        active_gate_artifact_path=active_gate_artifact_path,
        dry_run_artifact_path=dry_run_artifact_path,
        planned_pairs_path=planned_pairs_path,
        caveat_adjudication_path=caveat_adjudication_path,
        guarded_runner_artifact_path=guarded_runner_artifact_path,
        guarded_runner=guarded_runner,
        dry_run_artifact=dry_run_artifact,
        caveat_adjudication=caveat_adjudication,
        planned_stats=planned_stats,
        pairwise_model=pairwise_model,
        phase_config=phase_config,
    )
    guarded_runner_state = _guarded_runner_state(
        guarded_runner=guarded_runner,
        guarded_runner_artifact_path=guarded_runner_artifact_path,
    )
    guardrail_checks = _guardrail_checks(
        raw_config=raw_config,
        phase_config=phase_config,
        budget_fill_artifact=budget_fill_artifact,
        active_gate_artifact=active_gate_artifact,
        dry_run_artifact=dry_run_artifact,
        caveat_adjudication=caveat_adjudication,
        guarded_runner=guarded_runner,
        planned_stats=planned_stats,
        provider_model_availability=provider_model_availability,
        pairwise_model=pairwise_model,
        max_usd=max_usd,
    )
    ledger_path = Path(str((guarded_runner.get("ledger") or {}).get("path") or ""))
    ledger = JsonlLedger(ledger_path)
    estimated_additional_spend = float(
        (guarded_runner.get("totals") or {}).get("estimated_additional_spend_usd")
        or 0.0
    )
    final_go_no_go = _final_go_no_go(
        frozen_manifest_validation=frozen_manifest_validation,
        guarded_runner_state=guarded_runner_state,
        guardrail_checks=guardrail_checks,
        provider_model_availability=provider_model_availability,
        estimated_additional_spend=estimated_additional_spend,
        max_usd=max_usd,
    )
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "dry_run": True,
        "paid_calls_made": 0,
        "paid_spend_usd": 0.0,
        "pointwise_calls_made": 0,
        "method": {
            "summary": (
                "Final no-paid execution preflight over the reviewed frozen "
                "new-information challenger manifest and guarded-runner go/no-go."
            ),
            "paid_runner_invoked": False,
            "guarded_runner_execute_mode_invoked": False,
            "paid_labeling_invoked": False,
            "pointwise_runner_invoked": False,
            "label_generation_calls_made": 0,
            "chat_completions_calls_made": 0,
            "model_availability_check": "GET /models only",
            "secrets_printed_or_stored": False,
        },
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
            "guarded_runner_artifact_path": str(guarded_runner_artifact_path),
            "guarded_runner_artifact_sha256": _sha256(
                guarded_runner_artifact_path
            ),
            "guarded_runner_code_path": str(DEFAULT_GUARDED_RUNNER_SCRIPT),
            "guarded_runner_code_sha256": _sha256(DEFAULT_GUARDED_RUNNER_SCRIPT),
            "new_information_challenger_code_path": str(DEFAULT_CHALLENGER_CODE),
            "new_information_challenger_code_sha256": _sha256(
                DEFAULT_CHALLENGER_CODE
            ),
            "decision_memo_path": str(DEFAULT_DECISION_MEMO),
            "historical_results_doc_path": str(DEFAULT_HISTORICAL_RESULTS_DOC),
        },
        "provider_model_availability": provider_model_availability,
        "frozen_manifest_validation": frozen_manifest_validation,
        "guarded_runner_state": guarded_runner_state,
        "guardrail_checks": guardrail_checks,
        "ledger": {
            "path": str(ledger_path),
            "format": "jsonl",
            "line_count": _ledger_line_count(ledger_path),
            "existing_spend_usd": ledger.existing_spend_usd(),
            "expected_new_entries_in_later_cache_only_execution": 0,
            "historical_ledgers_rewritten": False,
        },
        "max_usd_cap": {
            "recommended_later_execution_cap_usd": max_usd,
            "guarded_runner_recommended_max_usd": (
                (guarded_runner.get("go_no_go") or {}).get("recommended_max_usd")
            ),
            "paid_cap_usd": DEFAULT_PAID_CAP_USD,
            "known_paid_spend_before_workflow_usd": CURRENT_KNOWN_SPEND_USD,
            "estimated_additional_spend_usd": estimated_additional_spend,
            "projected_known_paid_spend_after_workflow_usd": round(
                CURRENT_KNOWN_SPEND_USD + estimated_additional_spend,
                6,
            ),
        },
        "totals": {
            **planned_stats,
            "pairwise_calls_to_buy": planned_stats["unique_missing_pairwise_labels"],
            "estimated_additional_spend_usd": estimated_additional_spend,
            "known_paid_spend_before_workflow_usd": CURRENT_KNOWN_SPEND_USD,
            "paid_cap_usd": DEFAULT_PAID_CAP_USD,
            "projected_known_paid_spend_after_workflow_usd": round(
                CURRENT_KNOWN_SPEND_USD + estimated_additional_spend,
                6,
            ),
        },
        "final_go_no_go": final_go_no_go,
        "caveats": {
            "accepted_caveat_scope": (
                guarded_runner.get("caveat_scope") or {}
            ).get("scope_note"),
            "accepted_constraints": (
                guarded_runner.get("caveat_scope") or {}
            ).get("constraints")
            or caveat_adjudication.get("constraints")
            or [],
            "limitations": [
                "This preflight does not execute the guarded runner.",
                "This preflight does not buy labels or call chat/completions.",
                "The go decision is scoped only to the frozen zero-missing-label manifest and reviewed artifacts.",
            ],
        },
        "recommendation": final_go_no_go["recommendation"],
        "validation_commands": [
            "uv run python scripts/run_new_information_execution_preflight.py",
            "uv run python -m json.tool artifacts/backtest-arxiv-new-information-execution-preflight/execution-preflight-go-no-go.json >/dev/null",
            "uv run pytest tests/test_new_information_execution_preflight.py",
            "git diff --check",
            "uv run pytest -p no:cacheprovider",
        ],
        "output_path": str(output_path),
    }
    validate_execution_preflight_artifact_schema(payload)
    write_json_artifact(output_path, payload)
    return {**payload, "artifact_path": str(output_path)}


def validate_execution_preflight_artifact_schema(payload: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(payload))
    if missing:
        raise ValueError(
            "execution preflight artifact missing top-level keys: "
            + ", ".join(missing)
        )
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("execution preflight artifact has unexpected artifact_type")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("execution preflight artifact has unexpected schema_version")
    if payload.get("dry_run") is not True:
        raise ValueError("execution preflight must be a dry-run artifact")
    if payload.get("paid_calls_made") != 0:
        raise ValueError("execution preflight must make zero paid calls")
    if float(payload.get("paid_spend_usd") or 0.0) != 0.0:
        raise ValueError("execution preflight must spend zero USD")
    if payload.get("pointwise_calls_made") != 0:
        raise ValueError("execution preflight must make zero pointwise calls")
    final = payload.get("final_go_no_go")
    if not isinstance(final, Mapping):
        raise ValueError("execution preflight final_go_no_go must be an object")
    if final.get("decision") not in {"go", "no_go"}:
        raise ValueError("execution preflight decision must be go/no_go")
    if final.get("decision") == "go":
        if final.get("recommended_later_execution_cap_usd") != 0.01:
            raise ValueError("go decision requires explicit USD 0.01 cap")
        if final.get("expected_execution_mode") != "cache_only_zero_spend":
            raise ValueError("go decision requires cache-only zero-spend mode")
    serialized = json.dumps(payload, sort_keys=True)
    forbidden_fragments = ("Authorization", "Bearer ")
    if any(fragment in serialized for fragment in forbidden_fragments):
        raise ValueError("execution preflight artifact must not include secrets")


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
            "required_before_later_execution": True,
            "check_method": "GET /models",
            "requested_models": [pairwise_model],
            "missing_models": [pairwise_model],
            "available_requested_models": [],
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
            "required_before_later_execution": True,
            "check_method": "GET /models",
            "requested_models": [pairwise_model],
            "missing_models": [pairwise_model],
            "available_requested_models": [],
            "label_generation_calls_made": 0,
            "chat_completions_calls_made": 0,
            "api_key_env_present": bool(os.environ.get("SESTINA_LLM_API_KEY")),
            "base_url_env_present": bool(os.environ.get("SESTINA_LLM_BASE_URL")),
            "secrets_printed_or_stored": False,
            "error": str(exc),
        }
    return {
        **result,
        "required_before_later_execution": True,
        "check_method": "GET /models",
        "label_generation_calls_made": 0,
        "chat_completions_calls_made": 0,
        "api_key_env_present": bool(os.environ.get("SESTINA_LLM_API_KEY")),
        "base_url_env_present": bool(os.environ.get("SESTINA_LLM_BASE_URL")),
        "secrets_printed_or_stored": False,
    }


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
    guarded_runner_artifact_path: Path,
    guarded_runner: Mapping[str, Any],
    dry_run_artifact: Mapping[str, Any],
    caveat_adjudication: Mapping[str, Any],
    planned_stats: Mapping[str, Any],
    pairwise_model: str,
    phase_config: Mapping[str, Any],
) -> dict[str, Any]:
    guarded_inputs = guarded_runner.get("input_artifacts") or {}
    dry_inputs = dry_run_artifact.get("input_artifacts") or {}
    dry_frozen = dry_run_artifact.get("frozen_inputs") or {}
    dry_totals = dry_run_artifact.get("totals") or {}
    dry_planned = dry_run_artifact.get("planned_execution") or {}
    caveat_inputs = caveat_adjudication.get("input_artifacts") or {}
    guarded_frozen = guarded_runner.get("frozen_manifest_validation") or {}
    guarded_shape = guarded_frozen.get("shape") or {}
    guarded_totals = guarded_runner.get("totals") or {}

    checks = {
        "guarded_runner_artifact_type": guarded_runner.get("artifact_type")
        == "sestina-new-information-guarded-runner-go-no-go",
        "config_sha_matches_guarded": _sha_matches(
            config_path,
            guarded_inputs.get("config_sha256"),
        ),
        "manifest_sha_matches_guarded": _sha_matches(
            manifest_path,
            guarded_inputs.get("manifest_sha256"),
        ),
        "budget_fill_sha_matches_guarded": _sha_matches(
            budget_fill_artifact_path,
            guarded_inputs.get("budget_fill_artifact_sha256"),
        ),
        "active_gate_sha_matches_guarded": _sha_matches(
            active_gate_artifact_path,
            guarded_inputs.get("active_gate_artifact_sha256"),
        ),
        "dry_run_sha_matches_guarded": _sha_matches(
            dry_run_artifact_path,
            guarded_inputs.get("dry_run_artifact_sha256"),
        ),
        "planned_pairs_sha_matches_guarded": _sha_matches(
            planned_pairs_path,
            guarded_inputs.get("planned_pairs_sha256"),
        ),
        "caveat_sha_matches_guarded": _sha_matches(
            caveat_adjudication_path,
            guarded_inputs.get("caveat_adjudication_sha256"),
        ),
        "runner_code_sha_matches_guarded": _sha_matches(
            DEFAULT_GUARDED_RUNNER_SCRIPT,
            guarded_inputs.get("runner_code_sha256"),
        ),
        "config_sha_matches_dry_run": _sha_matches(
            config_path,
            dry_inputs.get("config_sha256"),
        ),
        "manifest_sha_matches_dry_run": _sha_matches(
            manifest_path,
            dry_inputs.get("manifest_sha256"),
        ),
        "budget_fill_sha_matches_dry_run": _sha_matches(
            budget_fill_artifact_path,
            dry_inputs.get("budget_fill_artifact_sha256"),
        ),
        "active_gate_sha_matches_dry_run": _sha_matches(
            active_gate_artifact_path,
            dry_inputs.get("active_gate_artifact_sha256"),
        ),
        "dry_run_sha_matches_caveat": _sha_matches(
            dry_run_artifact_path,
            caveat_inputs.get("dry_run_artifact_sha256"),
        ),
        "planned_pairs_sha_matches_caveat": _sha_matches(
            planned_pairs_path,
            caveat_inputs.get("planned_pairs_sha256"),
        ),
        "planned_pairs_path_matches_dry_run": _same_path(
            planned_pairs_path,
            dry_run_artifact.get("planned_pair_occurrences_path"),
        ),
        "source_artifact_dir_matches_dry_run": _same_path(
            source_artifact_dir,
            dry_inputs.get("source_artifact_dir"),
        ),
        "active_arm_name_matches": dry_frozen.get("active_arm_name") == ARM_NEW_INFO,
        "random_control_matches": dry_frozen.get("random_control_comparator")
        == ARM_EXACT,
        "pairwise_model_matches_config": pairwise_model
        == str(phase_config.get("pairwise_model")),
        "pairwise_model_matches_dry_run": pairwise_model
        == dry_frozen.get("pairwise_model"),
        "guarded_shape_matches_planned_rows": _int_field(
            guarded_shape,
            "planned_pair_occurrences",
            default=-1,
        )
        == planned_stats["pairwise_scheduled_occurrences"],
        "guarded_unique_pair_labels_match": _int_field(
            guarded_shape,
            "unique_planned_pair_labels",
            default=-1,
        )
        == planned_stats["unique_planned_pair_labels"],
        "guarded_unique_missing_labels_match": _int_field(
            guarded_shape,
            "unique_missing_pairwise_labels",
            default=-1,
        )
        == planned_stats["unique_missing_pairwise_labels"],
        "dry_run_pair_occurrences_match": int(
            dry_run_artifact.get("planned_pair_occurrence_count") or -1
        )
        == planned_stats["pairwise_scheduled_occurrences"],
        "dry_run_unique_pair_labels_match": _int_field(
            dry_totals,
            "unique_planned_pair_labels",
            default=-1,
        )
        == planned_stats["unique_planned_pair_labels"],
        "dry_run_unique_missing_labels_match": _int_field(
            dry_totals,
            "unique_missing_pairwise_labels",
            default=-1,
        )
        == planned_stats["unique_missing_pairwise_labels"],
        "guarded_totals_unique_missing_labels_match": _int_field(
            guarded_totals,
            "unique_missing_pairwise_labels",
            default=-1,
        )
        == planned_stats["unique_missing_pairwise_labels"],
        "planned_pairwise_label_kind_pairwise_active": dry_planned.get(
            "planned_pairwise_label_kind"
        )
        == PAIRWISE_CALL_KIND,
    }
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


def _guarded_runner_state(
    *,
    guarded_runner: Mapping[str, Any],
    guarded_runner_artifact_path: Path,
) -> dict[str, Any]:
    go_no_go = guarded_runner.get("go_no_go") or {}
    planned_execution = guarded_runner.get("planned_execution") or {}
    checks = {
        "guarded_artifact_schema_valid": True,
        "guarded_artifact_path_exists": guarded_runner_artifact_path.exists(),
        "guarded_artifact_dry_run": guarded_runner.get("dry_run") is True,
        "guarded_decision_go": go_no_go.get("decision") == "go",
        "runner_ready_for_later_execution": go_no_go.get(
            "runner_ready_for_later_execution"
        )
        is True,
        "runner_ready_mode_cache_only_zero_spend": go_no_go.get(
            "runner_ready_for_later_execution_mode"
        )
        == "cache_only_zero_spend",
        "guarded_expected_cache_only_zero_spend": planned_execution.get(
            "expected_execution_mode"
        )
        == "cache_only_zero_spend",
        "guarded_paid_label_purchase_not_authorized": go_no_go.get(
            "paid_label_purchase_authorized_by_this_artifact"
        )
        is False,
        "guarded_estimated_additional_spend_zero": float(
            go_no_go.get("estimated_additional_spend_usd") or 0.0
        )
        == 0.0,
        "guarded_recommended_max_usd_001": float(
            go_no_go.get("recommended_max_usd") or 0.0
        )
        == 0.01,
    }
    blocking = _false_keys(checks)
    return {
        "checks": checks,
        "blocking_reasons": blocking,
        "valid": not blocking,
        "prior_model_availability_status": go_no_go.get("model_availability_status"),
        "decision": go_no_go.get("decision"),
        "runner_ready_for_later_execution": go_no_go.get(
            "runner_ready_for_later_execution"
        ),
        "runner_ready_for_later_execution_mode": go_no_go.get(
            "runner_ready_for_later_execution_mode"
        ),
        "expected_execution_mode": planned_execution.get("expected_execution_mode"),
        "recommended_max_usd": go_no_go.get("recommended_max_usd"),
        "stop_rule": go_no_go.get("stop_rule"),
    }


def _guardrail_checks(
    *,
    raw_config: Mapping[str, Any],
    phase_config: Mapping[str, Any],
    budget_fill_artifact: Mapping[str, Any],
    active_gate_artifact: Mapping[str, Any],
    dry_run_artifact: Mapping[str, Any],
    caveat_adjudication: Mapping[str, Any],
    guarded_runner: Mapping[str, Any],
    planned_stats: Mapping[str, Any],
    provider_model_availability: Mapping[str, Any],
    pairwise_model: str,
    max_usd: float,
) -> dict[str, Any]:
    model_policy = raw_config.get("model_name_policy") or {}
    active_gate_verdict = active_gate_artifact.get("gate_verdict") or {}
    active_gate_caveats = active_gate_artifact.get("caveats") or {}
    missing_caveat = active_gate_caveats.get("missing_label_caveat") or {}
    budget_caveat = active_gate_caveats.get("budget_completeness_caveat") or {}
    dry_planned = dry_run_artifact.get("planned_execution") or {}
    dry_totals = dry_run_artifact.get("totals") or {}
    guarded_guardrails = guarded_runner.get("guardrails") or {}
    guarded_guardrail_policy = guarded_guardrails.get("guardrail_policy") or {}
    guarded_go = guarded_runner.get("go_no_go") or {}
    guarded_planned = guarded_runner.get("planned_execution") or {}
    guarded_ledger = guarded_runner.get("ledger") or {}
    ledger_path = Path(str(guarded_ledger.get("path") or ""))

    checks = {
        "model_name_policy_requires_provider_prefix": model_policy.get(
            "require_provider_prefix"
        )
        is True,
        "model_name_policy_requires_availability_check": model_policy.get(
            "availability_check_required_before_paid_run"
        )
        is True,
        "provider_prefixed_pairwise_model": "/" in pairwise_model,
        "pairwise_model_in_rate_card": pairwise_model
        in (raw_config.get("rate_card") or {}),
        "pairwise_model_matches_phase_config": pairwise_model
        == str(phase_config.get("pairwise_model")),
        "model_availability_available": provider_model_availability.get("status")
        == "available",
        "budget_fill_zero_paid": budget_fill_artifact.get("paid_calls_made") == 0
        and float(budget_fill_artifact.get("paid_spend_usd") or 0.0) == 0.0,
        "budget_fill_zero_pointwise": budget_fill_artifact.get(
            "pointwise_calls_made"
        )
        == 0,
        "active_gate_paid_followup_allowed": active_gate_artifact.get(
            "paid_followup_allowed"
        )
        is True
        and active_gate_verdict.get("paid_followup_allowed", True) is True,
        "active_gate_no_blocking_reasons": not bool(
            active_gate_verdict.get("blocking_reasons") or []
        ),
        "active_gate_no_missing_label_caveat": not bool(missing_caveat.get("present")),
        "active_gate_no_budget_completeness_caveat": not bool(
            budget_caveat.get("present")
        ),
        "caveat_accepted_with_constraints": caveat_adjudication.get("decision")
        == "caveat_accepted_with_constraints",
        "caveat_acceptance_scoped_current_manifest": (
            guarded_runner.get("caveat_scope") or {}
        ).get("accepted_for_current_manifest")
        is True,
        "dry_run_zero_paid": dry_run_artifact.get("paid_calls_made") == 0
        and float(dry_run_artifact.get("paid_spend_usd") or 0.0) == 0.0,
        "dry_run_zero_pointwise": dry_run_artifact.get("pointwise_calls_made") == 0
        and _int_field(dry_totals, "pointwise_calls", default=-1) == 0,
        "planned_rows_pairwise_only": planned_stats["pointwise_like_planned_rows"]
        == 0
        and planned_stats["non_pairwise_call_rows"] == 0,
        "planned_pointwise_calls_zero": planned_stats["pointwise_calls"] == 0
        and dry_planned.get("pointwise_calls_planned") == 0
        and guarded_planned.get("pointwise_calls_planned") == 0,
        "random_control_paid_labels_zero": dry_planned.get(
            "random_control_paid_labels_planned"
        )
        == 0
        and guarded_planned.get("random_control_paid_labels_planned") == 0,
        "zero_missing_pairwise_labels": planned_stats[
            "unique_missing_pairwise_labels"
        ]
        == 0,
        "expected_cache_only_execution": guarded_planned.get(
            "expected_execution_mode"
        )
        == "cache_only_zero_spend",
        "estimated_additional_spend_zero": float(
            guarded_go.get("estimated_additional_spend_usd") or 0.0
        )
        == 0.0,
        "max_usd_cap_is_001": max_usd == 0.01,
        "max_usd_cap_lte_paid_cap": max_usd <= DEFAULT_PAID_CAP_USD,
        "known_spend_preserved": float(
            (guarded_runner.get("totals") or {}).get(
                "known_paid_spend_before_workflow_usd"
            )
            or 0.0
        )
        == CURRENT_KNOWN_SPEND_USD,
        "ledger_jsonl": ledger_path.suffix == ".jsonl",
        "ledger_empty_before_execution": _ledger_line_count(ledger_path) == 0,
        "ledger_existing_spend_zero": JsonlLedger(ledger_path).existing_spend_usd()
        == 0.0,
        "artifact_directory_separate": (
            guarded_runner.get("guardrails") or {}
        ).get("checks", {}).get("separate_artifact_directory")
        is True,
        "pointwise_abort_guard_enabled": guarded_guardrail_policy.get(
            "abort_on_pointwise_call_attempt"
        )
        is True
        and (guarded_guardrails.get("checks") or {}).get("pointwise_call_trap_enabled")
        is True,
        "allowed_call_kind_pairwise_active": guarded_guardrail_policy.get(
            "allowed_network_call_kind"
        )
        == PAIRWISE_CALL_KIND,
        "no_paid_label_purchase_authorized": guarded_go.get(
            "paid_label_purchase_authorized_by_this_artifact"
        )
        is False,
    }
    blocking = _false_keys(checks)
    return {
        "checks": checks,
        "blocking_reasons": blocking,
        "valid": not blocking,
        "policy": {
            "paid_labels_allowed_in_this_preflight": False,
            "runner_execution_allowed_in_this_preflight": False,
            "later_execution_allowed_call_kind": PAIRWISE_CALL_KIND,
            "abort_on_pointwise_call_attempt": True,
            "hard_max_usd_cap": max_usd,
        },
    }


def _final_go_no_go(
    *,
    frozen_manifest_validation: Mapping[str, Any],
    guarded_runner_state: Mapping[str, Any],
    guardrail_checks: Mapping[str, Any],
    provider_model_availability: Mapping[str, Any],
    estimated_additional_spend: float,
    max_usd: float,
) -> dict[str, Any]:
    blockers = sorted(
        set(frozen_manifest_validation.get("blocking_reasons") or [])
        | {
            f"guarded_runner:{item}"
            for item in guarded_runner_state.get("blocking_reasons") or []
        }
        | {
            f"guardrail:{item}"
            for item in guardrail_checks.get("blocking_reasons") or []
        }
    )
    if provider_model_availability.get("status") != "available":
        blockers.append("provider_model_availability")
    blockers = sorted(set(blockers))
    decision = "no_go" if blockers else "go"
    if decision == "go":
        recommendation = (
            "Later reviewed execution may use the guarded runner for this exact "
            "frozen manifest with --max-usd 0.01. Expected behavior remains "
            "cache-only and zero-spend: no paid labels, no chat/completions "
            "label-generation calls, and zero pointwise calls."
        )
    else:
        recommendation = (
            "No-go for later execution until the listed preflight blockers are "
            "resolved. Do not run paid labels or execute the guarded runner."
        )
    return {
        "decision": decision,
        "blocking_reasons": blockers,
        "provider_model_availability_status": provider_model_availability.get(
            "status"
        ),
        "runner_ready_for_later_execution": decision == "go",
        "expected_execution_mode": "cache_only_zero_spend",
        "estimated_additional_spend_usd": estimated_additional_spend,
        "recommended_later_execution_cap_usd": max_usd if decision == "go" else 0.0,
        "expected_paid_calls_in_later_execution": 0,
        "expected_pointwise_calls_in_later_execution": 0,
        "paid_label_purchase_authorized_by_this_artifact": False,
        "recommendation": recommendation,
        "stop_rule": (
            "Do not execute in this preflight. In a later reviewed run, stop at "
            "the first pointwise-call attempt, model-availability failure, hard "
            "cap breach, ledger write failure, or mismatch from the frozen "
            "manifest identity."
        ),
    }


def _sha_matches(path: Path, expected: object) -> bool:
    return isinstance(expected, str) and _sha256(path) == expected


def _ledger_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _stdout_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_path": payload.get("artifact_path") or payload.get("output_path"),
        "artifact_type": payload["artifact_type"],
        "dry_run": payload["dry_run"],
        "paid_calls_made": payload["paid_calls_made"],
        "paid_spend_usd": payload["paid_spend_usd"],
        "pointwise_calls_made": payload["pointwise_calls_made"],
        "provider_model_availability_status": payload[
            "provider_model_availability"
        ]["status"],
        "decision": payload["final_go_no_go"]["decision"],
        "blocking_reasons": payload["final_go_no_go"]["blocking_reasons"],
        "expected_execution_mode": payload["final_go_no_go"][
            "expected_execution_mode"
        ],
        "recommended_later_execution_cap_usd": payload["final_go_no_go"][
            "recommended_later_execution_cap_usd"
        ],
        "unique_missing_pairwise_labels": payload["totals"][
            "unique_missing_pairwise_labels"
        ],
        "pairwise_calls_to_buy": payload["totals"]["pairwise_calls_to_buy"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
