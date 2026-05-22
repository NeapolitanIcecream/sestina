#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
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
    _chat_json_with_usage,
    _config_for_phase,
    _ledger_entry,
    _load_ok_call_artifact,
    _normalize_rates_from_config,
    _normalize_token_assumptions_from_config,
    _pointwise_payload,
    _safe_name,
    check_model_availability,
    load_dataset_manifest,
    usage_cost_payload,
    validate_model_names,
)
from sestina.diagnostics import fingerprint, write_json_artifact  # noqa: E402
from sestina.models import Paper, PointwiseAssessment  # noqa: E402


ARTIFACT_TYPE = "sestina-fresh-holdout-pointwise-artifacts"
SCHEMA_VERSION = 1
RunnerMode = Literal["planning", "execute"]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "arxiv_historical_pilot_budget_config.json"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "artifacts"
    / "backtest-datasets"
    / "arxiv-historical-coverage-floor-fresh-holdout-manifest.json"
)
DEFAULT_ARTIFACT_DIR = (
    REPO_ROOT / "artifacts" / "backtest-arxiv-coverage-floor-fresh-holdout-pointwise"
)
DEFAULT_LEDGER = DEFAULT_ARTIFACT_DIR / "fresh-holdout-pointwise-ledger.jsonl"
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_DIR / "fresh-holdout-pointwise-review.json"
DEFAULT_MAX_USD = round(DEFAULT_PAID_CAP_USD - CURRENT_KNOWN_SPEND_USD, 6)
FORBIDDEN_MODEL_VISIBLE_KEYS = {
    "good_paper",
    "citation_count",
    "citation_rank",
    "citation_percentile",
    "citation_positive",
    "citation_positive_cutoff_rank",
    "citation_match",
    "matched_title",
    "work_id",
    "label_source",
    "metadata_fetched_at",
    "evaluation_label",
    "evaluation_labels",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plan, execute, and review pointwise artifacts for the autonomous "
            "fresh holdout only. No pairwise calls are made by this script."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", default="pilot")
    parser.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--mode", choices=("planning", "execute"), default="planning")
    parser.add_argument(
        "--confirm-fresh-holdout-pointwise-generation",
        action="store_true",
        help="required with --mode execute before any pointwise LLM calls",
    )
    args = parser.parse_args(argv)

    payload = build_fresh_holdout_pointwise_artifacts(
        config_path=args.config,
        manifest_path=args.manifest,
        artifact_dir=args.artifact_dir,
        ledger_path=args.ledger,
        output_path=args.output,
        phase=args.phase,
        max_usd=args.max_usd,
        timeout_seconds=args.timeout_seconds,
        mode=args.mode,
        confirm_fresh_holdout_pointwise_generation=(
            args.confirm_fresh_holdout_pointwise_generation
        ),
    )
    sys.stdout.write(json.dumps(_stdout_summary(payload), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    if args.mode == "execute" and payload["final_status"]["status"] != "complete":
        return 2
    return 0


def build_fresh_holdout_pointwise_artifacts(
    *,
    config_path: Path,
    manifest_path: Path,
    artifact_dir: Path,
    ledger_path: Path,
    output_path: Path,
    phase: str = "pilot",
    max_usd: float = DEFAULT_MAX_USD,
    timeout_seconds: float = 60.0,
    mode: RunnerMode = "planning",
    confirm_fresh_holdout_pointwise_generation: bool = False,
    urlopen: Any = urllib.request.urlopen,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_config = load_config(config_path)
    phase_config = _config_for_phase(raw_config, phase=phase)["phases"][0]
    pointwise_model = str(phase_config["pointwise_model"])
    validate_model_names([pointwise_model])
    token_assumptions = _normalize_token_assumptions_from_config(raw_config)
    rates = _normalize_rates_from_config(raw_config)
    pointwise_estimate = _call_estimate(
        "pointwise",
        pointwise_model,
        token_assumptions,
        rates,
    )
    manifest = load_dataset_manifest(manifest_path)
    buckets = manifest.buckets_for_phase(phase)
    ledger = JsonlLedger(ledger_path)
    existing_spend = ledger.existing_spend_usd()
    planned = _planned_pointwise_rows(
        buckets,
        artifact_dir=artifact_dir,
        phase=phase,
    )
    missing_rows = [row for row in planned if row["status"] == "missing"]
    present_rows = [row for row in planned if row["status"] == "available"]
    leakage = _model_visible_leakage(
        buckets,
        model=pointwise_model,
    )
    projected_spend = round(
        existing_spend + (len(missing_rows) * pointwise_estimate.cost_usd),
        6,
    )
    provider = _provider_model_availability(
        pointwise_model=pointwise_model,
        timeout_seconds=timeout_seconds,
        required=mode == "execute",
        urlopen=urlopen,
    )
    guardrails = _guardrails(
        buckets=buckets,
        mode=mode,
        confirm_fresh_holdout_pointwise_generation=(
            confirm_fresh_holdout_pointwise_generation
        ),
        ledger_path=ledger_path,
        artifact_dir=artifact_dir,
        max_usd=max_usd,
        existing_spend=existing_spend,
        projected_spend=projected_spend,
        missing_count=len(missing_rows),
        leakage=leakage,
        provider=provider,
    )

    execution = {
        "attempted": False,
        "status": "not_attempted",
        "pointwise_calls_attempted": 0,
        "pointwise_calls_succeeded": 0,
        "new_ledger_entries": 0,
    }
    paid_calls_made = 0
    paid_spend_usd = 0.0
    if mode == "execute" and not guardrails["blocking_reasons"]:
        spend_before = ledger.existing_spend_usd()
        calls_before = _ledger_line_count(ledger_path)
        execution = _execute_missing_pointwise(
            missing_rows,
            pointwise_model=pointwise_model,
            pointwise_estimate=pointwise_estimate,
            rates=rates,
            ledger=ledger,
            max_usd=max_usd,
            timeout_seconds=timeout_seconds,
            urlopen=urlopen,
        )
        paid_calls_made = int(execution["pointwise_calls_succeeded"])
        paid_spend_usd = round(ledger.existing_spend_usd() - spend_before, 6)
        execution["new_ledger_entries"] = _ledger_line_count(ledger_path) - calls_before
        planned = _planned_pointwise_rows(
            buckets,
            artifact_dir=artifact_dir,
            phase=phase,
        )
        missing_rows = [row for row in planned if row["status"] == "missing"]
        present_rows = [row for row in planned if row["status"] == "available"]

    review = _review_artifacts(planned, leakage=leakage)
    planned_public = [_public_pointwise_row(row) for row in planned]
    final_status = {
        "status": "complete" if review["complete"] else "incomplete",
        "blocking_reasons": sorted(set(guardrails["blocking_reasons"]) | set(review["blocking_reasons"])),
        "pointwise_artifacts_reviewed": review["complete"],
        "safe_for_pairwise_preflight_source": review["complete"],
    }
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "dry_run": mode == "planning",
        "paid_calls_made": paid_calls_made,
        "paid_spend_usd": paid_spend_usd,
        "pointwise_calls_made": paid_calls_made,
        "input_artifacts": {
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "artifact_dir": str(artifact_dir),
            "ledger_path": str(ledger_path),
            "runner_code_path": str(
                REPO_ROOT / "scripts" / "run_fresh_holdout_pointwise_artifacts.py"
            ),
        },
        "method": {
            "summary": (
                "Fresh-holdout pointwise artifact generation and review for "
                "the autonomous coverage-floor campaign."
            ),
            "allowed_call_kind": "pointwise",
            "pairwise_calls_made": 0,
            "future_labels_used_as_model_features": False,
            "model_visible_prompt_source": (
                "manifest title, abstract, and sanitized metadata only"
            ),
            "raw_prompts_stored": False,
            "secrets_printed_or_stored": False,
        },
        "fresh_holdout": {
            "phase": phase,
            "bucket_count": len(buckets),
            "bucket_names": [bucket.name for bucket in buckets],
            "expected_pointwise_artifacts": len(planned),
            "available_pointwise_artifacts": len(present_rows),
            "missing_pointwise_artifacts": len(missing_rows),
        },
        "provider_model_availability": provider,
        "ledger": {
            "path": str(ledger_path),
            "format": "jsonl",
            "line_count": _ledger_line_count(ledger_path),
            "spend_usd_before_workflow": existing_spend,
            "spend_usd_after_workflow": ledger.existing_spend_usd(),
            "historical_ledgers_rewritten": False,
        },
        "cost_policy": {
            "cap_policy": "campaign_total_usd_100_including_known_prior_spend",
            "known_paid_spend_before_workflow_usd": CURRENT_KNOWN_SPEND_USD,
            "additional_max_usd": max_usd,
            "paid_cap_usd": DEFAULT_PAID_CAP_USD,
            "per_pointwise_call_estimate": {
                "input_tokens": pointwise_estimate.input_tokens,
                "output_tokens": pointwise_estimate.output_tokens,
                "cost_usd": pointwise_estimate.cost_usd,
            },
            "projected_additional_spend_usd": projected_spend,
            "cost_measurement": (
                "Ledger records upstream returned cost when present, otherwise "
                "cost estimated from upstream returned usage and configured "
                "rates, otherwise the conservative configured token estimate."
            ),
        },
        "model_visible_leakage_review": leakage,
        "planned_pointwise_rows": planned_public,
        "guardrails": guardrails,
        "execution_summary": execution,
        "review": review,
        "final_status": final_status,
        "validation_commands": [
            "uv run python scripts/run_fresh_holdout_pointwise_artifacts.py",
            "uv run pytest tests/test_fresh_holdout_pointwise_artifacts.py",
            "git diff --check",
        ],
        "output_path": str(output_path),
    }
    validate_fresh_holdout_pointwise_artifact(payload)
    write_json_artifact(output_path, payload)
    return {**payload, "artifact_path": str(output_path)}


def validate_fresh_holdout_pointwise_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("fresh pointwise artifact has unexpected artifact_type")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("fresh pointwise artifact has unexpected schema_version")
    if int(payload.get("pointwise_calls_made") or 0) != int(
        payload.get("paid_calls_made") or 0
    ):
        raise ValueError("fresh pointwise paid calls must equal pointwise calls")
    method = payload.get("method")
    if not isinstance(method, Mapping) or method.get("pairwise_calls_made") != 0:
        raise ValueError("fresh pointwise runner must not make pairwise calls")
    leakage = payload.get("model_visible_leakage_review")
    if not isinstance(leakage, Mapping):
        raise ValueError("fresh pointwise artifact missing leakage review")
    final_status = payload.get("final_status")
    if (
        leakage.get("present") is not False
        and isinstance(final_status, Mapping)
        and final_status.get("status") == "complete"
    ):
        raise ValueError("fresh pointwise artifact cannot be complete with leakage")
    serialized = json.dumps(payload, sort_keys=True)
    if "Authorization" in serialized or "Bearer " in serialized:
        raise ValueError("fresh pointwise artifact must not include secrets")


def _planned_pointwise_rows(
    buckets: Sequence[Any],
    *,
    artifact_dir: Path,
    phase: str,
) -> list[dict[str, Any]]:
    rows = []
    for bucket in buckets:
        calls_dir = artifact_dir / phase / _safe_name(bucket.name) / "calls"
        for index, paper in enumerate(bucket.papers, start=1):
            path = calls_dir / (
                f"{index:04d}-pointwise-{fingerprint(paper.paper_id)}.json"
            )
            cached = _load_ok_call_artifact(path)
            rows.append(
                {
                    "bucket": bucket.name,
                    "phase": phase,
                    "paper_id": paper.paper_id,
                    "paper": paper,
                    "artifact_path": str(path),
                    "status": "available" if cached is not None else "missing",
                    "call_kind": "pointwise",
                    "future_labels_used_as_model_features": False,
                    "good_paper_used_as_model_feature": False,
                    "citation_outcomes_used_as_model_features": False,
                }
            )
    return rows


def _public_pointwise_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"paper"}
    }


def _execute_missing_pointwise(
    rows: Sequence[Mapping[str, Any]],
    *,
    pointwise_model: str,
    pointwise_estimate: Any,
    rates: dict[str, dict[str, float]],
    ledger: JsonlLedger,
    max_usd: float,
    timeout_seconds: float,
    urlopen: Any,
) -> dict[str, Any]:
    _ensure_jsonl_ledger_exists(ledger.path)
    attempted = 0
    succeeded = 0
    for row in rows:
        paper = _paper_from_row(row)
        artifact_path = Path(str(row["artifact_path"]))
        ledger.guard_projected_spend(
            cap_usd=max_usd,
            next_cost_usd=pointwise_estimate.cost_usd,
        )
        _guard_campaign_total(
            existing_spend=ledger.existing_spend_usd(),
            next_cost_usd=pointwise_estimate.cost_usd,
        )
        attempted += 1
        try:
            response = _chat_json_with_usage(
                base_url=os.environ.get("SESTINA_LLM_BASE_URL") or "",
                api_key=os.environ.get("SESTINA_LLM_API_KEY") or "",
                payload=_pointwise_payload(model=pointwise_model, paper=paper),
                timeout_seconds=timeout_seconds,
                urlopen=urlopen,
            )
            assessment = PointwiseAssessment.from_dict(response.content)
            cost_payload = usage_cost_payload(
                model=pointwise_model,
                estimate=pointwise_estimate,
                rates=rates,
                response=response,
            )
            status = "ok"
            artifact = _call_artifact(
                phase=str(row["phase"]),
                bucket=str(row["bucket"]),
                model=pointwise_model,
                kind="pointwise",
                estimate=pointwise_estimate,
                status=status,
                response=assessment.to_dict(),
                subject=_pointwise_subject(paper),
                cost_payload=cost_payload,
            )
        except Exception as exc:
            status = "failed"
            cost_payload = usage_cost_payload(
                model=pointwise_model,
                estimate=pointwise_estimate,
                rates=rates,
                response=None,
            )
            artifact = _call_artifact(
                phase=str(row["phase"]),
                bucket=str(row["bucket"]),
                model=pointwise_model,
                kind="pointwise",
                estimate=pointwise_estimate,
                status=status,
                subject=_pointwise_subject(paper),
                error=exc,
                cost_payload=cost_payload,
            )
            write_json_artifact(artifact_path, artifact)
            ledger.append(
                _ledger_entry(
                    phase=str(row["phase"]),
                    bucket=str(row["bucket"]),
                    model=pointwise_model,
                    kind="pointwise",
                    estimate=pointwise_estimate,
                    status=status,
                    artifact_path=artifact_path,
                    cost_payload=cost_payload,
                )
            )
            raise
        write_json_artifact(artifact_path, artifact)
        ledger.append(
            _ledger_entry(
                phase=str(row["phase"]),
                bucket=str(row["bucket"]),
                model=pointwise_model,
                kind="pointwise",
                estimate=pointwise_estimate,
                status=status,
                artifact_path=artifact_path,
                cost_payload=cost_payload,
            )
        )
        succeeded += 1
    return {
        "attempted": True,
        "status": "completed_pointwise_generation",
        "pointwise_calls_attempted": attempted,
        "pointwise_calls_succeeded": succeeded,
        "new_ledger_entries": 0,
    }


def _paper_from_row(row: Mapping[str, Any]) -> Paper:
    paper = row.get("paper")
    if not isinstance(paper, Paper):
        raise ValueError("pointwise row is missing paper object")
    return paper


def _pointwise_subject(paper: Paper) -> dict[str, Any]:
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "abstract": paper.abstract[:6000],
        "metadata": dict(paper.metadata),
    }


def _model_visible_leakage(
    buckets: Sequence[Any],
    *,
    model: str,
) -> dict[str, Any]:
    offending_rows = []
    for bucket in buckets:
        for paper in bucket.papers:
            payload = _pointwise_payload(model=model, paper=paper)
            visible = json.dumps(payload["messages"], sort_keys=True).lower()
            forbidden = [
                key for key in sorted(FORBIDDEN_MODEL_VISIBLE_KEYS) if key in visible
            ]
            if forbidden:
                offending_rows.append(
                    {
                        "bucket": bucket.name,
                        "paper_id": paper.paper_id,
                        "forbidden_keys": forbidden,
                    }
                )
    return {
        "present": bool(offending_rows),
        "offending_rows": offending_rows[:20],
        "offending_row_count": len(offending_rows),
        "forbidden_model_visible_keys": sorted(FORBIDDEN_MODEL_VISIBLE_KEYS),
        "policy": (
            "Future labels, good_paper labels, citation outcomes, matched titles "
            "or work IDs, and evaluation labels are forbidden in pointwise "
            "model-visible prompts and features."
        ),
    }


def _guardrails(
    *,
    buckets: Sequence[Any],
    mode: RunnerMode,
    confirm_fresh_holdout_pointwise_generation: bool,
    ledger_path: Path,
    artifact_dir: Path,
    max_usd: float,
    existing_spend: float,
    projected_spend: float,
    missing_count: int,
    leakage: Mapping[str, Any],
    provider: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "fresh_holdout_phase_buckets_present": bool(buckets),
        "pointwise_only_runner": True,
        "ledger_path_is_jsonl": ledger_path.suffix == ".jsonl",
        "ledger_under_artifact_dir": _is_relative_to(
            ledger_path.resolve(),
            artifact_dir.resolve(),
        ),
        "max_usd_within_campaign_remaining": (
            0.0 < max_usd <= DEFAULT_MAX_USD
        ),
        "existing_ledger_spend_lte_max": existing_spend <= max_usd,
        "projected_ledger_spend_lte_max": projected_spend <= max_usd,
        "projected_campaign_total_lte_paid_cap": (
            CURRENT_KNOWN_SPEND_USD + projected_spend <= DEFAULT_PAID_CAP_USD
        ),
        "model_visible_leakage_absent": leakage.get("present") is False,
        "execute_mode_confirmed_if_requested": (
            confirm_fresh_holdout_pointwise_generation
            if mode == "execute"
            else True
        ),
        "provider_model_available_if_execute": (
            provider.get("status") == "available"
            if mode == "execute" and missing_count > 0
            else True
        ),
    }
    blocking = sorted(key for key, value in checks.items() if value is False)
    return {
        "checks": checks,
        "blocking_reasons": blocking,
        "policy": {
            "allowed_call_kind": "pointwise",
            "pairwise_calls_forbidden": True,
            "scope": "fresh_holdout_pointwise_artifact_generation_only",
            "hard_paid_cap_usd": DEFAULT_PAID_CAP_USD,
            "additional_max_usd": max_usd,
        },
    }


def _review_artifacts(
    rows: Sequence[Mapping[str, Any]],
    *,
    leakage: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [row for row in rows if row.get("status") != "available"]
    blockers = []
    if missing:
        blockers.append("pointwise_artifacts_missing")
    if leakage.get("present") is not False:
        blockers.append("model_visible_leakage_present")
    return {
        "complete": not blockers,
        "blocking_reasons": blockers,
        "reviewed_artifact_count": len(rows) - len(missing),
        "missing_artifact_count": len(missing),
        "missing_examples": [
            {
                "bucket": row.get("bucket"),
                "paper_id": row.get("paper_id"),
                "artifact_path": row.get("artifact_path"),
            }
            for row in missing[:20]
        ],
    }


def _provider_model_availability(
    *,
    pointwise_model: str,
    timeout_seconds: float,
    required: bool,
    urlopen: Any,
) -> dict[str, Any]:
    if not required:
        return {
            "status": "not_checked_planning",
            "required_before_paid_calls": True,
            "requested_models": [pointwise_model],
            "api_key_env_present": bool(os.environ.get("SESTINA_LLM_API_KEY")),
            "base_url_env_present": bool(os.environ.get("SESTINA_LLM_BASE_URL")),
            "secrets_printed_or_stored": False,
        }
    try:
        result = check_model_availability(
            base_url=os.environ.get("SESTINA_LLM_BASE_URL") or "",
            api_key=os.environ.get("SESTINA_LLM_API_KEY") or "",
            models=[pointwise_model],
            timeout_seconds=timeout_seconds,
            urlopen=urlopen,
        )
    except ModelAvailabilityError as exc:
        return {
            "status": "unavailable",
            "required_before_paid_calls": True,
            "requested_models": [pointwise_model],
            "api_key_env_present": bool(os.environ.get("SESTINA_LLM_API_KEY")),
            "base_url_env_present": bool(os.environ.get("SESTINA_LLM_BASE_URL")),
            "secrets_printed_or_stored": False,
            "error": str(exc),
        }
    return {
        **result,
        "required_before_paid_calls": True,
        "api_key_env_present": bool(os.environ.get("SESTINA_LLM_API_KEY")),
        "base_url_env_present": bool(os.environ.get("SESTINA_LLM_BASE_URL")),
        "secrets_printed_or_stored": False,
    }


def _ensure_jsonl_ledger_exists(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("")


def _guard_campaign_total(*, existing_spend: float, next_cost_usd: float) -> None:
    projected_total = CURRENT_KNOWN_SPEND_USD + existing_spend + next_cost_usd
    if projected_total > DEFAULT_PAID_CAP_USD:
        raise RuntimeError(
            "projected campaign spend exceeds USD 100 hard cap before next call"
        )


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stdout_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_path": payload.get("artifact_path") or payload.get("output_path"),
        "artifact_type": payload["artifact_type"],
        "mode": payload["mode"],
        "paid_calls_made": payload["paid_calls_made"],
        "paid_spend_usd": payload["paid_spend_usd"],
        "pointwise_calls_made": payload["pointwise_calls_made"],
        "expected_pointwise_artifacts": payload["fresh_holdout"][
            "expected_pointwise_artifacts"
        ],
        "available_pointwise_artifacts": payload["fresh_holdout"][
            "available_pointwise_artifacts"
        ],
        "missing_pointwise_artifacts": payload["fresh_holdout"][
            "missing_pointwise_artifacts"
        ],
        "provider_model_availability_status": payload[
            "provider_model_availability"
        ]["status"],
        "status": payload["final_status"]["status"],
        "blocking_reasons": payload["final_status"]["blocking_reasons"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
