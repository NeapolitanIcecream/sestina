#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sestina.active_arm_gate import (  # noqa: E402
    CURRENT_KNOWN_SPEND_USD,
    DEFAULT_PAID_CAP_USD,
    build_active_arm_gate,
    validate_active_arm_gate_artifact_schema,
)
from sestina.diagnostics import write_json_artifact  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a no-paid active-arm replay/simulation artifact against "
            "the Sestina active-arm paid-followup gate."
        )
    )
    parser.add_argument(
        "--active-artifact",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-ci-partition-gate"
            / "ci-partition-gate-analysis.json"
        ),
        help="No-paid active-arm comparison artifact to evaluate.",
    )
    parser.add_argument(
        "--random-variance-artifact",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-full-random-variance-completion"
            / "full-random-variance-completion.json"
        ),
        help="Completed 20-seed full-random variance reference artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-active-arm-gate-harness"
            / "active-arm-gate-smoke.json"
        ),
    )
    parser.add_argument("--active-arm", default=None)
    parser.add_argument("--random-control-arm", default=None)
    parser.add_argument(
        "--paid-followup-estimate-usd",
        type=float,
        default=0.0,
        help="Estimated additional paid spend for a proposed follow-up, if known.",
    )
    parser.add_argument(
        "--known-spend-usd",
        type=float,
        default=CURRENT_KNOWN_SPEND_USD,
    )
    parser.add_argument("--paid-cap-usd", type=float, default=DEFAULT_PAID_CAP_USD)
    args = parser.parse_args(argv)

    active_artifact = _read_json(args.active_artifact)
    random_variance_artifact = _read_json(args.random_variance_artifact)
    payload = build_active_arm_gate(
        active_artifact,
        random_variance_artifact,
        active_artifact_path=str(args.active_artifact),
        random_variance_artifact_path=str(args.random_variance_artifact),
        active_arm_name=args.active_arm,
        candidate_random_control_baseline=args.random_control_arm,
        paid_followup_estimate_usd=args.paid_followup_estimate_usd,
        known_spend_usd=args.known_spend_usd,
        paid_cap_usd=args.paid_cap_usd,
    )
    validate_active_arm_gate_artifact_schema(payload)
    write_json_artifact(args.output, payload)
    stdout = _stdout_summary(payload, output_path=args.output)
    sys.stdout.write(json.dumps(stdout, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _stdout_summary(payload: dict[str, Any], *, output_path: Path) -> dict[str, Any]:
    return {
        "artifact_path": str(output_path),
        "artifact_type": payload["artifact_type"],
        "active_arm_name": payload["active_arm_name"],
        "candidate_random_control_baseline": payload[
            "candidate_random_control_baseline"
        ],
        "paid_calls_made": payload["paid_calls_made"],
        "paid_spend_usd": payload["paid_spend_usd"],
        "paid_followup_allowed": payload["paid_followup_allowed"],
        "gate_verdict": payload["gate_verdict"],
        "paired_active_minus_random_deltas": payload[
            "paired_active_minus_random_deltas"
        ]["metric_deltas"],
        "spend_estimate": payload["spend_estimate"],
        "recommended_next_action": payload["recommended_next_action"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
