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

from sestina.diagnostics import write_json_artifact  # noqa: E402
from sestina.experiment_protocol import (  # noqa: E402
    build_next_experiment_protocol,
    validate_next_experiment_protocol,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "backtest-arxiv-next-experiment-protocol"
    / "next-experiment-protocol.json"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an offline next-experiment protocol artifact. This makes no "
            "paid LLM calls and does not edit historical ledgers or call artifacts."
        )
    )
    parser.add_argument("--no-paid-gate-artifact", type=Path, default=None)
    parser.add_argument("--fresh-holdout-request", type=Path, default=None)
    parser.add_argument(
        "--priority-direction",
        default="confidence_interval_top_k_partition_elimination",
        choices=[
            "confidence_interval_top_k_partition_elimination",
            "no_paid_replay_gate_randomized_coverage_floor",
        ],
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    gate_artifact = (
        _read_json(args.no_paid_gate_artifact)
        if args.no_paid_gate_artifact is not None
        else None
    )
    fresh_holdout_request = (
        _read_json(args.fresh_holdout_request)
        if args.fresh_holdout_request is not None
        else None
    )
    payload = build_next_experiment_protocol(
        no_paid_gate_artifact=gate_artifact,
        priority_direction=args.priority_direction,
        fresh_holdout_request=fresh_holdout_request,
    )
    validate_next_experiment_protocol(payload)
    write_json_artifact(args.output, payload)
    sys.stdout.write(json.dumps(_stdout_summary(payload, args.output), indent=2))
    sys.stdout.write("\n")
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _stdout_summary(payload: dict[str, Any], output_path: Path) -> dict[str, Any]:
    gate = payload["future_experiment_gate"]["no_paid_gate"]
    holdout = payload["fresh_holdout_validation_protocol"]
    return {
        "artifact_path": str(output_path),
        "artifact_type": payload["artifact_type"],
        "paid_calls_made": payload["paid_calls_made"],
        "paid_spend_usd": payload["paid_spend_usd"],
        "pointwise_calls_made": payload["pointwise_calls_made"],
        "campaign_status": payload["current_result_boundary"][
            "campaign_status"
        ],
        "current_best_result": payload["current_result_boundary"][
            "current_best_result"
        ],
        "no_paid_gate_passed": gate["passed"],
        "no_paid_gate_blocking_reasons": gate["blocking_reasons"],
        "fresh_holdout_allowed_to_begin": holdout["allowed_to_begin"],
        "fresh_holdout_blocking_reasons": holdout["blocking_reasons"],
        "paid_label_purchase_authorized_by_this_protocol": holdout[
            "paid_label_purchase_authorized_by_this_protocol"
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())

