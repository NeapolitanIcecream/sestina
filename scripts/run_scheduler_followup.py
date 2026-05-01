#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sestina.diagnostics import write_json_artifact  # noqa: E402
from sestina.scheduler_followup import SchedulerOnlyRunner  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a guarded scheduler-only follow-up. Pointwise assessments are "
            "loaded from historical artifacts; only novel active pairwise "
            "comparisons may call the configured LLM when --confirm-paid is set."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO_ROOT
            / "experiments"
            / "arxiv_historical_pilot_budget_config.json"
        ),
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
    parser.add_argument("--phase", default="pilot")
    parser.add_argument(
        "--max-usd",
        type=float,
        required=True,
        help="scheduler-only hard cap in USD; paid runs require <= 2.00",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="separate directory for follow-up estimates, calls, and summaries",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        required=True,
        help="separate JSONL ledger path for new paid pairwise calls",
    )
    parser.add_argument(
        "--confirm-paid",
        action="store_true",
        help="explicitly allow novel pairwise LLM calls after all guards pass",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--scheduler-kind",
        choices=[
            "quota",
            "evsi",
            "exact_pool_random",
            "sequential_evsi",
            "cctd_gf",
        ],
        default="quota",
        help="active pair scheduler to use for the follow-up",
    )
    parser.add_argument(
        "--aggregation-mode",
        choices=["score", "posterior_topk", "both"],
        default="score",
        help="which aggregation outputs to score in bucket results",
    )
    args = parser.parse_args(argv)

    runner = SchedulerOnlyRunner(
        config_path=args.config,
        manifest_path=args.manifest,
        source_artifact_dir=args.source_artifact_dir,
        artifact_dir=args.artifact_dir,
        ledger_path=args.ledger,
        phase=args.phase,
        max_usd=args.max_usd,
        confirm_paid=args.confirm_paid,
        seed=args.seed,
        scheduler_kind=args.scheduler_kind,
        aggregation_mode=args.aggregation_mode,
    )
    try:
        summary = runner.run()
    except Exception as exc:  # noqa: BLE001
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        error_path = args.artifact_dir / f"error-{args.phase}.json"
        write_json_artifact(
            error_path,
            {
                "artifact_type": "sestina-scheduler-only-followup-error",
                "phase": args.phase,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "paid_run_requested": args.confirm_paid,
            },
        )
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        sys.stderr.write(f"error_artifact={error_path}\n")
        return 2

    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
