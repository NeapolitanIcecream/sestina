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

from sestina.backtest_runner import BacktestRunner, PHASE_CHOICES  # noqa: E402
from sestina.diagnostics import write_json_artifact  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the guarded Sestina backtest experiment. The default mode is "
            "dry-run only and never calls an LLM."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "experiments" / "backtest_budget_config.json",
        help="backtest budget config JSON",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="frozen labeled dataset manifest; required with --confirm-paid",
    )
    parser.add_argument(
        "--phase",
        choices=PHASE_CHOICES,
        default="smoke",
        help="phase to estimate or execute",
    )
    parser.add_argument(
        "--max-usd",
        type=float,
        required=True,
        help="hard run cap in USD; must be <= 100 for paid runs",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="directory for estimates, call artifacts, and summaries",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        required=True,
        help="JSONL ledger path for paid calls",
    )
    parser.add_argument(
        "--confirm-paid",
        action="store_true",
        help="explicitly allow paid LLM calls after all safety checks pass",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)

    runner = BacktestRunner(
        config_path=args.config,
        phase=args.phase,
        max_usd=args.max_usd,
        artifact_dir=args.artifact_dir,
        ledger_path=args.ledger,
        manifest_path=args.manifest,
        confirm_paid=args.confirm_paid,
        seed=args.seed,
    )
    try:
        summary = runner.run()
    except Exception as exc:  # noqa: BLE001
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        error_path = args.artifact_dir / f"error-{args.phase}.json"
        write_json_artifact(
            error_path,
            {
                "artifact_type": "sestina-backtest-run-error",
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
