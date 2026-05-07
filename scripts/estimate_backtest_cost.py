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

from sestina.backtest_budget import (  # noqa: E402
    BudgetExceededError,
    estimate_from_config,
    load_config,
    render_text_summary,
    write_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate dry-run LLM token and dollar budget for the Sestina "
            "top-K backtest experiment. This script never calls an LLM."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "experiments" / "backtest_budget_config.json",
        help="experiment budget config JSON",
    )
    parser.add_argument(
        "--max-usd",
        type=float,
        help="hard budget cap override; defaults to budget_cap_usd in config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON artifact path for the full estimate/ledger",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="stdout format",
    )
    args = parser.parse_args(argv)

    try:
        report = estimate_from_config(
            load_config(args.config),
            max_usd=args.max_usd,
            validate_budget=True,
        )
    except BudgetExceededError as exc:
        if args.output:
            write_report(args.output, exc.report)
        sys.stderr.write(f"{exc}\n")
        if args.format == "json":
            sys.stdout.write(json.dumps(exc.report, indent=2, sort_keys=True) + "\n")
        else:
            sys.stdout.write(render_text_summary(exc.report) + "\n")
        return 2

    if args.output:
        write_report(args.output, report)
    if args.format == "json":
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(render_text_summary(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
