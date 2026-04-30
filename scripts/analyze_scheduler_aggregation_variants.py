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

from sestina.scheduler_followup import (  # noqa: E402
    DEFAULT_STRENGTH_SWEEP,
    analyze_aggregation_variants,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze historical scheduler aggregation variants using cached "
            "pilot pointwise and pairwise artifacts. This never calls an LLM."
        )
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
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "backtest-arxiv-pilot-live"
            / "aggregation-variant-analysis.json"
        ),
    )
    parser.add_argument("--phase", default="pilot")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--strength",
        action="append",
        type=float,
        help="pairwise strength value; may be repeated",
    )
    args = parser.parse_args(argv)

    strengths = tuple(args.strength) if args.strength else DEFAULT_STRENGTH_SWEEP
    result = analyze_aggregation_variants(
        manifest_path=args.manifest,
        source_artifact_dir=args.source_artifact_dir,
        output_path=args.output,
        phase=args.phase,
        seed=args.seed,
        strengths=strengths,
    )
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
