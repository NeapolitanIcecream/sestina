from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from sestina.diagnostics import DiagnosticRecorder, write_error_artifact, write_json_artifact
from sestina.io import dump_json, load_run_input
from sestina.pipeline import PipelineConfig, run_pipeline


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sestina",
        description="Run Sestina pointwise-first top-K paper discovery.",
    )
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="run offline JSON/JSONL ranking")
    run_parser.add_argument("input", type=Path, help="JSON or JSONL paper input")
    run_parser.add_argument("--comparisons", type=Path, help="optional comparisons JSON/JSONL")
    target = run_parser.add_mutually_exclusive_group()
    target.add_argument("--top-k", type=int)
    target.add_argument("--top-alpha", type=float)
    run_parser.add_argument("--mode", choices=["content_only", "metadata_aware"])
    run_parser.add_argument("--candidate-size", type=int)
    run_parser.add_argument("--pairwise-budget", type=int)
    run_parser.add_argument("--posterior-samples", type=int)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--output", type=Path, help="write JSON output to this path")
    run_parser.add_argument("--report", type=Path, help="write a Markdown summary report")
    run_parser.add_argument("--debug-dir", type=Path, help="write diagnostics/debug artifacts")
    return parser


def _run(args: argparse.Namespace) -> int:
    diagnostics = DiagnosticRecorder()
    try:
        run_input = load_run_input(
            args.input,
            top_k=args.top_k,
            top_alpha=args.top_alpha,
            comparisons_path=args.comparisons,
        )
        config = dict(run_input.config)
        if args.mode:
            config["mode"] = args.mode
        elif run_input.mode:
            config["mode"] = run_input.mode
        for attr in ("candidate_size", "pairwise_budget", "posterior_samples", "seed"):
            value = getattr(args, attr)
            if value is not None:
                config[attr] = value
        result = run_pipeline(
            run_input.papers,
            run_input.target,
            comparisons=run_input.comparisons,
            config=PipelineConfig.from_dict(config),
            diagnostics=diagnostics,
        )
        payload = result.to_dict()
        if args.output:
            dump_json(args.output, payload)
        else:
            sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(render_markdown_report(payload))
        if args.debug_dir:
            write_json_artifact(args.debug_dir / "sestina-run-diagnostics.json", payload["diagnostics"])
        return 0
    except Exception as exc:  # noqa: BLE001
        diagnostics.record(
            step="cli",
            code="cli_run_failed",
            level="error",
            message="sestina run failed",
            data={"error_type": type(exc).__name__, "error_message": str(exc)},
        )
        if args.debug_dir:
            write_error_artifact(args.debug_dir, error=exc, diagnostics=diagnostics)
        sys.stderr.write(f"sestina: {type(exc).__name__}: {exc}\n")
        return 1


def render_markdown_report(payload: dict[str, object]) -> str:
    target = payload["target"]  # type: ignore[index]
    recommendations = payload["recommendations"]  # type: ignore[index]
    pairwise_schedule = payload["pairwise_schedule"]  # type: ignore[index]
    caveats = payload.get("caveats", [])
    lines = [
        "# Sestina Report",
        "",
        f"- Papers: {target['n']}",  # type: ignore[index]
        f"- Target K: {target['k']}",  # type: ignore[index]
        f"- Pairwise budget: {pairwise_schedule['budget']['budget']}",  # type: ignore[index]
        "",
        "## Recommended Good Papers",
        "",
    ]
    for item in recommendations["recommended_good_papers"]:  # type: ignore[index]
        lines.append(
            "- {title} (`{paper_id}`): tier `{tier}`, top-K probability {top_k_probability:.3f}".format(
                **item
            )
        )
    if recommendations["near_misses"]:  # type: ignore[index]
        lines.extend(["", "## Near Misses", ""])
        for item in recommendations["near_misses"]:  # type: ignore[index]
            lines.append(
                "- {title} (`{paper_id}`): tier `{tier}`, top-K probability {top_k_probability:.3f}".format(
                    **item
                )
            )
    if caveats:
        lines.extend(["", "## Caveats", ""])
        for caveat in caveats:
            lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
