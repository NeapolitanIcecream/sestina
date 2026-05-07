#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sestina.diagnostics import write_json_artifact  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit pairwise labels against held-out citation labels."
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
    parser.add_argument("--phase", default="pilot")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="named artifact source as name=path",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    labels = _load_labels(args.manifest)
    sources = [_parse_source(value) for value in args.source]
    records = []
    for source_name, source_path in sources:
        records.extend(
            _load_pairwise_records(
                source_name=source_name,
                source_path=source_path,
                phase=args.phase,
                labels=labels,
            )
        )

    payload = {
        "artifact_type": "sestina-pairwise-noise-audit",
        "manifest_path": str(args.manifest),
        "phase": args.phase,
        "sources": [
            {"name": name, "artifact_dir": str(path)} for name, path in sources
        ],
        "records_total": len(records),
        "summary_by_source": _summaries(records, key="source_name"),
        "summary_by_source_and_purpose": _summaries(
            records,
            key=lambda row: f"{row['source_name']}::{row['purpose']}",
        ),
        "stratified": {
            "boundary_relevance": _stratified(records, "boundary_relevance"),
            "pair_information": _stratified(records, "pair_information"),
            "score_probability_gap": _stratified(records, "score_probability_gap"),
        },
        "evsi_boundary_vs_random": _evsi_boundary_vs_random(records),
        "limitations": _limitations(records),
    }
    write_json_artifact(args.output, payload)
    sys.stdout.write(json.dumps({**payload, "artifact_path": str(args.output)}, indent=2) + "\n")
    return 0


def _parse_source(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--source must use name=path")
    name, path = raw.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError("--source must use non-empty name=path")
    return name.strip(), Path(path)


def _load_labels(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    labels = {}
    for bucket in manifest.get("buckets", []):
        for paper in bucket.get("papers", []):
            paper_id = str(paper["paper_id"])
            paper_labels = paper.get("labels") or {}
            labels[paper_id] = {
                "bucket": str(bucket.get("name") or ""),
                "citation_count": _as_float(paper_labels.get("citation_count")),
                "good_paper": bool(paper_labels.get("good_paper", False)),
                "citation_rank": _as_float(paper_labels.get("citation_rank")),
            }
    return labels


def _load_pairwise_records(
    *,
    source_name: str,
    source_path: Path,
    phase: str,
    labels: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for path in sorted((source_path / phase).glob("*/calls/*pairwise*.json")):
        artifact = json.loads(path.read_text())
        if artifact.get("status") not in {"ok", "reused"}:
            continue
        kind = str(artifact.get("kind") or "")
        if source_name == "historical_original" and kind != "pairwise_active":
            continue
        if source_name == "historical_random" and kind != "pairwise_random":
            continue
        subject = artifact.get("subject") or {}
        left_id = str(subject.get("left_id") or "")
        right_id = str(subject.get("right_id") or "")
        if not left_id or not right_id:
            continue
        response = artifact.get("response") or {}
        response_metadata = response.get("metadata") or {}
        winner = _normalize_winner(response.get("winner"), left_id, right_id)
        left_labels = labels.get(left_id)
        right_labels = labels.get(right_id)
        scheduled_pair = artifact.get("scheduled_pair") or {}
        diagnostics = scheduled_pair.get("diagnostics") or {}
        purpose = str(
            scheduled_pair.get("purpose")
            or diagnostics.get("selected_cctd_gf_purpose")
            or diagnostics.get("source_evsi_purpose")
            or response_metadata.get("scheduled_pair_purpose")
            or artifact.get("kind")
            or "unknown"
        )
        records.append(
            {
                "source_name": source_name,
                "artifact_path": str(path),
                "bucket": str(artifact.get("bucket") or ""),
                "kind": kind,
                "purpose": purpose,
                "left_id": left_id,
                "right_id": right_id,
                "winner": winner,
                "left_citation_count": (
                    left_labels.get("citation_count") if left_labels else None
                ),
                "right_citation_count": (
                    right_labels.get("citation_count") if right_labels else None
                ),
                "left_good_paper": left_labels.get("good_paper") if left_labels else None,
                "right_good_paper": (
                    right_labels.get("good_paper") if right_labels else None
                ),
                "has_citation_labels": left_labels is not None and right_labels is not None,
                "has_scheduler_diagnostics": bool(diagnostics),
                "boundary_relevance": _optional_float(
                    diagnostics.get("boundary_relevance")
                ),
                "pair_information": _optional_float(
                    diagnostics.get("pair_information")
                ),
                "score_probability_gap": _optional_float(
                    diagnostics.get("score_probability_gap")
                ),
            }
        )
    return records


def _normalize_winner(value: Any, left_id: str, right_id: str) -> str:
    winner = str(value or "uncertain").strip().lower()
    if winner in {"left", "right", "tie", "uncertain"}:
        return winner
    if winner == left_id.lower():
        return "left"
    if winner == right_id.lower():
        return "right"
    return "uncertain"


def _summaries(
    records: list[dict[str, Any]],
    *,
    key: str | Any,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group_key = str(record[key] if isinstance(key, str) else key(record))
        grouped[group_key].append(record)
    return {
        group_key: _summary(rows)
        for group_key, rows in sorted(grouped.items())
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    citation_eligible = 0
    higher_citation_wins = 0
    positive_eligible = 0
    positive_wins = 0
    ties_or_uncertain = 0
    missing_citations = 0
    missing_diagnostics = 0
    for record in records:
        if not record["has_citation_labels"]:
            missing_citations += 1
            continue
        winner = record["winner"]
        if winner in {"tie", "uncertain"}:
            ties_or_uncertain += 1
        left_cites = record["left_citation_count"]
        right_cites = record["right_citation_count"]
        if left_cites is not None and right_cites is not None and left_cites != right_cites:
            citation_eligible += 1
            if (winner == "left" and left_cites > right_cites) or (
                winner == "right" and right_cites > left_cites
            ):
                higher_citation_wins += 1
        left_positive = record["left_good_paper"]
        right_positive = record["right_good_paper"]
        if left_positive is not None and right_positive is not None and left_positive != right_positive:
            positive_eligible += 1
            if (winner == "left" and left_positive) or (
                winner == "right" and right_positive
            ):
                positive_wins += 1
        if not record["has_scheduler_diagnostics"]:
            missing_diagnostics += 1
    return {
        "pairs_total": len(records),
        "higher_citation_comparable_pairs": citation_eligible,
        "judge_winner_higher_citation_total": higher_citation_wins,
        "judge_winner_higher_citation_rate": _rate(
            higher_citation_wins,
            citation_eligible,
        ),
        "future_positive_vs_nonpositive_pairs": positive_eligible,
        "future_positive_wins_total": positive_wins,
        "future_positive_win_rate": _rate(positive_wins, positive_eligible),
        "ties_or_uncertain_total": ties_or_uncertain,
        "missing_citation_label_pairs": missing_citations,
        "missing_scheduler_diagnostics_pairs": missing_diagnostics,
    }


def _stratified(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    rows = [record for record in records if record.get(field) is not None]
    if not rows:
        return {}
    values = sorted(float(record[field]) for record in rows)
    low_cut = values[len(values) // 3]
    high_cut = values[(2 * len(values)) // 3]
    grouped = {"low": [], "mid": [], "high": []}
    for record in rows:
        value = float(record[field])
        if value <= low_cut:
            grouped["low"].append(record)
        elif value <= high_cut:
            grouped["mid"].append(record)
        else:
            grouped["high"].append(record)
    return {
        name: {
            **_summary(group),
            "field": field,
            "lower_bound": min((float(record[field]) for record in group), default=None),
            "upper_bound": max((float(record[field]) for record in group), default=None),
        }
        for name, group in grouped.items()
    }


def _evsi_boundary_vs_random(records: list[dict[str, Any]]) -> dict[str, Any]:
    evsi_boundary = [
        record
        for record in records
        if record["source_name"] in {"evsi_followup", "sequential_evsi"}
        and record["purpose"] in {"evsi_boundary_duel", "calibration_discovery"}
    ]
    random_like = [
        record
        for record in records
        if record["source_name"] in {"historical_random", "exact_pool_random"}
        or record["purpose"] in {"pairwise_random", "exact_pool_random"}
    ]
    return {
        "evsi_boundary": _summary(evsi_boundary),
        "random_or_exact_pool": _summary(random_like),
        "citation_alignment_delta_evsi_minus_random": round(
            _summary(evsi_boundary)["judge_winner_higher_citation_rate"]
            - _summary(random_like)["judge_winner_higher_citation_rate"],
            8,
        ),
        "future_positive_win_delta_evsi_minus_random": round(
            _summary(evsi_boundary)["future_positive_win_rate"]
            - _summary(random_like)["future_positive_win_rate"],
            8,
        ),
    }


def _limitations(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    missing_diagnostics = sum(1 for record in records if not record["has_scheduler_diagnostics"])
    missing_info = sum(1 for record in records if record.get("pair_information") is None)
    return {
        "missing_scheduler_diagnostics_pairs": missing_diagnostics,
        "missing_scheduler_diagnostics_rate": _rate(missing_diagnostics, total),
        "missing_pair_information_pairs": missing_info,
        "missing_pair_information_rate": _rate(missing_info, total),
        "interpretation": (
            "Citation-alignment rates are available for completed arXiv pairwise "
            "artifacts. Boundary/information/gap stratification is best-effort "
            "because older historical pairwise artifacts do not store scheduled_pair "
            "diagnostics and exact-pool random artifacts predate CCTD-GF information "
            "fields."
        ),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
