from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sestina.models import PairwiseComparison, Paper, RunInput, TargetSpec


def load_run_input(
    input_path: Path,
    *,
    top_k: int | None = None,
    top_alpha: float | None = None,
    comparisons_path: Path | None = None,
) -> RunInput:
    if input_path.suffix.lower() == ".jsonl":
        papers = _load_jsonl_papers(input_path)
        target = TargetSpec(top_k=top_k, top_alpha=top_alpha)
        comparisons = _load_comparisons(comparisons_path) if comparisons_path else []
        return RunInput(papers=papers, target=target, comparisons=comparisons)

    payload = json.loads(input_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("JSON input must be an object with papers and target")
    papers = [Paper.from_dict(item) for item in payload.get("papers", [])]
    target_payload = dict(payload.get("target") or {})
    if top_k is not None:
        target_payload["top_k"] = top_k
        target_payload.pop("top_alpha", None)
    if top_alpha is not None:
        target_payload["top_alpha"] = top_alpha
        target_payload.pop("top_k", None)
    comparisons_payload = payload.get("comparisons", [])
    comparisons = [PairwiseComparison.from_dict(item) for item in comparisons_payload]
    if comparisons_path:
        comparisons.extend(_load_comparisons(comparisons_path))
    return RunInput(
        papers=papers,
        target=TargetSpec.from_dict(target_payload),
        comparisons=comparisons,
        mode=payload.get("mode", "content_only"),
        config=dict(payload.get("config") or {}),
    )


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_jsonl_papers(path: Path) -> list[Paper]:
    papers: list[Paper] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
        papers.append(Paper.from_dict(payload))
    return papers


def _load_comparisons(path: Path | None) -> list[PairwiseComparison]:
    if path is None:
        return []
    if path.suffix.lower() == ".jsonl":
        return [
            PairwiseComparison.from_dict(json.loads(line))
            for line in path.read_text().splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("comparisons", [])
    return [PairwiseComparison.from_dict(item) for item in payload]

