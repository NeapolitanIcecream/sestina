from __future__ import annotations

import math
from dataclasses import dataclass, field

from sestina.diagnostics import DiagnosticRecorder
from sestina.models import Paper, SelectionMode


@dataclass(frozen=True, slots=True)
class CandidateSelectionConfig:
    candidate_size: int | None = None
    mode: SelectionMode = "content_only"
    exploit_fraction: float = 0.5
    boundary_fraction: float = 0.35


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    candidate_ids: list[str]
    groups: dict[str, list[str]]
    scores: dict[str, float]
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_ids": list(self.candidate_ids),
            "groups": {key: list(value) for key, value in self.groups.items()},
            "scores": dict(self.scores),
            "diagnostics": dict(self.diagnostics),
        }


def default_candidate_size(n: int, k: int) -> int:
    return min(n, math.ceil((3 * k) + math.sqrt(n)))


def select_candidates(
    papers: list[Paper],
    *,
    k: int,
    config: CandidateSelectionConfig | None = None,
    diagnostics: DiagnosticRecorder | None = None,
) -> CandidateSelection:
    cfg = config or CandidateSelectionConfig()
    recorder = diagnostics or DiagnosticRecorder()
    n = len(papers)
    m_default = default_candidate_size(n, k)
    candidate_size_requested = (
        int(cfg.candidate_size) if cfg.candidate_size is not None else None
    )
    candidate_size_source = (
        "default" if candidate_size_requested is None else "override"
    )
    minimum_candidate_size = max(1, k)
    if (
        candidate_size_requested is not None
        and candidate_size_requested < minimum_candidate_size
    ):
        _record_invalid_candidate_size(
            recorder,
            n=n,
            k=k,
            candidate_size_requested=candidate_size_requested,
            candidate_size_default=m_default,
            minimum_candidate_size=minimum_candidate_size,
        )
        raise ValueError(
            "candidate_size must be at least resolved top K "
            f"(minimum {minimum_candidate_size}; got {candidate_size_requested})"
        )

    if n == 0 or k == 0:
        selection = CandidateSelection(
            candidate_ids=[],
            groups={"exploit": [], "boundary": [], "explore": []},
            scores={},
            diagnostics={
                "n": n,
                "k": k,
                "candidate_size_default": m_default,
                "candidate_size_requested": candidate_size_requested,
                "candidate_size_source": candidate_size_source,
                "candidate_size": 0,
                "mode": cfg.mode,
            },
        )
        recorder.record(
            step="candidate_selection",
            code="candidate_selection_empty",
            message="no candidates selected from empty input or zero target",
            data=selection.diagnostics,
        )
        return selection

    requested_or_default = (
        candidate_size_requested if candidate_size_requested is not None else m_default
    )
    m = min(n, requested_or_default)
    if m < k:
        _record_invalid_candidate_size(
            recorder,
            n=n,
            k=k,
            candidate_size_requested=candidate_size_requested,
            candidate_size_default=m_default,
            minimum_candidate_size=k,
        )
        raise ValueError(
            "candidate_size must be at least resolved top K "
            f"(minimum {k}; got {m})"
        )

    by_quality = sorted(
        papers,
        key=lambda paper: (
            paper.pointwise.good_probability,
            -paper.pointwise.uncertainty,
            paper.paper_id,
        ),
        reverse=True,
    )
    boundary_probability = by_quality[min(k - 1, len(by_quality) - 1)].pointwise.good_probability
    by_boundary = sorted(
        papers,
        key=lambda paper: (
            abs(paper.pointwise.good_probability - boundary_probability),
            -paper.pointwise.uncertainty,
            paper.paper_id,
        ),
    )
    by_uncertainty = sorted(
        papers,
        key=lambda paper: (
            paper.pointwise.uncertainty,
            paper.pointwise.good_probability,
            paper.paper_id,
        ),
        reverse=True,
    )

    exploit_target = max(k, math.ceil(m * cfg.exploit_fraction))
    boundary_target = max(0, math.ceil(m * cfg.boundary_fraction))
    exploit = _take_unique(by_quality, limit=exploit_target)
    boundary = _take_unique(by_boundary, limit=boundary_target)
    explore = _round_robin_by_metadata(
        by_uncertainty,
        existing=set(exploit) | set(boundary),
        limit=max(0, m - len(set(exploit) | set(boundary))),
    )

    ordered = _ordered_union(
        exploit,
        boundary,
        explore,
        [paper.paper_id for paper in by_quality],
        limit=m,
    )
    groups = {
        "exploit": [paper_id for paper_id in exploit if paper_id in ordered],
        "boundary": [paper_id for paper_id in boundary if paper_id in ordered],
        "explore": [paper_id for paper_id in explore if paper_id in ordered],
    }
    scores = {
        paper.paper_id: _candidate_score(paper, boundary_probability)
        for paper in papers
        if paper.paper_id in ordered
    }
    payload = {
        "n": n,
        "k": k,
        "candidate_size_default": m_default,
        "candidate_size_requested": candidate_size_requested,
        "candidate_size_source": candidate_size_source,
        "candidate_size": m,
        "mode": cfg.mode,
        "group_counts": {key: len(value) for key, value in groups.items()},
        "boundary_probability": round(boundary_probability, 6),
    }
    recorder.record(
        step="candidate_selection",
        code="candidate_selection_completed",
        message="selected exploit, boundary, and exploration candidates",
        data=payload,
    )
    return CandidateSelection(
        candidate_ids=ordered,
        groups=groups,
        scores=scores,
        diagnostics=payload,
    )


def _take_unique(papers: list[Paper], *, limit: int) -> list[str]:
    seen: set[str] = set()
    selected: list[str] = []
    for paper in papers:
        if paper.paper_id in seen:
            continue
        seen.add(paper.paper_id)
        selected.append(paper.paper_id)
        if len(selected) >= limit:
            break
    return selected


def _round_robin_by_metadata(
    papers: list[Paper],
    *,
    existing: set[str],
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []
    buckets: dict[str, list[Paper]] = {}
    for paper in papers:
        if paper.paper_id in existing:
            continue
        key = _bucket_key(paper)
        buckets.setdefault(key, []).append(paper)
    selected: list[str] = []
    keys = sorted(buckets)
    while keys and len(selected) < limit:
        next_keys: list[str] = []
        for key in keys:
            bucket = buckets[key]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop(0).paper_id)
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected


def _bucket_key(paper: Paper) -> str:
    for key in ("topic", "venue", "source", "field", "category"):
        value = paper.metadata.get(key)
        if value:
            return f"{key}:{value}"
    return "unknown"


def _ordered_union(*groups: list[str], limit: int) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for paper_id in group:
            if paper_id in seen:
                continue
            seen.add(paper_id)
            selected.append(paper_id)
            if len(selected) >= limit:
                return selected
    return selected


def _candidate_score(paper: Paper, boundary_probability: float) -> float:
    closeness = 1.0 - abs(paper.pointwise.good_probability - boundary_probability)
    return round(
        (0.55 * paper.pointwise.good_probability)
        + (0.25 * paper.pointwise.uncertainty)
        + (0.20 * closeness),
        6,
    )


def _record_invalid_candidate_size(
    recorder: DiagnosticRecorder,
    *,
    n: int,
    k: int,
    candidate_size_requested: int | None,
    candidate_size_default: int,
    minimum_candidate_size: int,
) -> None:
    recorder.record(
        step="candidate_selection",
        code="candidate_size_invalid",
        level="error",
        message="candidate_size must be at least resolved top K",
        data={
            "n": n,
            "k": k,
            "candidate_size_default": candidate_size_default,
            "candidate_size_requested": candidate_size_requested,
            "candidate_size_source": "override",
            "minimum_candidate_size": minimum_candidate_size,
        },
    )
