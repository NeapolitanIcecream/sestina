from __future__ import annotations

import math
from dataclasses import dataclass

from sestina.diagnostics import DiagnosticRecorder
from sestina.models import TargetSpec


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    n: int
    k: int
    requested: TargetSpec
    mode: str

    def to_dict(self) -> dict[str, int | float | str | None]:
        return {
            "n": self.n,
            "k": self.k,
            "top_k": self.requested.top_k,
            "top_alpha": self.requested.top_alpha,
            "mode": self.mode,
        }


def resolve_target(
    n: int,
    target: TargetSpec,
    *,
    diagnostics: DiagnosticRecorder | None = None,
) -> ResolvedTarget:
    recorder = diagnostics or DiagnosticRecorder()
    if n < 0:
        raise ValueError("n must be non-negative")
    has_k = target.top_k is not None
    has_alpha = target.top_alpha is not None
    if has_k == has_alpha:
        recorder.record(
            step="target",
            code="target_invalid",
            level="error",
            message="exactly one of top_k or top_alpha is required",
            data={"n": n, "top_k": target.top_k, "top_alpha": target.top_alpha},
        )
        raise ValueError("exactly one of top_k or top_alpha is required")

    if has_k:
        k = int(target.top_k or 0)
        if k < 1:
            raise ValueError("top_k must be at least 1")
        if k > n:
            raise ValueError("top_k cannot exceed number of papers")
        mode = "top_k"
    else:
        alpha = float(target.top_alpha or 0.0)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("top_alpha must be in the interval (0, 1]")
        k = 0 if n == 0 else max(1, math.ceil(alpha * n))
        mode = "top_alpha"

    resolved = ResolvedTarget(n=n, k=k, requested=target, mode=mode)
    recorder.record(
        step="target",
        code="target_resolved",
        message="resolved discovery target",
        data=resolved.to_dict(),
    )
    return resolved

