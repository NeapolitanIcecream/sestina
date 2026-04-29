from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

DiagnosticLevel = Literal["info", "warning", "error"]


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


@dataclass(slots=True)
class DiagnosticEvent:
    step: str
    code: str
    level: DiagnosticLevel
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "code": self.code,
            "level": self.level,
            "message": self.message,
            "data": _bounded_payload(self.data),
        }


@dataclass(slots=True)
class DiagnosticRecorder:
    events: list[DiagnosticEvent] = field(default_factory=list)

    def record(
        self,
        *,
        step: str,
        code: str,
        level: DiagnosticLevel = "info",
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            DiagnosticEvent(
                step=step,
                code=code,
                level=level,
                message=message,
                data=data or {},
            )
        )

    def extend(self, events: list[DiagnosticEvent]) -> None:
        self.events.extend(events)

    def to_dict(self) -> dict[str, Any]:
        counts = {"info": 0, "warning": 0, "error": 0}
        for event in self.events:
            counts[event.level] += 1
        return {
            "counts": counts,
            "events": [event.to_dict() for event in self.events],
        }


def write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_error_artifact(
    debug_dir: Path,
    *,
    error: BaseException,
    diagnostics: DiagnosticRecorder | None = None,
) -> Path:
    payload = {
        "artifact_type": "sestina-error-debug",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "diagnostics": (diagnostics or DiagnosticRecorder()).to_dict(),
    }
    path = debug_dir / "sestina-error-debug.json"
    write_json_artifact(path, payload)
    return path


def _bounded_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Keep diagnostics compact and avoid leaking full paper text by default."""
    bounded: dict[str, Any] = {}
    for key, value in data.items():
        if key in {"abstract", "full_text", "text", "prompt", "api_key", "token"}:
            bounded[key] = "[redacted]"
        elif isinstance(value, str) and len(value) > 220:
            bounded[key] = value[:217] + "..."
        elif isinstance(value, list) and len(value) > 40:
            bounded[key] = value[:40] + [{"truncated_items": len(value) - 40}]
        else:
            bounded[key] = value
    return bounded

