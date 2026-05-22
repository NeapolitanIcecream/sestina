from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sestina.candidates import default_candidate_size
from sestina.diagnostics import DiagnosticRecorder
from sestina.scheduler import default_pairwise_budget

POINTWISE_STRATEGIES = frozenset(
    {"pointwise_only", "pointwise_random_pairwise", "sestina_active_pairwise"}
)
PAIRWISE_STRATEGIES = frozenset(
    {"pointwise_random_pairwise", "sestina_active_pairwise"}
)
ZERO_LLM_STRATEGIES = frozenset({"random", "semantic_baseline"})
KNOWN_STRATEGIES = POINTWISE_STRATEGIES | PAIRWISE_STRATEGIES | ZERO_LLM_STRATEGIES

DEFAULT_MODEL_NAME_POLICY = {
    "require_provider_prefix": False,
    "availability_check_required_before_paid_run": True,
}
DEFAULT_PAIRWISE_BUDGET_ABLATION = {
    "enabled": False,
    "mode": "prefix_reuse",
    "reuse_scheduled_pairwise_prefixes": True,
    "points": ("0", "K", "K + sqrt(n)", "B_pair"),
    "metrics_artifact": "pairwise_budget_ablation_metrics.csv",
}


@dataclass(frozen=True, slots=True)
class BudgetExceededError(RuntimeError):
    cap_usd: float
    estimated_usd: float
    report: dict[str, Any]

    def __str__(self) -> str:
        return (
            f"estimated backtest cost ${self.estimated_usd:.4f} exceeds "
            f"budget cap ${self.cap_usd:.4f}"
        )


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def estimate_from_config(
    config: dict[str, Any],
    *,
    max_usd: float | None = None,
    validate_budget: bool = True,
) -> dict[str, Any]:
    recorder = DiagnosticRecorder()
    report = _estimate_from_config(config, max_usd=max_usd, diagnostics=recorder)
    cap = report["budget_cap_usd"]
    total = report["totals"]["cost_usd"]
    if cap is not None and total > cap:
        recorder.record(
            step="backtest_budget",
            code="backtest_budget_exceeded",
            level="error",
            message="estimated backtest cost exceeds configured budget cap",
            data={
                "budget_cap_usd": cap,
                "estimated_cost_usd": total,
                "overage_usd": round(total - cap, 6),
            },
        )
        report["diagnostics"] = recorder.to_dict()
        report["budget_status"] = "exceeds_cap"
        if validate_budget:
            raise BudgetExceededError(
                cap_usd=float(cap),
                estimated_usd=float(total),
                report=report,
            )
    else:
        recorder.record(
            step="backtest_budget",
            code="backtest_budget_within_cap",
            message="estimated backtest cost is within configured budget cap",
            data={"budget_cap_usd": cap, "estimated_cost_usd": total},
        )
        report["diagnostics"] = recorder.to_dict()
        report["budget_status"] = "within_cap"
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def render_text_summary(report: dict[str, Any]) -> str:
    totals = report["totals"]
    lines = [
        "Sestina backtest budget estimate",
        f"- Dry run only: {report['dry_run']}",
        f"- Budget cap: {_money_or_none(report['budget_cap_usd'])}",
        f"- Estimated cost: ${totals['cost_usd']:.4f}",
        (
            "- Calls: "
            f"{totals['pointwise_calls']} pointwise, "
            f"{totals['pairwise_calls']} pairwise, "
            f"{totals['audit_pairwise_calls']} audit pairwise"
        ),
        (
            "- Tokens: "
            f"{totals['input_tokens']} input, {totals['output_tokens']} output"
        ),
        f"- Budget status: {report['budget_status']}",
    ]
    ablation = report.get("pairwise_budget_ablation", {})
    if ablation.get("enabled"):
        lines.append(
            "- Pairwise budget ablation: {mode} at {points}; "
            "{extra} extra pairwise calls".format(
                mode=ablation["mode"],
                points=", ".join(ablation["configured_points"]),
                extra=ablation["extra_pairwise_calls"],
            )
        )
    lines.extend(["", "By phase:"])
    for phase in report["phases"]:
        lines.append(
            "  - {name}: ${cost:.4f} estimated vs {allocation} allocation; "
            "{pointwise} pointwise, {pairwise} pairwise, {audit} audit".format(
                name=phase["name"],
                cost=phase["totals"]["cost_usd"],
                allocation=_money_or_none(phase["allocated_usd"]),
                pointwise=phase["totals"]["pointwise_calls"],
                pairwise=phase["totals"]["pairwise_calls"],
                audit=phase["totals"]["audit_pairwise_calls"],
            )
        )
    return "\n".join(lines)


def _estimate_from_config(
    config: dict[str, Any],
    *,
    max_usd: float | None,
    diagnostics: DiagnosticRecorder,
) -> dict[str, Any]:
    rates = _normalize_rates(config.get("rate_card", {}))
    model_name_policy = _normalize_model_name_policy(
        config.get("model_name_policy", {})
    )
    pairwise_budget_ablation = _normalize_pairwise_budget_ablation(
        config.get("pairwise_budget_ablation", {})
    )
    token_assumptions = _normalize_token_assumptions(
        config.get("token_assumptions", {})
    )
    budget_cap = (
        float(max_usd)
        if max_usd is not None
        else _optional_float(config.get("budget_cap_usd"))
    )
    phases: list[dict[str, Any]] = []
    by_model: dict[str, dict[str, float | int | str | None]] = {}
    totals = _empty_totals()
    for raw_phase in config.get("phases", []):
        phase = _estimate_phase(
            raw_phase,
            rates=rates,
            model_name_policy=model_name_policy,
            pairwise_budget_ablation=pairwise_budget_ablation,
            token_assumptions=token_assumptions,
            budget_cap_usd=budget_cap,
            diagnostics=diagnostics,
        )
        phases.append(phase)
        _add_totals(totals, phase["totals"])
        for model, model_totals in phase["by_model"].items():
            aggregate = by_model.setdefault(
                model,
                {
                    **_empty_totals(),
                    "input_usd_per_1m_tokens": model_totals[
                        "input_usd_per_1m_tokens"
                    ],
                    "output_usd_per_1m_tokens": model_totals[
                        "output_usd_per_1m_tokens"
                    ],
                    "discount_multiplier": model_totals["discount_multiplier"],
                    "rate_note": model_totals.get("rate_note"),
                },
            )
            _add_totals(aggregate, model_totals)

    totals = _rounded_totals(totals)
    return {
        "artifact_type": "sestina-backtest-cost-estimate",
        "dry_run": True,
        "budget_cap_usd": budget_cap,
        "currency": config.get("currency", "USD"),
        "rate_assumption_note": config.get("rate_assumption_note", ""),
        "model_name_policy": dict(model_name_policy),
        "model_availability": {
            "status": "not_checked_dry_run",
            "required_before_paid_run": model_name_policy[
                "availability_check_required_before_paid_run"
            ],
            "models_requiring_check": sorted(by_model),
        },
        "formulae": {
            "candidate_size": "M = min(n, ceil(3K + sqrt(n)))",
            "pairwise_budget": "B_pair = min(ceil(1.25M), ceil(0.25n))",
            "pairwise_budget_ablation": (
                "evaluate scheduled pairwise prefixes at configured points such "
                "as 0, K, ceil(K + sqrt(n)), and B_pair"
            ),
            "pointwise_reuse": (
                "one pointwise pass per bucket is shared by all pointwise-based "
                "strategy arms"
            ),
        },
        "pairwise_budget_ablation": _pairwise_budget_ablation_report(
            pairwise_budget_ablation
        ),
        "totals": totals,
        "by_model": {
            model: _rounded_totals(model_totals)
            for model, model_totals in sorted(by_model.items())
        },
        "phases": phases,
        "budget_status": "not_validated",
        "diagnostics": diagnostics.to_dict(),
    }


def _estimate_phase(
    phase: dict[str, Any],
    *,
    rates: dict[str, dict[str, Any]],
    model_name_policy: dict[str, bool],
    pairwise_budget_ablation: dict[str, Any],
    token_assumptions: dict[str, dict[str, int]],
    budget_cap_usd: float | None,
    diagnostics: DiagnosticRecorder,
) -> dict[str, Any]:
    phase_name = str(phase["name"])
    strategies = [str(item) for item in phase.get("strategies", [])]
    unknown = sorted(set(strategies) - KNOWN_STRATEGIES)
    if unknown:
        diagnostics.record(
            step="backtest_budget",
            code="unknown_strategy",
            level="error",
            message="phase references unknown backtest strategy",
            data={"phase": phase_name, "unknown_strategies": unknown},
        )
        raise ValueError(
            f"unknown backtest strategies in phase {phase_name}: {unknown}"
        )

    pointwise_model = str(phase.get("pointwise_model", "gpt-5.4-mini"))
    pairwise_model = str(phase.get("pairwise_model", pointwise_model))
    audit_model = str(phase.get("audit_model", "gpt-5.4-mini"))
    audit_calls = int(phase.get("audit_pairwise_calls", 0))
    if audit_calls < 0:
        diagnostics.record(
            step="backtest_budget",
            code="invalid_audit_calls",
            level="error",
            message="audit_pairwise_calls must be non-negative",
            data={"phase": phase_name, "audit_pairwise_calls": audit_calls},
        )
        raise ValueError("audit_pairwise_calls must be non-negative")

    uses_pointwise = any(strategy in POINTWISE_STRATEGIES for strategy in strategies)
    uses_pairwise = any(strategy in PAIRWISE_STRATEGIES for strategy in strategies)
    required_models = set()
    if phase.get("buckets") and uses_pointwise:
        required_models.add(pointwise_model)
    if phase.get("buckets") and uses_pairwise:
        required_models.add(pairwise_model)
    if audit_calls:
        required_models.add(audit_model)
    for model in required_models:
        _validate_model_name(
            model,
            policy=model_name_policy,
            phase=phase_name,
            diagnostics=diagnostics,
        )
        if model not in rates:
            diagnostics.record(
                step="backtest_budget",
                code="missing_model_rate",
                level="error",
                message="model is missing from rate card",
                data={"phase": phase_name, "model": model},
            )
            raise ValueError(f"missing rate_card entry for model {model}")

    totals = _empty_totals()
    by_model: dict[str, dict[str, float | int | str | None]] = {}
    buckets = []
    for bucket in phase.get("buckets", []):
        bucket_estimate = _estimate_bucket(
            bucket,
            strategies=strategies,
            pointwise_model=pointwise_model,
            pairwise_model=pairwise_model,
            pairwise_budget_ablation=pairwise_budget_ablation,
            rates=rates,
            token_assumptions=token_assumptions,
            by_model=by_model,
        )
        buckets.append(bucket_estimate)
        _add_totals(totals, bucket_estimate["totals"])

    if audit_calls:
        audit_totals = _estimate_call_group(
            calls=audit_calls,
            assumption=token_assumptions["audit_pairwise"],
            model=audit_model,
            rates=rates,
            call_kind="audit_pairwise",
            by_model=by_model,
        )
        _add_totals(totals, audit_totals)

    allocated_usd = None
    if budget_cap_usd is not None and phase.get("allocation_fraction") is not None:
        allocated_usd = round(budget_cap_usd * float(phase["allocation_fraction"]), 6)
    rounded_totals = _rounded_totals(totals)
    return {
        "name": phase_name,
        "allocation_fraction": _optional_float(phase.get("allocation_fraction")),
        "allocated_usd": allocated_usd,
        "models": {
            "pointwise": pointwise_model,
            "pairwise": pairwise_model,
            "audit": audit_model,
        },
        "strategies": strategies,
        "buckets": buckets,
        "totals": rounded_totals,
        "by_model": {
            model: _rounded_totals(model_totals)
            for model, model_totals in sorted(by_model.items())
        },
        "within_phase_allocation": (
            None
            if allocated_usd is None
            else rounded_totals["cost_usd"] <= allocated_usd
        ),
    }


def _estimate_bucket(
    bucket: dict[str, Any],
    *,
    strategies: list[str],
    pointwise_model: str,
    pairwise_model: str,
    pairwise_budget_ablation: dict[str, Any],
    rates: dict[str, dict[str, Any]],
    token_assumptions: dict[str, dict[str, int]],
    by_model: dict[str, dict[str, float | int | str | None]],
) -> dict[str, Any]:
    name = str(bucket.get("name", "bucket"))
    n = int(bucket["n"])
    k = int(bucket["k"])
    count = int(bucket.get("count", 1))
    if n < 0 or k < 0 or count < 0:
        raise ValueError("bucket n, k, and count must be non-negative")
    candidate_size = default_candidate_size(n, k)
    pairwise_budget = default_pairwise_budget(n, candidate_size)
    uses_pointwise = any(strategy in POINTWISE_STRATEGIES for strategy in strategies)
    pairwise_strategy_count = sum(
        1 for strategy in strategies if strategy in PAIRWISE_STRATEGIES
    )
    pointwise_calls = n * count if uses_pointwise else 0
    pairwise_calls = pairwise_budget * pairwise_strategy_count * count
    ablation = _estimate_pairwise_budget_ablation(
        n=n,
        k=k,
        candidate_size=candidate_size,
        pairwise_budget=pairwise_budget,
        pairwise_strategy_count=pairwise_strategy_count,
        config=pairwise_budget_ablation,
    )

    totals = _empty_totals()
    if pointwise_calls:
        pointwise_totals = _estimate_call_group(
            calls=pointwise_calls,
            assumption=token_assumptions["pointwise"],
            model=pointwise_model,
            rates=rates,
            call_kind="pointwise",
            by_model=by_model,
        )
        _add_totals(totals, pointwise_totals)
    if pairwise_calls:
        pairwise_totals = _estimate_call_group(
            calls=pairwise_calls,
            assumption=token_assumptions["pairwise"],
            model=pairwise_model,
            rates=rates,
            call_kind="pairwise",
            by_model=by_model,
        )
        _add_totals(totals, pairwise_totals)

    estimate = {
        "name": name,
        "n": n,
        "k": k,
        "count": count,
        "candidate_size": candidate_size,
        "pairwise_budget_per_strategy": pairwise_budget,
        "pairwise_strategy_count": pairwise_strategy_count,
        "pointwise_calls": pointwise_calls,
        "pairwise_calls": pairwise_calls,
        "totals": _rounded_totals(totals),
    }
    if ablation is not None:
        estimate["pairwise_budget_ablation"] = ablation
    return estimate


def _estimate_pairwise_budget_ablation(
    *,
    n: int,
    k: int,
    candidate_size: int,
    pairwise_budget: int,
    pairwise_strategy_count: int,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    if not config["enabled"] or pairwise_strategy_count == 0:
        return None

    quarter_n_cap = math.ceil(0.25 * n) if n > 0 else 0
    one_point_two_five_m_budget = (
        math.ceil(1.25 * candidate_size) if candidate_size > 0 else 0
    )
    cap_source = "none"
    if pairwise_budget > 0:
        cap_source = (
            "0.25n"
            if quarter_n_cap <= one_point_two_five_m_budget
            else "1.25M"
        )
    points = []
    for label in config["points"]:
        calls = _resolve_ablation_point(
            label,
            n=n,
            k=k,
            pairwise_budget=pairwise_budget,
        )
        points.append(
            {
                "label": label,
                "pairwise_calls": min(pairwise_budget, max(0, calls)),
            }
        )

    return {
        "mode": config["mode"],
        "reuse_scheduled_pairwise_prefixes": config[
            "reuse_scheduled_pairwise_prefixes"
        ],
        "points_per_strategy": points,
        "default_pairwise_budget": pairwise_budget,
        "default_pairwise_budget_cap_source": cap_source,
        "quarter_n_cap": quarter_n_cap,
        "one_point_two_five_m_budget": one_point_two_five_m_budget,
        "extra_pairwise_calls": 0,
        "costing_note": (
            "ablation points are evaluated as prefixes of each already scheduled "
            "pairwise strategy, so they do not add LLM calls"
        ),
    }


def _estimate_call_group(
    *,
    calls: int,
    assumption: dict[str, int],
    model: str,
    rates: dict[str, dict[str, Any]],
    call_kind: str,
    by_model: dict[str, dict[str, float | int | str | None]],
) -> dict[str, float | int]:
    input_tokens = calls * assumption["input_tokens_per_call"]
    output_tokens = calls * assumption["output_tokens_per_call"]
    rate = rates[model]
    cost = (
        ((input_tokens / 1_000_000) * rate["input_usd_per_1m_tokens"])
        + ((output_tokens / 1_000_000) * rate["output_usd_per_1m_tokens"])
    ) * rate["discount_multiplier"]
    totals = _empty_totals()
    totals[f"{call_kind}_calls"] += calls
    totals["input_tokens"] += input_tokens
    totals["output_tokens"] += output_tokens
    totals["cost_usd"] += cost

    model_totals = by_model.setdefault(
        model,
        {
            **_empty_totals(),
            "input_usd_per_1m_tokens": rate["input_usd_per_1m_tokens"],
            "output_usd_per_1m_tokens": rate["output_usd_per_1m_tokens"],
            "discount_multiplier": rate["discount_multiplier"],
            "rate_note": rate.get("note"),
        },
    )
    _add_totals(model_totals, totals)
    return totals


def _normalize_rates(rate_card: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for model, raw in rate_card.items():
        normalized[str(model)] = {
            "input_usd_per_1m_tokens": float(raw["input_usd_per_1m_tokens"]),
            "output_usd_per_1m_tokens": float(raw["output_usd_per_1m_tokens"]),
            "discount_multiplier": float(raw.get("discount_multiplier", 1.0)),
            "note": raw.get("note"),
        }
    return normalized


def _normalize_model_name_policy(raw: dict[str, Any]) -> dict[str, bool]:
    policy = dict(DEFAULT_MODEL_NAME_POLICY)
    for key in policy:
        if key in raw:
            policy[key] = bool(raw[key])
    return policy


def _normalize_pairwise_budget_ablation(raw: dict[str, Any]) -> dict[str, Any]:
    config = dict(DEFAULT_PAIRWISE_BUDGET_ABLATION)
    for key in ("enabled", "reuse_scheduled_pairwise_prefixes"):
        if key in raw:
            config[key] = bool(raw[key])
    if "mode" in raw:
        config["mode"] = str(raw["mode"])
    if "points" in raw:
        config["points"] = tuple(str(point) for point in raw["points"])
    if "metrics_artifact" in raw:
        config["metrics_artifact"] = str(raw["metrics_artifact"])

    if config["enabled"]:
        if config["mode"] != "prefix_reuse":
            raise ValueError(
                "pairwise_budget_ablation.mode must be prefix_reuse unless "
                "extra ablation calls are explicitly costed"
            )
        if not config["reuse_scheduled_pairwise_prefixes"]:
            raise ValueError(
                "pairwise_budget_ablation must reuse scheduled prefixes unless "
                "extra ablation calls are explicitly costed"
            )
        if not config["points"]:
            raise ValueError("pairwise_budget_ablation.points must not be empty")
        for point in config["points"]:
            _resolve_ablation_point(point, n=1, k=1, pairwise_budget=1)
    return config


def _pairwise_budget_ablation_report(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": config["enabled"],
        "mode": config["mode"],
        "reuse_scheduled_pairwise_prefixes": config[
            "reuse_scheduled_pairwise_prefixes"
        ],
        "configured_points": list(config["points"]),
        "metrics_artifact": config["metrics_artifact"],
        "extra_pairwise_calls": 0,
        "costing": (
            "prefix_reuse evaluates metrics at prefixes of already scheduled "
            "pairwise calls; it does not multiply LLM spend"
        ),
    }


def _resolve_ablation_point(
    label: str,
    *,
    n: int,
    k: int,
    pairwise_budget: int,
) -> int:
    expression = label.replace(" ", "").lower()
    if expression == "0":
        return 0
    if expression == "k":
        return k
    if expression == "k+sqrt(n)":
        return math.ceil(k + math.sqrt(n))
    if expression in {"b_pair", "bpair"}:
        return pairwise_budget
    if expression in {"0.25n", "ceil(0.25n)"}:
        return math.ceil(0.25 * n)
    try:
        return int(expression)
    except ValueError as exc:
        raise ValueError(
            f"unknown pairwise_budget_ablation point {label!r}"
        ) from exc


def _validate_model_name(
    model: str,
    *,
    policy: dict[str, bool],
    phase: str,
    diagnostics: DiagnosticRecorder,
) -> None:
    if policy["require_provider_prefix"] and "/" not in model:
        diagnostics.record(
            step="backtest_budget",
            code="model_name_missing_provider_prefix",
            level="error",
            message="model names must include a provider prefix before paid use",
            data={
                "phase": phase,
                "model": model,
                "example_openai_prefix": "openai/",
            },
        )
        raise ValueError(
            "model names must include a provider prefix; "
            f"OpenAI-routed models should look like openai/{model}"
        )


def _normalize_token_assumptions(
    token_assumptions: dict[str, Any],
) -> dict[str, dict[str, int]]:
    defaults = {
        "pointwise": {"input_tokens_per_call": 900, "output_tokens_per_call": 220},
        "pairwise": {"input_tokens_per_call": 1500, "output_tokens_per_call": 180},
        "audit_pairwise": {
            "input_tokens_per_call": 1500,
            "output_tokens_per_call": 220,
        },
    }
    normalized = dict(defaults)
    for key, raw in token_assumptions.items():
        normalized[str(key)] = {
            "input_tokens_per_call": int(raw["input_tokens_per_call"]),
            "output_tokens_per_call": int(raw["output_tokens_per_call"]),
        }
    return normalized


def _empty_totals() -> dict[str, float | int]:
    return {
        "pointwise_calls": 0,
        "pairwise_calls": 0,
        "audit_pairwise_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }


def _add_totals(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in _empty_totals():
        target[key] += source.get(key, 0)


def _rounded_totals(totals: dict[str, Any]) -> dict[str, Any]:
    rounded = dict(totals)
    rounded["cost_usd"] = round(float(rounded["cost_usd"]), 6)
    for key in (
        "pointwise_calls",
        "pairwise_calls",
        "audit_pairwise_calls",
        "input_tokens",
        "output_tokens",
    ):
        rounded[key] = int(rounded[key])
    for key in (
        "input_usd_per_1m_tokens",
        "output_usd_per_1m_tokens",
        "discount_multiplier",
    ):
        if key in rounded:
            rounded[key] = round(float(rounded[key]), 6)
    return rounded


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _money_or_none(value: object) -> str:
    if value is None:
        return "none"
    return f"${float(value):.2f}"
