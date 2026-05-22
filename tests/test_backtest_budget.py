from __future__ import annotations

import pytest

from sestina.backtest_budget import BudgetExceededError, estimate_from_config


def test_estimator_reuses_pointwise_and_counts_pairwise_strategy_arms() -> None:
    report = estimate_from_config(
        {
            "budget_cap_usd": 10.0,
            "rate_card": {
                "openai/mini": {
                    "input_usd_per_1m_tokens": 1.0,
                    "output_usd_per_1m_tokens": 2.0,
                }
            },
            "token_assumptions": {
                "pointwise": {
                    "input_tokens_per_call": 1000,
                    "output_tokens_per_call": 100,
                },
                "pairwise": {
                    "input_tokens_per_call": 2000,
                    "output_tokens_per_call": 200,
                },
                "audit_pairwise": {
                    "input_tokens_per_call": 3000,
                    "output_tokens_per_call": 300,
                },
            },
            "phases": [
                {
                    "name": "pilot",
                    "allocation_fraction": 1.0,
                    "pointwise_model": "openai/mini",
                    "pairwise_model": "openai/mini",
                    "audit_model": "openai/mini",
                    "strategies": [
                        "random",
                        "semantic_baseline",
                        "pointwise_only",
                        "pointwise_random_pairwise",
                        "sestina_active_pairwise",
                    ],
                    "buckets": [{"name": "b", "n": 100, "k": 5, "count": 2}],
                    "audit_pairwise_calls": 3,
                }
            ],
        }
    )

    phase = report["phases"][0]
    bucket = phase["buckets"][0]
    assert bucket["candidate_size"] == 25
    assert bucket["pairwise_budget_per_strategy"] == 25
    assert bucket["pointwise_calls"] == 200
    assert bucket["pairwise_calls"] == 100

    totals = report["totals"]
    assert totals["pointwise_calls"] == 200
    assert totals["pairwise_calls"] == 100
    assert totals["audit_pairwise_calls"] == 3
    assert totals["input_tokens"] == 409000
    assert totals["output_tokens"] == 40900
    assert totals["cost_usd"] == pytest.approx(0.4908)
    assert report["budget_status"] == "within_cap"


def test_pairwise_budget_ablation_reuses_prefixes_without_extra_calls() -> None:
    report = estimate_from_config(
        {
            "budget_cap_usd": 10.0,
            "pairwise_budget_ablation": {
                "enabled": True,
                "mode": "prefix_reuse",
                "points": ["0", "K", "K + sqrt(n)", "B_pair"],
            },
            "rate_card": {
                "openai/mini": {
                    "input_usd_per_1m_tokens": 1.0,
                    "output_usd_per_1m_tokens": 2.0,
                }
            },
            "token_assumptions": {
                "pointwise": {
                    "input_tokens_per_call": 1000,
                    "output_tokens_per_call": 100,
                },
                "pairwise": {
                    "input_tokens_per_call": 2000,
                    "output_tokens_per_call": 200,
                },
            },
            "phases": [
                {
                    "name": "main",
                    "pointwise_model": "openai/mini",
                    "pairwise_model": "openai/mini",
                    "audit_model": "openai/mini",
                    "strategies": [
                        "pointwise_only",
                        "pointwise_random_pairwise",
                        "sestina_active_pairwise",
                    ],
                    "buckets": [{"name": "capped", "n": 300, "k": 15}],
                }
            ],
        }
    )

    bucket = report["phases"][0]["buckets"][0]
    ablation = bucket["pairwise_budget_ablation"]
    assert ablation["mode"] == "prefix_reuse"
    assert ablation["reuse_scheduled_pairwise_prefixes"] is True
    assert ablation["extra_pairwise_calls"] == 0
    assert ablation["default_pairwise_budget_cap_source"] == "0.25n"
    assert ablation["points_per_strategy"] == [
        {"label": "0", "pairwise_calls": 0},
        {"label": "K", "pairwise_calls": 15},
        {"label": "K + sqrt(n)", "pairwise_calls": 33},
        {"label": "B_pair", "pairwise_calls": 75},
    ]
    assert bucket["pairwise_calls"] == 150
    assert report["totals"]["pairwise_calls"] == 150
    assert report["pairwise_budget_ablation"]["extra_pairwise_calls"] == 0


def test_budget_guard_raises_and_records_diagnostic() -> None:
    config = {
        "budget_cap_usd": 0.001,
        "rate_card": {
            "openai/mini": {
                "input_usd_per_1m_tokens": 10.0,
                "output_usd_per_1m_tokens": 10.0,
            }
        },
        "token_assumptions": {
            "pointwise": {
                "input_tokens_per_call": 1000,
                "output_tokens_per_call": 1000,
            }
        },
        "phases": [
            {
                "name": "too_big",
                "pointwise_model": "openai/mini",
                "pairwise_model": "openai/mini",
                "audit_model": "openai/mini",
                "strategies": ["pointwise_only"],
                "buckets": [{"n": 10, "k": 2}],
            }
        ],
    }

    with pytest.raises(BudgetExceededError) as exc_info:
        estimate_from_config(config)

    report = exc_info.value.report
    assert report["budget_status"] == "exceeds_cap"
    events = report["diagnostics"]["events"]
    assert events[-1]["code"] == "backtest_budget_exceeded"
    assert events[-1]["level"] == "error"


def test_unknown_strategy_is_rejected() -> None:
    config = {
        "budget_cap_usd": 1.0,
        "rate_card": {
            "openai/mini": {
                "input_usd_per_1m_tokens": 1.0,
                "output_usd_per_1m_tokens": 1.0,
            }
        },
        "phases": [
            {
                "name": "bad",
                "pointwise_model": "openai/mini",
                "pairwise_model": "openai/mini",
                "audit_model": "openai/mini",
                "strategies": ["made_up"],
                "buckets": [],
            }
        ],
    }

    with pytest.raises(ValueError, match="unknown backtest strategies"):
        estimate_from_config(config)


def test_provider_prefix_policy_can_still_reject_unprefixed_model_names() -> None:
    config = {
        "budget_cap_usd": 1.0,
        "model_name_policy": {
            "require_provider_prefix": True,
            "availability_check_required_before_paid_run": True,
        },
        "rate_card": {
            "gpt-5.4-mini": {
                "input_usd_per_1m_tokens": 1.0,
                "output_usd_per_1m_tokens": 1.0,
            }
        },
        "phases": [
            {
                "name": "bad_model",
                "pointwise_model": "gpt-5.4-mini",
                "pairwise_model": "gpt-5.4-mini",
                "audit_model": "gpt-5.4-mini",
                "strategies": ["pointwise_only"],
                "buckets": [{"n": 10, "k": 2}],
            }
        ],
    }

    with pytest.raises(ValueError, match="provider prefix"):
        estimate_from_config(config)


def test_report_marks_model_availability_as_required_but_not_checked_in_dry_run() -> None:
    report = estimate_from_config(
        {
            "budget_cap_usd": 10.0,
            "rate_card": {
                "openai/mini": {
                    "input_usd_per_1m_tokens": 1.0,
                    "output_usd_per_1m_tokens": 2.0,
                }
            },
            "phases": [
                {
                    "name": "pilot",
                    "pointwise_model": "openai/mini",
                    "pairwise_model": "openai/mini",
                    "audit_model": "openai/mini",
                    "strategies": ["pointwise_only"],
                    "buckets": [{"n": 5, "k": 2}],
                }
            ],
        }
    )

    assert report["model_availability"]["status"] == "not_checked_dry_run"
    assert report["model_availability"]["required_before_paid_run"] is True
    assert report["model_availability"]["models_requiring_check"] == ["openai/mini"]
