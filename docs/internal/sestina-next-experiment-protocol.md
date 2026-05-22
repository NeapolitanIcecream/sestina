# Sestina Next Experiment Protocol

Date: 2026-05-08

Workflow: `sestina-next-experiment-loop`

Status: autonomous fresh-holdout campaign policy. This document and its
source/test counterparts make zero paid Sestina LLM calls, zero pointwise calls,
no paid labels, no historical ledger edits, and no raw call artifact edits.

## Current Result Boundary

The cached result boundary remains stopped for publication claims. The best
current result is still the
budget-filled `new_information_challenger_cached_replay` against
`exact_pool_random_cached_replay`, followed by reviewed guarded cache-only
execution of the exact frozen manifest. It is a cached/no-paid internal result,
not a fresh holdout validation and not a publication-ready paid-label result.

Separately, the next fresh-holdout campaign is now authorized to continue
autonomously under the USD 100 campaign cap. Missing fresh holdout manifests and
missing reviewed pointwise artifacts should be handled by automation, not
treated as user-permission blockers.

Cleanup and publication work may summarize the result, but public-facing
material must preserve these boundaries:

- Say cached/no-paid replay plus guarded cache-only execution, not fresh
  validation.
- Keep the weak-bucket caveat scoped to the exact frozen zero-missing-label
  manifest and reviewed artifacts.
- Do not publish raw paid-call JSON, historical paid ledgers, planned-pair
  JSONL manifests, stdout JSON, dataset manifests with work IDs, or
  `.codex-workflows/**` records without separate scrub/review.
- Do not edit historical paid ledgers or raw paid-call artifacts. Create
  sanitized derivatives if evidence must be shared.

## Required No-Paid Gate

Any future Sestina experiment starts with a new no-paid gate. The gate must be
an offline replay/simulation artifact that makes zero paid calls, spends USD
0.000000, and makes zero pointwise calls for the gate itself.

The gate must block unless it has:

- A paired random or exact-pool random control.
- At least 20 paired seeds over the historical buckets unless marked
  exploratory and non-decisive.
- Recall@K as the primary metric, with nDCG@K and AP as secondary metrics.
- Seed-level active-minus-random confidence intervals.
- The completed full-random variance reference.
- Weak-bucket diagnostics: pointwise-plus-touched oracle cap,
  positive-negative-pair oracle cap, observed positive-winner cap, unique
  future positives touched, graph connectivity, and degree around future
  positives and posterior top-K nodes.
- A randomized coverage floor or paired random-control schedule.
- Explicit no-leakage evidence: future labels, citation outcomes,
  `good_paper`, matched title/work ID, and cached label values cannot be used
  for scheduling, routing, prompts, or model-visible inputs.

`sestina.active_arm_gate` now treats future-label/cached-label leakage markers
as hard blockers for paid follow-up.

## Priority Direction

The next algorithmic direction should be one of:

- Confidence-interval top-K partition/elimination scheduler.
- No-paid replay gate with a randomized coverage floor.

This is a new design gate, not a continuation of the stopped campaign. Do not
resume CCTD-GF, sequential EVSI, expanded-pool random, targeted-outsider
random, posterior shrinkage, soft-strength calibration, or acquisition-score
micro-tweaks over the same information surface unless a new no-paid gate and
reviewer-approved protocol explicitly reopens that work.

## Fresh Holdout Protocol

Fresh holdout validation may only begin after the no-paid gate passes. Passing
the no-paid gate still does not authorize arbitrary paid label purchase by
itself.

The permitted sequence is:

1. Freeze a fresh historical holdout design before any fresh result analysis.
2. Build the fresh manifest with no overlap against the old 8 development
   buckets.
3. Generate and review pointwise artifacts only for that fresh holdout under the
   campaign cap.
4. Run the coverage-floor preflight.
5. If and only if the preflight says `go`, run guarded pairwise-only validation.
6. Analyze against `exact_pool_random_cached_replay` with Recall@K primary and
   nDCG@K/AP secondary.

The fresh holdout protocol must require:

- Provider/model availability check before any label-generation call.
- Separate artifact directory.
- JSONL ledger.
- Hard `--max-usd` cap under the remaining paid cap.
- Pointwise calls scoped only to fresh-holdout artifact generation/review.
- Pairwise-only validation guardrails and immediate abort on any pointwise-call
  attempt during validation.
- No future-label leakage and no cached label values used before scheduling.
- Immutable historical paid ledgers and raw paid-call artifacts.
- Ledger usage/cost fields sufficient to enforce the cap; use upstream returned
  cost when present, otherwise upstream returned usage with configured rates,
  otherwise the conservative configured token estimate.

Machine-readable contract:

```bash
uv run python scripts/validate_next_experiment_protocol.py
```

With no no-paid gate artifact, the command emits a blocked protocol. To test the
fresh-holdout path, pass `--no-paid-gate-artifact` and a
`--fresh-holdout-request` JSON object with
`standing_campaign_authorization=true` and
`fresh_holdout_pointwise_artifacts_authorized=true`; the protocol still writes
zero paid calls and does not authorize work outside the fresh-holdout campaign
scope.

## Verification

Focused executable specs:

```bash
uv run pytest tests/test_active_arm_gate.py tests/test_experiment_protocol.py
```

Full repo verification before PR:

```bash
uv run pytest -p no:cacheprovider
git diff --check
```
