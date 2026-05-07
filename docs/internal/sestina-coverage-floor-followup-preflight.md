# Sestina Coverage-Floor Follow-Up Preflight

Workflow: `sestina-coverage-floor-followup`

Status: blocked planning preflight. This made zero paid LLM calls, zero paid
spend, zero pointwise calls, and no paid pairwise validation.

## Scope

This follow-up freezes the merged no-paid sweep winner,
`randomized_coverage_floor_hybrid_cached_replay`, from:

- `artifacts/backtest-arxiv-no-paid-algorithm-sweep/no-paid-algorithm-sweep.json`
- `artifacts/backtest-arxiv-no-paid-algorithm-sweep/active-arm-gate.json`

Frozen controls and guardrails:

- Random control: `exact_pool_random_cached_replay`.
- Seeds: the 20-seed sweep set.
- Primary metric: Recall@K.
- Secondary metrics: nDCG@K and average precision.
- Policy: randomized coverage-floor hybrid with random floor 0.35, per-item
  cap 6, anchor multiplier 2, and challenger multiplier 5.
- No pointwise calls. Pointwise is not separately approved.
- No future labels, `good_paper`, citation outcomes, matched titles/work IDs,
  or cached label values may be used for scheduling, routing, prompts, or
  model-visible inputs.

## Artifact

Produced:

- `artifacts/backtest-arxiv-coverage-floor-followup-preflight/coverage-floor-followup-preflight.json`

The artifact is a dry-run/preflight summary. It does not include raw paid-call
JSON, paid ledgers, old planned-pair manifests, or historical workflow records.

Preflight result:

- Provider/model availability: available.
- Planned ledger path:
  `artifacts/backtest-arxiv-coverage-floor-followup-preflight/coverage-floor-followup-ledger.jsonl`.
- Hard cap: 2.00 USD.
- Paid calls/spend: 0 calls, 0.000000 USD.
- Pointwise calls: 0.
- Paid validation: blocked.

Blocking prerequisites:

- Fresh holdout manifest is missing.
- Therefore the fresh holdout cannot be used for planning.
- Reviewed pointwise artifacts for that fresh holdout are also unavailable, and
  this task does not approve creating them.

## Reproduction

```bash
uv run python scripts/run_coverage_floor_followup_preflight.py
uv run python -m json.tool artifacts/backtest-arxiv-coverage-floor-followup-preflight/coverage-floor-followup-preflight.json >/dev/null
uv run pytest tests/test_coverage_floor_followup_preflight.py
git diff --check
```

To unblock a future preflight, first create a predeclared fresh historical
holdout manifest, then provide reviewed pointwise artifacts for exactly that
holdout. The manifest builder is:

```bash
uv run python scripts/build_arxiv_historical_manifest.py \
  --bucket CATEGORY:YYYY-MM \
  --bucket CATEGORY:YYYY-MM \
  --limit 80 \
  --k 5 \
  --phase pilot \
  --metadata-provider auto \
  --unmatched-policy drop \
  --output artifacts/backtest-datasets/arxiv-historical-coverage-floor-fresh-holdout-manifest.json
```

Do not use the old pilot manifest as the fresh holdout. Do not run pointwise
label generation under this approval.
