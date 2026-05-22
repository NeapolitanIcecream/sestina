# Sestina Coverage-Floor Follow-Up Preflight

Workflow: `sestina-coverage-floor-followup`

Status: superseded by the autonomous fresh-holdout campaign policy. The PR #5
preflight remains a pairwise-only gate, but missing fresh holdout manifests and
missing reviewed pointwise artifacts are now automation tasks under the campaign
cap, not user-permission blockers.

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
- Pairwise validation remains pairwise-only. Pointwise calls are authorized only
  for the separate fresh-holdout pointwise artifact generation/review workflow.
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
- Hard cap: remaining campaign spend under the USD 100 total cap, including the
  known prior spend recorded by the repo.
- Paid calls/spend: 0 calls, 0.000000 USD.
- Pointwise calls: 0.
- Paid validation: blocked.

Original blocking prerequisites:

- Fresh holdout manifest is missing.
- Therefore the fresh holdout cannot be used for planning.
- Reviewed pointwise artifacts for that fresh holdout are also unavailable.

Current campaign status:

- The fresh holdout design is frozen at
  `artifacts/backtest-arxiv-autonomous-holdout-campaign/fresh-holdout-design.json`.
- Manifest construction is blocked by arXiv HTTP 429 before any paid calls. The
  builder now uses `https://arxiv.org/api/query` with conservative paging rather
  than `https://export.arxiv.org/api/query`:
  `artifacts/backtest-arxiv-autonomous-holdout-campaign/campaign-blocked-arxiv-429.json`.

## Reproduction

```bash
uv run python scripts/run_coverage_floor_followup_preflight.py
uv run python -m json.tool artifacts/backtest-arxiv-coverage-floor-followup-preflight/coverage-floor-followup-preflight.json >/dev/null
uv run pytest tests/test_coverage_floor_followup_preflight.py
git diff --check
```

To unblock the preflight, retry the frozen design command after the arXiv rate
limit cools down, then generate reviewed pointwise artifacts for exactly that
holdout under the campaign cap. The manifest builder command shape is:

```bash
uv run python scripts/build_arxiv_historical_manifest.py \
  --bucket CATEGORY:YYYY-MM \
  --bucket CATEGORY:YYYY-MM \
  --limit 80 \
  --arxiv-page-size 5 \
  --k 5 \
  --phase pilot \
  --metadata-provider auto \
  --unmatched-policy drop \
  --output artifacts/backtest-datasets/arxiv-historical-coverage-floor-fresh-holdout-manifest.json
```

Do not use the old pilot manifest as the fresh holdout. Do not run pointwise
calls for anything except the fresh-holdout artifact generation/review workflow.
