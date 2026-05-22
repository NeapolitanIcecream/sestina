# Sestina Autonomous Fresh-Holdout Campaign

Date: 2026-05-08

Workflow: `sestina-autonomous-holdout-campaign`

Status: blocked before paid calls by current arXiv rate limiting. This work made
zero paid Sestina LLM calls, spent USD 0.000000, generated zero pointwise
artifacts, and generated zero pairwise labels.

## Standing Policy

After PR #5, fresh-holdout campaign design is delegated to automation. Missing
fresh holdout manifests and missing reviewed pointwise artifacts are execution
tasks, not user-permission blockers.

The campaign policy is:

- Automation may select and freeze fresh historical arXiv holdout buckets.
- Pointwise calls are authorized only for generating and reviewing pointwise
  artifacts for the frozen fresh holdout.
- Pairwise validation is pairwise-only and may run only after the preflight says
  `go`.
- Stop before any call if the provider/model is unavailable, auth/balance fails,
  usage/cost cannot be measured well enough to enforce the cap, the USD 100
  campaign cap would be exceeded, a manifest identity mismatch is detected, or
  leakage is detected.
- Do not edit historical paid ledgers, raw paid-call artifacts, or old planned
  pair manifests.

## Frozen Holdout Design

Produced:

- `artifacts/backtest-arxiv-autonomous-holdout-campaign/fresh-holdout-design.json`

Selected buckets:

- `cs.LG:2023-03`
- `cs.LG:2023-04`
- `cs.CL:2023-03`
- `cs.CL:2023-04`
- `cs.AI:2023-03`
- `cs.AI:2023-04`
- `cs.CV:2023-03`
- `cs.CV:2023-04`

The design uses the same four categories as the old development replay for
comparability, but the next two chronological months after the old 2023-01 and
2023-02 buckets. The design artifact was written before any fresh result
analysis or paid label generation.

## Source Endpoint

The manifest builder now uses `https://arxiv.org/api/query` rather than
`https://export.arxiv.org/api/query`. A manual `arxiv.org` probe returned HTTP
200 for the frozen-style `cs.LG:2023-04` submitted-date query, but subsequent
larger requests hit HTTP 429 without a `Retry-After` header. To reduce request
weight, the frozen design command now pages arXiv requests with
`--arxiv-page-size 5`.

## Blocker

The manifest build stopped before writing the final manifest:

- Blocking reason: arXiv API returned HTTP 429 for `cs.LG:2023-04`.
- Retry-After header: not provided.
- Exit code: 75.
- Sanitized blocker artifact:
  `artifacts/backtest-arxiv-autonomous-holdout-campaign/campaign-blocked-arxiv-429.json`

The previous `cs.LG:2023-03` part has been written and can be reused by the next
manifest attempt.

No pointwise generation, pairwise preflight, guarded pairwise validation, or
fresh validation analysis was run because the fresh holdout manifest does not
exist.

## Reproduction

Freeze or inspect the design:

```bash
uv run python scripts/design_coverage_floor_fresh_holdout.py
```

Retry the exact manifest build after arXiv API access recovers:

```bash
uv run python scripts/build_arxiv_historical_manifest.py \
  --bucket cs.LG:2023-03 \
  --bucket cs.LG:2023-04 \
  --bucket cs.CL:2023-03 \
  --bucket cs.CL:2023-04 \
  --bucket cs.AI:2023-03 \
  --bucket cs.AI:2023-04 \
  --bucket cs.CV:2023-03 \
  --bucket cs.CV:2023-04 \
  --limit 80 \
  --arxiv-page-size 5 \
  --k 5 \
  --phase pilot \
  --metadata-provider auto \
  --unmatched-policy drop \
  --part-dir artifacts/backtest-datasets/arxiv-historical-coverage-floor-fresh-holdout-parts \
  --reuse-parts \
  --write-parts \
  --target-bucket-count 8 \
  --output artifacts/backtest-datasets/arxiv-historical-coverage-floor-fresh-holdout-manifest.json
```

Then continue in order:

```bash
uv run python scripts/run_fresh_holdout_pointwise_artifacts.py --mode execute --confirm-fresh-holdout-pointwise-generation
uv run python scripts/run_coverage_floor_followup_preflight.py
uv run python scripts/run_coverage_floor_followup_preflight.py --mode execute --confirm-guarded-pairwise-only-execution
uv run python scripts/analyze_coverage_floor_fresh_validation.py
```

Do not claim fresh validation success unless the analysis artifact reports
`fresh_validation_claim.complete=true`.
