# Sestina No-Paid Algorithm Sweep

Date: 2026-05-07

Workflow: `sestina-no-paid-algorithm-sweep`

Status: no-paid offline replay over cached/historical local artifacts. This is
not a fresh holdout validation and does not authorize paid label purchase. The
current campaign remains stopped.

## Boundary

This sweep made zero paid LLM calls, zero pointwise calls, no fresh holdout, and
no historical ledger or raw paid-call artifact edits. It read local cached
pointwise/pairwise artifacts and wrote derived artifacts only:

- `artifacts/backtest-arxiv-no-paid-algorithm-sweep/no-paid-algorithm-sweep.json`
- `artifacts/backtest-arxiv-no-paid-algorithm-sweep/active-arm-gate.json`
- `artifacts/backtest-arxiv-no-paid-algorithm-sweep/next-experiment-protocol-from-sweep.json`

The active-arm gate passed for a no-paid replay candidate, but the embedded next
experiment protocol still authorizes no label purchase. The only protocol next
step is review of a separate fresh-holdout dry-run/preflight protocol with
provider availability, JSONL ledger, separate artifact directory, hard max-USD
cap, pairwise-only guardrails, and no pointwise calls.

## Inputs

- Development replay set: old 8 historical arXiv buckets from
  `artifacts/backtest-datasets/arxiv-historical-pilot-manifest.json`.
- Cached pointwise source: `artifacts/backtest-arxiv-pilot-live`.
- Cached pairwise sources: local arXiv live caches plus the completed
  full-random variance cache directory.
- Control reference:
  `artifacts/backtest-arxiv-full-random-variance-completion/full-random-variance-completion.json`.

## Arms Tried

- `ci_partition_elimination_cached_replay`
- `paper_borda_lcb_cached_replay`
- `randomized_coverage_floor_hybrid_cached_replay`
- `challenger_outsider_hybrid_cached_replay`

Controls:

- `exact_pool_random_cached_replay`
- `historical_random_cached_replay`
- `posterior_topk_pointwise_prior_control`

## Results

Primary metric is Recall@K. Secondary metrics are nDCG@K and average precision.
All rows below are means over 20 seeds x 8 buckets.

| Arm | Recall@K | nDCG@K | AP |
| --- | ---: | ---: | ---: |
| randomized coverage-floor hybrid | 0.367500 | 0.403420 | 0.395577 |
| challenger/outsider hybrid | 0.338750 | 0.376057 | 0.377537 |
| historical random control | 0.336250 | 0.371382 | 0.369225 |
| exact-pool random control | 0.332500 | 0.369220 | 0.377826 |
| posterior top-K prior control | 0.316250 | 0.360111 | 0.372521 |
| CI partition/elimination | 0.310000 | 0.323514 | 0.339847 |
| paper Borda/LCB | 0.198750 | 0.255758 | 0.248768 |

Best candidate: `randomized_coverage_floor_hybrid_cached_replay`.

Paired active-minus-exact deltas:

- Recall@K mean delta: +0.035000, 95 percent normal CI `[0.01934258, 0.05065742]`.
- nDCG@K mean delta: +0.03419979.
- Average precision mean delta: +0.01775105.
- Seed count: 20.
- Missing pairwise labels: 0.
- Budget shortfall: 0.
- Paid calls: 0.
- Pointwise calls: 0.

## Diagnostics

For the winning coverage-floor hybrid:

- Randomized floor: 1120 floor pairs, mean floor rate 0.35.
- Scheduled pairwise rows: 3200 scheduled, 3200 cached labels available,
  0 missing, 0 partial rows.
- Confidence-bound unresolved count: mean 79.25 rows.
- Graph connectivity proxy: mean largest component size 13.76875; mean component
  count 2.86875.
- Degree around future positives: mean 1.21; mean zero-degree future positives
  2.675.
- Degree around posterior top-K nodes: mean 1.88625; mean zero-degree posterior
  top-K nodes 1.35625.
- Unique future positives touched: 372 total, mean touch rate 0.465.

Weak-bucket diagnostics versus exact-pool random:

- Weak rows: 160.
- Selected-positive delta total: +28.
- Unique future positives touched delta total: -65.
- Mean pointwise-plus-touched oracle cap delta: +0.00375.
- Mean positive-negative-pair oracle cap delta: +0.01000.

Interpretation: the winning hybrid improved selected positives and ranking
metrics despite touching fewer unique future positives than exact-pool random.
That makes the gate pass under the merged protocol, but it is still a cached
development replay result, not evidence from fresh paid labels.

## Gate Outcome

`artifacts/backtest-arxiv-no-paid-algorithm-sweep/active-arm-gate.json` is
schema-valid and passed:

- `paid_followup_allowed`: true.
- `paid_calls_made`: 0.
- `paid_spend_usd`: 0.0.
- `pointwise_calls_made`: 0.
- No future-label or cached-label leakage markers.
- Full 20-seed random variance reference complete.
- Current result boundary remains cached/no-paid and stopped.

The passed gate does not buy labels and does not start fresh holdout validation.

## Reproduction

```bash
uv run python scripts/run_no_paid_algorithm_sweep.py
uv run pytest tests/test_active_arm_gate.py tests/test_no_paid_algorithm_sweep.py tests/test_experiment_protocol.py
uv run python scripts/validate_next_experiment_protocol.py \
  --no-paid-gate-artifact artifacts/backtest-arxiv-no-paid-algorithm-sweep/active-arm-gate.json \
  --priority-direction no_paid_replay_gate_randomized_coverage_floor \
  --output artifacts/backtest-arxiv-no-paid-algorithm-sweep/next-experiment-protocol-from-sweep.json
git diff --check
```

The pre-existing untracked `docs/internal/sestina-next-experiment-report.md`
from the prior merged loop was left untouched.
