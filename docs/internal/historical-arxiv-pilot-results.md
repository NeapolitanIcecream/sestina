# Historical arXiv Pilot Results

This note records the Sestina backtest work completed so far and the reason we
are pausing before a larger run.

## Current Status

Sestina's core design is still supported:

- Run pointwise LLM scoring over every paper in a historical bucket.
- Use a small pairwise budget to refine the top-K decision boundary.
- Report top-K discovery metrics rather than a full ranking.

The current open problem is narrower: the active pair scheduler has not yet
shown that it spends the pairwise budget better than a random pairwise control.

## Data

The historical arXiv pilot used 8 category-month buckets:

| Bucket | Papers | Positives |
|---|---:|---:|
| `cs.LG:2023-01` | 80 | 5 |
| `cs.LG:2023-02` | 79 | 5 |
| `cs.CL:2023-01` | 80 | 5 |
| `cs.CL:2023-02` | 80 | 5 |
| `cs.AI:2023-01` | 79 | 5 |
| `cs.AI:2023-02` | 77 | 5 |
| `cs.CV:2023-01` | 80 | 5 |
| `cs.CV:2023-02` | 79 | 5 |

Total: 634 papers and 40 citation-derived positives.

Labels came from future public citation metadata. Model-visible metadata was
restricted to bucket dates, categories, source, and publication/update dates.
Citation counts, citation rank, DOI, arXiv ID, matched work ID, and matched
title were not visible to the model.

## Cost And Guardrails

Known paid spend:

| Work | Spend |
|---|---:|
| Earlier smoke and historical arXiv setup | USD 0.207235 |
| 8-bucket historical arXiv pilot | USD 0.662340 |
| Scheduler-only follow-up | USD 0.075705 |
| Posterior top-K EVSI follow-up | USD 0.063945 |
| Total known paid spend | USD 1.009225 |

Remaining from the USD 100 cap: USD 98.990775.

All paid phases used explicit dry-run estimates, provider-prefixed model names,
model availability checks, JSONL ledgers, artifact directories, and hard
`--max-usd` limits. The scheduler-only follow-ups made 0 pointwise calls.

## Pilot Metrics

Mean metrics across the 8 buckets:

| Strategy | Recall@K | Precision@K | nDCG@K | AP |
|---|---:|---:|---:|---:|
| Random | 0.075 | 0.075 | 0.090341 | 0.108121 |
| Neutral baseline | 0.025 | 0.025 | 0.018259 | 0.076909 |
| Pointwise-only | 0.300 | 0.300 | 0.339587 | 0.356506 |
| Pointwise + random pairwise | 0.350 | 0.350 | 0.393837 | 0.401054 |
| Sestina active pairwise, original scheduler | 0.325 | 0.325 | 0.371886 | 0.398478 |

Interpretation:

- Pointwise LLM scoring is useful. It is far above random and the neutral
  baseline.
- Pairwise refinement is useful. Both pairwise arms improve over pointwise-only.
- The original active scheduler did not beat random pairwise at the same budget.

## Post-Pilot Diagnosis

The active scheduler underperformed because the original schedule was too narrow.

Observed from offline reconstruction:

- Candidate recall was 25/40 positives, or 0.625.
- `cs.LG:2023-02`, `cs.AI:2023-02`, and `cs.CV:2023-02` selected only 1-2 of
  the 5 positives into the candidate set.
- The original active schedule used 160/160 candidate-internal pairs.
- It scheduled 0 candidate-outsider sentinel pairs.
- It compared 0 positive outsiders.
- It assigned 128 pairs to closeness and 32 to boundary.
- The diversity component was effectively 0 because scheduler metadata bucketing
  did not recognize `primary_category` or `categories` and fell back to
  `source=arxiv`.

This means the old active scheduler was mostly comparing papers the pointwise
model already thought were close. It did not spend enough budget on recall audit
or cross-category calibration.

## Scheduler Changes

The scheduler now uses purpose-budgeted proposal pools:

- `boundary_anchor`
- `candidate_internal`
- `sentinel_outsider`
- `audit_diversity`

It also:

- adds high-uncertainty/high-quality outsider pairs against boundary anchors;
- fixes metadata bucketing to prefer `primary_category` and `categories`;
- reduces the pure closeness weight;
- increases uncertainty and diversity influence;
- emits machine-readable coverage diagnostics, including purpose counts,
  proposal counts, budget utilization, candidate-outsider counts, metadata
  bucket coverage, average rank midpoint, and average probability gap.

Offline preview for the same 160-pair budget allocated:

| Purpose | Pairs |
|---|---:|
| `boundary_anchor` | 48 |
| `candidate_internal` | 48 |
| `sentinel_outsider` | 32 |
| `audit_diversity` | 32 |

The preview included 62 candidate-outsider pairs.

## Aggregation Variants

Offline aggregation sweeps over existing pilot comparisons:

| Pairwise strength | Active Recall@K | Active nDCG@K | Active AP | Random Recall@K | Random nDCG@K | Random AP |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.300 | 0.339587 | 0.356506 | 0.300 | 0.339587 | 0.356506 |
| 1.0 | 0.325 | 0.370028 | 0.394699 | 0.325 | 0.377437 | 0.394092 |
| 2.5 | 0.325 | 0.371886 | 0.398478 | 0.350 | 0.393837 | 0.401054 |
| 5.0 | 0.350 | 0.396776 | 0.397268 | 0.350 | 0.390145 | 0.401390 |

Higher pairwise strength recovers active recall and nDCG on the historical
schedule, but it does not fix the schedule coverage problem. This suggests there
are two separate issues: aggregation weighting and pair selection.

## Scheduler-Only Follow-Up

The follow-up reused historical pointwise artifacts and historical pairwise
artifacts where the same canonical pair was already judged. It made no pointwise
calls.

Dry-run plan:

- 160 scheduled active pairs.
- 57 reused historical pairwise artifacts.
- 103 novel active pairwise calls.
- Estimated spend: USD 0.075705.

Live follow-up:

- 103 `pairwise_active` calls.
- 103 ok.
- Spend: USD 0.075705.

Metrics:

| Strategy | Recall@K | Precision@K | nDCG@K | AP |
|---|---:|---:|---:|---:|
| Pointwise-only | 0.300 | 0.300 | 0.339587 | 0.356506 |
| Revised active pairwise | 0.325 | 0.325 | 0.372640 | 0.384669 |
| Historical random pairwise reference | 0.350 | 0.350 | 0.393837 | 0.401054 |

The revised scheduler still beats pointwise-only, but it still trails the
historical random pairwise reference.

## Posterior Top-K EVSI Follow-Up

After the scheduler-only follow-up, we implemented a decision-aware follow-up
based on posterior top-K membership:

- aggregate pointwise priors and pairwise labels with the existing Bayesian
  Bradley-Terry MAP model;
- sample approximate posterior top-K membership probabilities;
- schedule pairs with a top-K EVSI-style boundary-duel heuristic;
- score both the existing aggregate score and posterior top-K membership.

Dry-run plan:

- 160 scheduled active pairs.
- 73 reused historical pairwise artifacts.
- 87 novel active pairwise calls.
- Estimated spend: USD 0.063945.

Live follow-up:

- 87 `pairwise_active` calls.
- 87 ok.
- Spend: USD 0.063945.

Metrics:

| Strategy | Recall@K | Precision@K | nDCG@K | AP | Brier |
|---|---:|---:|---:|---:|---:|
| Pointwise-only | 0.300 | 0.300 | 0.339587 | 0.356506 | 0.701230 |
| Historical random pairwise, score aggregation | 0.350 | 0.350 | 0.393837 | 0.401054 | 0.702172 |
| Historical random pairwise, posterior top-K | 0.375 | 0.375 | 0.412096 | 0.407579 | 0.053854 |
| EVSI active pairwise, score aggregation | 0.325 | 0.325 | 0.366335 | 0.382875 | 0.701293 |
| EVSI active pairwise, posterior top-K | 0.325 | 0.325 | 0.368193 | 0.389797 | 0.054995 |

Interpretation:

- Posterior top-K aggregation is useful. On historical random pairwise artifacts
  it improves Recall@K from 0.350 to 0.375 and improves nDCG/AP.
- Posterior top-K aggregation improves AP and probability calibration for EVSI
  active pairwise, but it does not change the recovered EVSI top-K set enough
  to improve Recall@K.
- EVSI boundary dueling still beats pointwise-only, but it still trails the
  historical random pairwise reference on Recall@K, nDCG@K, and AP.
- The next obstacle is probably the acquisition policy, not posterior scoring.
  The pair selector still needs a stronger way to identify decision-changing
  outsider challengers or to use earlier pairwise outcomes adaptively within
  each bucket.

## Sequential EVSI Isolation Dry-Run

Implemented the minimal isolation experiment for the acquisition-policy
diagnosis:

- A: keep the existing historical random pairwise + posterior top-K baseline
  fixed at Recall@K 0.375, nDCG@K 0.412096, AP 0.407579.
- B: add `--scheduler-kind exact_pool_random`, which builds the same feasible
  EVSI proposal pool as the EVSI scheduler and samples randomly from that pool.
- C: add `--scheduler-kind sequential_evsi`, which uses 5 rounds x 4 pairs per
  bucket, refits the posterior before each batch, selects the batch, and only
  then reveals cached historical labels. Novel uncached pairs stop the dry-run
  path rather than being treated as known.

New scheduler diagnostics are machine-readable in the follow-up estimate,
summary, and offline bucket-result artifacts. They include unique papers
touched, plausible top-K degree distribution, connected component and anchor
coverage, high-UCB outsider exposure, per-batch top-K entropy/churn, EVSI
zero/tie score rates, retrospective future-positive exposure, and
positive-vs-negative pairwise win rate when pairwise labels are available.

Dry-run commands, both with no paid calls:

```bash
uv run python scripts/run_scheduler_followup.py \
  --max-usd 0.50 \
  --artifact-dir artifacts/backtest-arxiv-exact-pool-random-dry-run \
  --ledger artifacts/backtest-arxiv-exact-pool-random-dry-run/ledger.jsonl \
  --scheduler-kind exact_pool_random \
  --aggregation-mode posterior_topk \
  --seed 17

uv run python scripts/run_scheduler_followup.py \
  --max-usd 0.50 \
  --artifact-dir artifacts/backtest-arxiv-sequential-evsi-dry-run \
  --ledger artifacts/backtest-arxiv-sequential-evsi-dry-run/ledger.jsonl \
  --scheduler-kind sequential_evsi \
  --aggregation-mode posterior_topk \
  --seed 17
```

Dry-run results:

| Arm | Scheduled pairs | Cached labels | Novel labels | Estimated cost | Status |
|---|---:|---:|---:|---:|---|
| Exact-pool random | 160 | 27 | 133 | USD 0.097755 | Blocked for valid full metrics |
| Sequential EVSI | 32 | 11 | 21 | USD 0.015435 | Blocked after first novel batch per bucket |

Partial offline metrics were computed only from cached labels and are not valid
full-arm results:

| Arm | Aggregation | Recall@K | nDCG@K | AP |
|---|---|---:|---:|---:|
| Exact-pool random partial | Score | 0.325 | 0.360457 | 0.376952 |
| Exact-pool random partial | Posterior top-K | 0.300 | 0.335894 | 0.364423 |
| Sequential EVSI partial | Score | 0.300 | 0.336648 | 0.351298 |
| Sequential EVSI partial | Posterior top-K | 0.300 | 0.341445 | 0.363506 |

Blocker: a valid exact-pool random result needs 133 new pairwise labels under the
current cache and seed. A valid sequential EVSI continuation needs at least the
21 first-blocking novel labels, then the sequential dry-run must be rerun because
later batches depend on those revealed outcomes. No paid calls were made.

## Conclusion

The pointwise-first and pairwise-light design is worth keeping. The evidence is
not strong enough to claim that the current active scheduler is better than a
random pairwise control.

The likely algorithmic obstacle is that pairwise comparisons are only helpful
when they expose pointwise errors that affect the top-K boundary. The current
candidate construction and active schedule still do not reliably find enough
positive outsiders or enough informative cross-cluster comparisons. Increasing
pairwise strength can improve active nDCG on the old schedule, but it does not
solve candidate recall.

Do not start a larger main run until the active scheduler has a stronger design
argument or a better small-scale result against random pairwise.

## Current Next Question

The next external algorithmic question should include the posterior top-K EVSI
result, not just the quota scheduler result. A compact prompt is:

```text
We are building Sestina, a pointwise-first, pairwise-light system for finding
top-K high-impact arXiv papers. In an 8-bucket historical arXiv pilot
(634 papers, 40 future-citation top-K positives), pointwise-only beat random
baselines, and pairwise refinement beat pointwise-only. However, active pair
scheduling did not beat a random pairwise control at the same budget.

Metrics: pointwise-only Recall@K 0.300, nDCG@K 0.3396, AP 0.3565; random
pairwise Recall@K 0.350, nDCG@K 0.3938, AP 0.4011; original active pairwise
Recall@K 0.325, nDCG@K 0.3719, AP 0.3985. Diagnostics showed candidate recall
was 25/40 positives, the original active schedule used 160/160 candidate-internal
pairs, 0 sentinel/outsider pairs, and diversity was effectively disabled by
metadata bucketing. We revised the scheduler to allocate boundary/candidate/
sentinel/diversity quotas and ran a scheduler-only follow-up: revised active
still beat pointwise-only but trailed random pairwise (Recall@K 0.325, nDCG@K
0.3726, AP 0.3847).

We then implemented posterior top-K aggregation and an EVSI-style boundary-duel
scheduler. It scheduled 160 active pairs, reused 73 historical pairwise
artifacts, made 87 new pairwise calls, and cost USD 0.063945. EVSI active
pairwise with score aggregation reached Recall@K 0.325, nDCG@K 0.3663, AP
0.3829. EVSI active pairwise with posterior top-K aggregation reached Recall@K
0.325, nDCG@K 0.3682, AP 0.3898, with much better Brier score. It still did not
beat the historical random pairwise reference. Offline, the same posterior
top-K aggregation on historical random pairwise artifacts improved Recall@K to
0.375, nDCG@K to 0.4121, and AP to 0.4076, so posterior scoring appears useful;
the remaining failure is more likely in active pair acquisition.

What algorithmic change should we try next? Focus on active pair selection,
candidate construction, aggregation/posterior modeling, or a different
evaluation design. We want a low-pairwise-budget method that can beat random
pairwise without relying on full ranking.
```
