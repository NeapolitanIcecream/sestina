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
| Total known paid spend | USD 0.945280 |

Remaining from the USD 100 cap: USD 99.054720.

All paid phases used explicit dry-run estimates, provider-prefixed model names,
model availability checks, JSONL ledgers, artifact directories, and hard
`--max-usd` limits. The scheduler-only follow-up made 0 pointwise calls and only
103 novel active pairwise calls.

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

## Recommended Next Question

Ask for algorithmic advice before spending more budget. A compact prompt is:

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

What algorithmic change should we try next? Focus on active pair selection,
candidate construction, aggregation/posterior modeling, or a different
evaluation design. We want a low-pairwise-budget method that can beat random
pairwise without relying on full ranking.
```
