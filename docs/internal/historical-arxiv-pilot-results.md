# Historical arXiv Pilot Results

This note records the Sestina backtest work completed so far and the reason we
are pausing before a larger run.

Decision memo: `docs/internal/sestina-experiment-decision-memo.md` consolidates
the current evidence, random-baseline variance result, hard protocol for future
active arms, and repo-hygiene handoff.

Final handoff: `docs/internal/sestina-final-results-handoff.md` records the
reviewed guarded execution closure, stop decision, PR/publication cleanup
readiness, and repo hygiene checklist.

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
| Exact-pool random isolation | USD 0.098490 |
| Sequential EVSI isolation | USD 0.063945 |
| CCTD-GF isolation | USD 0.083790 |
| Expanded-pool random isolation | USD 0.109515 |
| Targeted-outsider random isolation | USD 0.111720 |
| Posterior decision shrinkage offline analysis | USD 0.000000 |
| Pairwise soft-strength calibration offline analysis | USD 0.000000 |
| Random-control gap/oracle diagnostic offline analysis | USD 0.000000 |
| CI top-K partition replay gate offline analysis | USD 0.000000 |
| Active-arm gate harness smoke | USD 0.000000 |
| Full-schedule random variance completion | USD 1.269345 |
| Total known paid spend | USD 2.746030 |

Remaining from the USD 100 cap: USD 97.253970.

All paid phases used explicit dry-run estimates, provider-prefixed model names,
model availability checks, JSONL ledgers, artifact directories, and hard
`--max-usd` limits. The scheduler-only follow-ups made 0 pointwise calls.

Future active-arm proposals must pass the executable no-paid gate before any
paid pairwise follow-up:

```bash
uv run python scripts/run_active_arm_gate.py \
  --active-artifact <no-paid-active-comparison.json> \
  --random-variance-artifact artifacts/backtest-arxiv-full-random-variance-completion/full-random-variance-completion.json \
  --output artifacts/backtest-arxiv-active-arm-gate-harness/<gate-result>.json
```

The default smoke used the cached CI-partition replay artifact, made zero paid
LLM calls, and wrote
`artifacts/backtest-arxiv-active-arm-gate-harness/active-arm-gate-smoke.json`.
It blocked paid follow-up: paired Recall@K delta was +0.003750 with 95% normal
approximation CI [-0.009681, +0.017181], while nDCG@K and AP deltas were
negative.

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

## Sequential EVSI Isolation Results

Implemented the minimal isolation experiment for the acquisition-policy
diagnosis:

- A: keep the existing historical random pairwise + posterior top-K baseline
  fixed at Recall@K 0.375, nDCG@K 0.412096, AP 0.407579.
- B: add `--scheduler-kind exact_pool_random`, which builds the same feasible
  EVSI proposal pool as the EVSI scheduler and samples randomly from that pool.
- C: add `--scheduler-kind sequential_evsi`, which uses 5 rounds x 4 pairs per
  bucket, refits the posterior before each batch, selects the batch, and only
  then reveals cached historical or previously paid labels.

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
| Exact-pool random | 160 | 27 | 133 | USD 0.097755 | Ready for paid labeling |
| Sequential EVSI first pass | 32 | 11 | 21 | USD 0.015435 | Ready for adaptive paid labeling |

Partial offline metrics were computed only from cached labels and are not valid
full-arm results:

| Arm | Aggregation | Recall@K | nDCG@K | AP |
|---|---|---:|---:|---:|
| Exact-pool random partial | Score | 0.325 | 0.360457 | 0.376952 |
| Exact-pool random partial | Posterior top-K | 0.300 | 0.335894 | 0.364423 |
| Sequential EVSI partial | Score | 0.300 | 0.336648 | 0.351298 |
| Sequential EVSI partial | Posterior top-K | 0.300 | 0.341445 | 0.363506 |

Live exact-pool random:

- Artifact directory:
  `artifacts/backtest-arxiv-exact-pool-random-live`.
- Summary:
  `artifacts/backtest-arxiv-exact-pool-random-live/summary-pilot.json`.
- Ledger:
  `artifacts/backtest-arxiv-exact-pool-random-live/ledger.jsonl`.
- 160 scheduled pairs.
- 27 historical pairwise labels reused.
- 133 successful paid labels.
- 1 malformed response was ledgered as `parse_error` and retried successfully.
- Spend: USD 0.098490.

Live sequential EVSI:

- Artifact directory:
  `artifacts/backtest-arxiv-sequential-evsi-live`.
- Summary:
  `artifacts/backtest-arxiv-sequential-evsi-live/summary-pilot.json`.
- Ledger:
  `artifacts/backtest-arxiv-sequential-evsi-live/ledger.jsonl`.
- 160 scheduled pairs, 20 per bucket.
- 74 historical pairwise labels reused.
- 86 successful paid labels.
- 1 provider response failure was ledgered and retried successfully.
- Spend: USD 0.063945.
- Completion status: `complete`.

Full paid metrics:

| Arm | Aggregation | Recall@K | Precision@K | nDCG@K | AP | Brier |
|---|---|---:|---:|---:|---:|---:|
| Pointwise-only | Score | 0.300 | 0.300 | 0.339587 | 0.356506 | 0.701230 |
| Historical random pairwise | Posterior top-K | 0.375 | 0.375 | 0.412096 | 0.407579 | 0.053854 |
| One-shot EVSI active | Posterior top-K | 0.325 | 0.325 | 0.368193 | 0.389797 | 0.054995 |
| Exact-pool random | Score | 0.350 | 0.350 | 0.382736 | 0.384309 | 0.701440 |
| Exact-pool random | Posterior top-K | 0.375 | 0.375 | 0.404687 | 0.381836 | 0.054963 |
| Sequential EVSI | Score | 0.350 | 0.350 | 0.379797 | 0.384419 | 0.701404 |
| Sequential EVSI | Posterior top-K | 0.325 | 0.325 | 0.365254 | 0.383471 | 0.053958 |

Interpretation:

- Exact-pool random matching the historical random posterior-top-K Recall@K
  means the feasible EVSI pool contains enough useful comparisons to recover
  the random-control hit rate. The pool is not empty or hopeless.
- Exact-pool random still trails the historical random posterior-top-K nDCG and
  AP, so pool construction and graph coverage may still be weaker than the
  original global random control.
- Sequential EVSI did not improve over one-shot EVSI. It refit the posterior
  after each paid batch, but the final posterior-top-K result remained at
  Recall@K 0.325 and slightly lower nDCG/AP than one-shot EVSI.
- The stale one-shot explanation is therefore weak. The stronger diagnosis is an
  acquisition-score issue: EVSI's ranking of pairs inside its own feasible pool
  is worse than random sampling from that pool for this pilot.
- Remaining uncertainty: this is still one seed and one 8-bucket pilot. However,
  the isolation result is strong enough that the next scheduler change should
  alter acquisition scoring, not only add adaptivity.

## Conclusion

The pointwise-first and pairwise-light design is worth keeping. The evidence is
not strong enough to claim that the current active scheduler is better than a
random pairwise control.

The likely algorithmic obstacle is that pairwise comparisons are only helpful
when they expose pointwise errors that affect the top-K boundary. The current
candidate construction and active schedule still do not reliably find enough
decision-changing comparisons. Increasing pairwise strength can improve active
nDCG on the old schedule, but it does not solve candidate recall.

The sequential isolation experiment narrows the problem. Random sampling from
the exact EVSI feasible pool reaches Recall@K 0.375 with posterior top-K, but
EVSI's own acquisition ranking reaches only Recall@K 0.325 even after adaptive
refits. That makes acquisition scoring the lead suspect. Pool/graph quality may
still matter because exact-pool random trails the historical random posterior
top-K nDCG/AP, but stale one-shot scoring is no longer a compelling primary
explanation.

## CCTD-GF Isolation Results

Implemented `--scheduler-kind cctd_gf`: Coverage-Constrained Thompson Top-K
Disagreement with Graph Floor.

The scheduler keeps the EVSI/exact-pool feasible proposal pool and posterior
top-K aggregation, but changes acquisition inside the pool:

- 4 adaptive mini-batches x 5 pairs per bucket.
- Per bucket target mix: 12 sampled top-K disagreement pairs, 4 graph-floor
  anchor pairs, and 4 exact-pool random floor pairs.
- Posterior latent samples estimate top-K disagreement and BALD-style pair
  information.
- Graph diagnostics track active degrees, cross-component pairs, decision
  boundary pairs, top-K disagreement, information, and score/probability gaps.
- The guarded follow-up runner stops after any batch with novel labels, pays
  only through `scripts/run_scheduler_followup.py`, and resumes after those
  labels become cache-revealable.

Dry-run plan for the first adaptive pass:

- 40 scheduled pairs.
- 16 historical/follow-up labels reusable.
- 24 novel pairwise labels.
- Estimated spend: USD 0.017640.

Live CCTD-GF:

- Artifact directory: `artifacts/backtest-arxiv-cctd-gf-live`.
- Summary: `artifacts/backtest-arxiv-cctd-gf-live/summary-pilot.json`.
- Ledger: `artifacts/backtest-arxiv-cctd-gf-live/ledger.jsonl`.
- Noise audit: `artifacts/backtest-arxiv-cctd-gf-live/pairwise-noise-audit.json`.
- 160 scheduled pairs, 20 per bucket.
- 46 historical/follow-up pairwise labels reused.
- 114 successful paid pairwise labels.
- Spend: USD 0.083790.
- Completion status: `complete`.

Full CCTD-GF metrics:

| Arm | Aggregation | Recall@K | Precision@K | nDCG@K | AP | Brier |
|---|---|---:|---:|---:|---:|---:|
| CCTD-GF | Score | 0.325 | 0.325 | 0.368947 | 0.382628 | 0.701625 |
| CCTD-GF | Posterior top-K | 0.325 | 0.325 | 0.367089 | 0.375631 | 0.054283 |
| Exact-pool random reference | Posterior top-K | 0.375 | 0.375 | 0.404687 | 0.381836 | 0.054963 |
| Historical random reference | Posterior top-K | 0.375 | 0.375 | 0.412096 | 0.407579 | 0.053854 |

Interpretation:

- CCTD-GF did not beat exact-pool random on Recall@K or nDCG@K.
- CCTD-GF posterior top-K AP also fell below exact-pool random.
- The graph floor did not fix the active-acquisition failure. In this pilot,
  concentrating on sampled disagreement plus graph anchors still chose a worse
  set of comparisons than random sampling from the same feasible pool.
- Do not start a larger CCTD-GF main run.

Offline noise audit:

- The audit is best-effort and citation-retrospective. It uses future citation
  labels only after the fact and does not change the model-visible data.
- Citation alignment is available for completed pairwise artifacts. Boundary,
  information, and score/probability-gap stratification is limited because
  older historical artifacts do not store `scheduled_pair` diagnostics and some
  reused labels lack CCTD-GF diagnostics.
- CCTD-GF pairwise judgments were not obviously less citation-aligned than
  random controls: the judge winner had higher future citations on 75/157
  comparable CCTD-GF pairs, or 0.477707, versus 53/156, or 0.339744, for
  exact-pool random.
- Future positives beat nonpositives on 30/62 CCTD-GF positive-vs-nonpositive
  pairs, or 0.483871, versus 16/44, or 0.363636, for exact-pool random.
- EVSI-selected boundary pairs did not appear less citation-aligned than
  random/exact-pool pairs in this audit. The EVSI-boundary citation-alignment
  rate was 0.418502 versus 0.359736 for random/exact-pool; however, many older
  EVSI/random artifacts lack scheduler diagnostics, so this is not a clean
  causal diagnosis.
- The audit therefore does not support pairwise-label noise as the main
  explanation for CCTD-GF underperformance. The stronger explanation remains
  that the acquisition policy is surfacing citation-aligned comparisons that do
  not move the posterior top-K decision in the right buckets.

## Expanded-Pool Random Candidate-Construction Result

Implemented `--scheduler-kind expanded_pool_random` as the next diagnostic
candidate-construction arm. It keeps random sampling inside the feasible pool,
but widens the EVSI dynamic proposal pool instead of changing the acquisition
score:

- `pool_multiplier=4` instead of the exact-pool random default of 2.
- `diverse_outsider_count=max(20, 4K)`, which is 20 for this K=5 pilot.
- same 20-pair per-bucket budget and posterior top-K aggregation.
- no pointwise calls.

The purpose was to test whether exact-pool random's remaining nDCG/AP gap
against historical random was mostly caused by a too-narrow candidate/proposal
pool. This is intentionally not another CCTD-GF-style within-pool acquisition
score.

Dry-run command:

```bash
uv run python scripts/run_scheduler_followup.py \
  --max-usd 2.00 \
  --artifact-dir artifacts/backtest-arxiv-expanded-pool-random-dry-run \
  --ledger artifacts/backtest-arxiv-expanded-pool-random-dry-run/ledger.jsonl \
  --scheduler-kind expanded_pool_random \
  --aggregation-mode posterior_topk \
  --seed 17
```

Dry-run result:

- 160 scheduled pairs.
- 11 reusable historical labels.
- 149 novel pairwise labels required.
- Estimated spend: USD 0.109515.
- 0 pointwise calls.
- Model availability was not checked in dry-run, as expected.
- Mean proposal-pool size increased from 17.25 for exact-pool random to 29.25.
- Mean scheduled unique papers increased from 15.875 to 22.875.

Paid command:

```bash
uv run python scripts/run_scheduler_followup.py \
  --max-usd 2.00 \
  --artifact-dir artifacts/backtest-arxiv-expanded-pool-random-live \
  --ledger artifacts/backtest-arxiv-expanded-pool-random-live/ledger.jsonl \
  --scheduler-kind expanded_pool_random \
  --aggregation-mode posterior_topk \
  --seed 17 \
  --confirm-paid
```

Live result:

- Artifact directory:
  `artifacts/backtest-arxiv-expanded-pool-random-live`.
- Summary:
  `artifacts/backtest-arxiv-expanded-pool-random-live/summary-pilot.json`.
- Ledger:
  `artifacts/backtest-arxiv-expanded-pool-random-live/ledger.jsonl`.
- Model availability: `openai/gpt-5.4-mini` available.
- 149 `pairwise_active` ledger entries, all `ok`.
- 0 pointwise calls.
- Spend: USD 0.109515.

Full expanded-pool metrics:

| Arm | Aggregation | Recall@K | Precision@K | nDCG@K | AP | Brier |
|---|---|---:|---:|---:|---:|---:|
| Expanded-pool random | Score | 0.300 | 0.300 | 0.363396 | 0.383487 | 0.701486 |
| Expanded-pool random | Posterior top-K | 0.325 | 0.325 | 0.374246 | 0.377321 | 0.056189 |
| Exact-pool random reference | Posterior top-K | 0.375 | 0.375 | 0.404687 | 0.381836 | 0.054963 |
| Historical random reference | Posterior top-K | 0.375 | 0.375 | 0.412096 | 0.407579 | 0.053854 |
| CCTD-GF reference | Posterior top-K | 0.325 | 0.325 | 0.367089 | 0.375631 | 0.054283 |

Interpretation:

- Widening the proposal pool did not beat exact-pool random or historical
  random under the same 160-label budget.
- It matched CCTD-GF Recall@K and slightly exceeded CCTD-GF nDCG/AP, but this is
  not enough to justify scaling the arm.
- The widened pool touched more papers but diluted the label budget. The dry-run
  retrospective future-positive exposure was not higher than exact-pool random,
  even though that exposure is an after-the-fact diagnostic and was not
  model-visible.
- The failure weakens the hypothesis that the remaining gap is fixed by simply
  broadening the candidate pool. A future candidate-construction arm needs a
  more targeted challenger-construction rule, or the next arm should instead
  change the posterior decision model.

## Targeted-Outsider Random Candidate-Construction Result

Implemented `--scheduler-kind targeted_outsider_random` to test a more selective
outsider-challenger construction while keeping within-pool selection random.
The arm constructs, per bucket:

- 5 current posterior top-K anchors and 5 boundary/UCB-fill anchors.
- 10 outsider challengers selected by model-visible posterior top-K probability,
  boundary mass, UCB, uncertainty, metadata-bucket diversity, and prior pairwise
  degree.
- All anchor-outsider proposals, then a random 20-pair schedule from that
  constructed pool.

The scheduling rule does not use future labels, citations, `good_paper`, or
retrospective diagnostics. In this one-shot run, prior pairwise degree is
available in the diagnostic/scoring path but neutral because no pairwise labels
are revealed before scheduling.

Dry-run command:

```bash
uv run python scripts/run_scheduler_followup.py \
  --max-usd 2.00 \
  --artifact-dir artifacts/backtest-arxiv-targeted-outsider-random-dry-run \
  --ledger artifacts/backtest-arxiv-targeted-outsider-random-dry-run/ledger.jsonl \
  --scheduler-kind targeted_outsider_random \
  --aggregation-mode posterior_topk \
  --seed 17
```

Dry-run result:

- 160 scheduled pairs.
- 8 reusable historical labels.
- 152 novel pairwise labels required.
- Estimated spend: USD 0.111720.
- 0 pointwise calls.
- Model availability was not checked in dry-run, as expected.
- Mean anchors per bucket: 10.
- Mean targeted outsiders per bucket: 10.
- Mean outsider-anchor proposal pairs per bucket: 100.
- Mean scheduled outsider-anchor pairs per bucket: 20.
- Mean scheduled unique papers per bucket: 18.125.
- Mean targeted pool size was 0.691174 of expanded-pool random.
- Mean targeted pool item delta versus exact-pool random: +2.75.
- Mean targeted pool item delta versus expanded-pool random: -9.25.
- Mean targeted proposal delta versus exact-pool random: -40.5.
- Mean targeted proposal delta versus expanded-pool random: -317.875.
- Retrospective future-positive exposure averaged 5.625 pairs touching a future
  positive and 2.5 unique future positives touched per bucket. This was measured
  after scheduling and was not model-visible.

Paid command:

```bash
uv run python scripts/run_scheduler_followup.py \
  --max-usd 2.00 \
  --artifact-dir artifacts/backtest-arxiv-targeted-outsider-random-live \
  --ledger artifacts/backtest-arxiv-targeted-outsider-random-live/ledger.jsonl \
  --scheduler-kind targeted_outsider_random \
  --aggregation-mode posterior_topk \
  --seed 17 \
  --confirm-paid
```

Live result:

- Artifact directory:
  `artifacts/backtest-arxiv-targeted-outsider-random-live`.
- Summary:
  `artifacts/backtest-arxiv-targeted-outsider-random-live/summary-pilot.json`.
- Ledger:
  `artifacts/backtest-arxiv-targeted-outsider-random-live/ledger.jsonl`.
- Model availability: `openai/gpt-5.4-mini` available.
- 152 `pairwise_active` ledger entries, all `ok`.
- 8 historical pairwise labels reused.
- 0 pointwise calls.
- Spend: USD 0.111720.

Full targeted-outsider metrics:

| Arm | Aggregation | Recall@K | Precision@K | nDCG@K | AP | Brier |
|---|---|---:|---:|---:|---:|---:|
| Targeted-outsider random | Score | 0.325 | 0.325 | 0.370028 | 0.383710 | 0.701595 |
| Targeted-outsider random | Posterior top-K | 0.325 | 0.325 | 0.371886 | 0.386160 | 0.054523 |
| Exact-pool random reference | Posterior top-K | 0.375 | 0.375 | 0.404687 | 0.381836 | 0.054963 |
| Historical random reference | Posterior top-K | 0.375 | 0.375 | 0.412096 | 0.407579 | 0.053854 |
| CCTD-GF reference | Posterior top-K | 0.325 | 0.325 | 0.367089 | 0.375631 | 0.054283 |
| Expanded-pool random reference | Posterior top-K | 0.325 | 0.325 | 0.374246 | 0.377321 | 0.056189 |

Interpretation:

- Targeted outsider construction did what it was designed to do mechanically:
  it was much narrower than expanded-pool random, sampled only anchor-outsider
  pairs, and exposed more unique papers than exact-pool random without the full
  dilution of expanded-pool random.
- It did not beat exact-pool random or historical random on Recall@K or nDCG@K.
- Posterior-top-K AP improved versus exact-pool random, expanded-pool random,
  and CCTD-GF, but that single AP gain is not enough to justify scaling the arm
  because the top-K hit rate stayed at 0.325.
- Retrospective future-positive exposure was better than expanded-pool random
  and matched exact-pool random on unique future positives touched, but it did
  not translate into better Recall@K or nDCG@K.
- This weakens the hypothesis that a simple targeted outsider-anchor schedule is
  sufficient. The next useful change should alter the posterior decision model,
  label interpretation, or evaluation design rather than trying another
  one-shot random candidate-construction variant.

## Posterior Decision-Model Shrinkage Result

Tested one offline posterior-layer change:
degree-aware shrinkage of posterior top-K membership toward the pointwise-only
posterior top-K membership. The rule uses:

```text
weight = comparisons_used / (comparisons_used + 2.0)
score = weight * pairwise_posterior_top_k_probability
      + (1 - weight) * pointwise_only_posterior_top_k_probability
```

The intent was to test whether sparse pairwise evidence was being trusted too
hard for low-degree papers. This changed only the final decision rule; it did
not alter candidate construction, pair acquisition, paid labels, or evaluation
labels. Future citation labels were used only for retrospective metrics.

Offline command:

```bash
uv run python scripts/analyze_posterior_decision_shrinkage.py \
  --output artifacts/backtest-arxiv-posterior-decision-shrinkage/decision-shrinkage-analysis.json
```

Artifact:

- `artifacts/backtest-arxiv-posterior-decision-shrinkage/decision-shrinkage-analysis.json`

Spend:

- Paid calls: 0.
- Added spend: USD 0.000000.
- Known paid spend remains USD 1.476685.

Machine-readable diagnostics include rule parameters, pairwise model name
validation from config, per-bucket/per-arm decision outputs for every paper,
coverage statistics, tie statistics, uncertainty deltas, arm-level metrics, and
explicit limitations. The artifact also records aggregate comparison-source
completeness and excludes partial-label legacy rows from aggregate comparisons.

Aggregate metrics:

| Arm | Baseline posterior Recall@K | Shrunk Recall@K | Baseline nDCG@K | Shrunk nDCG@K | Baseline AP | Shrunk AP |
|---|---:|---:|---:|---:|---:|---:|
| Historical active | 0.350 | 0.325 | 0.382736 | 0.363396 | 0.384868 | 0.379433 |
| Historical random | 0.375 | 0.350 | 0.412096 | 0.393837 | 0.407579 | 0.402017 |
| Exact-pool random | 0.375 | 0.375 | 0.404687 | 0.404687 | 0.381836 | 0.390420 |
| Sequential EVSI | 0.325 | 0.325 | 0.365254 | 0.363396 | 0.383471 | 0.382902 |
| CCTD-GF | 0.325 | 0.325 | 0.367089 | 0.368947 | 0.375631 | 0.381514 |
| Expanded-pool random | 0.325 | 0.325 | 0.374246 | 0.376104 | 0.377321 | 0.384166 |
| Targeted-outsider random | 0.325 | 0.300 | 0.371886 | 0.355485 | 0.386160 | 0.387529 |

Excluded from aggregate table:

- Revised active: partial cached labels in all 8 buckets, 57/160 scheduled
  pairwise labels available, 103 missing.
- Posterior top-K EVSI: partial cached labels in all 8 buckets, 73/160
  scheduled pairwise labels available, 87 missing.

The per-bucket diagnostics for these two arms remain in the JSON artifact, but
the aggregate rows are intentionally omitted because they are not comparable to
complete-label arms or to earlier full-run metrics documented above.

Interpretation:

- Degree-aware shrinkage is not a promising replacement for posterior top-K.
- It did not improve Recall@K on any arm.
- It preserved exact-pool random Recall@K and nDCG@K while improving AP from
  0.381836 to 0.390420, but exact-pool random still did not beat historical
  random posterior top-K on nDCG/AP.
- It hurt the strongest historical random reference, reducing Recall@K from
  0.375 to 0.350 and nDCG@K from 0.412096 to 0.393837.
- It also hurt targeted-outsider random Recall@K and nDCG@K, despite a tiny AP
  gain.
- The diagnostics show the rule changed few top-K decisions in most arms
  (mean changed vs posterior top-K was 0.125 to 0.5 papers per bucket), so the
  conservative shrinkage was too weak or aimed at the wrong failure mode.

Recommendation from this workflow:

- Do not adopt degree-aware posterior top-K shrinkage as the default decision
  rule.
- Keep posterior top-K as the default decision rule for small runs.
- The next useful posterior-layer test should change label interpretation or
  calibration more directly, for example a pairwise confidence/soft-probability
  calibration model, a tie/uncertain-label model, or a bucket-level pairwise
  strength model. Do not spend paid calls for another acquisition-only arm
  unless it uses genuinely new information.

## Pairwise Soft-Strength Calibration Result

Tested one offline pairwise label-interpretation change:
soft-probability strength calibration. The existing Bradley-Terry aggregation
already uses pairwise `soft_probability` as the fractional winner target. This
experiment also uses the same soft margin as the likelihood-strength multiplier
for decisive left/right labels:

```text
margin = abs(soft_probability - 0.5) / 0.5
strength_multiplier = 0.5 + 0.5 * margin
calibrated_confidence = original_confidence * strength_multiplier
```

With the fixed defaults, a close 0.54 win is interpreted as a 0.54 target and a
0.54 likelihood-strength multiplier, while a 0.90 win keeps 0.90 of the original
confidence. Ties and uncertain labels keep the existing limited-weight
aggregation behavior. The rule does not use future labels, citations, or
`good_paper` in calibration; retrospective labels are used only for metrics.

Offline command:

```bash
uv run python scripts/analyze_pairwise_strength_calibration.py \
  --output artifacts/backtest-arxiv-pairwise-strength-calibration/strength-calibration-analysis.json
```

Artifact:

- `artifacts/backtest-arxiv-pairwise-strength-calibration/strength-calibration-analysis.json`

Spend:

- Paid calls: 0.
- Added spend: USD 0.000000.
- Known paid spend remains USD 1.476685.

Machine-readable diagnostics include rule parameters, pairwise model name
validation from config, per-bucket/per-arm comparison-strength rows, aggregate
strength summaries, per-paper decision outputs, aggregate metrics, explicit
partial-label exclusions, and limitations. As in the shrinkage workflow, revised
active and posterior-top-K EVSI partial legacy rows are excluded from aggregate
metrics because the current reconstruction has only 57/160 and 73/160 cached
labels respectively.

Aggregate metrics:

| Arm | Baseline posterior Recall@K | Calibrated Recall@K | Baseline nDCG@K | Calibrated nDCG@K | Baseline AP | Calibrated AP |
|---|---:|---:|---:|---:|---:|---:|
| Historical active | 0.350 | 0.325 | 0.382736 | 0.368947 | 0.384868 | 0.387400 |
| Historical random | 0.375 | 0.375 | 0.412096 | 0.410238 | 0.407579 | 0.406245 |
| Exact-pool random | 0.375 | 0.375 | 0.404687 | 0.404687 | 0.381836 | 0.388651 |
| Sequential EVSI | 0.325 | 0.325 | 0.365254 | 0.366335 | 0.383471 | 0.384435 |
| CCTD-GF | 0.325 | 0.325 | 0.367089 | 0.368947 | 0.375631 | 0.381575 |
| Expanded-pool random | 0.325 | 0.325 | 0.374246 | 0.374246 | 0.377321 | 0.378803 |
| Targeted-outsider random | 0.325 | 0.300 | 0.371886 | 0.355485 | 0.386160 | 0.387258 |

Aggregate strength diagnostics:

| Arm | Comparisons | Decisive labels | Mean strength multiplier | Mean calibrated confidence |
|---|---:|---:|---:|---:|
| Historical active | 160 | 119 | 0.707813 | 0.586479 |
| Historical random | 160 | 147 | 0.766875 | 0.710004 |
| Exact-pool random | 160 | 142 | 0.675438 | 0.589336 |
| Sequential EVSI | 160 | 123 | 0.696313 | 0.591939 |
| CCTD-GF | 160 | 145 | 0.669625 | 0.597194 |
| Expanded-pool random | 160 | 139 | 0.681063 | 0.602716 |
| Targeted-outsider random | 160 | 147 | 0.680375 | 0.622031 |

Interpretation:

- Soft-strength calibration is not a promising default replacement for
  posterior top-K on this pilot.
- It did not improve Recall@K for any complete-label arm.
- It preserved exact-pool random Recall@K and nDCG@K while improving AP from
  0.381836 to 0.388651, but exact-pool random still trails historical random
  posterior top-K on nDCG/AP.
- It slightly hurt the strongest historical random reference on nDCG/AP, though
  Recall@K stayed at 0.375.
- It hurt historical active and targeted-outsider random top-K membership,
  dropping each by 0.025 mean Recall@K.
- Compared with degree-aware posterior shrinkage, soft-strength calibration is
  less damaging to historical random Recall@K and keeps exact-pool Recall@K, but
  it still does not produce a top-K gain or beat the random controls.

Recommendation from this workflow:

- Do not adopt soft-probability strength calibration as the default decision
  rule.
- Keep posterior top-K as the default small-run decision rule.
- Treat the AP gains on exact-pool, CCTD-GF, and targeted-outsider random as
  diagnostic only; they do not overcome the missing Recall@K/nDCG gains.
- This was workflow 3 of 3 maximum overnight workflows. After review, stop and
  report rather than launching another arm.

## Random-Control Gap And Oracle Decomposition

Implemented one offline diagnostic pass to explain why random/exact-pool random
remains the strongest small-budget baseline.

Offline command:

```bash
uv run python scripts/analyze_random_control_gap.py \
  --output artifacts/backtest-arxiv-random-control-diagnosis/random-control-gap-analysis.json
```

Artifact:

- `artifacts/backtest-arxiv-random-control-diagnosis/random-control-gap-analysis.json`

Spend:

- Paid calls: 0.
- Added spend: USD 0.000000.
- Known paid spend remains USD 1.476685.

Method:

- Reconstructed complete-label arms from existing pointwise artifacts, pairwise
  artifacts, scheduler diagnostics, and citation labels.
- Used future citation labels only after scheduling/scoring for retrospective
  metrics, exposure counts, graph diagnostics, false-positive/false-negative
  decomposition, and oracle upper bounds.
- Excluded current partial-label reconstructions from aggregate diagnostics.
  Revised active has 57/160 cached labels under the current reconstruction, and
  one-shot posterior EVSI has 73/160. Their earlier live summaries remain
  documented above, but this diagnostic does not mix those partial
  reconstructions with complete-arm aggregate rows.
- Loaded the existing posterior-decision shrinkage and pairwise-strength
  calibration artifacts as decision-layer context.

Complete-arm posterior top-K metrics:

| Arm | Recall@K | nDCG@K | AP |
|---|---:|---:|---:|
| Historical random | 0.375 | 0.412096 | 0.407579 |
| Exact-pool random | 0.375 | 0.404687 | 0.381836 |
| Historical active | 0.350 | 0.382736 | 0.384868 |
| Targeted-outsider random | 0.325 | 0.371886 | 0.386160 |
| Expanded-pool random | 0.325 | 0.374246 | 0.377321 |
| CCTD-GF | 0.325 | 0.367089 | 0.375631 |
| Sequential EVSI | 0.325 | 0.365254 | 0.383471 |

Exposure and oracle diagnostics:

| Arm | Unique positives touched | Pos-neg pairs | Positive win rate | Pointwise + touched cap | Pos-neg oracle cap | Observed-winner cap |
|---|---:|---:|---:|---:|---:|---:|
| Historical random | 24 | 46 | 0.782609 | 0.600 | 0.600 | 0.575 |
| Exact-pool random | 20 | 44 | 0.772727 | 0.525 | 0.525 | 0.500 |
| Historical active | 23 | 68 | 0.573529 | 0.575 | 0.550 | 0.350 |
| Sequential EVSI | 23 | 73 | 0.602740 | 0.575 | 0.475 | 0.350 |
| CCTD-GF | 23 | 62 | 0.758065 | 0.575 | 0.550 | 0.450 |
| Expanded-pool random | 19 | 36 | 0.694444 | 0.525 | 0.525 | 0.500 |
| Targeted-outsider random | 20 | 44 | 0.863636 | 0.550 | 0.550 | 0.500 |

Graph diagnostics do not make exact-pool random look uniquely privileged:

| Arm | Mean papers touched | Mean largest component | Mean future-positive degree | Mean posterior-top-K degree |
|---|---:|---:|---:|---:|
| Historical random | 21.000 | 16.000 | 1.200 | 1.975 |
| Exact-pool random | 15.875 | 15.500 | 1.300 | 2.050 |
| Historical active | 10.375 | 10.375 | 2.300 | 4.375 |
| Sequential EVSI | 11.375 | 11.375 | 2.475 | 5.700 |
| CCTD-GF | 17.625 | 17.375 | 1.900 | 3.500 |
| Expanded-pool random | 22.875 | 15.500 | 0.950 | 1.450 |
| Targeted-outsider random | 18.125 | 16.625 | 1.150 | 2.000 |

Interpretation:

- Historical random and exact-pool random recover 15/40 positives. The active
  variants recover 13/40 or 14/40 positives under posterior top-K.
- The exact-pool advantage over sequential EVSI, CCTD-GF, and
  targeted-outsider random is 2 positives out of 40, split across
  `cs.LG:2023-01` and `cs.CL:2023-01`. The expanded-pool gap is also 2
  positives, split across `cs.LG:2023-01` and `cs.CV:2023-02`. This is not
  dominated by a single bucket, but it is still a small one-seed gap.
- Exact-pool random does not win because it touches the most future positives.
  Historical random, CCTD-GF, historical active, and sequential EVSI all touch
  more unique future positives than exact-pool random.
- Exact-pool random also does not win because it has the best graph topology.
  CCTD-GF and targeted-outsider random have larger connected components, and
  historical active/sequential EVSI give future positives and posterior-top-K
  papers higher mean degree.
- Pairwise label interpretation is part of the story but not the whole story.
  Historical random has the strongest AP/nDCG and a high positive-vs-negative
  win rate. Targeted-outsider random has the highest positive win rate, but it
  still misses Recall@K/nDCG. Degree shrinkage and soft-strength calibration
  did not improve Recall@K for any complete-label arm.
- The strongest diagnosis is that random controls provide useful stochastic
  boundary and false-negative coverage without over-concentrating labels. The
  active policies often spend many labels around plausible top-K nodes, but the
  resulting evidence does not move the right false negatives into top-K.

Recommendation from this diagnostic:

- Keep historical random or exact-pool random plus posterior top-K as the
  default small-budget baseline.
- Do not spend on another one-shot acquisition-score tweak, naive pool widening,
  simple posterior shrinkage, or soft-strength calibration default.
- Before any future paid arm, require an offline gate: the proposed schedule
  must improve weak-bucket pointwise-plus-touched and positive-negative-pair
  oracle caps versus exact-pool random while preserving a randomized floor or
  paired random-control seed.
- Future paid artifacts should store `scheduled_pair` diagnostics for every
  reused and newly paid label so later diagnostics do not depend on
  reconstructing schedules after scheduler code changes.

## Related-Work Audit

The related-work audit in `docs/internal/related-work-audit.md` and
`artifacts/backtest-arxiv-related-work/related-work-matrix.json` supports the
same small-budget recommendation, with one important caveat.

The literature supports keeping random or exact-pool random plus posterior top-K
as the default baseline because low-budget active selection can be
variance-sensitive, sparse noisy pairwise graphs can make BT/PL-style posterior
tweaks insufficient, and strong random or uniform controls are standard in
crowdsourced top-K and active-evaluation studies. It also challenges any broader
claim that active scheduling is hopeless: active ranking, top-K aggregation, and
dueling-bandit algorithms can beat passive selection under stochastic
preference, separation, confidence, or geometry assumptions that Sestina has not
yet validated.

The next paid acquisition change should therefore be blocked by a no-paid design
gate. The most concrete candidate is a confidence-interval top-K
partition/elimination scheduler with a randomized coverage floor, replayed
against cached labels and exact-pool random across multiple seeds. It should
proceed only if it improves Recall@K/nDCG@K or weak-bucket oracle caps without
dropping the random-control floor.

## CI Partition Replay Gate

Implemented the related-work recommendation as a no-paid cached-label replay
gate.

Command:

```bash
uv run python scripts/run_ci_partition_gate.py
```

Artifacts:

- `artifacts/backtest-arxiv-ci-partition-gate/ci-partition-gate-analysis.json`
- `artifacts/backtest-arxiv-ci-partition-gate/ci-partition-gate-smoke.json`

Spend:

- Paid calls: 0.
- Pointwise paid calls: 0.
- Added spend: USD 0.000000.
- Known paid spend remains USD 1.476685.

Method:

- Built `sestina/ci_partition_gate.py`, which treats pairwise labels as noisy
  stochastic evidence. Each item gets a beta-style interval from pointwise prior
  mass plus fractional pairwise wins/losses weighted by label confidence and
  soft probability.
- Tracked unresolved items at the K boundary by comparing the Kth item's lower
  bound with the best outsider upper bound.
- Replayed an adaptive CI partition/elimination scheduler against cached
  pairwise labels only. No retrospective citation labels were visible to the
  scheduler.
- Added a randomized coverage floor. The full gate scheduled 640 random-floor
  pairs out of 3,200 CI replay pairs, a 0.200 floor rate.
- Compared against `exact_pool_random_cached_replay`, a clearly documented
  approximation that samples randomly from cached labels intersected with the
  exact EVSI feasible proposal pool. CI replay updates its feasible pool after
  cached labels are revealed; exact-pool random is the one-shot cached-pool
  control.
- Ran 20 seeds across the 8 historical buckets. Retrospective citation labels
  were used only for metrics, weak-bucket deltas, exposure diagnostics, graph
  diagnostics, and oracle caps.

Gate criteria:

- Preserve the randomized coverage floor.
- Allow paid follow-up only if mean Recall@K improves by at least 0.025 while
  nDCG@K is nonnegative and AP does not drop by more than 0.01, or if
  weak-bucket oracle headroom improves materially without losing the random
  floor.

20-seed posterior top-K metrics:

| Arm | Recall@K | nDCG@K | AP |
|---|---:|---:|---:|
| CI partition/elimination replay | 0.313750 | 0.326249 | 0.336553 |
| Cached exact-pool random replay | 0.310000 | 0.343602 | 0.360758 |
| CI minus cached exact-pool random | +0.003750 | -0.017352 | -0.024205 |

Gate diagnostics:

| Diagnostic | CI replay | Cached exact-pool random |
|---|---:|---:|
| Unique future positives touched | 461 | 447 |
| Mean future-positive touch rate | 0.576250 | 0.558750 |
| Mean largest component size | 13.825000 | 15.262500 |
| Mean future-positive degree | 2.071250 | 1.721250 |
| Mean unresolved CI boundary count | 79.250000 | 79.250000 |
| Pointwise + touched oracle recall cap | 0.576250 | 0.566250 |
| Positive-negative pair oracle recall cap | 0.527500 | 0.558750 |
| Observed positive-winner recall cap | 0.490000 | 0.535000 |

Weak-bucket deltas versus cached exact-pool random:

- Selected-positive total delta: +3 across 160 seed/bucket rows.
- Unique future positives touched delta: +14.
- Mean pointwise-plus-touched oracle recall-cap delta: +0.010000.
- Mean positive-negative-pair oracle recall-cap delta: -0.031250.

Verdict:

- The randomized floor was preserved.
- The metric gate failed: Recall@K improved only +0.003750, while nDCG@K and
  AP dropped.
- The weak-bucket oracle fallback failed: pointwise-plus-touched headroom rose
  only +0.010000 and positive-negative pair oracle headroom fell.
- Paid follow-up is blocked. No paid pairwise calls were run.

Limitations:

- This is an offline cached-label replay, not a fresh paid acquisition run.
- The exact comparison pool is approximated by cached labels intersected with
  the exact EVSI feasible proposal pool.
- The CI intervals remain very wide; the mean unresolved boundary count is
  essentially the whole bucket, so the current interval model is not producing
  useful eliminations at this budget.
- The pilot is still 8 historical arXiv buckets, and paired seed variance is
  material.

Recommendation from this gate:

- Do not spend on a paid CI partition arm in its current form.
- Keep historical random or exact-pool random plus posterior top-K as the
  default small-budget baseline.
- The next non-paid direction should improve the reliability model or candidate
  information enough to shrink boundary intervals before it asks for paid
  labels. Another acquisition heuristic over the same cached pool is unlikely
  to be credible by itself.

## Random Variance Replication Audit

Implemented a no-paid cached-label variance audit to test whether the
random/exact-pool random advantage is robust or a lucky seed-17 artifact.

Command:

```bash
uv run python scripts/analyze_random_variance_replication.py
```

Artifact:

- `artifacts/backtest-arxiv-random-variance-replication/random-variance-replication.json`

Spend:

- Paid calls: 0.
- Pointwise paid calls: 0.
- Pairwise paid calls: 0.
- Added spend: USD 0.000000.
- Known paid spend remains USD 1.476685.

Method:

- Replayed 20 seeds over the same 8 historical buckets.
- Scanned cached pairwise labels from all existing live arXiv artifact
  directories.
- Separated two questions:
  - Full-schedule cache probe: generate the historical-random and exact-pool
    random schedules for each seed and measure whether every pair already has a
    cached label. Incomplete rows are not used as headline metrics.
  - Cached-label constrained replay: sample historical-random and exact-pool
    random schedules only from already labeled feasible pairs, then compute
    posterior top-K metrics and seed-level 95% intervals.
- The exact-pool cached replay applies the cached-label filter to the exact EVSI
  feasible proposal pool first, then uses the same random selector and dynamic
  per-item-cap policy as `schedule_exact_pool_random`. It is not the CI
  partition replay helper's capped proxy.
- Used future citation labels only for retrospective metrics and diagnostics,
  not for scheduling or scoring.

Full-schedule cache completeness:

| Full schedule probe | Scheduled pairs | Cached labels | Missing labels | Cache reuse | Complete seed/bucket rows |
|---|---:|---:|---:|---:|---:|
| Historical random | 3,200 | 864 | 2,336 | 0.270000 | 8/160 |
| Exact-pool random | 3,200 | 1,770 | 1,430 | 0.553125 | 8/160 |

Only seed 17 is complete for all 8 buckets in both full-schedule probes.
Completing the 20-seed full schedules would require 1,724 unique additional
pair labels, estimated at USD 1.267140 under the current config. This audit did
not spend that money because the cached replay is already enough to show that a
single-seed claim is unsafe; a full paid completion should be a separate guarded
pairwise-only labeling workflow if needed.

Cached-label constrained posterior top-K replay:

| Arm | Recall@K mean | Recall@K 95% bootstrap CI | nDCG@K mean | nDCG@K 95% bootstrap CI | AP mean | AP 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|
| Historical random cached replay | 0.321250 | [0.305000, 0.336250] | 0.350769 | [0.336956, 0.363924] | 0.342584 | [0.333069, 0.352232] |
| Exact-pool random cached replay | 0.315000 | [0.300000, 0.328750] | 0.348280 | [0.333132, 0.362113] | 0.358501 | [0.348886, 0.368077] |

Paired historical-random minus exact-pool random deltas:

| Metric | Mean delta | 95% bootstrap CI |
|---|---:|---:|
| Recall@K | +0.006250 | [-0.017500, +0.028750] |
| nDCG@K | +0.002489 | [-0.018645, +0.023537] |
| AP | -0.015917 | [-0.030002, -0.000786] |

Interpretation:

- The original complete seed-17 historical-random and exact-pool-random rows
  both reached Recall@K 0.375, but full multi-seed replication is not available
  from cache except for seed 17.
- The cached constrained replay shows material seed variance. A one-positive
  swing is 0.025 mean Recall@K in this 40-positive pilot, which is the same
  order as several active-vs-random gaps.
- The paired historical-vs-exact replay Recall@K interval crosses zero. These
  two random baselines should be treated as comparable controls, not as a
  stable ranking between random variants.
- The exact-pool cached replay now matches the current scheduler policy within
  the cached feasible pool, but it is still not a perfect substitute for a full
  paid random replication because unlabeled exact-pool proposal pairs are
  unavailable and the replay pool is constrained by which labels prior
  workflows happened to buy. It is sufficient for the current decision:
  single-seed random/exact-pool advantage should not be overread.

Recommendation from this audit:

- Keep historical random or exact-pool random plus posterior top-K as the
  default small-budget baseline.
- A randomized floor should be mandatory in future paid active-arm comparisons.
- Future active claims should report per-seed/per-bucket metrics, seed-unit 95%
  confidence intervals for active-minus-random deltas, label reuse and missing
  label diagnostics, new paid calls, ledger spend, and weak-bucket
  exposure/oracle-cap diagnostics.
- Do not claim an active arm beats random unless the paired
  active-minus-random Recall@K interval is positive, or the mean Recall@K gain
  is at least 0.025 with nonnegative nDCG/AP deltas and no missing-label caveat.

## Full-Schedule Random Variance Completion

Completed the paid follow-up identified by the random variance replication
audit. This workflow bought only the deduped missing pairwise labels needed to
make the 20-seed historical-random and exact-pool-random full schedules
complete.

No-paid planning command:

```bash
uv run python scripts/run_full_random_variance_completion.py --max-usd 5.00
```

Guarded paid/resume command:

```bash
uv run python scripts/run_full_random_variance_completion.py \
  --max-usd 5.00 \
  --confirm-paid
```

Artifacts:

- Initial plan:
  `artifacts/backtest-arxiv-full-random-variance-completion/initial-missing-label-plan.json`
- Latest zero-missing plan:
  `artifacts/backtest-arxiv-full-random-variance-completion/missing-label-plan.json`
- Ledger:
  `artifacts/backtest-arxiv-full-random-variance-completion/ledger.jsonl`
- Labeling summary:
  `artifacts/backtest-arxiv-full-random-variance-completion/labeling-summary-pilot.json`
- Final metrics:
  `artifacts/backtest-arxiv-full-random-variance-completion/full-random-variance-completion.json`

Spend and guardrails:

- Initial no-paid plan: 1,724 unique missing pair labels, estimated at
  USD 1.267140.
- Paid completion: 1,727 ledger entries, all `pairwise_full_random_variance`;
  1,724 `ok` labels and 3 preserved `parse_error` retry attempts.
- Actual ledger spend: USD 1.269345.
- Pointwise paid calls: 0.
- Provider-prefixed model: `openai/gpt-5.4-mini`.
- Model availability check passed before paid calls.
- Paid call artifacts were written under a separate artifact directory. Failed
  parse artifacts were preserved and retries used distinct attempt paths.

Complete full-schedule posterior top-K metrics:

| Arm | Recall@K mean | Recall@K 95% bootstrap CI | nDCG@K mean | nDCG@K 95% bootstrap CI | AP mean | AP 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|
| Historical random full schedule | 0.332500 | [0.320000, 0.345000] | 0.366567 | [0.355858, 0.376360] | 0.368876 | [0.359224, 0.376462] |
| Exact-pool random full schedule | 0.322500 | [0.310000, 0.336250] | 0.362799 | [0.352992, 0.373035] | 0.373689 | [0.369669, 0.377526] |

Paired historical-random minus exact-pool random deltas:

| Metric | Mean delta | 95% bootstrap CI |
|---|---:|---:|
| Recall@K | +0.010000 | [-0.010000, +0.028750] |
| nDCG@K | +0.003768 | [-0.010960, +0.018203] |
| AP | -0.004813 | [-0.014868, +0.004117] |

Interpretation:

- Complete random baseline robustness is supported in the specific sense needed
  here: random/exact-pool random plus posterior top-K remain mandatory
  baselines, and future active arms need paired random-seed intervals.
- The seed-17 Recall@K 0.375 references were high relative to the complete
  multi-seed means. They should not be treated as stable point estimates.
- Historical random and exact-pool random are statistically comparable at this
  sample size; their paired Recall@K, nDCG@K, and AP intervals all cross zero.
- Expanded-pool random, targeted-outsider random, and CCTD-GF remain prior
  single-seed context only. They are not methodologically valid headline
  full-schedule interval controls.

Recommendation from the completion:

- Stop spending on random-baseline completion. The completed artifact is the
  variance reference for future simulator gates or a predeclared active-arm
  comparison.
- Keep the active-claim threshold unchanged: require a positive paired
  active-minus-random Recall@K interval, or a mean Recall@K gain of at least
  0.025 with nonnegative nDCG/AP deltas and no missing-label caveat.

## Recommendation

Use random or exact-pool random plus posterior top-K as the default small-run
baseline. Do not start a larger main run for CCTD-GF, expanded-pool random,
targeted-outsider random, or the current CI partition replay gate. The completed
20-seed random variance artifact replaces the old single-seed random reference:
future paid active comparisons should include a paired randomized floor and
seed-level uncertainty intervals.
The next active arm should not be another within-pool acquisition score tweak or
naive pool widening. The degree-aware posterior decision shrinkage and
soft-strength pairwise calibration tested here were negative/inconclusive, so a
future posterior-layer change needs a stronger reason than simple sparse-label
shrinkage or soft-probability downweighting.

## Active-Arm Shortlist No-Paid Gate

The next shortlist was gated with the reviewed active-arm harness and no paid
calls:

```bash
uv run python scripts/run_active_arm_shortlist_gate.py
```

Artifact:
`artifacts/backtest-arxiv-active-arm-shortlist-gate/shortlist-gate-study.json`

Original verdict at shortlist time: no paid active-arm follow-up was allowed.
The cached CI partition replay was the only shortlist item with a
methodologically valid active-gate input, and it remained blocked: paired
Recall@K delta was +0.003750 with 95% normal CI [-0.009681, +0.017181],
nDCG/AP deltas were negative, the unresolved-boundary count did not drop, and
positive-negative oracle headroom fell. The new-information challenger and
standard-ranking aggregation directions were blocked until they produced
complete no-paid replay artifacts. The new-information prerequisite has since
been produced and is summarized below. The simulator harness is usable
infrastructure, but it is not itself an active policy and does not justify paid
labels.

## Reliability-Aware CI Partition V2 No-Paid Replay

Implemented the missing no-paid replay/simulator prerequisite for the
reliability-aware CI partition v2 candidate. The workflow made zero paid
Sestina LLM calls, zero pointwise calls, zero paid labeling calls, and no paid
ledger or paid-call artifact rewrites. Known paid spend remains USD 2.746030.

Command:

```bash
uv run python scripts/run_ci_partition_v2_gate_replay.py
uv run python scripts/run_active_arm_gate.py \
  --active-artifact artifacts/backtest-arxiv-ci-partition-v2-gate-replay/ci-partition-v2-gate-replay.json \
  --random-variance-artifact artifacts/backtest-arxiv-full-random-variance-completion/full-random-variance-completion.json \
  --output artifacts/backtest-arxiv-ci-partition-v2-gate-replay/active-arm-gate.json \
  --active-arm reliability_aware_ci_partition_v2_cached_replay \
  --random-control-arm exact_pool_random_cached_replay
```

Artifacts:

- `artifacts/backtest-arxiv-ci-partition-v2-gate-replay/ci-partition-v2-gate-replay.json`
- `artifacts/backtest-arxiv-ci-partition-v2-gate-replay/active-arm-gate.json`

Method:

- Reused existing cached/reviewed pointwise and pairwise artifacts only.
- Kept the paired cached exact-pool random control mandatory.
- Used cached feasible incident support and revealed effective pairwise evidence
  to downweight or exclude low-reliability CI boundary decisions.
- Raised randomized fallback when unresolved boundary reliability stayed low.
- Used future citation labels only for retrospective metrics, weak-bucket
  deltas, oracle caps, and diagnostics.

20-seed paired posterior top-K result versus cached exact-pool random:

| Metric | V2 delta | 95% normal CI |
|---|---:|---|
| Recall@K | +0.003750 | [-0.013400, +0.020900] |
| nDCG@K | -0.008870 | [-0.024598, +0.006859] |
| AP | -0.013815 | [-0.026494, -0.001136] |

Diagnostics:

- Missing-label caveat: false for both V2 and exact-pool random.
- Randomized fallback rate: 0.600.
- V2 reliability rows: 640; low-reliability fallback row rate: 1.000; mean
  boundary item reliability: 0.171093; mean unresolved fraction: 1.000.
- Compared with original CI partition replay, V2 kept Recall@K unchanged and
  improved nDCG by +0.008483 and AP by +0.010390, but remained below the gate.

Active-arm gate verdict: paid follow-up is blocked. The blocking reason is that
mean Recall@K delta remains below +0.025 and the Recall@K CI is not credibly
positive. The replay-local gate also blocks because positive-negative oracle
recall cap still falls versus exact-pool random.

## New-Information Challenger No-Paid Replay

Implemented the missing no-paid replay/simulator prerequisite for the
new-information challenger construction direction. The workflow made zero paid
Sestina LLM calls, zero pointwise calls, zero paid labeling calls, and no paid
ledger or paid-call artifact rewrites. Known paid spend remains USD 2.746030.

Command:

```bash
uv run python scripts/run_new_information_challenger_simulator.py
uv run python scripts/run_active_arm_gate.py \
  --active-artifact artifacts/backtest-arxiv-new-information-challenger-simulator/new-information-challenger-simulator.json \
  --random-variance-artifact artifacts/backtest-arxiv-full-random-variance-completion/full-random-variance-completion.json \
  --output artifacts/backtest-arxiv-new-information-challenger-simulator/active-arm-gate.json \
  --active-arm new_information_challenger_cached_replay \
  --random-control-arm exact_pool_random_cached_replay
```

Artifacts:

- `artifacts/backtest-arxiv-new-information-challenger-simulator/new-information-challenger-simulator.json`
- `artifacts/backtest-arxiv-new-information-challenger-simulator/active-arm-gate.json`

Method:

- Reused existing cached/reviewed pointwise and pairwise artifacts only.
- Kept the paired cached exact-pool random control mandatory.
- Used pointwise rubric residuals, uncertainty, lexical novelty, and metadata
  diversity to expose model-visible possible pointwise false negatives.
- Restricted replay pairs to cached labels to avoid paid labeling and missing
  labels; budget completeness is reported separately from label completeness.
- Used future citation labels only for retrospective metrics, weak-bucket
  deltas, oracle caps, and diagnostics.

20-seed paired posterior top-K result versus cached exact-pool random:

| Metric | New-information delta | 95% normal CI |
|---|---:|---|
| Recall@K | +0.026250 | [+0.010165, +0.042335] |
| nDCG@K | +0.024842 | [+0.010131, +0.039552] |
| AP | +0.002569 | [-0.006515, +0.011653] |

Diagnostics:

- Missing-label caveat: false for both new-information and exact-pool random.
- Budget-completeness caveat: true for the active arm. The
  `arxiv_cs_AI_2023_01_historical_citation_pilot` row schedules 16 of the
  resolved 20 active comparisons for each of the 20 seeds, an active shortfall
  of 80 comparisons. The exact-pool random control has no shortfall.
- Randomized floor rate: 0.198438.
- Mean scheduled challenger count: 7.300 per seed/bucket row.
- Mean challenger rubric residual: 0.038389; mean lexical novelty: 0.955456.
- Weak-bucket selected-positive delta total: +21.
- Unique future positives touched fell from 447 for cached exact-pool random to
  412 for the new-information arm.
- Weak-bucket pointwise-plus-touched and positive-negative oracle cap deltas are
  -0.051250 and -0.043750.

Active-arm gate verdict: paid follow-up is blocked. Paired Recall@K is
credibly positive and the mean Recall@K delta exceeds +0.025 with nonnegative
nDCG/AP, but the budget-completeness caveat is blocking because the active
schedule does not fill the resolved per-row pairwise budget in 20 rows. The
replay-local false-negative diagnostic also blocks because weak-bucket oracle
headroom fell versus exact-pool random. Do not buy labels for this arm until a
later no-paid replay fills the schedule with a predeclared no-future-label
cached fallback or explicitly scopes a paid follow-up to close this shortfall.

## New-Information Budget-Fill Replay

Filled the active shortfall from the prior no-paid replay with a predeclared
cached frontier fallback. The fallback uses only model-visible or
schedule-available information: pointwise probability, uncertainty, rubric
scores, title/abstract lexical novelty, metadata diversity, and cache-key
availability. Future citation labels and cached pairwise label values are used
only after scheduling for retrospective metrics and diagnostics. The workflow
made zero paid Sestina LLM calls, zero pointwise calls, zero paid labeling
calls, and no paid ledger or paid-call artifact rewrites. Known paid spend
remains USD 2.746030.

Command:

```bash
uv run python scripts/run_new_information_challenger_simulator.py \
  --output artifacts/backtest-arxiv-new-information-budget-fill-gate/new-information-budget-fill-gate.json \
  --active-gate-output artifacts/backtest-arxiv-new-information-budget-fill-gate/active-arm-gate.json
uv run python scripts/run_active_arm_gate.py \
  --active-artifact artifacts/backtest-arxiv-new-information-budget-fill-gate/new-information-budget-fill-gate.json \
  --random-variance-artifact artifacts/backtest-arxiv-full-random-variance-completion/full-random-variance-completion.json \
  --output artifacts/backtest-arxiv-new-information-budget-fill-gate/active-arm-gate.json \
  --active-arm new_information_challenger_cached_replay \
  --random-control-arm exact_pool_random_cached_replay
```

Artifacts:

- `artifacts/backtest-arxiv-new-information-budget-fill-gate/new-information-budget-fill-gate.json`
- `artifacts/backtest-arxiv-new-information-budget-fill-gate/active-arm-gate.json`

Budget-fill diagnostics:

- Primary active budget shortfall before fallback: 80 comparisons across 20
  seed/bucket rows.
- Fallback completed shortfall: 80 comparisons; remaining active shortfall: 0.
- Random-control shortfall: 0.
- Missing-label caveat: false for both arms.
- Reviewed active-arm gate budget-completeness caveat: false.

20-seed paired posterior top-K result versus cached exact-pool random:

| Metric | Budget-filled new-information delta | 95% normal CI |
|---|---:|---|
| Recall@K | +0.028750 | [+0.011972, +0.045528] |
| nDCG@K | +0.026204 | [+0.011096, +0.041313] |
| AP | +0.002436 | [-0.006659, +0.011532] |

Aggregate active metrics are Recall@K 0.338750, nDCG@K 0.369806, and AP
0.363194. The paired exact-pool random cached replay remains Recall@K 0.310000,
nDCG@K 0.343602, and AP 0.360758. Compared with the prior incomplete
new-information replay, the budget-filled replay changes active Recall@K by
+0.002500, nDCG@K by +0.001363, and AP by -0.000133 while removing the
80-comparison active shortfall.

The reviewed active-arm gate passes on the completed no-paid artifact:
`paid_followup_allowed` is true, paired Recall@K is credibly positive, mean
Recall@K delta exceeds +0.025 with nonnegative nDCG/AP, paired seed count is 20,
the completed full-random reference is available, no missing-label caveat is
present, and no budget-completeness caveat remains. The later paid-workflow
dry-run below still recommends no-go because the replay-local false-negative
diagnostic blocks on weak-bucket oracle headroom.

## New-Information Paid Dry-Run Gate

This is the final no-paid planning gate for the budget-filled new-information
challenger. It made zero paid Sestina LLM calls, zero pointwise calls, and did
not create or append the planned JSONL ledger. Known paid spend remains USD
2.746030.

Command:

```bash
uv run python scripts/run_new_information_paid_dry_run.py
```

Artifacts:

- `artifacts/backtest-arxiv-new-information-paid-dry-run/paid-dry-run-go-no-go.json`
- `artifacts/backtest-arxiv-new-information-paid-dry-run/planned-pair-occurrences.jsonl`

Frozen inputs:

- Seeds: 17, 101, 211, 307, 401, 503, 607, 709, 811, 907, 1009, 1103,
  1201, 1301, 1409, 1511, 1601, 1709, 1801, 1901.
- Rows: the 8 pilot buckets, each with K=5 and a resolved 20-pair budget per
  seed/bucket row.
- Active arm: `new_information_challenger_cached_replay`.
- Comparator: `exact_pool_random_cached_replay`.
- Pairwise model: `openai/gpt-5.4-mini`; provider-prefix validation passed,
  and model availability is marked required before any paid call.
- Planned artifact directory:
  `artifacts/backtest-arxiv-new-information-paid-dry-run`.
- Planned ledger:
  `artifacts/backtest-arxiv-new-information-paid-dry-run/ledger.jsonl`.

Dry-run estimate:

- Planned pair occurrences: 3,200.
- Unique same-bucket canonical pair labels: 269.
- Cached/reused occurrences: 3,200.
- Unique missing pairwise labels: 0.
- Estimated additional spend: USD 0.000000.
- Active shortfall: 0; random-control shortfall: 0; missing-label caveat:
  false; planned pointwise calls: 0.

Go/no-go: no-go. The unresolved caveat is the replay-local false-negative
diagnostic: weak-bucket pointwise-plus-touched cap delta is -0.051250,
positive-negative-pair cap delta is -0.043750, and unique future positives
touched delta is -35 versus cached exact-pool random. The guardrail gap is that
no reviewed guarded 20-seed paid runner is wired to execute this frozen
new-information manifest. Required fix before any paid execution: resolve or
explicitly reviewer-accept the weak-bucket caveat, then add a guarded
pairwise-only runner with provider model availability check, JSONL ledger, hard
cap, separate artifact directory, and an abort on any pointwise-call attempt.
That follow-up runner go/no-go now exists below and clears the runner guardrail
only for this exact zero-missing frozen manifest.

## New-Information Weak-Bucket Caveat Adjudication

The replay-local weak-bucket oracle-headroom caveat has now been adjudicated in
a no-paid workflow. The adjudication made zero paid Sestina LLM calls, zero
pointwise calls, zero paid labeling calls, and no paid ledger or paid-call
artifact rewrites. It used future citation labels only for retrospective
diagnostics after the frozen schedules.

Command:

```bash
uv run python scripts/adjudicate_new_information_caveat.py
```

Artifact:

- `artifacts/backtest-arxiv-new-information-caveat-adjudication/caveat-adjudication.json`

Decision: `caveat_accepted_with_constraints`.

Reviewer-auditable rationale:

- The reviewed active-arm gate remains intact and passes: Recall@K delta
  +0.028750 with CI [+0.011972, +0.045528], nDCG@K delta +0.026204, AP delta
  +0.002436, 20 seeds, no missing-label caveat, and no budget-completeness
  caveat.
- The weak-bucket caveat is real: pointwise-plus-touched cap is -41 selected
  positives in count-equivalent terms, positive-negative-pair cap is -35, and
  unique future-positive touch occurrences are -35.
- The lost-touch positives are real active false negatives, not a metric
  artifact: 49 lost-touch occurrences are no longer selected by the
  new-information posterior and all 49 are zero-degree active false negatives.
- The accepted-risk reason is that the top-K objective still improves:
  selected future-positive occurrences rise by +23 overall, 40 rows gain
  selected positives despite no unique-touch gain, and 23 rows improve nDCG with
  the same selected-positive count.
- Fallback comparisons do not explain the weak-bucket loss. The 20 fallback
  rows, all in `arxiv_cs_AI_2023_01_historical_citation_pilot`, have selected
  positive delta +8 and touch delta 0. Nonfallback rows still have selected
  positive delta +15 while carrying the full -35 touch delta.
- Prior complete-arm diagnostics do not support using oracle headroom as a hard
  override by itself: in the one-seed random-control diagnosis, Spearman
  correlation between Recall@K and pointwise-plus-touched cap is +0.187867, and
  several arms have high oracle caps but lower Recall@K than exact-pool random.

Constraints:

- Acceptance is scoped only to the frozen budget-filled new-information
  manifest and the current reviewed artifacts.
- Acceptance does not authorize paid label purchase; the dry-run found zero
  unique missing labels and USD 0.000000 estimated additional spend.
- Do not weaken or bypass the reviewed active-arm gate.
- Any later workflow tied to this manifest must remain pairwise-only, make zero
  pointwise calls, use no future labels for scheduling/model-visible selection,
  and preserve the JSONL ledger, hard cap, separate artifact directory, provider
  model availability check, and abort-on-pointwise-call guardrails.
- If a later manifest has missing labels or changes the schedule, rerun the
  dry-run and this caveat adjudication before buying labels.
- The guarded runner go/no-go below clears the runner-infrastructure guardrail
  only for this exact frozen zero-missing manifest. It does not authorize paid
  label purchase.

Recommended next workflow: do not run paid calls now. Treat the weak-bucket
oracle-headroom caveat as an accepted, documented risk for this frozen
zero-missing-label manifest only. A later reviewed execution may use the guarded
runner path below as cache-only, zero-spend validation of the frozen manifest.

## New-Information Guarded Runner Go/No-Go

This no-paid workflow adds the guarded pairwise-only runner infrastructure for
the frozen budget-filled new-information manifest. It ran in planning mode only:
zero paid Sestina LLM calls, zero pointwise calls, zero paid labeling calls, and
no historical paid ledger or paid-call artifact rewrites.

Command:

```bash
uv run python scripts/run_new_information_guarded_runner.py
```

Artifacts:

- `artifacts/backtest-arxiv-new-information-guarded-runner/guarded-runner-go-no-go.json`
- `artifacts/backtest-arxiv-new-information-guarded-runner/guarded-runner-ledger.jsonl`

Go/no-go: `go` for later reviewed execution of this exact frozen manifest, with
`runner_ready_for_later_execution=true`. The expected execution mode is
`cache_only_zero_spend`: all 3,200 planned pair occurrences and 269 unique
same-bucket pair labels are cached, unique missing labels are 0, pairwise calls
to buy are 0, estimated additional spend is USD 0.000000, and the JSONL ledger
is a separate empty file under the guarded-runner artifact directory.

Guardrails validated:

- Frozen dry-run, planned-pair manifest, budget-fill artifact, active-gate
  artifact, and caveat adjudication hashes and shape match the reviewed inputs.
- The reviewed active-arm gate is not weakened or bypassed.
- The caveat acceptance is scoped to the current frozen zero-missing-label
  manifest and current reviewed artifacts.
- Planned rows are pairwise-only, future labels are not used for scheduling or
  model-visible selection, and cached label values are not used before
  scheduling.
- The pairwise model is provider-prefixed (`openai/gpt-5.4-mini`), model
  availability is required before any future execution call, the ledger is
  JSONL, the artifact directory is separate, the hard cap recommendation is USD
  0.01, parse retries are recorded, and any pointwise-call attempt aborts.

This artifact does not buy labels and does not authorize paid label purchase.
If a later manifest differs or has missing labels, do not transfer this go
decision; rerun the paid dry-run, caveat adjudication, and guarded runner
go/no-go for the changed manifest.

## New-Information Execution Preflight

This final no-paid preflight was run immediately before any possible execution
handoff. It did not execute the guarded runner, did not call chat/completions,
did not buy labels, made zero paid Sestina LLM labeling calls, and made zero
pointwise calls. It used the configured provider endpoint only for a `/models`
availability check.

Command:

```bash
uv run python scripts/run_new_information_execution_preflight.py
```

Artifact:

- `artifacts/backtest-arxiv-new-information-execution-preflight/execution-preflight-go-no-go.json`

Preflight decision: `go` for a later reviewed execution workflow of this exact
frozen manifest only. Provider/model availability is now closed:
`openai/gpt-5.4-mini` is available. The preflight found no frozen-manifest
drift, revalidated the reviewed active-arm gate and caveat scope, confirmed the
plan is pairwise-only with zero pointwise rows, confirmed zero unique missing
pairwise labels, and confirmed expected `cache_only_zero_spend` behavior.

Handoff constraints:

- Later execution cap: USD 0.01.
- Expected paid calls: 0.
- Expected pointwise calls: 0.
- Expected additional spend: USD 0.000000.
- Guarded ledger remains
  `artifacts/backtest-arxiv-new-information-guarded-runner/guarded-runner-ledger.jsonl`
  with 0 lines and USD 0.000000 existing spend.
- This preflight authorizes no paid label purchase. If the manifest, schedule,
  cache status, model, caveat scope, or guarded-runner state changes, rerun the
  paid dry-run, caveat adjudication, guarded runner go/no-go, and this
  preflight before any paid labeling.

## New-Information Guarded Execution

This reviewed execution used the guarded runner in execute mode for the exact
frozen budget-filled new-information manifest cleared by the preflight above.
The run preserved the provider-prefixed pairwise model
`openai/gpt-5.4-mini`, the USD 0.01 hard cap, the pairwise-only guard, the
abort-on-pointwise guard, and the separate execution artifact directory.

Command:

```bash
uv run python scripts/run_new_information_guarded_runner.py --mode execute --max-usd 0.01 --confirm-guarded-pairwise-only-execution --artifact-dir artifacts/backtest-arxiv-new-information-guarded-execution --ledger artifacts/backtest-arxiv-new-information-guarded-execution/guarded-execution-ledger.jsonl --output artifacts/backtest-arxiv-new-information-guarded-execution/guarded-execution-go-no-go.json
```

Artifacts:

- `artifacts/backtest-arxiv-new-information-guarded-execution/guarded-execution-go-no-go.json`
- `artifacts/backtest-arxiv-new-information-guarded-execution/guarded-execution-ledger.jsonl`

Execution result: `go`, `mode=execute`, `dry_run=false`, and
`cache_only_zero_spend`. The provider model availability check reports
`available` for `openai/gpt-5.4-mini`. All 3,200 planned pair occurrences were
cache reuses; all 269 unique planned pair labels were already cached; unique
missing pairwise labels were 0; pairwise calls to buy were 0; pointwise-like
planned rows were 0; paid calls made were 0; pointwise calls made were 0; and
paid spend was USD 0.000000. The guarded-execution JSONL ledger has 0 lines,
0 new entries, and USD 0.000000 spend before and after the invocation.

Recommendation: treat this as the cache-only execution handoff for the current
frozen zero-missing-label manifest only. It buys no labels and authorizes no
paid label purchase. If the manifest, cache state, model, caveat scope, runner,
or gate artifacts drift, rerun the dry-run, caveat adjudication, guarded runner
go/no-go, execution preflight, and guarded execution before any paid labeling.

## Current Next Question

The experiment campaign's current next question is no longer “which Sestina
active arm should buy labels next?” The answered decision for this branch is:
stop experiments, preserve the current best internal result, and move to
PR/publication cleanup. Any future algorithmic work should start as a separate
no-paid design gate with a fresh protocol.

Use the compact prompt below only for external algorithmic review or future
research planning after the cleanup boundary is decided. It is not approval to
run paid labels, alter the frozen manifest, publish internal artifacts, or
continue this campaign.

```text
We are building Sestina, a pointwise-first, pairwise-light system for finding
top-K high-impact arXiv papers. In an 8-bucket historical arXiv pilot
(634 papers, 40 future-citation top-K positives), pointwise-only beat random
baselines, and pairwise refinement beat pointwise-only. However, the original
active pair scheduler did not beat a random pairwise control at the same budget.

Initial metrics: pointwise-only Recall@K 0.300, nDCG@K 0.3396, AP 0.3565;
random pairwise Recall@K 0.350, nDCG@K 0.3938, AP 0.4011; original active
pairwise Recall@K 0.325, nDCG@K 0.3719, AP 0.3985. Diagnostics showed
candidate recall was only 25/40 positives, the original active schedule used
160/160 candidate-internal pairs, 0 sentinel/outsider pairs, and diversity was
effectively disabled by metadata bucketing.

We tried scheduler and posterior variants: quota-based scheduler-only follow-up,
posterior top-K EVSI, exact-pool random, cache-aware sequential EVSI, CCTD-GF,
expanded-pool random, targeted-outsider random, degree-aware posterior shrinkage,
and soft-probability pairwise-strength calibration. The useful recurring lesson
was that posterior top-K aggregation is worth keeping, but active acquisition
heuristics over the same information surface did not reliably beat random or
exact-pool random controls. Simple posterior shrinkage and soft-strength
calibration did not improve Recall@K on complete-label arms.

We then completed the full 20-seed historical-random and exact-pool-random
variance baseline with a guarded pairwise-only labeling pass. The paid completion
made 1,724 successful pairwise labels plus 3 preserved parse-error retry
attempts, 0 pointwise calls, and cost USD 1.269345. Complete full-schedule
posterior top-K means were: historical random Recall@K 0.3325, nDCG@K 0.3666,
AP 0.3689; exact-pool random Recall@K 0.3225, nDCG@K 0.3628, AP 0.3737. The
paired historical-minus-exact intervals crossed zero for Recall@K, nDCG@K, and
AP, so paired seed-level random controls are mandatory.

The best internal active result is the budget-filled new-information challenger
cached replay. It uses pointwise rubric residuals, uncertainty, lexical novelty,
and metadata diversity to expose possible pointwise false negatives, then fills
the earlier 80-comparison active shortfall with a predeclared cached frontier
fallback that does not use future labels, citation outcomes, or cached label
values for scheduling. Against cached exact-pool random over 20 paired seeds and
8 buckets, it reaches Recall@K 0.338750 versus 0.310000 (delta +0.028750, 95%
CI [+0.011972, +0.045528]), nDCG@K 0.369806 versus 0.343602 (delta +0.026204,
CI [+0.011096, +0.041313]), and AP 0.363194 versus 0.360758 (delta +0.002436,
CI [-0.006659, +0.011532]). The credible positive claim is Recall@K/nDCG, not
AP.

This best result has important caveats. It is a reviewed cache-only
replay/execution result, not a fresh paid-label validation. The replay-local
weak-bucket oracle-headroom caveat is accepted only for the exact frozen
zero-missing-label manifest and current reviewed artifacts. The final guarded
execution enumerated 3,200 planned pair occurrences and 269 unique same-bucket
pair labels; all were cached, 0 labels were missing, 0 pairwise calls were
bought, 0 pointwise calls were made, the separate execution ledger stayed empty,
and additional spend was USD 0.000000. No artifact authorizes paid label
purchase. If the manifest, cache state, model, runner, gate artifacts, or caveat
scope changes, rerun the paid dry-run, caveat adjudication, guarded runner
go/no-go, execution preflight, and guarded execution before making any paid-run
claim.

Campaign decision: stop Sestina experiments on this branch. Move to
PR/publication cleanup: isolate coherent source/test/doc changes, keep raw
artifacts, ledgers, stdout JSON, planned pair manifests, dataset manifests, and
.codex-workflows records internal unless separately scrubbed/reviewed, and write
any public claim as “budget-filled cached replay plus guarded cache-only
execution” with paired uncertainty and caveats.

Future research question, for a new no-paid protocol rather than this branch:
what low-pairwise-budget design could beat paired random/exact-pool random
without relying on full ranking and without reusing the same failed information
surface? Promising directions likely need a stronger pairwise-label reliability
model, a candidate-construction signal that improves weak-bucket positive
exposure, or a different evaluation design; more acquisition-score micro-tweaks
over the current cached pool are not justified by this campaign.
```
