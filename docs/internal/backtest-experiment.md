# Sestina Backtest Experiment

## Objective

Validate whether Sestina improves top-K good-paper discovery over the current
semantic-ranking style baseline under a hard USD 100 LLM budget.

The first pass is abstract/summary-level only. Full-text reading is reserved for
top-candidate audit or future work, because full text would spend the budget on
context length before the ranking comparison is statistically useful.

No live backtest should run without a dry-run estimate, a `--max-usd` cap, and a
JSON ledger artifact. This repository currently ships only the dry-run estimator.

## Hypotheses

- H1: Sestina active pairwise improves recall@K and nDCG@K over the semantic
  baseline on historical buckets with known accepted/selected good papers.
- H2: Sestina active pairwise improves over pointwise-only ranking, especially
  near the K boundary where pointwise probabilities are close or uncertain.
- H3: Active pair scheduling beats a pointwise + random-pairwise strategy at the
  same pairwise budget.
- H4: Active pairwise lift remains visible at smaller pairwise budget prefixes,
  not only after spending the full default `B_pair`.
- H5: The improvement is robust across topic, venue, and time buckets, not only
  on one cherry-picked subset.

## Dataset And Bucket Selection

Use historical paper pools where a final "good paper" outcome is already known.
Each bucket should be a realistic retrieval set that the existing semantic
baseline would have ranked at the time.

Recommended bucket sources:

- Topic-month buckets: a query/topic over a fixed month or quarter.
- Venue-year buckets: a venue or conference track over a year.
- Recoleta-style semantic pools: archived candidate sets from prior triage runs,
  if they include enough source metadata to reconstruct the baseline score.

Bucket rules:

- Target 40-50 buckets for the main comparison after smoke and pilot validation.
- Keep each first-pass bucket in the 60-300 paper range.
- Use K values matching the real user workflow, usually 5, 10, 12, or 15.
- Freeze bucket membership, paper text, labels, prompt versions, model names,
  and rate assumptions before each paid phase.
- Exclude buckets with fewer than K label-positive papers unless the analysis is
  explicitly recall@available-positive.

For each bucket with `n` papers and target `K`, Sestina uses the existing
formulae:

```text
M = min(n, ceil(3K + sqrt(n)))
B_pair = min(ceil(1.25M), ceil(0.25n))
```

### Historical arXiv Citation Buckets

The historical arXiv builder creates buckets directly from arXiv category/date
windows and labels later impact with public scholarly citation metadata:

```bash
.venv/bin/python scripts/build_arxiv_historical_manifest.py \
  --category cs.LG \
  --month 2023-01 \
  --limit 80 \
  --k 5 \
  --metadata-provider auto \
  --unmatched-policy drop \
  --min-papers 60 \
  --output artifacts/backtest-datasets/arxiv-historical-smoke-manifest.json
```

Data path:

- Paper discovery uses the arXiv API with a category filter and submitted-date
  window. The model-visible paper evidence is limited to title, abstract,
  primary category, category list, published/updated dates, and bucket dates.
- Metadata matching uses Semantic Scholar by arXiv ID or DOI, with OpenAlex as
  the `auto` fallback by DOI/title search. Tests use static fixtures only and
  must not call live APIs.
- Citation counts are stored under `papers[].labels`, not
  `papers[].metadata`, so the runner does not include future citation evidence
  in pointwise or pairwise prompts.
- Manifest paper IDs are anonymized per bucket. Raw arXiv IDs and matched work
  IDs are retained only in `labels` for auditability.
- `good_paper` is assigned to the top-K cited papers within the bucket by
  default, or to `ceil(alpha * n)` papers when `--top-alpha` is supplied.
- The manifest preserves arXiv chronological bucket order after assigning
  citation ranks, avoiding an order leak through equal baseline scores.
- `baseline_score` is a constant `0.5` neutral baseline. It is not a citation
  proxy and should not be interpreted as an oracle or Recoleta-derived ranking.

Smoke quality checks before any paid run:

- At least K positive labels are present and every positive is derived from the
  public citation metadata payload.
- `source.diagnostics.papers_in_manifest` is large enough for the smoke bucket,
  normally 60-100 papers after unmatched-paper handling.
- No `citation_count`, `citation_rank`, DOI, arXiv ID, or matched work ID is in
  `papers[].metadata`.
- The runner dry-run summary and cost estimate exist before `--confirm-paid` is
  used.

## Labels And Success Criteria

Primary label:

- `good_paper = 1` when the paper was accepted, selected, cited in the final
  evidence set, or otherwise recorded as a real positive by the historical
  workflow.

Secondary labels:

- `strong_positive` for the highest-confidence positives, used in audit.
- `near_miss` for papers reviewed but not selected, used to inspect false
  positives and boundary behavior.

Primary metrics:

- recall@K
- nDCG@K
- average precision
- pairwise-budget ablation curve for recall@K, nDCG@K, and average precision

Secondary metrics:

- precision@K
- Brier score for calibrated pointwise probabilities
- near-miss positive rate just below K
- win rate by bucket against the semantic baseline

Success threshold:

- Ship to a larger live evaluation only if Sestina active pairwise improves
  mean recall@K and nDCG@K over the semantic baseline, improves over
  pointwise-only, and wins on at least 60% of non-smoke buckets.
- Treat the result as inconclusive if the mean lift is positive but dominated by
  one or two buckets.
- Reject or redesign if active pairwise does not beat random pairwise at the
  same pairwise budget.
- Treat active pairwise as budget-sensitive if lift appears only at the full
  `B_pair` point and disappears at the smaller `K` or `K + sqrt(n)` prefixes.

## Strategy Matrix

| Strategy | Purpose | LLM calls counted by estimator | Notes |
|---|---:|---:|---|
| Random | Lower bound | 0 | Shuffle bucket papers with fixed seeds; run multiple seeds for confidence intervals. |
| Semantic baseline | Current system baseline | 0 | Use archived semantic score or re-run the existing semantic ranker outside the LLM budget if needed. |
| Pointwise-only | Measures value of abstract-level LLM scoring alone | `n` pointwise calls per bucket | Shared pointwise pass is reused by the pairwise strategies. |
| Pointwise + random pairwise | Controls for pairwise-call count without active scheduling | `B_pair` pairwise calls per bucket | Pairs sampled from Sestina candidates with fixed seeds. |
| Sestina active pairwise | Treatment | `B_pair` pairwise calls per bucket | Uses Sestina candidate selection and active pair scheduling near the top-K boundary. |

The estimator intentionally counts pointwise scoring once per bucket, not once
per strategy arm, because the pointwise outputs are shared by pointwise-only,
random-pairwise, and active-pairwise arms. It counts random-pairwise and
active-pairwise comparisons separately because they are different pair schedules.

## Pairwise Budget Ablation

Budget ablation is evaluated as a prefix analysis over each already scheduled
pairwise strategy. The runner should first schedule the full random-pairwise and
active-pairwise lists up to `B_pair`, judge those pairs once, then recompute the
strategy metrics after consuming only the first `b` judged pairs from each list.
This keeps the default ablation at zero extra pairwise LLM calls.

Required ablation points per bucket:

| Point | Pairwise calls per pairwise strategy | Interpretation |
|---|---:|---|
| `0` | 0 | No pairwise judgments; equivalent to pointwise-only ranking for pairwise arms. |
| `K` | `K` | Minimal top-K-sized refinement budget. |
| `K + sqrt(n)` | `ceil(K + sqrt(n))` | Boundary-light budget that grows slowly with bucket size. |
| `B_pair` | `min(ceil(1.25M), ceil(0.25n))` | Default full pairwise budget, including the `0.25n` cap when it binds. |

If a point exceeds `B_pair`, report it as capped to `B_pair` for that bucket.
Every ablation row must record the resolved numeric point and whether the
default budget was capped by `ceil(0.25n)` or by `ceil(1.25M)`.

If a future design requires non-prefix pair sets or any extra pairwise calls,
those calls must be added to `experiments/backtest_budget_config.json` and
counted by `scripts/estimate_backtest_cost.py` before the paid run is allowed
under the USD 100 cap.

## Token And Cost Model

The dry-run estimator lives at:

```bash
python scripts/estimate_backtest_cost.py \
  --config experiments/backtest_budget_config.json \
  --max-usd 100 \
  --output artifacts/backtest-budget/estimate.json
```

The estimator reports pointwise calls, pairwise calls, audit pairwise calls,
input tokens, output tokens, cost by model, phase allocation, pairwise budget
ablation configuration, resolved per-bucket ablation points, and whether the
estimate exceeds the cap. It never calls an LLM.

Default planning assumptions in `experiments/backtest_budget_config.json`:

- Broad pointwise and pairwise passes use `openai/gpt-5.4-mini` or an equivalent
  cheap model with the provider prefix required by the endpoint.
- The stronger-model audit uses `openai/gpt-5.4` or equivalent only on a small
  sample.
- Model names must be confirmed before paid execution. For OpenAI-routed models
  on this endpoint, use the `openai/` prefix, such as `openai/gpt-5.4-mini`, not
  bare `gpt-5.4-mini`.
- Before any live run, query the endpoint's model list or run an explicit
  low-cost availability probe for every configured model, then freeze the
  verified names in the run manifest.
- Pointwise prompt: 900 input tokens, 220 output tokens.
- Pairwise prompt: 1,500 input tokens, 180 output tokens.
- Audit pairwise prompt: 1,500 input tokens, 220 output tokens.
- Rate-card values are explicit planning estimates. Replace them with the actual
  endpoint rates before any paid run.
- Batch/Flex discounts may be represented with `discount_multiplier`, but the
  multiplier is an estimate and not guaranteed spend.
- Pairwise budget ablation is enabled as `prefix_reuse` at `0`, `K`,
  `K + sqrt(n)`, and `B_pair`; this adds no LLM calls unless the config is
  changed to require additional non-prefix comparisons.

Initial USD 100 allocation:

| Phase | Allocation | Intent |
|---|---:|---|
| Smoke | USD 5 | Validate prompts, JSON parsing, labels, and ledgers on tiny buckets. |
| Pilot | USD 25 | Run several small historical buckets and check metric variance. |
| Main | USD 55 | Run enough buckets to compare all strategies. |
| Audit | USD 10 | Stronger-model re-judge of a small pairwise/label sample. |
| Reserve | USD 5 | Reruns after parser failures, provider failures, or prompt fixes. |

## Phased Run Plan

### 1. Smoke

- Use two small buckets around 60 papers each.
- Run dry-run estimate first and save the JSON artifact.
- Run pointwise prompts on abstracts/summaries only.
- Judge only the scheduled pairwise calls needed for random-pairwise and active
  pairwise.
- Verify that all outputs are valid JSON, no secret values appear in artifacts,
  and the budget ledger matches request counts.

Abort smoke if JSON parse failures exceed 5%, any prompt includes full text by
mistake, or the ledger cannot reconcile calls to saved artifacts.

### 2. Pilot

- Use 8-12 buckets across at least two bucket families.
- Compare all five strategies.
- Produce the pairwise budget ablation table for random-pairwise and active
  pairwise using schedule prefixes.
- Inspect per-bucket deltas, near-miss papers, and active-pair purposes.
- Re-estimate before expanding if prompt token counts differ from assumptions by
  more than 20%.

Proceed only if active pairwise beats pointwise-only and random pairwise on at
least the primary metrics or the failure mode is clearly fixable.

### 3. Main Run

- Use the frozen prompt, model, rate card, and dataset manifest from pilot.
- Run 35-50 total non-smoke buckets if the budget estimate stays below cap.
- Save per-phase ledgers and per-bucket metrics.
- Save the pairwise budget ablation metrics table before aggregate
  interpretation.
- Compute confidence intervals with bootstrap over buckets, not over individual
  papers.

### 4. Audit

- Sample model disagreements, semantic-baseline wins, Sestina wins, and boundary
  near misses.
- Use a stronger model for a small pairwise re-judge sample only if budget
  remains.
- Check whether wins come from real relevance gains or prompt artifacts.

## Abort Criteria And Budget Guardrails

Abort before more paid calls if any condition is met:

- Dry-run estimate exceeds `--max-usd`.
- Actual ledger spend exceeds 80% of the phase allocation before 80% of planned
  buckets finish.
- Paid calls happen without a ledger path.
- More than 5% of responses fail schema validation after one retry.
- Labels are missing, leaky, or discovered to encode the model output.
- A provider/rate-card change would push projected total cost over USD 100.
- Prompt token counts exceed assumptions by more than 20% without a re-estimate.

Live-call runner requirements before implementation:

- Dry-run default.
- Required `--max-usd`.
- Required ledger/artifact path.
- Required model-name validation: every configured model must include a provider
  prefix, and OpenAI models must use `openai/`.
- Required model-availability check before the first paid call in each run.
- Per-call artifact with phase, bucket, model, prompt version, token estimate,
  response status, and cost estimate.
- No environment variable values, API keys, prompts containing full text, or
  sensitive metadata in logs.

`SESTINA_LLM_API_KEY` and `SESTINA_LLM_BASE_URL` are expected from the user's
interactive zsh environment for any future live runner. Scripts must check
presence without printing values.

## Output Artifacts

Required artifacts for a paid run:

- Dataset manifest with bucket IDs, paper IDs, labels, K, and frozen timestamps.
- Prompt version files for pointwise, pairwise, and audit prompts.
- Dry-run estimate JSON from `scripts/estimate_backtest_cost.py`.
- Per-phase spend ledger JSONL.
- Per-bucket strategy predictions and metrics JSON.
- Pairwise budget ablation metrics table, for example
  `pairwise_budget_ablation_metrics.csv`, with columns for phase, bucket ID,
  strategy, ablation label, resolved pairwise budget point, cap source,
  recall@K, nDCG@K, average precision, precision@K, and estimated pairwise spend
  consumed by the prefix.
- Aggregate metrics table with bootstrap confidence intervals.
- Audit sample manifest and audit judgments.
- Final interpretation memo.

## Interpretation Checklist

- Did Sestina active pairwise beat the semantic baseline on recall@K and nDCG@K?
- Did active pairwise beat pointwise-only and random pairwise at the same
  pairwise budget?
- How much lift remains at pairwise budget points `0`, `K`,
  `K + sqrt(n)`, and `B_pair`?
- Is the active-pairwise lift robust before the `0.25n` cap is fully spent?
- Are gains distributed across buckets, topics, and K values?
- Are semantic-baseline wins explainable by missing abstracts, weak pointwise
  prompts, or label mismatch?
- Did pairwise calls concentrate near the top-K boundary as intended?
- Did the stronger-model audit agree with the cheap-model pairwise judgments?
- Was total estimated and actual spend below USD 100?
- Are prompt, model, rate-card, dataset, and seed versions sufficient for rerun?
