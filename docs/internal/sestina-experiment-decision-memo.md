# Sestina Experiment Decision Memo

Date: 2026-05-02

Workflow: `sestina-experiment-consolidation`

Status: internal decision memo and repo-hygiene handoff. This consolidation made
zero paid Sestina LLM calls.

Detailed evidence log:
`docs/internal/historical-arxiv-pilot-results.md`

Related-work audit:
`docs/internal/related-work-audit.md`

Final internal handoff:
`docs/internal/sestina-final-results-handoff.md`

Next experiment protocol:
`docs/internal/sestina-next-experiment-protocol.md`

Machine-readable consolidation:
`artifacts/backtest-arxiv-experiment-consolidation/consolidation-summary.json`

Caveat adjudication:
`artifacts/backtest-arxiv-new-information-caveat-adjudication/caveat-adjudication.json`

Guarded runner go/no-go:
`artifacts/backtest-arxiv-new-information-guarded-runner/guarded-runner-go-no-go.json`

Execution preflight:
`artifacts/backtest-arxiv-new-information-execution-preflight/execution-preflight-go-no-go.json`

Guarded execution:
`artifacts/backtest-arxiv-new-information-guarded-execution/guarded-execution-go-no-go.json`

## Decision

Stop spending on random-baseline completion. The completed 20-seed
historical-random and exact-pool-random artifact is now the random variance
reference for future active-arm comparisons.

Do not scale the current paid active arms. The reviewed active, candidate,
posterior-layer, and CI-partition attempts are negative or inconclusive under
the current low-budget arXiv pilot. A future paid active arm must first pass a
no-paid replay/simulation gate against paired random controls.

The new-information challenger simulator now has a budget-filled no-paid replay.
A predeclared cached frontier fallback filled the prior 80-comparison active
shortfall without using future labels, citation outcomes, or cached label values
for scheduling. The reviewed active-arm gate passes on the completed artifact.
The no-paid paid-workflow dry-run froze 3,200 planned pair occurrences across
the 20 seeds and 8 buckets, found all 269 unique planned pair labels already
reusable from reviewed caches, estimated USD 0.000000 additional spend, and kept
pointwise calls at zero. A separate no-paid adjudication accepts the replay-local
weak-bucket oracle-headroom caveat as a scoped, reviewer-auditable risk for this
frozen zero-missing-label manifest. The guarded runner go/no-go now validates a
pairwise-only runner path and reports `runner_ready_for_later_execution=true`
for cache-only, zero-spend execution of this exact frozen manifest. The final
execution preflight closes the remaining provider-model gap:
`openai/gpt-5.4-mini` is available via a `/models` check, no frozen-manifest
drift was found, and the later handoff cap is USD 0.01. The reviewed guarded
execution then ran for the same frozen manifest with `--mode execute`,
`--max-usd 0.01`, and the provider-prefixed model. It reused cache only, wrote
the separate guarded-execution JSONL ledger with 0 entries, made zero paid label
calls, made zero pointwise calls, and added USD 0.000000 spend. This does not
authorize paid label purchase; if the manifest changes or has missing labels,
rerun the dry-run, caveat adjudication, guarded runner go/no-go, execution
preflight, and guarded execution before buying anything.

Use random or exact-pool random plus posterior top-K as the mandatory
small-budget baseline. The seed-17 Recall@K 0.375 random reference is not stable
enough to stand alone.

## Current Evidence And Spend

The pilot covers 8 historical arXiv category-month buckets, 634 papers, and 40
future-citation top-K positives. Future citation labels are used only for
retrospective evaluation and diagnostics, not for scheduling or model-visible
features.

Known paid spend is USD 2.746030 of the USD 100 cap:

| Spend component | USD |
|---|---:|
| Known spend before full-random completion | 1.476685 |
| Full-random variance completion | 1.269345 |
| Total known spend | 2.746030 |
| Remaining cap | 97.253970 |

The full-random completion was pairwise-only: 1,727 ledger entries, 1,724 `ok`,
3 preserved `parse_error` retry attempts, zero pointwise-like entries, and USD
1.269345 spend under the USD 5 cap.

## What Failed

The repeated pattern is that active or more targeted schedules create exposure,
degree, or cleaner-looking pair pools, but they do not move enough future
positives into the final posterior top-K.

| Workflow / arm | Status | Spend | Headline posterior top-K result | Interpretation |
|---|---|---:|---|---|
| Original/revised active scheduling | Negative | paid in earlier runs | Complete active variants recover 13/40 or 14/40 positives, while random/exact-pool random recover 15/40 | Active degree and exposure did not become top-K movement. |
| Sequential EVSI | Negative | USD 0.063945 | Recall@K 0.325, nDCG@K 0.365254, AP 0.383471 | Adaptive refits did not fix EVSI acquisition. |
| CCTD-GF | Negative | USD 0.083790 | Recall@K 0.325, nDCG@K 0.367089, AP 0.375631 | Graph floor and sampled disagreement did not beat exact-pool random. |
| Expanded-pool random | Negative/inconclusive | USD 0.109515 | Recall@K 0.325, nDCG@K 0.374246, AP 0.377321 | Naive pool widening touched more papers but diluted useful labels. |
| Targeted-outsider random | Negative/inconclusive | USD 0.111720 | Recall@K 0.325, nDCG@K 0.371886, AP 0.386160 | Better outsider mechanics and AP did not recover top-K hits. |
| Degree-aware posterior shrinkage | Negative/inconclusive | USD 0.000000 | No complete-label arm improved Recall@K | Simple shrinkage toward pointwise priors is not enough. |
| Pairwise soft-strength calibration | Negative/inconclusive | USD 0.000000 | No complete-label arm improved Recall@K | Soft-probability downweighting is diagnostic only, not a default. |
| CI partition replay gate | Blocked paid follow-up | USD 0.000000 | CI replay Recall@K 0.313750, nDCG@K 0.326249, AP 0.336553 versus cached exact-pool Recall@K 0.310000, nDCG@K 0.343602, AP 0.360758 | Recall gain was too small and nDCG/AP dropped; current intervals remain too wide. |
| New-information challenger budget-filled replay | Caveat accepted; guarded execution delivered cache-only zero-spend | USD 0.000000 | Recall@K 0.338750, nDCG@K 0.369806, AP 0.363194 versus cached exact-pool Recall@K 0.310000, nDCG@K 0.343602, AP 0.360758 | Predeclared cached frontier fallback filled 80/80 active shortfall comparisons and the reviewed active-arm gate passes. The replay-local weak-bucket oracle-headroom caveat is accepted with constraints for the frozen zero-missing-label manifest. The guarded runner, final execution-preflight, and guarded-execution artifacts clear the runner/model-availability guardrails for this exact manifest, but they buy no labels and authorize no paid label purchase. |

The random-control diagnosis explains the failure mode: historical random and
exact-pool random recover 15/40 positives, while complete active arms recover
13/40 or 14/40. Active policies often touch future positives and raise degree
around plausible top-K nodes, but the evidence does not promote the right false
negatives in the right buckets.

The related-work audit supports this conservative reading. Strong random
controls are standard, low-budget active gains are assumption-sensitive, and
sparse noisy pairwise graphs can make posterior or BT/PL-style tweaks
insufficient. The literature does not prove active scheduling is hopeless; it
does say Sestina needs stronger assumptions or a better no-paid gate before
buying more active labels.

## Strongest Baseline

The strongest baseline is the completed 20-seed random variance reference using
posterior top-K aggregation:

| Arm | Recall@K mean | Recall@K 95% CI | nDCG@K mean | nDCG@K 95% CI | AP mean | AP 95% CI |
|---|---:|---|---:|---|---:|---|
| Historical random full schedule | 0.332500 | [0.320000, 0.345000] | 0.366567 | [0.355858, 0.376360] | 0.368876 | [0.359224, 0.376462] |
| Exact-pool random full schedule | 0.322500 | [0.310000, 0.336250] | 0.362799 | [0.352992, 0.373035] | 0.373689 | [0.369669, 0.377526] |

Paired historical-minus-exact deltas:

| Metric | Mean delta | 95% CI |
|---|---:|---|
| Recall@K | +0.010000 | [-0.010000, +0.028750] |
| nDCG@K | +0.003768 | [-0.010960, +0.018203] |
| AP | -0.004813 | [-0.014868, +0.004117] |

Use historical random when the question is broad global random coverage. Use
exact-pool random when the question is whether an active acquisition policy
beats random sampling from the same feasible proposal pool. Treat them as
comparable random-control variants at this sample size because all paired delta
intervals cross zero.

## Full-Random Variance Implications

The seed-17 Recall@K 0.375 reference was high relative to the complete 20-seed
means. It remains useful as a historical row, but not as a stable point estimate
or as the only comparator.

A one-positive swing is 0.025 mean Recall@K in this 40-positive pilot. That is
the same order as several active-vs-random differences. Future claims need
paired seed-level uncertainty, not a single seed or an unpaired comparison.

The completed random artifact is sufficient for baseline robustness. Spending
more on random baselines would be lower value than improving the active-arm
design gate.

## Hard Protocol For Future Active Arms

2026-05-07 operationalization: the machine-readable gate/protocol contract now
lives in `sestina.active_arm_gate`, `sestina.experiment_protocol`, and
`scripts/validate_next_experiment_protocol.py`. It preserves the stop decision:
the current best result is cached/no-paid internal evidence, not fresh
validation.

Before any paid labels:

- Predeclare the active policy, feasible proposal pool, random-control variant,
  seed set, metrics, and stopping rule.
- Run a no-paid replay/simulation gate over cached labels and the completed
  random variance artifact.
- Require at least 20 seeds over the 8 buckets unless a smaller diagnostic is
  explicitly marked exploratory and non-decisive.
- Preserve a randomized coverage floor or a paired random-control schedule.
- Report Recall@K as the primary metric, with nDCG@K and AP as secondary
  metrics.
- Block paid follow-up unless paired active-minus-random Recall@K is credibly
  positive, or mean Recall@K improves by at least 0.025 with nonnegative
  nDCG/AP deltas and no missing-label caveat.
- Also require weak-bucket diagnostics: pointwise-plus-touched oracle cap,
  positive-negative-pair oracle cap, observed positive-winner cap, unique future
  positives touched, graph connectivity, and degree around positives and
  posterior top-K nodes.

Executable gate harness:

```bash
uv run python scripts/run_active_arm_gate.py \
  --active-artifact artifacts/backtest-arxiv-ci-partition-gate/ci-partition-gate-analysis.json \
  --random-variance-artifact artifacts/backtest-arxiv-full-random-variance-completion/full-random-variance-completion.json \
  --output artifacts/backtest-arxiv-active-arm-gate-harness/active-arm-gate-smoke.json
```

The harness is offline-only. It makes zero Sestina paid LLM calls, requires a
paired random/exact-pool control, checks the completed 20-seed full-random
variance artifact, reports seed-level active-minus-random confidence intervals,
and emits `paid_followup_allowed`.

Current smoke result for the cached CI-partition replay remains blocked:
active-minus-exact-pool-random Recall@K delta is +0.003750 with 95% normal
approximation CI [-0.009681, +0.017181], nDCG@K delta is -0.017352, and AP
delta is -0.024205. Missing-label caveat is false, and
`paid_followup_allowed` is false.

During any paid run:

- Make zero pointwise calls unless the run is explicitly approved as a new
  pointwise experiment.
- Use provider-prefixed model names, model availability checks, dry-run
  estimates, JSONL ledgers, hard `--max-usd` caps, and artifact directories.
- Store `scheduled_pair` diagnostics for reused and newly paid labels so future
  audits do not depend on reconstructing schedules after code changes.
- Keep future labels, citations, `good_paper`, arXiv ID, matched title, and
  matched work ID out of scheduling and model-visible inputs.

After the run:

- Publish per-seed and per-bucket metrics, paired deltas and confidence
  intervals, label reuse/missing diagnostics, spend, and parse/retry counts.
- Do not mix partial-label reconstructions with complete-label aggregate rows.
- Do not claim an active arm beats random if the result only beats
  pointwise-only or only improves AP while Recall@K/nDCG@K remain below random.

## Archived No-Paid Shortlist (Superseded By Stop Decision)

This shortlist is retained as chronology, not as an instruction to launch more
experiments on this branch. The campaign-level decision is now to stop
experiments and proceed with PR/publication cleanup. Any future algorithmic work
starts outside this campaign with a new no-paid gate and explicit approval before
paid labels.

| Candidate | Why it is worth testing | No-paid gate |
|---|---|---|
| Reliability-aware CI partition v2 | The current CI partition gate failed because intervals stayed wide and positive-negative oracle headroom fell. A better reliability model may be needed before top-K elimination can work. | Cached replay must reduce unresolved boundary count, improve or preserve positive-negative oracle cap, and beat exact-pool random on paired Recall@K/nDCG without losing the randomized floor. |
| New-information challenger construction | Naive pool widening and targeted outsiders failed, so another pool change needs genuinely new model-visible signal rather than more papers or another anchor mix. | Prerequisite and budget-fill replay now exist, the weak-bucket oracle-headroom caveat has been accepted with constraints for the frozen zero-missing-label manifest, and the reviewed guarded execution has completed in cache-only zero-spend mode. Any changed or nonzero-missing manifest must rerun these gates before paid labels. |
| Aggregation cross-check against standard ranking models | Degree shrinkage and soft-strength calibration failed, but an offline check against standard BT/PL/Rank Centrality implementations could expose implementation or regularization issues without paid labels. | Must improve Recall@K for at least one strong random-control baseline without hurting historical/exact random nDCG/AP or active-arm comparability. |
| Active-arm simulator harness | The full random artifact made seed variance visible; future active work needs the same paired reporting before paid calls. | Harness must produce seed/bucket rows, paired deltas, CIs, missing-label caveats, and spend estimates from cached or simulated labels before any new runner is allowed to pay. |

## Shortlist No-Paid Gate Result

Artifact:
`artifacts/backtest-arxiv-active-arm-shortlist-gate/shortlist-gate-study.json`

Command:

```bash
uv run python scripts/run_active_arm_shortlist_gate.py
```

Original result at shortlist time: no candidate allowed paid active-arm
follow-up. The study made zero paid Sestina LLM calls and zero pointwise calls.

| Candidate | Shortlist gate status | Paid follow-up | Reason |
|---|---|---:|---|
| Reliability-aware CI partition v2 | Evaluated via complete no-paid cached replay and blocked | No | V2 has paired Recall@K delta +0.003750 with 95% normal CI [-0.013400, +0.020900], nDCG/AP deltas remain negative, and positive-negative oracle cap still falls despite a larger randomized fallback. |
| New-information challenger construction | Superseded by later replay, dry-run, caveat adjudication, preflight, and guarded execution | No paid run here | The shortlist correctly blocked this candidate before a no-paid artifact existed. The later budget-fill replay clears the budget-completeness caveat, the weak-bucket oracle-headroom caveat is accepted with constraints for the frozen zero-missing-label manifest, and the reviewed guarded execution completed as cache-only zero-spend. There are still no labels to buy. |
| Aggregation cross-check against standard ranking models | Blocked missing prerequisite | No | Degree shrinkage and soft-strength calibration are zero-paid context but not standard BT/PL/Rank Centrality checks; neither improved Recall@K on complete-label arms. |
| Active-arm simulator harness / gate integration | Infrastructure ready, no paid arm | No | The reviewed harness produces seed-level deltas, confidence intervals, missing-label caveats, spend estimates, and random-variance checks, but infrastructure alone is not an active policy and the smoke input remains blocked. |

Updated next action after the budget-filled new-information replay and reviewed
guarded execution: stop experiments and proceed with PR/publication cleanup. Do
not buy labels in this workflow. The active shortfall is filled by a predeclared
no-future-label cached fallback, the reviewed active-arm gate passes, the
weak-bucket caveat acceptance is scoped only to the current frozen
zero-missing-label manifest, and the guarded execution completed with zero paid
calls, zero pointwise calls, an empty ledger, and zero labels to buy.

## Reliability-Aware CI Partition V2 Replay

Artifact:
`artifacts/backtest-arxiv-ci-partition-v2-gate-replay/ci-partition-v2-gate-replay.json`

Active-arm gate artifact:
`artifacts/backtest-arxiv-ci-partition-v2-gate-replay/active-arm-gate.json`

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

Result: the prerequisite no-paid v2 replay now exists, but paid follow-up is
still blocked. The workflow made zero paid Sestina LLM calls, zero pointwise
calls, and no paid labeling calls. Known paid spend remains USD 2.746030.

V2 is conservative: it uses cached feasible incident support and revealed
effective pairwise evidence to identify unreliable CI boundary decisions. When
nearly the whole bucket remains unresolved, low-reliability pairs enter through
the randomized fallback instead of active CI priority selection. No cached
pairwise label values are read before the pair is scheduled for replay.

20-seed paired result versus cached exact-pool random:

| Metric | V2 minus exact-pool random | 95% normal CI |
|---|---:|---|
| Recall@K | +0.003750 | [-0.013400, +0.020900] |
| nDCG@K | -0.008870 | [-0.024598, +0.006859] |
| AP | -0.013815 | [-0.026494, -0.001136] |

V2 preserved a larger randomized fallback rate (0.600) and had no missing-label
caveat, but the active-arm gate blocked it because the Recall@K gain is below
+0.025 and not credibly positive. The replay-local gate also blocked because
positive-negative oracle recall cap still fell versus exact-pool random.

Compared with the original CI partition replay, v2 kept Recall@K unchanged
while improving nDCG by +0.008483 and AP by +0.010390. That is useful
diagnostically but not enough to justify paid labels.

## New-Information Challenger No-Paid Replay

Artifacts:

- `artifacts/backtest-arxiv-new-information-challenger-simulator/new-information-challenger-simulator.json`
- `artifacts/backtest-arxiv-new-information-challenger-simulator/active-arm-gate.json`

Commands:

```bash
uv run python scripts/run_new_information_challenger_simulator.py
uv run python scripts/run_active_arm_gate.py \
  --active-artifact artifacts/backtest-arxiv-new-information-challenger-simulator/new-information-challenger-simulator.json \
  --random-variance-artifact artifacts/backtest-arxiv-full-random-variance-completion/full-random-variance-completion.json \
  --output artifacts/backtest-arxiv-new-information-challenger-simulator/active-arm-gate.json \
  --active-arm new_information_challenger_cached_replay \
  --random-control-arm exact_pool_random_cached_replay
```

The simulator uses pointwise rubric residuals, uncertainty, lexical novelty, and
metadata diversity to expose possible pointwise false negatives. It is distinct
from expanded-pool random and targeted-outsider random because it does not
change the pool by size or posterior anchor mix alone. Scheduling uses only
model-visible pointwise artifacts, paper text/metadata, and cache availability;
future citation labels are used only for retrospective metrics and diagnostics.

20-seed paired posterior top-K result versus cached exact-pool random:

| Metric | New-information delta | 95% normal CI |
|---|---:|---|
| Recall@K | +0.026250 | [+0.010165, +0.042335] |
| nDCG@K | +0.024842 | [+0.010131, +0.039552] |
| AP | +0.002569 | [-0.006515, +0.011653] |

The reviewed active-arm metric gate would pass: paired Recall@K is credibly
positive, mean Recall@K exceeds +0.025, nDCG/AP deltas are nonnegative, no
missing-label caveat is present, paired seed count is 20, and the completed
full-random reference is available. Paid follow-up is nevertheless blocked by a
budget-completeness caveat: the active arm schedules 16 of the resolved 20 pairs
in the `arxiv_cs_AI_2023_01_historical_citation_pilot` row for all 20 seeds, an
80-comparison shortfall. The workflow made zero paid Sestina LLM calls, zero
pointwise calls, and zero paid labeling calls. Known paid spend remains USD
2.746030.

Caveat: the replay-local false-negative diagnostic also blocks because
weak-bucket oracle headroom and exposure fall versus exact-pool random. The
new-information arm touches 412 unique future positives versus 447 for cached
exact-pool random, and weak-bucket pointwise-plus-touched and positive-negative
oracle cap deltas are -0.051250 and -0.043750. Therefore this is not permission
to buy labels in this workflow.

## New-Information Budget-Fill Replay

Artifacts:

- `artifacts/backtest-arxiv-new-information-budget-fill-gate/new-information-budget-fill-gate.json`
- `artifacts/backtest-arxiv-new-information-budget-fill-gate/active-arm-gate.json`

Commands:

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

The fallback is predeclared and cached-only. It fills only rows still short after
the primary new-information schedule, requires at least one preselected
new-information challenger endpoint, uses a top `4*K` boundary-frontier
comparator selected from pointwise/rubric/text/metadata signals, excludes all
primary proposal keys, and uses cache-key availability only as a feasibility
filter. It does not inspect future citation labels, cached pairwise label
values, or citation outcomes before scheduling.

Budget fill result: 80/80 active shortfall comparisons filled across 20
seed/bucket rows; remaining active shortfall 0; random-control shortfall 0;
missing-label caveat false. The workflow made zero paid Sestina LLM calls, zero
pointwise calls, and zero paid labeling calls. Known paid spend remains USD
2.746030.

20-seed paired posterior top-K result versus cached exact-pool random:

| Metric | Budget-filled delta | 95% normal CI |
|---|---:|---|
| Recall@K | +0.028750 | [+0.011972, +0.045528] |
| nDCG@K | +0.026204 | [+0.011096, +0.041313] |
| AP | +0.002436 | [-0.006659, +0.011532] |

The reviewed active-arm gate passes on the completed no-paid artifact:
`paid_followup_allowed` is true, no budget-completeness or missing-label caveat
is present, paired seed count is 20, the full-random reference is complete, and
known spend remains within the USD 100 cap. The later paid-workflow dry-run
below was nevertheless no-go because it judged the simulator-local
false-negative diagnostic as unresolved and found no guarded paid-runner path.
The follow-on caveat adjudication accepts that diagnostic as a scoped risk, but
it does not authorize paid calls.

## New-Information Paid Dry-Run Gate

Artifact:

- `artifacts/backtest-arxiv-new-information-paid-dry-run/paid-dry-run-go-no-go.json`
- `artifacts/backtest-arxiv-new-information-paid-dry-run/planned-pair-occurrences.jsonl`

Command:

```bash
uv run python scripts/run_new_information_paid_dry_run.py
```

The dry-run made zero paid Sestina LLM calls, zero pointwise calls, and did not
create or append the planned JSONL ledger. It froze the same 20 seeds, 8 bucket
rows, `openai/gpt-5.4-mini` pairwise model, exact-pool random comparator,
budget-filled cached frontier fallback, separate artifact directory, and planned
ledger path for any later review. It enumerated 3,200 planned pair occurrences
and 269 unique same-bucket canonical pair labels. All planned labels are already
reusable from reviewed pairwise caches, so unique missing labels are 0 and the
estimated additional spend is USD 0.000000.

Go/no-go: no-go. The active-arm gate remains green, active shortfall is 0,
random-control shortfall is 0, missing-label caveat is false, provider-prefixed
model and JSONL-ledger checks are configured, and model availability is marked
required before paid calls. Dry-run blocking reasons were:

- At dry-run time, the replay-local false-negative diagnostic was unresolved:
  weak-bucket pointwise-plus-touched cap delta is -0.051250,
  positive-negative-pair cap delta is -0.043750, and unique future positives
  touched delta is -35 versus cached exact-pool random.
- No reviewed guarded 20-seed paid runner is wired to execute this frozen
  new-information manifest.

Post-adjudication, preflight, and execution status: the weak-bucket caveat is
accepted with constraints for this frozen zero-missing-label manifest only. The
guarded runner go/no-go, final preflight, and reviewed guarded execution now
close this path for the exact manifest. The planning artifact validates the
frozen identity, pairwise-only rows, provider-prefixed model, separate JSONL
ledger, hard USD 0.01 cap, cache/missing-label counts, and
abort-on-pointwise guard. The final execution preflight additionally confirms
provider availability for `openai/gpt-5.4-mini`, detects no manifest drift, and
keeps the later cap at USD 0.01 with expected cache-only zero-spend behavior.
The reviewed guarded execution has now run in execute mode for the exact same
manifest and produced `cache_only_zero_missing_labels`: it made zero paid calls,
made zero pointwise calls, wrote zero ledger entries, and found zero labels to
buy. If the manifest changes or has missing labels, rerun the dry-run, caveat
adjudication, guarded runner go/no-go, execution preflight, and guarded
execution before paid labeling.

Guarded runner artifact:

- `artifacts/backtest-arxiv-new-information-guarded-runner/guarded-runner-go-no-go.json`
- `artifacts/backtest-arxiv-new-information-guarded-runner/guarded-runner-ledger.jsonl`
- `artifacts/backtest-arxiv-new-information-execution-preflight/execution-preflight-go-no-go.json`
- `artifacts/backtest-arxiv-new-information-guarded-execution/guarded-execution-go-no-go.json`
- `artifacts/backtest-arxiv-new-information-guarded-execution/guarded-execution-ledger.jsonl`

Command:

```bash
uv run python scripts/run_new_information_guarded_runner.py
uv run python scripts/run_new_information_execution_preflight.py
uv run python scripts/run_new_information_guarded_runner.py --mode execute --max-usd 0.01 --confirm-guarded-pairwise-only-execution --artifact-dir artifacts/backtest-arxiv-new-information-guarded-execution --ledger artifacts/backtest-arxiv-new-information-guarded-execution/guarded-execution-ledger.jsonl --output artifacts/backtest-arxiv-new-information-guarded-execution/guarded-execution-go-no-go.json
```

## What Not To Do Next

- Do not scale CCTD-GF. Its paid run missed exact-pool random on Recall@K and
  nDCG@K, and the graph floor did not fix the acquisition failure.
- Do not run naive pool widening. Expanded-pool random increased pool size and
  unique papers touched but failed to beat exact-pool random.
- Do not run another acquisition-score micro-tweak over the same information
  surface. EVSI, sequential EVSI, and CCTD-GF already show that plausible active
  scoring can choose worse comparisons than random from the same feasible pool.
- Do not spend more on random-baseline completion. The 20-seed full-random
  artifact is the baseline reference.
- Do not adopt degree-aware shrinkage or soft-strength calibration as defaults.
  They are negative/inconclusive and did not improve Recall@K.
- Do not compare future paid arms only to pointwise-only, a partial cached
  reconstruction, or seed-17 random alone.
- Do not edit historical paid ledgers or call artifacts while consolidating or
  preparing the next design gate.

## Repo Hygiene Handoff

No commits, staging, pushes, PR updates, reverts, destructive operations, paid
calls, paid-ledger edits, paid-call artifact edits, or historical experiment
artifact rewrites were performed by this consolidation.

Current changed or ignored handoff files are grouped below. The source/test/doc
lists reflect the reviewed uncommitted work already present in the worktree plus
the new memo and historical cross-link from this consolidation. The artifact and
workflow paths are ignored by git, so plain `git status` does not surface them.

Source:

- `M scripts/run_scheduler_followup.py`
- `M sestina/evsi_scheduler.py`
- `M sestina/scheduler_followup.py`
- `?? scripts/analyze_pairwise_strength_calibration.py`
- `?? scripts/analyze_posterior_decision_shrinkage.py`
- `?? scripts/analyze_random_control_gap.py`
- `?? scripts/analyze_random_variance_replication.py`
- `?? scripts/run_active_arm_gate.py`
- `?? scripts/run_ci_partition_gate.py`
- `?? scripts/run_new_information_challenger_simulator.py`
- `?? scripts/run_full_random_variance_completion.py`
- `?? sestina/active_arm_gate.py`
- `?? sestina/ci_partition_gate.py`
- `?? sestina/new_information_challenger.py`
- `?? sestina/pairwise_strength.py`
- `?? sestina/posterior_decision.py`

Tests:

- `M tests/test_evsi_scheduler.py`
- `M tests/test_scheduler_followup.py`
- `?? tests/test_active_arm_gate.py`
- `?? tests/test_ci_partition_gate.py`
- `?? tests/test_new_information_challenger.py`
- `?? tests/test_new_information_challenger_simulator.py`
- `?? tests/test_full_random_variance_completion.py`
- `?? tests/test_pairwise_strength.py`
- `?? tests/test_posterior_decision.py`
- `?? tests/test_posterior_decision_analysis.py`
- `?? tests/test_random_control_gap_analysis.py`
- `?? tests/test_random_variance_replication.py`

Docs:

- `M docs/internal/historical-arxiv-pilot-results.md`
- `?? docs/internal/related-work-audit.md`
- `?? docs/internal/sestina-experiment-decision-memo.md`

Artifacts:

- `artifacts/backtest-arxiv-active-arm-shortlist-gate/shortlist-gate-study.json`
  created by the no-paid active-arm shortlist gate study.
- `artifacts/backtest-arxiv-active-arm-gate-harness/active-arm-gate-smoke.json`
  created by the no-paid active-arm gate harness.
- `artifacts/backtest-arxiv-experiment-consolidation/consolidation-summary.json`
  created by this consolidation.
- `artifacts/backtest-arxiv-new-information-challenger-simulator/new-information-challenger-simulator.json`
  created by the new-information challenger no-paid replay.
- `artifacts/backtest-arxiv-new-information-challenger-simulator/active-arm-gate.json`
  created by the reviewed active-arm gate harness for that replay.
- `artifacts/backtest-arxiv-new-information-guarded-execution/guarded-execution-go-no-go.json`
  created by the guarded execute-mode cache-only handoff.
- `artifacts/backtest-arxiv-new-information-guarded-execution/guarded-execution-ledger.jsonl`
  created by the guarded execute-mode cache-only handoff with 0 entries.
- Existing ignored `artifacts/backtest-arxiv-*` experiment outputs, ledgers,
  and call artifacts were left untouched.

Workflow coordination:

- `.codex-workflows/sestina-active-arm-gate-harness/worker/process.md`
  updated by the active-arm gate harness worker.
- `.codex-workflows/sestina-active-arm-shortlist-gate/worker/process.md`
  updated by the shortlist gate worker.
- `.codex-workflows/sestina-experiment-consolidation/worker/process.md`
  updated by this consolidation.
- `.codex-workflows/sestina-new-information-guarded-execution/worker/process.md`
  updated by the guarded execution worker.
- Existing ignored `.codex-workflows/*` coordination files were left intact.
