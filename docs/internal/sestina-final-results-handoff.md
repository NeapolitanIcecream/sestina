# Sestina Final Results Handoff

Date: 2026-05-02

Workflow: `sestina-final-results-handoff`

Status: final internal handoff after the reviewed guarded
`sestina-new-information-guarded-execution` result. This handoff made zero paid
Sestina LLM calls, zero pointwise calls, and no paid labels.

Primary evidence:

- Decision memo:
  `docs/internal/sestina-experiment-decision-memo.md`
- Historical result log:
  `docs/internal/historical-arxiv-pilot-results.md`
- Machine-readable handoff:
  `artifacts/backtest-arxiv-final-results-handoff/final-results-handoff-summary.json`
- PR/publication cleanup audit:
  `docs/internal/sestina-pr-cleanup-plan.md`
- Internal-review-only cleanup package:
  `docs/internal/sestina-internal-review-package.md`
- Reviewed guarded execution:
  `artifacts/backtest-arxiv-new-information-guarded-execution/guarded-execution-go-no-go.json`
- Empty guarded execution ledger:
  `artifacts/backtest-arxiv-new-information-guarded-execution/guarded-execution-ledger.jsonl`

## Bottom Line

The Sestina experiment campaign should stop here. The current best active-arm
result is the budget-filled new-information challenger cached replay, followed
by reviewed guarded cache-only execution of the exact frozen manifest. It is a
credible internal result, not a publication-ready paid-label result.

The branch is ready for PR/publication cleanup work because the experimental
decision is closed, the guarded execution reviewer reported `no_issues`, and
the final result adds no new spend. It is not ready to publish as-is. Cleanup
must curate the broad uncommitted source/test/docs changes, decide what ignored
artifacts should be retained or excluded, remove or quarantine internal-only
workflow files, and preserve the caveat wording exactly.

Known paid spend remains USD 2.746030 of the USD 100 cap, leaving USD
97.253970. This handoff added USD 0.000000.

## Current Best Result

The best current Sestina result is:
`new_information_challenger_cached_replay` with the predeclared cached frontier
budget fill, evaluated against `exact_pool_random_cached_replay` over 20 paired
seeds and 8 historical arXiv buckets, then executed through the guarded runner
in cache-only mode.

| Metric | New-information replay | Exact-pool random replay | Active minus random |
|---|---:|---:|---:|
| Recall@K | 0.338750 | 0.310000 | +0.028750 |
| nDCG@K | 0.369806 | 0.343602 | +0.026204 |
| AP | 0.363194 | 0.360758 | +0.002436 |

Paired seed-level 95% normal approximation intervals:

| Delta | 95% CI |
|---|---:|
| Recall@K | [+0.011972, +0.045528] |
| nDCG@K | [+0.011096, +0.041313] |
| AP | [-0.006659, +0.011532] |

The active-arm gate reports `paid_followup_allowed=true` for this frozen
no-paid artifact. The later guarded execution does not strengthen the metric
claim; it confirms the frozen manifest can execute without buying labels.

Why the result is credible internally:

- It uses 20 paired seeds rather than the unstable seed-17 point estimate.
- It compares against the mandatory exact-pool random cached replay baseline.
- The full 20-seed random variance reference exists and remains the baseline
  robustness anchor.
- The prior 80-comparison active shortfall was filled by a predeclared
  cached-only frontier fallback without using future labels, citation outcomes,
  or cached label values for scheduling.
- The frozen manifest enumerates 3,200 planned pair occurrences and 269 unique
  same-bucket pair labels.
- Missing pairwise labels are 0, pairwise calls to buy are 0, pointwise calls
  are 0, and estimated additional spend is USD 0.000000.
- The final preflight confirmed provider availability for
  `openai/gpt-5.4-mini` and found no manifest drift.
- The guarded execution ran with `--mode execute`, `--max-usd 0.01`, a
  separate artifact directory, and an empty JSONL ledger.
- The guarded execution reviewer closed with `status: no_issues`.

## Constraints And Caveats

This is a cache-only replay/execution result. It should be described as a
reviewed no-paid/cached handoff, not as a fresh paid-label validation.

The weak-bucket caveat is accepted only for the exact frozen zero-missing-label
budget-filled manifest and the current reviewed artifacts. The caveat does not
transfer to any changed manifest, cache state, model, runner, gate artifact, or
caveat text. Any drift requires rerunning, in order:

1. New-information paid dry-run.
2. Weak-bucket caveat adjudication.
3. Guarded runner go/no-go.
4. Execution preflight.
5. Guarded execution.

The AP improvement is small and its paired interval crosses zero. The credible
positive part of the active result is Recall@K and nDCG@K, not AP.

Future labels and citation-derived positives remain retrospective evaluation
data only. They must not enter scheduling, model-visible features, prompt text,
or paid-run routing.

No artifact in this chain authorizes paid label purchase. Because the frozen
manifest has zero missing labels, there are no labels to buy.

## What Should Stop Now

Stop additional Sestina experiments in this campaign. Specifically:

- Stop paid labeling and paid active-arm execution for this branch.
- Stop random-baseline spending; the 20-seed full-random artifact is the
  baseline reference.
- Stop CCTD-GF, sequential EVSI, expanded-pool random, targeted-outsider
  random, CI partition v1/v2, posterior shrinkage, and soft-strength
  calibration as candidates for immediate scale-up.
- Stop acquisition-score micro-tweaks over the same information surface unless
  a future design starts with a new no-paid gate and reviewer-approved protocol.
- Stop comparing future active arms only to pointwise-only, partial cached
  reconstructions, or seed-17 random.
- Stop editing historical paid ledgers, paid-call artifacts, or paid-run
  outputs during cleanup.

The next work should be cleanup and review readiness, not more experiment
execution.

## PR And Publication Cleanup Readiness

Decision: ready for PR/publication cleanup, not ready for publication without
cleanup.

Required cleanup checklist:

- Keep the decision statement stable: current best result is the frozen
  budget-filled new-information cached replay plus cache-only guarded execution.
- Preserve the stop decision and caveat scope in README/PR text/public notes.
- Decide which ignored `artifacts/backtest-arxiv-*` outputs are retained as
  internal evidence, summarized, or excluded from a public branch.
- Scrub public-facing artifacts for absolute local paths such as
  `/Users/chenmohan/gits/sestina`, provider metadata, and internal workflow
  details before publication.
- Keep `.codex-workflows/*` as coordination records, not public-facing result
  documentation, unless deliberately publishing internal workflow history.
- Stage and commit source, tests, docs, and selected artifacts in coherent
  groups; do not mix generated ledgers/calls with code changes accidentally.
- Re-run full tests after any cleanup that moves, removes, or rewrites
  artifacts.
- If publishing claims externally, include paired seed-level uncertainty,
  baseline definitions, zero-paid/no-pointwise status for the guarded execution,
  and the weak-bucket caveat.
- Do not publish raw paid-call artifacts unless their prompts/responses and
  metadata have been reviewed for disclosure risk.

## Repo Hygiene

No staging, commits, pushes, PR updates, destructive operations, historical paid
ledger rewrites, paid-call artifact rewrites, paid calls, pointwise calls, or
paid labels were performed by this handoff.

Files added or modified by this handoff:

- Docs:
  `docs/internal/sestina-final-results-handoff.md`,
  `docs/internal/sestina-experiment-decision-memo.md`,
  `docs/internal/historical-arxiv-pilot-results.md`
- Artifact:
  `artifacts/backtest-arxiv-final-results-handoff/final-results-handoff-summary.json`
- Workflow coordination:
  `.codex-workflows/sestina-final-results-handoff/worker/process.md`

Pre-existing reviewed source changes in the current worktree:

- `M scripts/run_scheduler_followup.py`
- `M sestina/evsi_scheduler.py`
- `M sestina/scheduler_followup.py`
- `?? scripts/adjudicate_new_information_caveat.py`
- `?? scripts/analyze_pairwise_strength_calibration.py`
- `?? scripts/analyze_posterior_decision_shrinkage.py`
- `?? scripts/analyze_random_control_gap.py`
- `?? scripts/analyze_random_variance_replication.py`
- `?? scripts/run_active_arm_gate.py`
- `?? scripts/run_active_arm_shortlist_gate.py`
- `?? scripts/run_ci_partition_gate.py`
- `?? scripts/run_ci_partition_v2_gate_replay.py`
- `?? scripts/run_full_random_variance_completion.py`
- `?? scripts/run_new_information_challenger_simulator.py`
- `?? scripts/run_new_information_execution_preflight.py`
- `?? scripts/run_new_information_guarded_runner.py`
- `?? scripts/run_new_information_paid_dry_run.py`
- `?? sestina/active_arm_gate.py`
- `?? sestina/ci_partition_gate.py`
- `?? sestina/new_information_challenger.py`
- `?? sestina/pairwise_strength.py`
- `?? sestina/posterior_decision.py`

Pre-existing reviewed test changes in the current worktree:

- `M tests/test_evsi_scheduler.py`
- `M tests/test_scheduler_followup.py`
- `?? tests/test_active_arm_gate.py`
- `?? tests/test_active_arm_shortlist_gate.py`
- `?? tests/test_ci_partition_gate.py`
- `?? tests/test_ci_partition_v2_gate_replay.py`
- `?? tests/test_full_random_variance_completion.py`
- `?? tests/test_new_information_caveat_adjudication.py`
- `?? tests/test_new_information_challenger.py`
- `?? tests/test_new_information_challenger_simulator.py`
- `?? tests/test_new_information_execution_preflight.py`
- `?? tests/test_new_information_guarded_runner.py`
- `?? tests/test_new_information_paid_dry_run.py`
- `?? tests/test_pairwise_strength.py`
- `?? tests/test_posterior_decision.py`
- `?? tests/test_posterior_decision_analysis.py`
- `?? tests/test_random_control_gap_analysis.py`
- `?? tests/test_random_variance_replication.py`

Pre-existing reviewed docs in the current worktree:

- `M docs/internal/historical-arxiv-pilot-results.md`
- `?? docs/internal/related-work-audit.md`
- `?? docs/internal/sestina-experiment-decision-memo.md`

Key ignored/internal artifacts for the campaign:

- `artifacts/backtest-arxiv-experiment-consolidation/consolidation-summary.json`
- `artifacts/backtest-arxiv-full-random-variance-completion/full-random-variance-completion.json`
- `artifacts/backtest-arxiv-active-arm-gate-harness/active-arm-gate-smoke.json`
- `artifacts/backtest-arxiv-active-arm-shortlist-gate/shortlist-gate-study.json`
- `artifacts/backtest-arxiv-ci-partition-v2-gate-replay/ci-partition-v2-gate-replay.json`
- `artifacts/backtest-arxiv-ci-partition-v2-gate-replay/active-arm-gate.json`
- `artifacts/backtest-arxiv-new-information-challenger-simulator/new-information-challenger-simulator.json`
- `artifacts/backtest-arxiv-new-information-challenger-simulator/active-arm-gate.json`
- `artifacts/backtest-arxiv-new-information-budget-fill-gate/new-information-budget-fill-gate.json`
- `artifacts/backtest-arxiv-new-information-budget-fill-gate/active-arm-gate.json`
- `artifacts/backtest-arxiv-new-information-paid-dry-run/paid-dry-run-go-no-go.json`
- `artifacts/backtest-arxiv-new-information-paid-dry-run/planned-pair-occurrences.jsonl`
- `artifacts/backtest-arxiv-new-information-caveat-adjudication/caveat-adjudication.json`
- `artifacts/backtest-arxiv-new-information-guarded-runner/guarded-runner-go-no-go.json`
- `artifacts/backtest-arxiv-new-information-guarded-runner/guarded-runner-ledger.jsonl`
- `artifacts/backtest-arxiv-new-information-execution-preflight/execution-preflight-go-no-go.json`
- `artifacts/backtest-arxiv-new-information-guarded-execution/guarded-execution-go-no-go.json`
- `artifacts/backtest-arxiv-new-information-guarded-execution/guarded-execution-ledger.jsonl`
- `artifacts/backtest-arxiv-final-results-handoff/final-results-handoff-summary.json`

Relevant workflow coordination records:

- `.codex-workflows/sestina-experiment-consolidation/worker/process.md`
- `.codex-workflows/sestina-new-information-paid-dry-run/worker/process.md`
- `.codex-workflows/sestina-new-information-caveat-adjudication/worker/process.md`
- `.codex-workflows/sestina-new-information-guarded-runner/worker/process.md`
- `.codex-workflows/sestina-new-information-execution-preflight/worker/process.md`
- `.codex-workflows/sestina-new-information-guarded-execution/worker/process.md`
- `.codex-workflows/sestina-new-information-guarded-execution/reviewer/review-to-worker.md`
- `.codex-workflows/sestina-final-results-handoff/worker/process.md`

## Validation

Validation performed for this handoff:

- `uv run python -m json.tool artifacts/backtest-arxiv-final-results-handoff/final-results-handoff-summary.json`
  passed.
- `git diff --check` passed.
- `uv run pytest -p no:cacheprovider` passed: 134 tests.

Ruff was not run for this docs-only handoff.
