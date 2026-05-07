# Sestina PR And Publication Cleanup Plan

Date: 2026-05-02

Workflow: `sestina-pr-cleanup-audit`

Status: cleanup readiness audit, not cleanup execution. This audit made zero
paid Sestina LLM calls, zero pointwise calls, no paid labels, no staging, no
commits, no pushes, no artifact deletion, and no paid-call artifact rewrites.

Machine-readable audit:
`artifacts/backtest-arxiv-pr-cleanup-audit/pr-cleanup-audit.json`

Internal-review-only cleanup package:
`docs/internal/sestina-internal-review-package.md` and
`artifacts/backtest-arxiv-internal-review-cleanup/internal-review-cleanup-index.json`

Next experiment protocol:
`docs/internal/sestina-next-experiment-protocol.md`

## Bottom Line

The branch is ready for a PR/publication cleanup workflow, but it should not be
published as-is.

The final experiment decision is stable: stop experiments now. The current best
result is the budget-filled new-information challenger cached replay, followed
by guarded cache-only execution of the exact frozen manifest. The result is
credible internal evidence, not a fresh paid-label validation.

Known campaign spend remains USD 2.746030 / USD 100. This cleanup audit added
USD 0.000000.

2026-05-06 documentation reconciliation: the historical compact prompt and
decision memo have been refreshed to reflect the budget-fill replay, caveat
adjudication, execution preflight, and reviewed guarded cache-only execution.
This does not change the artifact boundary: no raw `artifacts/**/calls/*.json`,
ledgers/stdout JSON, planned pair manifests, dataset manifests, or
`.codex-workflows/**` records should enter public-facing outputs without a
separate scrub/review.

2026-05-07 next-experiment protocol update: future experiment work must begin
with a new no-paid gate and must preserve the current result as cached/no-paid,
not fresh validation. Fresh holdout validation is dry-run/preflight-only until
the no-paid gate passes and all provider availability, JSONL ledger, max-usd,
zero-pointwise, no-leakage, and immutable-artifact guardrails clear.

## Current Worktree Shape

Git-visible changes are mixed across source, tests, and docs:

- Tracked modified source: `sestina/evsi_scheduler.py`,
  `sestina/scheduler_followup.py`, `scripts/run_scheduler_followup.py`.
- Tracked modified tests: `tests/test_evsi_scheduler.py`,
  `tests/test_scheduler_followup.py`.
- Tracked modified docs:
  `docs/internal/historical-arxiv-pilot-results.md`.
- Untracked source modules: `sestina/active_arm_gate.py`,
  `sestina/ci_partition_gate.py`, `sestina/new_information_challenger.py`,
  `sestina/pairwise_strength.py`, `sestina/posterior_decision.py`.
- Untracked experiment and analysis scripts: `scripts/run_active_arm_gate.py`,
  `scripts/run_active_arm_shortlist_gate.py`,
  `scripts/run_ci_partition_gate.py`,
  `scripts/run_ci_partition_v2_gate_replay.py`,
  `scripts/run_full_random_variance_completion.py`,
  `scripts/run_new_information_challenger_simulator.py`,
  `scripts/run_new_information_execution_preflight.py`,
  `scripts/run_new_information_guarded_runner.py`,
  `scripts/run_new_information_paid_dry_run.py`,
  `scripts/adjudicate_new_information_caveat.py`,
  `scripts/analyze_pairwise_strength_calibration.py`,
  `scripts/analyze_posterior_decision_shrinkage.py`,
  `scripts/analyze_random_control_gap.py`,
  `scripts/analyze_random_variance_replication.py`.
- Untracked tests: active-arm gate, active-arm shortlist gate, CI partition,
  CI partition v2 replay, full random completion, new-information challenger,
  caveat adjudication, execution preflight, guarded runner, paid dry-run,
  pairwise strength, posterior decision, random-control gap, and random
  variance replication tests.
- Untracked internal docs:
  `docs/internal/related-work-audit.md`,
  `docs/internal/sestina-experiment-decision-memo.md`,
  `docs/internal/sestina-final-results-handoff.md`, and this cleanup plan.

Ignored directories are significant:

- `.codex-workflows/` is ignored and currently about 147M.
- `artifacts/` is ignored and currently about 268M.
- `artifacts/` contains 4,283 files.
- Raw call artifacts under `*/calls/*.json`: 4,100 files total, including
  833 pointwise and 3,267 pairwise call JSON files.
- Largest generated JSONs include CI partition and new-information replay
  artifacts in the 44M to 54M range.

## Proposed PR / Commit Groups

Use small, reviewable commits. Do not mix raw paid-call outputs with code.

1. Scheduler follow-up baseline variants.
   Include `sestina/evsi_scheduler.py`, `sestina/scheduler_followup.py`,
   `scripts/run_scheduler_followup.py`, `tests/test_evsi_scheduler.py`, and
   `tests/test_scheduler_followup.py`. This is the cleanest first code group:
   expanded-pool random and targeted-outsider random scheduler support.

2. Active-arm gate harness.
   Include `sestina/active_arm_gate.py`, `scripts/run_active_arm_gate.py`,
   `scripts/run_active_arm_shortlist_gate.py`,
   `tests/test_active_arm_gate.py`, and
   `tests/test_active_arm_shortlist_gate.py`. Keep generated gate outputs out
   of this code commit unless a sanitized summary artifact is intentionally
   force-added.

3. CI partition replay infrastructure.
   Include `sestina/ci_partition_gate.py`,
   `scripts/run_ci_partition_gate.py`,
   `scripts/run_ci_partition_v2_gate_replay.py`,
   `tests/test_ci_partition_gate.py`, and
   `tests/test_ci_partition_v2_gate_replay.py`.

4. Posterior and pairwise-strength diagnostics.
   Include `sestina/posterior_decision.py`, `sestina/pairwise_strength.py`,
   the posterior/strength/random-control analysis scripts, and their tests.
   Treat this as diagnostic tooling unless public docs need one specific result.

5. Random variance completion tooling.
   Include `scripts/run_full_random_variance_completion.py`,
   `scripts/analyze_random_variance_replication.py`,
   `tests/test_full_random_variance_completion.py`, and
   `tests/test_random_variance_replication.py`. Do not include raw full-random
   call artifacts in the same commit.

6. New-information challenger and guarded execution tooling.
   Include `sestina/new_information_challenger.py`,
   `scripts/run_new_information_challenger_simulator.py`,
   `scripts/run_new_information_paid_dry_run.py`,
   `scripts/adjudicate_new_information_caveat.py`,
   `scripts/run_new_information_execution_preflight.py`,
   `scripts/run_new_information_guarded_runner.py`, and corresponding tests.
   This group should carry the strict zero-pointwise and cache-only guardrail
   tests.

7. Internal results documentation.
   Include `docs/internal/historical-arxiv-pilot-results.md`,
   `docs/internal/related-work-audit.md`,
   `docs/internal/sestina-experiment-decision-memo.md`,
   `docs/internal/sestina-final-results-handoff.md`, and this cleanup plan.
   Rewrite public-facing wording separately before moving any claim into
   README, release notes, a PR body, or publication text.

8. Selected sanitized evidence artifacts, if needed.
   Because `artifacts/` is ignored, this must be a deliberate force-add or a
   separate publication bundle. Prefer small sanitized summaries over raw JSON.
   Recommended candidates are the final handoff summary, consolidation summary,
   active gate JSON, budget-fill gate summary, paid dry-run go/no-go, caveat
   adjudication, guarded runner/preflight/execution go/no-go files, and the
   full-random completion summary. Scrub local paths and provider metadata
   first.

9. Workflow coordination records.
   Keep `.codex-workflows/*` internal. Do not include in public PRs unless the
   project explicitly wants to publish process transcripts, prompts, and local
   run logs.

## Artifact Handling Decisions

Recommended handling before PR/publication:

| Artifact class | Examples | Decision |
|---|---|---|
| Source and tests | `sestina/*.py`, `scripts/*.py`, `tests/*.py` | Include in coherent code commits after full test pass. |
| Internal result docs | `docs/internal/*handoff*.md`, decision memo, historical log | Include for reviewer context, but rewrite any public-facing version separately. |
| Small summary artifacts | final handoff summary, consolidation summary, go/no-go JSONs | Summarize in docs; optionally include sanitized copies if reviewers need machine-readable evidence. |
| Large replay JSONs | CI partition gate, CI partition v2, new-information challenger, budget-fill replay | Keep internal by default. Publish only compressed/sanitized summaries or external artifact bundle after review. |
| Raw paid-call JSON | `artifacts/**/calls/*.json` | Exclude from public PR/publication by default. Review prompts, responses, model metadata, paper metadata, and usage before any disclosure. |
| Ledgers and stdout JSON | `ledger.jsonl`, `*ledger*.jsonl`, `*.stdout.json` | Keep internal. Report aggregate spend and parse/retry counts in docs instead. |
| Planned pair manifest | `planned-pair-occurrences.jsonl` | Keep internal or publish a sanitized manifest hash/summary. Full manifest may reveal scheduling choices and local paths. |
| Workflow files | `.codex-workflows/**` | Keep internal coordination records. Exclude from public branch unless intentionally publishing workflow history. |
| Dataset manifests | `artifacts/backtest-datasets/**` | Review for arXiv/work IDs, local paths, and license/disclosure expectations before publication. |

## Disclosure And Scrub Checklist

Before public release, make these decisions explicitly:

- Preserve the exact claim scope: budget-filled cached replay plus guarded
  cache-only execution, not fresh paid-label validation.
- Include paired seed-level uncertainty for Recall@K, nDCG@K, and AP.
- State that AP improvement is small and its paired interval crosses zero.
- State that the weak-bucket caveat is accepted only for the exact frozen
  zero-missing-label manifest and reviewed artifacts.
- State that the guarded execution made zero paid calls, zero pointwise calls,
  and authorized no paid label purchase.
- Scrub absolute local paths such as `/Users/chenmohan/gits/sestina` from any
  artifact selected for publication.
- Decide whether provider/model metadata should be disclosed or stripped. Some
  generated artifacts mention provider-prefixed model names and model
  availability checks.
- Do not disclose environment presence fields such as API key/base URL presence
  booleans unless deliberately reviewed.
- Exclude or scrub raw prompts, responses, usage metadata, scheduled pair rows,
  arXiv/work identifiers, matched titles, and future-label diagnostics where
  they are not needed for the public claim.
- Keep historical paid ledgers and paid-call artifacts immutable. If a public
  artifact is needed, create a sanitized derivative rather than editing the raw
  paid-call artifact.
- Decide whether the publication bundle should include hashes of internal raw
  artifacts to support auditability without exposing raw data.

## Scan Results

Safe scans were path-only or count-only and did not print secret values.

- High-confidence secret pattern scan over `docs/internal`, `artifacts`, and
  `.codex-workflows` found no matches for common API key/private-key/Bearer
  token patterns.
- Absolute-path scan found many matches, especially ignored artifacts and
  workflow files. These require scrub or exclusion before publication.
- Provider/model metadata scan found matches in generated artifacts and workflow
  files, including model availability and API environment presence fields.
- Raw-call shape inspection found call JSON keys including `response`,
  `scheduled_pair`, `estimated_tokens`, `estimated_cost_usd`, `model`, `kind`,
  `subject`, and `comparison`; treat these as disclosure-sensitive.

## Recommended Next Cleanup Workflow

1. Freeze the cleanup scope in a short issue/PR checklist using the commit
   groups above.
2. Decide whether the PR is internal-review-only or publication-facing. If it is
   publication-facing, create sanitized derivative artifacts instead of force
   adding ignored raw artifacts.
3. Rewrite public-facing wording in README/PR text around the stop decision,
   cached replay scope, paired uncertainty, zero-spend guarded execution, and
   weak-bucket caveat.
4. Stage source/test groups first, then docs, then any selected sanitized
   artifact bundle. Do not accidentally stage `.codex-workflows/` or raw
   `artifacts/**/calls/*.json`.
5. Run `uv run pytest -p no:cacheprovider`, `git diff --check`, JSON
   validation for selected artifacts, and another path/secret marker scan after
   scrub.
6. In the PR body, include the exact validation commands, spend statement, raw
   artifact handling decision, and any disclosure limitations.

## Validation Performed

- `jq empty artifacts/backtest-arxiv-pr-cleanup-audit/pr-cleanup-audit.json`
  passed.
- Key referenced cleanup/handoff artifacts exist.
- High-confidence secret scan over the new cleanup plan, new audit JSON, and
  final handoff found no API key/private-key/Bearer-token style matches.
- Placeholder-marker scan over the cleanup plan, final handoff, and decision
  memo found no unresolved placeholders.
- `git diff --check` passed.
- `uv run pytest -p no:cacheprovider` passed: 134 tests.
- `uv run ruff --version` failed because `ruff` is not installed in the uv
  environment; ruff was skipped.
