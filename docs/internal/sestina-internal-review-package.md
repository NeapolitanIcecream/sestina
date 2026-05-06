# Sestina Internal Review Cleanup Package

Date: 2026-05-02

Workflow: `sestina-internal-review-cleanup`

Status: internal-review-only cleanup package. This package is for local branch
inspection before any staging, commit, PR, publication, or artifact-retention
decision. It made zero paid Sestina LLM calls, zero pointwise calls, no paid
labels, no staging, no commits, no pushes, no publication updates, no artifact
deletion, no historical paid-ledger rewrites, and no raw paid-call artifact
rewrites.

Machine-readable package:
`artifacts/backtest-arxiv-internal-review-cleanup/internal-review-cleanup-index.json`

## Bottom Line

Use the conservative internal-review-only path. The experiment campaign is
stopped, but the branch is not publication-ready as-is.

The current best result remains the budget-filled new-information challenger
cached replay, followed by guarded cache-only execution of the exact frozen
manifest. This is credible internal evidence, not a fresh paid-label
validation. Known spend remains USD 2.746030 / USD 100, with this cleanup
package adding USD 0.000000.

2026-05-06 reconciliation note: the internal historical prompt now matches the
final handoff. The campaign remains stopped; the guarded execution is evidence
that the frozen manifest is cache-only and zero-spend, not authorization for new
labels or publication of raw artifacts. Treat this package as internal-review
context until a separate public wording and artifact scrub is approved.

The main cleanup risk is not source code alone. The ignored artifact and
workflow trees contain large internal evidence, raw paid-call records, ledgers,
stdout JSON, provider/model metadata, local path markers, workflow records, and
large replay files. Keep them internal unless a sanitized derivative is created
and reviewed separately.

## Review Starting Points

Review these internal docs first:

| Purpose | Path | Handling |
|---|---|---|
| Cleanup audit and initial PR grouping | `docs/internal/sestina-pr-cleanup-plan.md` | Internal reviewer context. Not public wording. |
| Final result handoff | `docs/internal/sestina-final-results-handoff.md` | Internal reviewer context. Preserve caveats. |
| Experiment decision memo | `docs/internal/sestina-experiment-decision-memo.md` | Internal reviewer context. Preserve stop decision. |
| Historical evidence log | `docs/internal/historical-arxiv-pilot-results.md` | Internal reviewer context. Public excerpts require review. |
| Related-work audit | `docs/internal/related-work-audit.md` | Internal reviewer context. Public excerpts require review. |

## Artifact Include / Exclude Checklist

| Artifact class | Examples | Internal-review handling | Public handling |
|---|---|---|---|
| Small final summaries | final handoff summary, consolidation summary, active gate JSON, caveat adjudication, guarded runner, execution preflight, guarded execution go/no-go | Include by reference for internal review. | Publish only after sanitization review; preserve claim scope and caveats. |
| Large replay JSON | CI partition replay, CI partition v2 replay, new-information challenger simulator, budget-fill gate replay | Keep internal. Summarize only. | Exclude by default; publish only a sanitized derivative or separate reviewed bundle. |
| Planned pair manifest | `planned-pair-occurrences.jsonl` | Keep internal; useful for frozen-manifest audit. | Exclude by default or publish only hash/count summary after review. |
| Raw paid-call JSON | `artifacts/**/calls/*.json` | Internal-only. Do not rewrite. | Exclude unless prompts, responses, paper metadata, model metadata, usage, and cost fields receive explicit disclosure review. |
| Ledgers and stdout JSON | `ledger.jsonl`, `*ledger*.jsonl`, `*.stdout.json` | Internal-only. Do not rewrite historical ledgers. | Exclude by default; report aggregate spend and parse/retry counts instead. |
| Workflow records | `.codex-workflows/**` | Internal coordination records. | Exclude unless the project explicitly decides to publish workflow history. |
| Dataset manifests | `artifacts/backtest-datasets/**` | Internal review required for IDs, paths, and disclosure expectations. | Exclude or sanitize before publication. |
| Internal docs | `docs/internal/*.md` | Include for branch review. | Rewrite separate public-facing wording before README, PR body, release notes, or paper text. |

## Recommended PR / Commit Groups

Use small, reviewable groups. Do not stage anything from this package until the
user chooses a scope.

1. Scheduler follow-up baseline variants:
   `sestina/evsi_scheduler.py`, `sestina/scheduler_followup.py`,
   `scripts/run_scheduler_followup.py`, `tests/test_evsi_scheduler.py`, and
   `tests/test_scheduler_followup.py`.

2. Active-arm gate harness:
   `sestina/active_arm_gate.py`, `scripts/run_active_arm_gate.py`,
   `scripts/run_active_arm_shortlist_gate.py`,
   `tests/test_active_arm_gate.py`, and
   `tests/test_active_arm_shortlist_gate.py`.

3. CI partition replay infrastructure:
   `sestina/ci_partition_gate.py`, `scripts/run_ci_partition_gate.py`,
   `scripts/run_ci_partition_v2_gate_replay.py`,
   `tests/test_ci_partition_gate.py`, and
   `tests/test_ci_partition_v2_gate_replay.py`.

4. Posterior and pairwise-strength diagnostics:
   `sestina/posterior_decision.py`, `sestina/pairwise_strength.py`, the
   posterior/strength/random-control analysis scripts, and their tests.

5. Random variance completion tooling:
   `scripts/run_full_random_variance_completion.py`,
   `scripts/analyze_random_variance_replication.py`,
   `tests/test_full_random_variance_completion.py`, and
   `tests/test_random_variance_replication.py`.

6. New-information challenger and guarded execution tooling:
   `sestina/new_information_challenger.py`, the new-information dry-run,
   caveat, preflight, guarded-runner, and simulator scripts, plus matching
   tests. This group should carry the strict zero-pointwise and cache-only
   guardrail tests.

7. Internal results documentation:
   `docs/internal/historical-arxiv-pilot-results.md`,
   `docs/internal/related-work-audit.md`,
   `docs/internal/sestina-experiment-decision-memo.md`,
   `docs/internal/sestina-final-results-handoff.md`,
   `docs/internal/sestina-pr-cleanup-plan.md`, and this package doc.

8. Internal-review cleanup package:
   `docs/internal/sestina-internal-review-package.md` and, only if the user
   deliberately wants ignored artifacts force-added for review,
   `artifacts/backtest-arxiv-internal-review-cleanup/internal-review-cleanup-index.json`.

9. Selected sanitized evidence artifacts:
   optional and separate. Prefer small sanitized derivatives over raw artifacts.
   Do not mix this with source/test commits unless the reviewer explicitly asks.

## Remaining Approval Gates

- User approval before staging, committing, pushing, opening or updating a PR,
  or publishing any claim.
- User approval before force-adding anything under ignored `artifacts/`.
- User approval before deleting or moving large artifacts.
- Separate disclosure review before publishing raw paid-call JSON, ledgers,
  stdout JSON, planned pair manifests, dataset manifests, provider/model
  metadata, or workflow records.
- Separate public wording review before moving internal claims into README, PR
  body, release notes, public docs, or paper text.
- If any frozen manifest, cache state, runner, model, or caveat wording changes,
  rerun the dry-run, caveat adjudication, guarded runner go/no-go, execution
  preflight, and guarded execution sequence before making any paid-run claim.

## Validation

Validation performed for this package:

- `jq empty artifacts/backtest-arxiv-internal-review-cleanup/internal-review-cleanup-index.json`
  passed.
- Required review docs and referenced summary artifacts exist.
- Unresolved-marker scan over generated cleanup docs/artifacts found no marker
  matches.
- High-confidence credential scan over generated cleanup docs/artifacts found
  no matches.
- Sensitive-marker scan over generated cleanup docs/artifacts found no absolute
  local workspace path markers or provider environment-presence markers.
- `git diff --check` passed.
- `uv run pytest -p no:cacheprovider` passed: 134 tests.
- `uv run ruff --version` failed because `ruff` is not installed in the uv
  environment; ruff was skipped.
