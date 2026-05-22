# Sestina Next Experiment Report

Date: 2026-05-07

Workflow: `sestina-next-experiment-loop`

PR: https://github.com/NeapolitanIcecream/sestina/pull/3

Merge commit: `e1c4a55ccc77dbc4069c7283552861db28e92064`

Status: merged into `main`. This round made zero paid Sestina LLM calls, zero
pointwise calls, no paid labels, no historical paid-ledger edits, and no raw
paid-call artifact edits.

## Bottom Line

This round successfully converted the prior "stop and gate" decision into
enforceable source, tests, CLI validation, and internal documentation.

It did not run a fresh paid-label validation and did not strengthen the current
metric claim. The current best experimental result remains the cached/no-paid
`new_information_challenger_cached_replay` against
`exact_pool_random_cached_replay`, followed by guarded cache-only execution of
the frozen manifest. That result stays internal evidence, not a publication-ready
fresh validation.

The outcome is operational readiness for the next experiment round: future work
now has a machine-readable protocol that blocks fresh holdout work until a real
no-paid active-arm gate passes, and the active-arm gate blocks future-label or
cached-label leakage markers before any paid follow-up can be considered.

## What Was Delivered

New protocol module:

- `sestina/experiment_protocol.py`
- Builds a zero-paid `sestina-next-experiment-protocol` artifact.
- Preserves the stopped campaign boundary.
- Requires a no-paid active-arm gate before future experiments.
- Restricts fresh holdout validation to dry-run/preflight until all guardrails
  pass.
- Rejects arbitrary JSON masquerading as a passing no-paid gate by validating
  the active-arm gate schema before trusting the artifact.

New CLI:

- `scripts/validate_next_experiment_protocol.py`
- Emits the protocol artifact without paid calls.
- With no gate artifact, reports a blocked protocol.
- With future inputs, can validate a no-paid gate artifact and optional fresh
  holdout request.

Active-arm gate hardening:

- `sestina/active_arm_gate.py`
- Adds recursive leakage-marker scanning.
- Blocks paid follow-up when future labels, citation outcomes, `good_paper`,
  matched title/work IDs, or cached label values are marked as used for
  scheduling, routing, prompts, model-visible inputs, decision, or calibration.
- Covers marker aliases already emitted by existing Sestina code, including
  `uses_future_labels_for_scheduling`,
  `future_citation_labels_used_for_scheduling`,
  `uses_future_labels_for_decision`, and
  `uses_future_labels_for_calibration`.

New internal report/protocol docs:

- `docs/internal/sestina-next-experiment-protocol.md`
- Cross-links added from the decision memo, final handoff, and cleanup plan.

New executable specs:

- `tests/test_experiment_protocol.py`
- Additional active-arm gate specs in `tests/test_active_arm_gate.py`
- Regression coverage for inconsistent nested gate verdicts, non-gate artifact
  spoofing, unapproved pointwise calls, unapproved priority directions, and
  leakage-marker aliases.

## Five Requested Operating Points

1. Cleanup and publication boundary:
   The protocol preserves the current result as cached/no-paid internal
   evidence. It does not allow raw paid-call JSON, historical ledgers,
   planned-pair manifests, stdout JSON, dataset manifests with work IDs, or
   workflow records to become public artifacts without separate scrub/review.

2. No-paid gate before future experiments:
   `build_next_experiment_protocol(...)` blocks future experiment progression
   unless a schema-valid `sestina-active-arm-gate` artifact passes.

3. Priority experiment direction:
   The only approved next directions are
   `confidence_interval_top_k_partition_elimination` and
   `no_paid_replay_gate_randomized_coverage_floor`.

4. Hard gate standards:
   The machine-readable standards require paired random/exact-pool controls,
   at least 20 seeds, Recall@K primary, nDCG@K/AP secondary metrics,
   weak-bucket diagnostics, the completed full-random variance reference,
   randomized coverage or paired control, and no future-label/cached-label
   leakage.

5. Fresh holdout protocol:
   Fresh holdout work can only begin as dry-run/preflight after the no-paid gate
   passes. It requires provider/model availability checks, a separate artifact
   directory, JSONL ledger, hard `--max-usd` cap, zero pointwise calls unless
   explicitly approved, immutable historical artifacts, and no label leakage.
   The protocol still does not authorize paid label purchase by itself.

## Review And CI Outcome

Local worker/reviewer loop:

- Worker delivered the protocol patch.
- Local reviewer found two gate gaps.
- Worker revision closed both gaps.
- Local reviewer recheck reported `no_issues`.

Cloud Codex review:

- Initial cloud review on commit `8ecef78` found two P1 issues:
  - missing leakage aliases for decision/calibration;
  - no-paid gate artifact type/schema not validated before trust.
- Both were fixed in commit `3e4dd77`.
- Both review threads were replied to and resolved.
- A follow-up `@codex review` reported no major issues.

CI:

- GitHub Actions `test` passed on the latest commit.
- Two CI runs were green before merge.

Local verification:

```bash
uv run pytest -p no:cacheprovider
uv run python scripts/validate_next_experiment_protocol.py --output /tmp/sestina-next-experiment-protocol-final.json
uv run python -m json.tool /tmp/sestina-next-experiment-protocol-final.json >/dev/null
git diff --check
```

Observed final local test result before merge: `162 passed`.

## Current Protocol Output

With no no-paid gate artifact supplied, the protocol intentionally blocks the
next step:

- `paid_calls_made`: `0`
- `paid_spend_usd`: `0.0`
- `pointwise_calls_made`: `0`
- `campaign_status`: `stopped`
- `no_paid_gate_passed`: `false`
- `no_paid_gate_blocking_reasons`: `["no_paid_gate_artifact_missing"]`
- `fresh_holdout_allowed_to_begin`: `false`
- `paid_label_purchase_authorized_by_this_protocol`: `false`

This is the expected result. It means the next experiment cannot start by
accident; it must first produce and pass a no-paid gate artifact.

## Interpretation

The experiment loop succeeded as an operational experiment. It proved the
process can take an internal experiment decision, turn it into executable
guardrails, run local worker/reviewer closure, survive cloud Codex review, pass
CI, and merge.

It did not answer whether a confidence-interval top-K scheduler will beat the
random/exact-pool baseline. That is deliberately future work. The main result is
that such future work now has to clear a stronger, audited entry gate before it
can consume paid labels or claim fresh validation.

## Limitations

- No new paper-ranking labels were collected.
- No new active-arm performance metric was produced.
- No fresh holdout bucket was run.
- No external publication claim should be made from this PR alone.
- Older active-gate artifacts that lack explicit no-leakage metadata are
  intentionally blocked by the new protocol and should be regenerated through
  the updated harness if reused.

## Recommended Next Step

Do not start paid work. The next useful task is to implement a no-paid replay
for one approved priority direction:

1. Confidence-interval top-K partition/elimination scheduler, or
2. No-paid replay gate with a randomized coverage floor.

That replay should emit a schema-valid active-arm gate artifact and then be
checked with:

```bash
uv run python scripts/validate_next_experiment_protocol.py \
  --no-paid-gate-artifact <gate-artifact.json>
```

Only if the protocol reports the no-paid gate passed should fresh holdout
dry-run/preflight be considered.
