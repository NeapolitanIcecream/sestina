# Sestina Fresh-Holdout Validation Runbook

Use this runbook after the Huldra-backed manifest build has completed. It
generates and reviews fresh-holdout pointwise artifacts, runs the
coverage-floor pairwise-only preflight, optionally executes guarded pairwise
labels, and analyzes the fresh validation result.

Workspace:

```text
<sestina-repo>
```

Huldra API, if metadata needs to be checked again:

```text
http://127.0.0.1:8766
```

Default fresh-holdout manifest:

```text
artifacts/backtest-datasets/arxiv-historical-coverage-floor-fresh-holdout-manifest.json
```

## Current Campaign Result (2026-05-22 18:43 CST)

Paid Sestina label generation was explicitly authorized for this campaign.
`SESTINA_LLM_BASE_URL` from `.zshrc` needed the OpenAI-compatible `/v1` suffix
for these scripts, so the paid workflow shell normalized the configured provider
URL to include `/v1` without printing or storing the secret endpoint.

Current result in `<sestina-repo>`:

- State inspection passed: the manifest exists, all 8 bucket part manifests are
  present, and Huldra `/v1/status` was reachable with `upstream_429_total=0`,
  `cache_completed_total=53`, and `papers_total=1094`.
- Manifest validation passed: `bucket_count=8`, `positive_labels=40`, and
  `papers=636`; every bucket has `positive_labels=5` and enough papers for K.
- Paid pointwise Step 4 completed: `mode=execute`,
  `paid_calls_made=636`, `pointwise_calls_made=636`,
  `paid_spend_usd=0.371653`, `available_pointwise_artifacts=636`,
  `missing_pointwise_artifacts=0`, and final status `complete`.
- Pairwise-only Step 5 returned `decision=go` with no blockers after pointwise
  completion.
- Paid pairwise Step 6 completed after a pause/resume: one transient SSL EOF
  occurred after 758 successful labels, the runner resume guard was fixed, the
  user-requested pause checkpoint at 1369 labels was clean, and the final resume
  bought the remaining 1012 labels. Final pairwise execution artifact:
  `mode=execute`, `pointwise_calls_made=0`, `paid_calls_made=1012`,
  `paid_spend_usd=0.51143`, `paid_pairwise_calls_succeeded=1012`,
  `new_ledger_entries=1012`, and `decision=go`.
- Final pairwise ledger: `line_count=2381`, call artifacts `2381`, statuses
  `{"ok": 2381}`, pairwise ledger spend `1.199969`, and no historical ledgers
  were rewritten.
- Final pairwise bucket counts:

  ```text
  arxiv_cs_LG_2023_03_historical_citation_pilot 316
  arxiv_cs_LG_2023_04_historical_citation_pilot 295
  arxiv_cs_CL_2023_03_historical_citation_pilot 292
  arxiv_cs_CL_2023_04_historical_citation_pilot 313
  arxiv_cs_AI_2023_03_historical_citation_pilot 282
  arxiv_cs_AI_2023_04_historical_citation_pilot 317
  arxiv_cs_CV_2023_03_historical_citation_pilot 285
  arxiv_cs_CV_2023_04_historical_citation_pilot 281
  ```

- Fresh validation Step 7 completed and can claim fresh paid validation:
  `complete=true`, `can_claim_fresh_paid_validation=true`,
  `missing_pairwise_occurrences=0`, and `scheduled_pairwise_occurrences=6400`.
  The fresh result does not support the coverage-floor arm over exact-pool
  random: active-minus-random mean Recall@K delta is `-0.00625` with 95% CI
  `[-0.01797287, 0.00547287]`; mean nDCG@K delta is `-0.02092364`, and mean
  average-precision delta is `-0.01473001`.
- Aggregate fresh metrics:

  ```text
  randomized_coverage_floor_hybrid_cached_replay recall_at_k=0.225
  exact_pool_random_cached_replay recall_at_k=0.23125
  randomized_coverage_floor_hybrid_cached_replay ndcg_at_k=0.24878006
  exact_pool_random_cached_replay ndcg_at_k=0.2697037
  randomized_coverage_floor_hybrid_cached_replay average_precision=0.25272942
  exact_pool_random_cached_replay average_precision=0.26745943
  ```

- Campaign spend remains below the USD 100 cap. Known prior spend was
  `2.74603`; fresh pointwise spend was `0.371653`; fresh pairwise ledger spend
  was `1.199969`; projected campaign total is `4.317652`.
- Verification passed after the resume fix and final analysis:

  ```bash
  uv run pytest tests/test_coverage_floor_followup_preflight.py -q
  uv run pytest tests/test_fresh_holdout_pointwise_artifacts.py \
    tests/test_coverage_floor_followup_preflight.py \
    tests/test_coverage_floor_fresh_validation_analysis.py
  git diff --check -- scripts/run_coverage_floor_followup_preflight.py \
    tests/test_coverage_floor_followup_preflight.py \
    docs/internal/sestina-fresh-holdout-validation-runbook.md
  git diff --check
  ```

  The resume-focused pytest run returned 6 passing tests. The runbook-focused
  pytest run returned 11 passing tests. Both diff checks passed.

Next action: report the completed fresh validation as a negative/failed
coverage-floor validation against exact-pool random, not as a win for the active
arm. Do not rerun paid labels unless a new experiment is explicitly authorized.

## Authorization Boundary

This runbook contains paid execution steps. Only run paid steps when the
assignment explicitly authorizes paid Sestina label generation under the
campaign cap.

Planning commands are safe to run without paid-call authorization:

```bash
uv run python scripts/run_fresh_holdout_pointwise_artifacts.py --mode planning
uv run python scripts/run_coverage_floor_followup_preflight.py
uv run python scripts/analyze_coverage_floor_fresh_validation.py
```

Paid commands require explicit authorization:

```bash
uv run python scripts/run_fresh_holdout_pointwise_artifacts.py \
  --mode execute \
  --confirm-fresh-holdout-pointwise-generation

uv run python scripts/run_coverage_floor_followup_preflight.py \
  --mode execute \
  --confirm-guarded-pairwise-only-execution
```

Do not run paid commands if the assignment only asks for analysis, planning,
or a dry run.

## Guardrails

- Preserve existing user changes. Inspect the worktree before editing or
  running long commands.
- Do not edit historical paid ledgers, raw paid-call artifacts, old planned-pair
  manifests, or `.codex-workflows/**`.
- Do not change the manifest, labels, frozen no-paid sweep, active gate, seed
  set, pairwise policy, or evaluation metrics to make a run pass.
- Do not place `labels`, `good_paper`, citation counts, citation ranks, matched
  titles, or work IDs into model-visible prompts.
- Stop before any paid call if provider/model availability fails, auth or
  balance fails, leakage is detected, cost cannot be measured, the USD 100
  campaign cap would be exceeded, or manifest identity checks fail.
- If a paid run fails mid-stream, rerun the same command. The scripts inspect
  existing call artifacts and ledgers so they can continue from missing rows.

## Step 1: Inspect The State

Run:

```bash
cd <sestina-repo>
git status --short
test -f artifacts/backtest-datasets/arxiv-historical-coverage-floor-fresh-holdout-manifest.json
find artifacts/backtest-datasets/arxiv-historical-coverage-floor-fresh-holdout-parts \
  -maxdepth 1 -type f -name '*.json' | sort
curl -fsS http://127.0.0.1:8766/v1/status || true
```

Acceptance:

- The fresh-holdout manifest exists.
- The parts directory includes all 8 bucket part manifests.
- Huldra status is optional for this run if the manifest already exists. If
  Huldra is unreachable, record it, but continue with pointwise planning.

## Step 2: Validate The Fresh-Holdout Manifest

Run:

```bash
uv run python -m json.tool \
  artifacts/backtest-datasets/arxiv-historical-coverage-floor-fresh-holdout-manifest.json \
  >/dev/null
uv run python - <<'PY'
import json
from pathlib import Path

path = Path(
    "artifacts/backtest-datasets/"
    "arxiv-historical-coverage-floor-fresh-holdout-manifest.json"
)
manifest = json.loads(path.read_text())
buckets = manifest["buckets"]
print("bucket_count", len(buckets))
print("bucket_names", [bucket["name"] for bucket in buckets])
print(
    "positive_labels",
    sum(bucket["source"]["diagnostics"]["positive_labels"] for bucket in buckets),
)
print(
    "papers",
    sum(bucket["source"]["diagnostics"]["papers_in_manifest"] for bucket in buckets),
)
for bucket in buckets:
    diagnostics = bucket["source"]["diagnostics"]
    if diagnostics["positive_labels"] != 5:
        raise SystemExit(f"{bucket['name']} has wrong positive label count")
    if diagnostics["papers_in_manifest"] < bucket["k"]:
        raise SystemExit(f"{bucket['name']} has fewer than K papers")
PY
```

Acceptance:

- `bucket_count` is `8`.
- Total positive labels is `40`.
- Each bucket has `positive_labels=5`.
- Each bucket has at least `K` papers.

## Step 3: Run Pointwise Planning

Run:

```bash
uv run python scripts/run_fresh_holdout_pointwise_artifacts.py --mode planning
```

Inspect the review artifact:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

path = Path(
    "artifacts/backtest-arxiv-coverage-floor-fresh-holdout-pointwise/"
    "fresh-holdout-pointwise-review.json"
)
payload = json.loads(path.read_text())
print("mode", payload["mode"])
print("paid_calls_made", payload["paid_calls_made"])
print("pointwise_calls_made", payload["pointwise_calls_made"])
print("status", payload["final_status"]["status"])
print("blocking_reasons", payload["final_status"]["blocking_reasons"])
print("expected", payload["fresh_holdout"]["expected_pointwise_artifacts"])
print("available", payload["fresh_holdout"]["available_pointwise_artifacts"])
print("missing", payload["fresh_holdout"]["missing_pointwise_artifacts"])
print("leakage", payload["model_visible_leakage_review"]["present"])
PY
```

Acceptance:

- `paid_calls_made` is `0`.
- `pointwise_calls_made` is `0`.
- `model_visible_leakage_review.present` is `false`.
- If artifacts are missing, the only expected blocker is
  `pointwise_artifacts_missing`.
- Do not run pointwise execute unless paid pointwise generation is explicitly
  authorized.

## Step 4: Execute Pointwise Artifact Generation

Run this step only with explicit paid-call authorization.

Before running, verify provider env vars exist without printing secrets:

```bash
test -n "${SESTINA_LLM_API_KEY:-}" && echo "SESTINA_LLM_API_KEY is set"
test -n "${SESTINA_LLM_BASE_URL:-}" && echo "SESTINA_LLM_BASE_URL is set"
```

Run:

```bash
uv run python scripts/run_fresh_holdout_pointwise_artifacts.py \
  --mode execute \
  --confirm-fresh-holdout-pointwise-generation \
  --timeout-seconds 60
```

This can take a long time because the full fresh holdout has hundreds of papers.
If the command fails after writing some artifacts, inspect the error, then rerun
the same command after fixing the cause. Do not delete successful call artifacts
or ledger rows.

Inspect the result:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

path = Path(
    "artifacts/backtest-arxiv-coverage-floor-fresh-holdout-pointwise/"
    "fresh-holdout-pointwise-review.json"
)
payload = json.loads(path.read_text())
print("status", payload["final_status"]["status"])
print("blocking_reasons", payload["final_status"]["blocking_reasons"])
print("paid_calls_made", payload["paid_calls_made"])
print("paid_spend_usd", payload["paid_spend_usd"])
print("pointwise_calls_made", payload["pointwise_calls_made"])
print("available", payload["fresh_holdout"]["available_pointwise_artifacts"])
print("missing", payload["fresh_holdout"]["missing_pointwise_artifacts"])
print("ledger", payload["ledger"])
PY
```

Acceptance:

- `final_status.status` is `complete`.
- `missing_pointwise_artifacts` is `0`.
- `pointwise_calls_made` equals `paid_calls_made`.
- `method.pairwise_calls_made` is `0`.
- The pointwise ledger is JSONL and under
  `artifacts/backtest-arxiv-coverage-floor-fresh-holdout-pointwise`.
- The projected campaign spend remains under USD 100.

## Step 5: Run Pairwise-Only Preflight

Run:

```bash
uv run python scripts/run_coverage_floor_followup_preflight.py
```

Inspect the preflight artifact:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

path = Path(
    "artifacts/backtest-arxiv-coverage-floor-followup-preflight/"
    "coverage-floor-followup-preflight.json"
)
payload = json.loads(path.read_text())
print("mode", payload["mode"])
print("dry_run", payload["dry_run"])
print("paid_calls_made", payload["paid_calls_made"])
print("pointwise_calls_made", payload["pointwise_calls_made"])
print("provider", payload["provider_model_availability"]["status"])
print("fresh_holdout_status", payload["fresh_holdout"]["status"])
print("pointwise_artifacts", payload["fresh_holdout"]["pointwise_artifacts"]["status"])
print("decision", payload["final_go_no_go"]["decision"])
print("blocking_reasons", payload["final_go_no_go"]["blocking_reasons"])
print("pairwise_calls_to_buy", payload["totals"]["pairwise_calls_to_buy"])
print("estimated_additional_spend_usd", payload["totals"]["estimated_additional_spend_usd"])
PY
```

Acceptance:

- `paid_calls_made` is `0`.
- `pointwise_calls_made` is `0`.
- `fresh_holdout.status` is `loaded`.
- `fresh_holdout.pointwise_artifacts.status` is `available`.
- `final_go_no_go.decision` is `go`.
- `provider_model_availability.status` is `available`.
- `planned-pair-occurrences.jsonl` is written in
  `artifacts/backtest-arxiv-coverage-floor-followup-preflight`.

If the decision is `no_go`, stop and report the blocking reasons. Do not edit
the planned pairs or artifacts by hand.

## Step 6: Execute Guarded Pairwise Labels

Run this step only with explicit paid pairwise authorization and only after
Step 5 reports `decision=go`.

Run:

```bash
uv run python scripts/run_coverage_floor_followup_preflight.py \
  --mode execute \
  --confirm-guarded-pairwise-only-execution \
  --timeout-seconds 60
```

If the command fails after writing some pairwise artifacts, rerun the same
command after fixing the cause. The runner should reuse existing successful
artifacts and buy only missing unique pairwise labels.

Inspect the execution artifact:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

path = Path(
    "artifacts/backtest-arxiv-coverage-floor-followup-preflight/"
    "coverage-floor-followup-preflight.json"
)
payload = json.loads(path.read_text())
print("mode", payload["mode"])
print("paid_calls_made", payload["paid_calls_made"])
print("paid_spend_usd", payload["paid_spend_usd"])
print("pointwise_calls_made", payload["pointwise_calls_made"])
print("execution", payload["execution_summary"])
print("ledger", payload["ledger"])
print("decision", payload["final_go_no_go"]["decision"])
print("blocking_reasons", payload["final_go_no_go"]["blocking_reasons"])
PY
```

Acceptance:

- `mode` is `execute`.
- `pointwise_calls_made` is `0`.
- `paid_calls_made` equals the number of successful pairwise labels bought in
  this execution pass.
- `execution_summary.paid_pairwise_calls_succeeded` matches `paid_calls_made`.
- The pairwise ledger is JSONL and under
  `artifacts/backtest-arxiv-coverage-floor-followup-preflight`.
- No pointwise-like call was attempted.
- Spend remains under the requested `--max-usd` and the USD 100 campaign cap.

## Step 7: Analyze Fresh Validation

Run:

```bash
uv run python scripts/analyze_coverage_floor_fresh_validation.py
```

This command may exit `2` when the analysis is incomplete. Inspect the artifact
either way:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

path = Path(
    "artifacts/backtest-arxiv-autonomous-holdout-campaign/"
    "fresh-validation-analysis.json"
)
payload = json.loads(path.read_text())
print("complete", payload["fresh_validation_claim"]["complete"])
print(
    "can_claim_fresh_paid_validation",
    payload["fresh_validation_claim"]["can_claim_fresh_paid_validation"],
)
print("completeness", payload["completeness"])
print("primary_metric", payload["fresh_validation_claim"]["primary_metric"])
print("paired_deltas", payload["paired_deltas_vs_exact_pool_random"])
PY
```

Acceptance:

- `paid_calls_made` is `0`.
- `pointwise_calls_made` is `0`.
- `fresh_validation_claim.complete` is `true`.
- `fresh_validation_claim.can_claim_fresh_paid_validation` is `true`.
- `completeness.missing_pointwise_buckets` is `0`.
- `completeness.missing_pairwise_occurrences` is `0`.

If `fresh_validation_claim.complete` is `false`, report the completeness fields
and do not claim fresh validation success.

## Step 8: Run Focused Tests

Run:

```bash
uv run pytest tests/test_fresh_holdout_pointwise_artifacts.py \
  tests/test_coverage_floor_followup_preflight.py \
  tests/test_coverage_floor_fresh_validation_analysis.py
git diff --check
```

Acceptance:

- All listed tests pass.
- `git diff --check` passes.

## Step 9: Final Report

Report:

- Whether paid pointwise execution was authorized and run.
- Pointwise artifact count, missing count, paid calls, and spend.
- Whether pairwise preflight returned `go`.
- Whether paid pairwise execution was authorized and run.
- Pairwise calls bought, ledger path, and spend.
- Whether fresh validation analysis is complete.
- Primary Recall@K result and paired active-minus-random deltas.
- Any blockers, especially provider availability, leakage, missing artifacts,
  missing pairwise labels, ledger errors, or cap failures.
- Confirmation that Huldra metadata is no longer the active blocker.
