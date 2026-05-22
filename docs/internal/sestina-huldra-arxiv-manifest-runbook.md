# Sestina Huldra arXiv Manifest Runbook

Use this runbook to switch Sestina's historical arXiv manifest builder from
direct arXiv API requests to the local Huldra broker, then verify the frozen
fresh-holdout manifest can be built. The goal is to unblock the manifest build
that previously stopped on arXiv HTTP 429.

Workspace:

```text
<sestina-repo>
```

Huldra repo:

```text
<huldra-repo>
```

Expected local Huldra API:

```text
http://127.0.0.1:8766
```

## Guardrails

- Preserve existing user changes. Inspect the worktree before editing.
- Do not edit historical paid ledgers, raw paid-call artifacts, old planned-pair
  manifests, or `.codex-workflows/**`.
- Do not change the frozen fresh-holdout bucket list, `K`, labels, scheduling
  policy, or evaluation logic.
- Do not run paid execution commands in this runbook.
- Huldra is a metadata transport and cache. It does not change the experiment
  design.

Forbidden commands for this runbook:

```bash
uv run python scripts/run_fresh_holdout_pointwise_artifacts.py --mode execute
uv run python scripts/run_coverage_floor_followup_preflight.py --mode execute
```

## Step 1: Inspect The Current State

Run:

```bash
cd <sestina-repo>
git status --short
git diff -- scripts/build_arxiv_historical_manifest.py \
  tests/test_arxiv_historical_manifest.py \
  scripts/design_coverage_floor_fresh_holdout.py \
  docs/internal/sestina-autonomous-holdout-campaign.md
find artifacts/backtest-datasets/arxiv-historical-coverage-floor-fresh-holdout-parts \
  -maxdepth 1 -type f -name '*.json' | sort
curl -fsS http://127.0.0.1:8766/healthz
curl -fsS http://127.0.0.1:8766/v1/status
```

Acceptance:

- The Huldra health endpoint returns `{"status":"ok"}`.
- The parts directory shows 7 existing bucket part manifests.
- Existing changes are understood before editing.

If Huldra is not reachable, continue with code and unit-test work, but do not
attempt the live manifest build. Report that live verification is blocked by the
local Huldra service state.

## Step 2: Add Huldra CLI Options

Edit `scripts/build_arxiv_historical_manifest.py`.

Add these CLI options:

```text
--arxiv-metadata-source direct|huldra
--huldra-base-url http://127.0.0.1:8766
--huldra-wait-timeout-seconds 600
--huldra-client-id sestina-historical-arxiv-manifest
```

Keep `direct` as the default source for backward compatibility. The frozen
fresh-holdout command will pass `--arxiv-metadata-source huldra` explicitly.

Acceptance:

- Existing direct arXiv behavior remains available.
- Huldra settings are exposed as explicit CLI flags.
- No new third-party dependency is added to `pyproject.toml`.

## Step 3: Implement Huldra Fetching

Add a Huldra fetch helper in `scripts/build_arxiv_historical_manifest.py` using
stdlib HTTP calls. Do not import Huldra Python modules from `<huldra-repo>`.

For each bucket, submit a Huldra request through `/v1/sync` with `wait=true`,
then read the same request through `/v1/requests` with `cache_policy=cache_only`
and `readiness=analysis_ready`.

Use this request shape for a bucket:

```json
{
  "client_id": "sestina-historical-arxiv-manifest",
  "search_query": "cat:cs.CV",
  "submitted_start": "2023-04-01T00:00:00+00:00",
  "submitted_end": "2023-05-01T00:00:00+00:00",
  "start": 0,
  "max_results": 80,
  "sort_by": "submittedDate",
  "sort_order": "ascending",
  "cache_policy": "cache_or_enqueue",
  "readiness": "analysis_ready",
  "timeout_seconds": 600
}
```

Rules:

- Use `search_query="cat:<CATEGORY>"`.
- Put the historical submitted-date window in `submitted_start` and
  `submitted_end`.
- Do not embed `submittedDate:[...]` in `search_query` for Huldra mode.
- Use an exclusive `submitted_end`, for example `2023-05-01T00:00:00+00:00`
  for the April 2023 bucket.
- Preserve `sort_by="submittedDate"` and `sort_order="ascending"`.
- Preserve the builder's `limit` as Huldra `max_results`.

Map Huldra paper JSON back to the existing local `ArxivPaper` dataclass:

- `arxiv_id`: versionless ID, using existing `strip_arxiv_version`.
- `versioned_arxiv_id`: Huldra `arxiv_id`.
- `title`: Huldra `title`.
- `abstract`: Huldra `abstract` or empty string.
- `primary_category`: Huldra `primary_category` or empty string.
- `categories`: Huldra `categories` or an empty list.
- `published_at`: Huldra `published_at` as an ISO string.
- `updated_at`: Huldra `updated_at` as an ISO string.
- `doi`: Huldra `doi`.
- `authors`: Huldra `authors` as a tuple.

Error behavior:

- If Huldra returns `cooling_down`, `rate_limited`, upstream status `429`, or a
  request-level cooldown, fail like the existing arXiv 429 path and return exit
  code `75` from the CLI.
- If Huldra returns `cache_miss`, `queued`, `timeout`, `failed`, or any response
  without usable papers after waiting, fail with a clear message containing
  Huldra `status`, `cache_key`, and `blocked_reason` when present.
- Direct mode must keep the current retry and 429 behavior.

Acceptance:

- `fetch_arxiv_papers(...)` or a nearby wrapper can fetch from Huldra when
  requested.
- Existing manifest construction, citation matching, labels, and part manifest
  reuse are unchanged.
- A Huldra 429/cooldown path exits with code `75`, not a generic traceback.

## Step 4: Add Unit Tests

Update `tests/test_arxiv_historical_manifest.py`.

Add focused tests for Huldra mode:

1. Huldra mode posts to `/v1/sync`, then `/v1/requests`.
2. The Huldra request uses `search_query="cat:cs.LG"` and
   `submitted_start`/`submitted_end`.
3. The Huldra request does not put `submittedDate` inside `search_query`.
4. Huldra paper JSON converts to local `ArxivPaper`, including versioned and
   versionless IDs.
5. Huldra `cooling_down`, `rate_limited`, or upstream `429` maps to the existing
   rate-limit failure path.
6. Existing direct arXiv fetch tests still pass.

Use fake `urlopen` responses or another in-test HTTP fake. Do not require a live
Huldra daemon for unit tests.

Run:

```bash
uv run pytest tests/test_arxiv_historical_manifest.py
```

Acceptance:

- The focused test file passes.
- Tests do not make network calls.

## Step 5: Update The Fresh-Holdout Design Command

Edit `scripts/design_coverage_floor_fresh_holdout.py` so generated builder
commands include:

```text
--arxiv-metadata-source huldra
--huldra-base-url http://127.0.0.1:8766
--huldra-wait-timeout-seconds 600
```

Do not change the selected buckets.

Expected buckets:

```text
cs.LG:2023-03
cs.LG:2023-04
cs.CL:2023-03
cs.CL:2023-04
cs.AI:2023-03
cs.AI:2023-04
cs.CV:2023-03
cs.CV:2023-04
```

Update `tests/test_coverage_floor_fresh_holdout_design.py` to assert the Huldra
flags are present in the generated builder command.

Run:

```bash
uv run pytest tests/test_coverage_floor_fresh_holdout_design.py
```

Acceptance:

- The generated builder command uses Huldra.
- Bucket identities remain unchanged.

## Step 6: Run Focused Verification

Run:

```bash
uv run pytest tests/test_arxiv_historical_manifest.py \
  tests/test_coverage_floor_fresh_holdout_design.py
git diff --check
```

Acceptance:

- Both focused test files pass.
- `git diff --check` passes.

## Step 7: Build The Fresh-Holdout Manifest Through Huldra

Run the frozen manifest build with part reuse:

```bash
uv run python scripts/build_arxiv_historical_manifest.py \
  --bucket cs.LG:2023-03 \
  --bucket cs.LG:2023-04 \
  --bucket cs.CL:2023-03 \
  --bucket cs.CL:2023-04 \
  --bucket cs.AI:2023-03 \
  --bucket cs.AI:2023-04 \
  --bucket cs.CV:2023-03 \
  --bucket cs.CV:2023-04 \
  --limit 80 \
  --arxiv-metadata-source huldra \
  --huldra-base-url http://127.0.0.1:8766 \
  --huldra-wait-timeout-seconds 600 \
  --k 5 \
  --phase pilot \
  --metadata-provider auto \
  --unmatched-policy drop \
  --part-dir artifacts/backtest-datasets/arxiv-historical-coverage-floor-fresh-holdout-parts \
  --reuse-parts \
  --write-parts \
  --target-bucket-count 8 \
  --output artifacts/backtest-datasets/arxiv-historical-coverage-floor-fresh-holdout-manifest.json
```

Expected runtime:

- If 7 existing part manifests are reusable and Huldra has no cooldown, the only
  missing arXiv bucket should be `cs.CV:2023-04`.
- The arXiv metadata portion should usually take seconds to a few minutes.
- Citation metadata enrichment for the missing bucket can take several minutes.
- If Huldra hits arXiv 429, the command should exit `75`; Huldra may enter a
  cooldown, commonly around 1 hour depending on settings.

Acceptance:

- The command exits `0`.
- The output reports 8 buckets.
- This file exists:

```text
artifacts/backtest-datasets/arxiv-historical-coverage-floor-fresh-holdout-parts/cs_CV_2023-04.json
```

- This final manifest exists:

```text
artifacts/backtest-datasets/arxiv-historical-coverage-floor-fresh-holdout-manifest.json
```

If the command exits `75`, run:

```bash
curl -fsS http://127.0.0.1:8766/v1/status
```

Report `cooldown_active`, `cooldown_until`, `upstream_429_total`, and the failed
bucket. Do not retry in a tight loop.

## Step 8: Validate The Manifest

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
- All expected bucket names are present.
- Total positive labels is `40`.
- Every bucket has `positive_labels=5`.
- Every bucket has at least `K` papers.

## Step 9: Run Downstream Planning Only

Run:

```bash
uv run python scripts/run_fresh_holdout_pointwise_artifacts.py --mode planning
uv run python scripts/run_coverage_floor_followup_preflight.py
```

Acceptance:

- Both commands report `paid_calls_made=0`.
- Both commands report `pointwise_calls_made=0`.
- The preflight no longer reports `fresh_holdout_manifest_missing`.
- It is acceptable for reviewed pointwise artifacts to be missing at this stage.
- Do not proceed to paid execution from this runbook.

## Step 10: Run Final Focused Tests

Run:

```bash
uv run pytest tests/test_arxiv_historical_manifest.py \
  tests/test_coverage_floor_fresh_holdout_design.py \
  tests/test_fresh_holdout_pointwise_artifacts.py \
  tests/test_coverage_floor_followup_preflight.py \
  tests/test_coverage_floor_fresh_validation_analysis.py
git diff --check
```

Acceptance:

- All listed tests pass.
- `git diff --check` passes.

## Step 11: Final Report

Report these items:

- Files changed.
- Whether Huldra health and status endpoints were reachable.
- Whether the fresh-holdout manifest was built.
- Number of buckets and total positive labels in the final manifest.
- Whether `cs_CV_2023-04.json` was created or already present.
- Whether downstream planning advanced past `fresh_holdout_manifest_missing`.
- Any blocker, especially Huldra cooldown, failed Huldra status, or citation
  metadata failure.
- Confirmation that no paid pointwise or pairwise execute command was run.
