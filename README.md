# Sestina

Sestina runs pointwise-first, pairwise-light top-K paper discovery. It starts
from one assessment per paper, schedules only a small number of pairwise checks
around the decision boundary, and returns recommended good papers with
probabilities, tiers, near misses, caveats, and structured diagnostics.

This is an offline v1 package. Tests and examples do not call an LLM.

## Install

Use Python 3.11 or newer.

```bash
python -m pip install -e ".[dev]"
```

Run the tests:

```bash
python -m pytest
```

## Input Format

JSON input contains a target, papers, and optional judged comparisons:

```json
{
  "target": {"top_k": 3},
  "mode": "content_only",
  "papers": [
    {
      "paper_id": "p1",
      "title": "Paper title",
      "abstract": "Optional abstract or local summary",
      "pointwise": {
        "pointwise_good_probability": 0.82,
        "uncertainty": 0.25,
        "rubric_scores": {"novelty": 4.0, "evidence": 3.5},
        "summary": "Short assessment summary.",
        "reasons": ["why it may be good"]
      },
      "metadata": {"topic": "agents", "venue": "arXiv"}
    }
  ],
  "comparisons": [
    {
      "left_id": "p1",
      "right_id": "p2",
      "winner": "left",
      "soft_probability": 0.78,
      "confidence": 0.8,
      "reasons": ["clearer evidence"],
      "order": {
        "shown_first_id": "p2",
        "shown_second_id": "p1",
        "randomized": true,
        "seed": 17,
        "position_bias_audit": true
      }
    }
  ]
}
```

Targets use exactly one of:

- `{"top_k": 10}`
- `{"top_alpha": 0.1}`

JSONL input is also supported. Each line is one paper object. Pass `--top-k` or
`--top-alpha` on the CLI, and pass comparisons separately with `--comparisons`.

## CLI

Run the sample fixture and write JSON plus a Markdown report:

```bash
python -m sestina.cli run examples/sample_papers.json \
  --output output/sample-result.json \
  --report output/sample-report.md \
  --debug-dir output/debug \
  --posterior-samples 1000 \
  --seed 17
```

After installation, the console script is:

```bash
sestina run examples/sample_papers.json --output output/sample-result.json
```

The JSON output includes:

- `target`: resolved `K*`
- `candidate_selection`: exploit, boundary, and exploration candidate IDs
- `pairwise_schedule`: suggested comparisons, randomized A/B order, and audit
  metadata
- `aggregation`: MAP Bayesian Bradley-Terry estimates from pointwise priors and
  optional pairwise evidence
- `posterior`: approximate top-K probabilities from posterior sampling
- `recommendations`: recommended good papers, near misses, tiers, reasons, and
  caveats
- `diagnostics`: machine-readable events for every runtime stage

## Budgets

For `n` papers and resolved `K*`, the default candidate size is:

```text
M = min(n, ceil(3K* + sqrt(n)))
```

The default pairwise budget is:

```text
B_pair = min(ceil(1.25M), ceil(0.25n))
```

This keeps fixed top-K runs near `O(K + sqrt(n))` and top-alpha runs near
`O(alpha n + sqrt(n))`. The scheduler uses the budget to sample boundary,
closeness, uncertainty, diversity, and position-bias audit comparisons. It does
not schedule a full ranking.

Override budgets when needed:

```bash
sestina run papers.json --top-k 20 --candidate-size 80 --pairwise-budget 25
```

`--candidate-size` is optional. Omit it to use the default formula; explicit
overrides must be at least the resolved `K*`.

## Modeling Notes

Pointwise probabilities become logit priors. Pairwise comparisons are ingested as
fractional Bradley-Terry evidence:

- `winner: left` or `right` uses `soft_probability` and `confidence`
- `winner: tie` pulls both papers mildly together
- `winner: uncertain` has low weight

The v1 optimizer computes a MAP estimate and uses a diagonal Laplace
approximation for posterior sampling. This is intentionally inspectable and does
not pretend to provide an exact full ranking.

## LLM Judge Hook

`sestina.llm` defines a `PairwiseJudge` protocol, a deterministic mock judge for
tests, and an optional OpenAI-compatible judge. The network-backed judge reads:

- `SESTINA_LLM_API_KEY`
- `SESTINA_LLM_BASE_URL`
- `SESTINA_LLM_MODEL` (optional)

Do not put secret values in input files. Sestina diagnostics redact obvious
secret fields and do not store paper full text by default.

Use `ScheduledPairwiseJudgeAdapter` or `compare_scheduled_pair(...)` when
executing `pairwise_schedule` entries. The adapter presents papers in the
scheduled randomized A/B order, maps A/B/tie/uncertain results back to canonical
`left_id` / `right_id`, and carries the schedule's position-bias audit metadata
onto the resulting comparison.

## Development

Useful checks:

```bash
python -m pytest
python -m sestina.cli run examples/sample_papers.json --output output/sample-result.json
```
