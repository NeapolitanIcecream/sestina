# Recoleta Integration Notes

This note records the read-only integration seam inspected in
`/Users/chenmohan/gits/recoleta`.

## Current Recoleta Triage Shape

Recoleta's semantic triage lives in `recoleta/triage.py`.

- `TriageSelectionRequest` carries `run_id`, candidates, topics, limit, mode,
  query mode, embedding model config, threshold, exploration rate, recency floor,
  and debug settings.
- `SemanticTriage.select(...)` returns `TriageOutput` with selected candidates,
  stats, and optional artifacts.
- The current scoring path tries embeddings and cosine similarity first, then
  falls back to lexical scoring if embeddings fail.
- Artifacts already include request metadata, response/error debug data, and a
  bounded triage summary.

## Replacement Shape

An adapter can map Recoleta items into Sestina papers:

- `Item.id` to `Paper.paper_id`
- title and local excerpt to `Paper.title` / `Paper.abstract`
- existing LLM analysis output to `PointwiseAssessment`
- source, venue, date, and topic tags to `Paper.metadata`

Sestina then returns recommended IDs, near misses, scheduled pairwise
comparisons, and diagnostics. Recoleta can use those IDs where it currently uses
semantic ranking output, while preserving its existing fail-open behavior and
debug artifact conventions.

This repository does not depend on Recoleta and did not modify it.

