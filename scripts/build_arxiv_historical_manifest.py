#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from difflib import SequenceMatcher
from http.client import HTTPMessage
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
ARXIV_API_URL = "https://export.arxiv.org/api/query"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_GRAPH_URL = "https://api.semanticscholar.org/graph/v1/paper"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

UrlOpen = Callable[..., Any]
Sleep = Callable[[float], None]
Clock = Callable[[], float]
MetadataProvider = Literal["semantic_scholar", "openalex", "auto"]
UnmatchedPolicy = Literal["drop", "zero", "fail"]


@dataclass(frozen=True, slots=True)
class DateBucket:
    label: str
    start: date
    end: date


@dataclass(frozen=True, slots=True)
class BucketSpec:
    category: str
    date_bucket: DateBucket


@dataclass(frozen=True, slots=True)
class ArxivPacingConfig:
    delays_seconds: tuple[float, ...] = (15.0, 10.0, 6.0, 3.5)
    successes_per_step: int = 3
    min_delay_seconds: float = 3.5

    def __post_init__(self) -> None:
        if self.successes_per_step <= 0:
            raise ValueError("arXiv pacing successes per step must be greater than zero")
        if not self.delays_seconds:
            raise ValueError("at least one arXiv pacing delay is required")
        delays = tuple(float(delay) for delay in self.delays_seconds)
        too_fast = [delay for delay in delays if delay < self.min_delay_seconds]
        if too_fast:
            raise ValueError(
                "arXiv pacing delays must be at least "
                f"{self.min_delay_seconds:g} seconds"
            )
        object.__setattr__(self, "delays_seconds", delays)


class ArxivRequestPacer:
    def __init__(
        self,
        config: ArxivPacingConfig,
        *,
        sleep: Sleep = time.sleep,
        monotonic: Clock = time.monotonic,
    ) -> None:
        self._config = config
        self._sleep = sleep
        self._monotonic = monotonic
        self._delay_index = 0
        self._consecutive_successes = 0
        self._last_request_at: float | None = None

    @property
    def current_delay_seconds(self) -> float:
        return self._config.delays_seconds[self._delay_index]

    def before_request(self) -> None:
        now = self._monotonic()
        if self._last_request_at is not None:
            elapsed = now - self._last_request_at
            wait_seconds = self.current_delay_seconds - elapsed
            if wait_seconds > 0:
                self._sleep(wait_seconds)
                now = self._monotonic()
        self._last_request_at = now

    def record_success(self) -> None:
        self._consecutive_successes += 1
        if (
            self._consecutive_successes >= self._config.successes_per_step
            and self._delay_index < len(self._config.delays_seconds) - 1
        ):
            self._delay_index += 1
            self._consecutive_successes = 0

    def record_rate_limited(self) -> None:
        self._consecutive_successes = 0


@dataclass(frozen=True, slots=True)
class ArxivPaper:
    arxiv_id: str
    versioned_arxiv_id: str
    title: str
    abstract: str
    primary_category: str
    categories: list[str]
    published_at: str
    updated_at: str
    doi: str | None = None
    authors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CitationMatch:
    provider: str
    matched: bool
    cited_by_count: int | None = None
    work_id: str | None = None
    title: str | None = None
    doi: str | None = None
    publication_date: str | None = None
    method: str | None = None
    match_score: float | None = None
    error: str | None = None

    def to_label_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": self.provider,
            "matched": self.matched,
        }
        if self.cited_by_count is not None:
            payload["cited_by_count"] = self.cited_by_count
        if self.work_id:
            payload["work_id"] = self.work_id
        if self.title:
            payload["matched_title"] = self.title
        if self.doi:
            payload["doi"] = self.doi
        if self.publication_date:
            payload["publication_date"] = self.publication_date
        if self.method:
            payload["method"] = self.method
        if self.match_score is not None:
            payload["match_score"] = round(self.match_score, 6)
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True, slots=True)
class LabeledPaper:
    paper: ArxivPaper
    match: CitationMatch
    good_paper: bool
    citation_rank: int
    citation_percentile: float


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Sestina backtest manifest from historical arXiv buckets "
            "and later public citation metadata."
        )
    )
    parser.add_argument("--category", help="arXiv category, e.g. cs.LG")
    parser.add_argument(
        "--bucket",
        action="append",
        default=[],
        help=(
            "repeatable pilot bucket spec CATEGORY:YYYY-MM or CATEGORY:YYYY-QN; "
            "cannot be combined with --category/--month/--quarter/date range"
        ),
    )
    parser.add_argument("--month", help="UTC month bucket in YYYY-MM format")
    parser.add_argument("--quarter", help="UTC quarter bucket in YYYY-QN format")
    parser.add_argument("--start-date", type=_parse_date, help="inclusive YYYY-MM-DD")
    parser.add_argument("--end-date", type=_parse_date, help="inclusive YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=80, help="max arXiv papers to fetch")
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="top-K citation positives within the matched bucket",
    )
    parser.add_argument(
        "--top-alpha",
        type=float,
        help="alternative positive rate; positives = ceil(alpha * n)",
    )
    parser.add_argument("--phase", default="smoke")
    parser.add_argument(
        "--metadata-provider",
        choices=["semantic_scholar", "openalex", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--unmatched-policy",
        choices=["drop", "zero", "fail"],
        default="drop",
        help="how to handle papers without citation metadata",
    )
    parser.add_argument(
        "--min-papers",
        type=int,
        help="minimum labeled papers after metadata matching; defaults to K",
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=1.0,
        help="delay between metadata API calls",
    )
    parser.add_argument(
        "--arxiv-retry-attempts",
        type=int,
        default=5,
        help="retry attempts for transient arXiv API fetch failures",
    )
    parser.add_argument(
        "--arxiv-retry-delay-seconds",
        type=float,
        default=5.0,
        help="initial retry delay for transient arXiv API fetch failures",
    )
    parser.add_argument(
        "--arxiv-pacing-delays-seconds",
        default="15,10,6,3.5",
        help=(
            "comma-separated arXiv request spacing tiers; each value must be "
            "at least 3.5 seconds"
        ),
    )
    parser.add_argument(
        "--arxiv-pacing-successes-per-step",
        type=int,
        default=3,
        help="consecutive successful arXiv requests before moving to the next tier",
    )
    parser.add_argument(
        "--part-dir",
        type=Path,
        help="directory of per-bucket manifests for resumable pilot builds",
    )
    parser.add_argument(
        "--reuse-parts",
        action="store_true",
        help="reuse valid per-bucket manifests from --part-dir when present",
    )
    parser.add_argument(
        "--write-parts",
        action="store_true",
        help="write each successfully fetched bucket manifest to --part-dir",
    )
    parser.add_argument(
        "--target-bucket-count",
        type=int,
        help="stop after this many valid bucket manifests have been collected",
    )
    parser.add_argument(
        "--openalex-mailto",
        help="optional mailto parameter for OpenAlex polite pool",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "artifacts"
        / "backtest-datasets"
        / "arxiv-historical-smoke-manifest.json",
    )
    args = parser.parse_args(argv)

    try:
        bucket_specs = resolve_cli_bucket_specs(
            category=args.category,
            bucket_values=args.bucket,
            month=args.month,
            quarter=args.quarter,
            start=args.start_date,
            end=args.end_date,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if (args.reuse_parts or args.write_parts) and args.part_dir is None:
        parser.error("--part-dir is required with --reuse-parts or --write-parts")
    if args.target_bucket_count is not None and args.target_bucket_count <= 0:
        parser.error("--target-bucket-count must be greater than zero")
    try:
        arxiv_pacing = parse_arxiv_pacing_config(
            args.arxiv_pacing_delays_seconds,
            successes_per_step=args.arxiv_pacing_successes_per_step,
        )
    except ValueError as exc:
        parser.error(str(exc))

    pacer = ArxivRequestPacer(arxiv_pacing)
    bucket_manifests = []
    reused_parts = 0
    fetched_buckets = 0
    for spec in bucket_specs:
        if (
            args.target_bucket_count is not None
            and len(bucket_manifests) >= args.target_bucket_count
        ):
            break

        part_path = part_manifest_path(args.part_dir, spec) if args.part_dir else None
        if args.reuse_parts and part_path is not None and part_path.exists():
            bucket_manifests.append(
                load_reusable_part_manifest(
                    part_path,
                    spec=spec,
                    phase=args.phase,
                    k=args.k,
                )
            )
            reused_parts += 1
            continue

        try:
            bucket_manifest = build_manifest(
                category=spec.category,
                date_bucket=spec.date_bucket,
                limit=args.limit,
                k=args.k,
                top_alpha=args.top_alpha,
                phase=args.phase,
                metadata_provider=args.metadata_provider,
                unmatched_policy=args.unmatched_policy,
                min_papers=args.min_papers,
                request_delay_seconds=args.request_delay_seconds,
                openalex_mailto=args.openalex_mailto,
                arxiv_retry_attempts=args.arxiv_retry_attempts,
                arxiv_retry_delay_seconds=args.arxiv_retry_delay_seconds,
                arxiv_pacer=pacer,
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                sys.stderr.write(
                    "stopped after arXiv HTTP 429 for "
                    f"{spec.category}:{spec.date_bucket.label}; "
                    f"retry_after={retry_after or 'not_provided'}\n"
                )
                return 75
            raise

        if args.write_parts and part_path is not None:
            part_path.parent.mkdir(parents=True, exist_ok=True)
            part_path.write_text(
                json.dumps(bucket_manifest, indent=2, sort_keys=True) + "\n"
            )
        bucket_manifests.append(bucket_manifest)
        fetched_buckets += 1

    if (
        args.target_bucket_count is not None
        and len(bucket_manifests) < args.target_bucket_count
    ):
        raise RuntimeError(
            f"only collected {len(bucket_manifests)} bucket manifests; "
            f"target is {args.target_bucket_count}"
        )

    manifest = combine_manifests(bucket_manifests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    diagnostics = [
        bucket["source"]["diagnostics"] for bucket in manifest.get("buckets", [])
    ]
    sys.stdout.write(
        "wrote "
        f"{args.output} with {len(diagnostics)} bucket(s), "
        f"{sum(item['papers_in_manifest'] for item in diagnostics)} papers, "
        f"{sum(item['positive_labels'] for item in diagnostics)} citation positives, "
        f"{sum(item['metadata_matches'] for item in diagnostics)} metadata matches, "
        f"{reused_parts} reused part(s), {fetched_buckets} fetched bucket(s)\n"
    )
    return 0


def build_manifest(
    *,
    category: str,
    date_bucket: DateBucket,
    limit: int,
    k: int,
    top_alpha: float | None,
    phase: str,
    metadata_provider: MetadataProvider,
    unmatched_policy: UnmatchedPolicy,
    min_papers: int | None = None,
    request_delay_seconds: float = 1.0,
    openalex_mailto: str | None = None,
    arxiv_retry_attempts: int = 5,
    arxiv_retry_delay_seconds: float = 5.0,
    arxiv_pacer: ArxivRequestPacer | None = None,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> dict[str, Any]:
    if arxiv_pacer is not None:
        arxiv_pacer.before_request()
    try:
        papers = fetch_arxiv_papers(
            category=category,
            start=date_bucket.start,
            end=date_bucket.end,
            limit=limit,
            urlopen=urlopen,
            retry_attempts=arxiv_retry_attempts,
            retry_delay_seconds=arxiv_retry_delay_seconds,
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 429 and arxiv_pacer is not None:
            arxiv_pacer.record_rate_limited()
        raise
    if arxiv_pacer is not None:
        arxiv_pacer.record_success()
    matches = fetch_citation_matches(
        papers,
        provider=metadata_provider,
        request_delay_seconds=request_delay_seconds,
        openalex_mailto=openalex_mailto,
        urlopen=urlopen,
    )
    return build_manifest_from_records(
        papers=papers,
        matches=matches,
        category=category,
        date_bucket=date_bucket,
        requested_limit=limit,
        k=k,
        top_alpha=top_alpha,
        phase=phase,
        metadata_provider=metadata_provider,
        unmatched_policy=unmatched_policy,
        min_papers=min_papers,
    )


def build_manifest_from_records(
    *,
    papers: list[ArxivPaper],
    matches: dict[str, CitationMatch],
    category: str,
    date_bucket: DateBucket,
    requested_limit: int,
    k: int,
    top_alpha: float | None,
    phase: str,
    metadata_provider: MetadataProvider,
    unmatched_policy: UnmatchedPolicy,
    min_papers: int | None = None,
) -> dict[str, Any]:
    if k <= 0:
        raise ValueError("K must be greater than zero")
    if top_alpha is not None and not 0 < top_alpha <= 1:
        raise ValueError("--top-alpha must be in (0, 1]")

    usable: list[tuple[ArxivPaper, CitationMatch]] = []
    unmatched: list[tuple[ArxivPaper, CitationMatch]] = []
    for paper in papers:
        match = matches.get(
            paper.arxiv_id,
            CitationMatch(provider=str(metadata_provider), matched=False, error="missing"),
        )
        if match.matched and match.cited_by_count is not None:
            usable.append((paper, match))
        elif unmatched_policy == "zero":
            usable.append(
                (
                    paper,
                    CitationMatch(
                        provider=match.provider,
                        matched=False,
                        cited_by_count=0,
                        method=match.method,
                        error=match.error or "unmatched_as_zero",
                    ),
                )
            )
        elif unmatched_policy == "fail":
            raise RuntimeError(
                f"citation metadata missing for arXiv paper {paper.arxiv_id}"
            )
        else:
            unmatched.append((paper, match))

    min_required = min_papers if min_papers is not None else k
    if len(usable) < min_required:
        raise RuntimeError(
            f"only {len(usable)} papers have usable citation metadata; "
            f"minimum required is {min_required}"
        )
    if len(usable) < k:
        raise RuntimeError(
            f"only {len(usable)} labeled papers available; cannot assign K={k}"
        )

    labeled, positive_count = assign_citation_labels(usable, k=k, top_alpha=top_alpha)
    if positive_count < k and top_alpha is None:
        raise RuntimeError(
            f"only {positive_count} positive labels assigned; expected at least K={k}"
        )
    generated_at = datetime.now(UTC).isoformat()
    bucket_name = _bucket_name(
        category=category,
        label=date_bucket.label,
        phase=phase,
    )

    return {
        "artifact_type": "sestina-backtest-dataset-manifest",
        "version": 1,
        "generated_at": generated_at,
        "read_only_source": True,
        "label_definition": {
            "good_paper": (
                "Top cited papers within the same historical arXiv category/date "
                "bucket using later public citation metadata."
            ),
            "citation_source": str(metadata_provider),
            "citation_count_field": "labels.citation_count",
            "positive_rule": (
                f"top {positive_count} by citation count within bucket"
                if top_alpha is not None
                else f"top K={k} by citation count within bucket"
            ),
            "metadata_fetch_time": generated_at,
            "model_visible_evidence": (
                "title, abstract, primary category, category list, publication "
                "date, update date, and bucket date range only"
            ),
            "non_oracle_baseline": (
                "baseline_score is a constant 0.5 and does not use future "
                "citation metadata"
            ),
        },
        "schema": {
            "bucket_required_fields": ["name", "phase", "k", "papers"],
            "paper_required_fields": [
                "paper_id",
                "title",
                "abstract",
                "baseline_score",
                "labels.good_paper",
                "metadata",
            ],
        },
        "buckets": [
            {
                "name": bucket_name,
                "phase": phase,
                "k": k,
                "source": {
                    "kind": "historical_arxiv_citation_backtest",
                    "paper_source": "arxiv_api",
                    "metadata_provider": str(metadata_provider),
                    "category": category,
                    "date_bucket": date_bucket.label,
                    "start_date": date_bucket.start.isoformat(),
                    "end_date": date_bucket.end.isoformat(),
                    "requested_limit": requested_limit,
                    "selection": (
                        "arXiv API category/date bucket ordered by submitted date; "
                        "papers without usable citation metadata are omitted unless "
                        "the manifest was built with --unmatched-policy zero"
                    ),
                    "label_source": (
                        "later public citation counts fetched after the historical "
                        "publication window; citation counts are stored only under "
                        "labels and are not placed in model-visible metadata"
                    ),
                    "baseline": "neutral_non_oracle_constant_0.5",
                    "diagnostics": {
                        "arxiv_papers_fetched": len(papers),
                        "papers_in_manifest": len(labeled),
                        "metadata_matches": sum(
                            1
                            for match in matches.values()
                            if match.matched and match.cited_by_count is not None
                        ),
                        "metadata_unmatched": len(papers) - len(usable),
                        "positive_labels": positive_count,
                        "unmatched_policy": unmatched_policy,
                        "dropped_unmatched_arxiv_ids": [
                            paper.arxiv_id for paper, _match in unmatched
                        ][:100],
                    },
                },
                "papers": [
                    _paper_payload(
                        labeled_paper,
                        bucket_name=bucket_name,
                        ordinal=index,
                        positive_count=positive_count,
                        generated_at=generated_at,
                        date_bucket=date_bucket,
                    )
                    for index, labeled_paper in enumerate(labeled, start=1)
                ],
            }
        ],
    }


def combine_manifests(manifests: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not manifests:
        raise ValueError("at least one manifest is required")
    combined = dict(manifests[0])
    combined["generated_at"] = datetime.now(UTC).isoformat()
    combined["buckets"] = [
        bucket for manifest in manifests for bucket in manifest.get("buckets", [])
    ]
    combined["label_definition"] = dict(combined.get("label_definition") or {})
    combined["label_definition"]["bucket_count"] = len(combined["buckets"])
    combined["label_definition"]["positive_rule"] = (
        "top-K citation labels assigned independently within each bucket"
    )
    return combined


def parse_arxiv_pacing_config(
    raw_delays: str,
    *,
    successes_per_step: int,
) -> ArxivPacingConfig:
    try:
        delays = tuple(
            float(value.strip())
            for value in raw_delays.split(",")
            if value.strip()
        )
    except ValueError as exc:
        raise ValueError("--arxiv-pacing-delays-seconds must be numeric") from exc
    return ArxivPacingConfig(
        delays_seconds=delays,
        successes_per_step=successes_per_step,
    )


def part_manifest_path(part_dir: Path, spec: BucketSpec) -> Path:
    safe_category = re.sub(r"[^A-Za-z0-9]+", "_", spec.category).strip("_")
    safe_label = re.sub(r"[^A-Za-z0-9-]+", "_", spec.date_bucket.label).strip("_")
    return part_dir / f"{safe_category}_{safe_label}.json"


def load_reusable_part_manifest(
    path: Path,
    *,
    spec: BucketSpec,
    phase: str,
    k: int,
) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    buckets = manifest.get("buckets")
    if not isinstance(buckets, list) or len(buckets) != 1:
        raise ValueError(f"reusable part {path} must contain exactly one bucket")
    bucket = buckets[0]
    source = bucket.get("source") if isinstance(bucket, dict) else None
    if not isinstance(source, dict):
        raise ValueError(f"reusable part {path} is missing bucket source metadata")

    actual = {
        "category": source.get("category"),
        "date_bucket": source.get("date_bucket"),
        "phase": bucket.get("phase"),
        "k": bucket.get("k"),
    }
    expected = {
        "category": spec.category,
        "date_bucket": spec.date_bucket.label,
        "phase": phase,
        "k": k,
    }
    if actual != expected:
        raise ValueError(
            f"reusable part {path} does not match requested bucket: "
            f"expected {expected}, found {actual}"
        )
    diagnostics = source.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError(f"reusable part {path} is missing diagnostics")
    if int(diagnostics.get("positive_labels") or 0) < k:
        raise ValueError(f"reusable part {path} has fewer than K positives")
    if int(diagnostics.get("papers_in_manifest") or 0) < k:
        raise ValueError(f"reusable part {path} has fewer than K papers")
    return manifest


def fetch_arxiv_papers(
    *,
    category: str,
    start: date,
    end: date,
    limit: int,
    urlopen: UrlOpen = urllib.request.urlopen,
    sleep: Sleep = time.sleep,
    retry_attempts: int = 5,
    retry_delay_seconds: float = 5.0,
) -> list[ArxivPaper]:
    if limit <= 0:
        raise ValueError("--limit must be greater than zero")
    if end < start:
        raise ValueError("end date must be on or after start date")
    query = (
        f"cat:{category} AND submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
    )
    params = {
        "search_query": query,
        "start": 0,
        "max_results": limit,
        "sortBy": "submittedDate",
        "sortOrder": "ascending",
    }
    url = ARXIV_API_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "sestina-historical-arxiv-manifest/0.1"},
        method="GET",
    )
    feed = _read_bytes_with_retries(
        request,
        urlopen=urlopen,
        sleep=sleep,
        retry_attempts=retry_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )
    return parse_arxiv_feed(feed)


def parse_arxiv_feed(feed: bytes | str) -> list[ArxivPaper]:
    root = ET.fromstring(feed)
    papers = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        entry_id = _text(entry.find(f"{ATOM_NS}id"))
        versioned = entry_id.rstrip("/").rsplit("/", 1)[-1]
        arxiv_id = strip_arxiv_version(versioned)
        title = _normalize_space(_text(entry.find(f"{ATOM_NS}title")))
        abstract = _normalize_space(_text(entry.find(f"{ATOM_NS}summary")))
        published_at = _text(entry.find(f"{ATOM_NS}published"))
        updated_at = _text(entry.find(f"{ATOM_NS}updated"))
        primary_node = entry.find(f"{ARXIV_NS}primary_category")
        primary_category = (
            primary_node.attrib.get("term", "") if primary_node is not None else ""
        )
        categories = [
            node.attrib["term"]
            for node in entry.findall(f"{ATOM_NS}category")
            if node.attrib.get("term")
        ]
        doi = _text(entry.find(f"{ARXIV_NS}doi")) or None
        authors = tuple(
            _normalize_space(_text(author.find(f"{ATOM_NS}name")))
            for author in entry.findall(f"{ATOM_NS}author")
        )
        if not arxiv_id or not title:
            continue
        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                versioned_arxiv_id=versioned,
                title=title,
                abstract=abstract,
                primary_category=primary_category,
                categories=categories,
                published_at=published_at,
                updated_at=updated_at,
                doi=doi,
                authors=authors,
            )
        )
    return papers


def fetch_citation_matches(
    papers: Iterable[ArxivPaper],
    *,
    provider: MetadataProvider,
    request_delay_seconds: float,
    openalex_mailto: str | None,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> dict[str, CitationMatch]:
    results = {}
    for index, paper in enumerate(papers):
        if index and request_delay_seconds > 0:
            time.sleep(request_delay_seconds)
        if provider == "semantic_scholar":
            match = fetch_semantic_scholar_match(paper, urlopen=urlopen)
        elif provider == "openalex":
            match = fetch_openalex_match(
                paper,
                mailto=openalex_mailto,
                urlopen=urlopen,
            )
        else:
            match = fetch_semantic_scholar_match(paper, urlopen=urlopen)
            if not match.matched:
                match = fetch_openalex_match(
                    paper,
                    mailto=openalex_mailto,
                    urlopen=urlopen,
                )
        results[paper.arxiv_id] = match
    return results


def fetch_semantic_scholar_match(
    paper: ArxivPaper,
    *,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> CitationMatch:
    fields = (
        "paperId,title,externalIds,citationCount,year,publicationDate,url,venue"
    )
    for method, external_id in _semantic_scholar_lookup_ids(paper):
        url = (
            SEMANTIC_SCHOLAR_GRAPH_URL
            + "/"
            + urllib.parse.quote(external_id, safe=":")
            + "?"
            + urllib.parse.urlencode({"fields": fields})
        )
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "sestina-historical-arxiv-manifest/0.1"},
            method="GET",
        )
        payload = _read_json_or_none(request, urlopen=urlopen)
        if not payload:
            continue
        score = title_match_score(paper.title, str(payload.get("title") or ""))
        external_ids = payload.get("externalIds") or {}
        arxiv_match = (
            _normalize_arxiv_id(external_ids.get("ArXiv")) == paper.arxiv_id
            if isinstance(external_ids, dict)
            else False
        )
        if method.startswith("arxiv") or arxiv_match or score >= 0.88:
            return CitationMatch(
                provider="semantic_scholar",
                matched=True,
                cited_by_count=_as_int(payload.get("citationCount")),
                work_id=str(payload.get("paperId") or ""),
                title=str(payload.get("title") or ""),
                doi=(
                    str(external_ids.get("DOI"))
                    if isinstance(external_ids, dict) and external_ids.get("DOI")
                    else None
                ),
                publication_date=(
                    str(payload.get("publicationDate"))
                    if payload.get("publicationDate")
                    else None
                ),
                method=method,
                match_score=score,
            )
    return CitationMatch(
        provider="semantic_scholar",
        matched=False,
        method="arxiv_or_doi_lookup",
        error="not_found_or_title_mismatch",
    )


def fetch_openalex_match(
    paper: ArxivPaper,
    *,
    mailto: str | None = None,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> CitationMatch:
    if paper.doi:
        payload = _openalex_lookup_by_doi(paper.doi, mailto=mailto, urlopen=urlopen)
        if payload:
            score = title_match_score(paper.title, _openalex_title(payload))
            return _openalex_match_from_work(payload, method="doi_lookup", score=score)

    search_payload = _openalex_search_title(paper.title, mailto=mailto, urlopen=urlopen)
    candidates = search_payload.get("results", []) if search_payload else []
    best_payload: dict[str, Any] | None = None
    best_score = 0.0
    best_method = "title_search"
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        score = title_match_score(paper.title, _openalex_title(candidate))
        if _work_mentions_arxiv_id(candidate, paper.arxiv_id):
            score = max(score, 0.98)
            best_method = "title_search_arxiv_url"
        if score > best_score:
            best_score = score
            best_payload = candidate

    if best_payload is not None and best_score >= 0.88:
        return _openalex_match_from_work(
            best_payload,
            method=best_method,
            score=best_score,
        )
    return CitationMatch(
        provider="openalex",
        matched=False,
        method="doi_or_title_search",
        error="not_found_or_title_mismatch",
    )


def assign_citation_labels(
    usable: list[tuple[ArxivPaper, CitationMatch]],
    *,
    k: int,
    top_alpha: float | None,
) -> tuple[list[LabeledPaper], int]:
    if not usable:
        raise RuntimeError("no papers available for citation labels")
    ranked_items = sorted(
        usable,
        key=lambda item: (
            -(item[1].cited_by_count or 0),
            normalize_title(item[0].title),
            item[0].arxiv_id,
        ),
    )
    positive_count = (
        max(1, math.ceil(len(ranked_items) * top_alpha))
        if top_alpha is not None
        else k
    )
    positive_count = min(len(ranked_items), positive_count)
    denominator = max(1, len(ranked_items) - 1)
    ranks = {
        paper.arxiv_id: index
        for index, (paper, _match) in enumerate(ranked_items, start=1)
    }
    labeled = [
        LabeledPaper(
            paper=paper,
            match=match,
            good_paper=ranks[paper.arxiv_id] <= positive_count,
            citation_rank=ranks[paper.arxiv_id],
            citation_percentile=1.0 - ((ranks[paper.arxiv_id] - 1) / denominator),
        )
        for paper, match in usable
    ]
    return labeled, positive_count


def resolve_date_bucket(
    *,
    month: str | None,
    quarter: str | None,
    start: date | None,
    end: date | None,
) -> DateBucket:
    selected = [bool(month), bool(quarter), bool(start or end)]
    if sum(selected) != 1:
        raise ValueError(
            "choose exactly one of --month, --quarter, or --start-date/--end-date"
        )
    if month:
        match = re.fullmatch(r"(\d{4})-(\d{2})", month)
        if not match:
            raise ValueError("--month must look like YYYY-MM")
        year, month_number = int(match.group(1)), int(match.group(2))
        start_date = date(year, month_number, 1)
        if month_number == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month_number + 1, 1)
        return DateBucket(
            label=month,
            start=start_date,
            end=next_month - timedelta(days=1),
        )
    if quarter:
        match = re.fullmatch(r"(\d{4})-Q([1-4])", quarter)
        if not match:
            raise ValueError("--quarter must look like YYYY-QN")
        year, quarter_number = int(match.group(1)), int(match.group(2))
        first_month = 1 + ((quarter_number - 1) * 3)
        start_date = date(year, first_month, 1)
        if quarter_number == 4:
            next_quarter = date(year + 1, 1, 1)
        else:
            next_quarter = date(year, first_month + 3, 1)
        return DateBucket(
            label=quarter,
            start=start_date,
            end=next_quarter - timedelta(days=1),
        )
    if start is None or end is None:
        raise ValueError("--start-date and --end-date must be supplied together")
    if end < start:
        raise ValueError("--end-date must be on or after --start-date")
    return DateBucket(
        label=f"{start.isoformat()}_{end.isoformat()}",
        start=start,
        end=end,
    )


def resolve_cli_bucket_specs(
    *,
    category: str | None,
    bucket_values: Sequence[str],
    month: str | None,
    quarter: str | None,
    start: date | None,
    end: date | None,
) -> list[BucketSpec]:
    if bucket_values:
        if category or month or quarter or start or end:
            raise ValueError(
                "--bucket cannot be combined with --category, --month, "
                "--quarter, --start-date, or --end-date"
            )
        return [parse_bucket_spec(value) for value in bucket_values]
    if not category:
        raise ValueError("--category is required unless --bucket is supplied")
    return [
        BucketSpec(
            category=category,
            date_bucket=resolve_date_bucket(
                month=month,
                quarter=quarter,
                start=start,
                end=end,
            ),
        )
    ]


def parse_bucket_spec(value: str) -> BucketSpec:
    if ":" not in value:
        raise ValueError("--bucket must look like CATEGORY:YYYY-MM or CATEGORY:YYYY-QN")
    category, raw_date_bucket = value.split(":", 1)
    category = category.strip()
    raw_date_bucket = raw_date_bucket.strip()
    if not category:
        raise ValueError("--bucket category must not be empty")
    if re.fullmatch(r"\d{4}-\d{2}", raw_date_bucket):
        date_bucket = resolve_date_bucket(
            month=raw_date_bucket,
            quarter=None,
            start=None,
            end=None,
        )
    elif re.fullmatch(r"\d{4}-Q[1-4]", raw_date_bucket):
        date_bucket = resolve_date_bucket(
            month=None,
            quarter=raw_date_bucket,
            start=None,
            end=None,
        )
    else:
        raise ValueError("--bucket date must look like YYYY-MM or YYYY-QN")
    return BucketSpec(category=category, date_bucket=date_bucket)


def strip_arxiv_version(value: str) -> str:
    return re.sub(r"v\d+$", "", _normalize_arxiv_id(value))


def normalize_title(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(normalized.split())


def title_match_score(left: str, right: str) -> float:
    normalized_left = normalize_title(left)
    normalized_right = normalize_title(right)
    if not normalized_left or not normalized_right:
        return 0.0
    sequence_score = SequenceMatcher(None, normalized_left, normalized_right).ratio()
    left_tokens = set(normalized_left.split())
    right_tokens = set(normalized_right.split())
    token_score = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    return max(sequence_score, token_score)


def _paper_payload(
    labeled: LabeledPaper,
    *,
    bucket_name: str,
    ordinal: int,
    positive_count: int,
    generated_at: str,
    date_bucket: DateBucket,
) -> dict[str, Any]:
    paper = labeled.paper
    citation_count = int(labeled.match.cited_by_count or 0)
    return {
        "paper_id": f"arxiv-historical:{bucket_name}:{ordinal:04d}",
        "title": paper.title,
        "abstract": paper.abstract[:4000],
        "baseline_score": 0.5,
        "uncertainty": 0.5,
        "baseline_summary": "Neutral non-oracle baseline; does not use citations.",
        "baseline_reasons": ["constant_non_oracle_baseline"],
        "labels": {
            "good_paper": labeled.good_paper,
            "citation_count": citation_count,
            "citation_rank": labeled.citation_rank,
            "citation_percentile_within_bucket": round(
                labeled.citation_percentile,
                6,
            ),
            "citation_positive_cutoff_rank": positive_count,
            "citation_positive": labeled.good_paper,
            "label_source": "future_public_citation_metadata",
            "metadata_fetched_at": generated_at,
            "arxiv_id": paper.arxiv_id,
            "versioned_arxiv_id": paper.versioned_arxiv_id,
            "citation_match": labeled.match.to_label_payload(),
        },
        "metadata": {
            "source": "arxiv",
            "primary_category": paper.primary_category,
            "categories": paper.categories,
            "published_at": paper.published_at,
            "updated_at": paper.updated_at,
            "bucket_start_date": date_bucket.start.isoformat(),
            "bucket_end_date": date_bucket.end.isoformat(),
        },
    }


def _semantic_scholar_lookup_ids(paper: ArxivPaper) -> list[tuple[str, str]]:
    ids = [("arxiv_lookup", f"arXiv:{paper.arxiv_id}")]
    if paper.doi:
        ids.append(("doi_lookup", f"DOI:{paper.doi}"))
    return ids


def _openalex_lookup_by_doi(
    doi: str,
    *,
    mailto: str | None,
    urlopen: UrlOpen,
) -> dict[str, Any] | None:
    params = {"mailto": mailto} if mailto else {}
    suffix = urllib.parse.urlencode(params)
    url = OPENALEX_WORKS_URL + "/https://doi.org/" + doi
    if suffix:
        url += "?" + suffix
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "sestina-historical-arxiv-manifest/0.1"},
        method="GET",
    )
    return _read_json_or_none(request, urlopen=urlopen)


def _openalex_search_title(
    title: str,
    *,
    mailto: str | None,
    urlopen: UrlOpen,
) -> dict[str, Any] | None:
    params = {
        "search": title,
        "per-page": 5,
        "select": (
            "id,doi,title,display_name,cited_by_count,publication_date,"
            "publication_year,ids,locations,primary_location"
        ),
    }
    if mailto:
        params["mailto"] = mailto
    request = urllib.request.Request(
        OPENALEX_WORKS_URL + "?" + urllib.parse.urlencode(params),
        headers={"User-Agent": "sestina-historical-arxiv-manifest/0.1"},
        method="GET",
    )
    return _read_json_or_none(request, urlopen=urlopen)


def _openalex_match_from_work(
    payload: dict[str, Any],
    *,
    method: str,
    score: float,
) -> CitationMatch:
    return CitationMatch(
        provider="openalex",
        matched=True,
        cited_by_count=_as_int(payload.get("cited_by_count")),
        work_id=str(payload.get("id") or ""),
        title=_openalex_title(payload),
        doi=str(payload.get("doi") or "") or None,
        publication_date=(
            str(payload.get("publication_date"))
            if payload.get("publication_date")
            else None
        ),
        method=method,
        match_score=score,
    )


def _openalex_title(payload: dict[str, Any]) -> str:
    return str(payload.get("title") or payload.get("display_name") or "")


def _work_mentions_arxiv_id(payload: dict[str, Any], arxiv_id: str) -> bool:
    needle = arxiv_id.lower()
    haystacks = []
    ids = payload.get("ids")
    if isinstance(ids, dict):
        haystacks.extend(str(value).lower() for value in ids.values())
    primary = payload.get("primary_location")
    if isinstance(primary, dict):
        haystacks.extend(str(value).lower() for value in primary.values())
    locations = payload.get("locations")
    if isinstance(locations, list):
        for location in locations[:20]:
            if isinstance(location, dict):
                haystacks.extend(str(value).lower() for value in location.values())
    return any(needle in value for value in haystacks)


def _read_json_or_none(
    request: urllib.request.Request,
    *,
    urlopen: UrlOpen,
) -> dict[str, Any] | None:
    try:
        with urlopen(request, timeout=60.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in {404, 429}:
            return None
        raise
    except urllib.error.URLError:
        return None
    return payload if isinstance(payload, dict) else None


def _read_bytes_with_retries(
    request: urllib.request.Request,
    *,
    urlopen: UrlOpen,
    sleep: Sleep,
    retry_attempts: int,
    retry_delay_seconds: float,
) -> bytes:
    retryable_statuses = {500, 502, 503, 504}
    for attempt in range(1, retry_attempts + 1):
        try:
            with urlopen(request, timeout=60.0) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in retryable_statuses or attempt >= retry_attempts:
                raise
            delay = _retry_delay_seconds(
                headers=exc.headers,
                fallback_seconds=retry_delay_seconds,
                attempt=attempt,
            )
            sleep(delay)
        except (TimeoutError, urllib.error.URLError):
            if attempt >= retry_attempts:
                raise
            delay = _retry_delay_seconds(
                headers=None,
                fallback_seconds=retry_delay_seconds,
                attempt=attempt,
            )
            sleep(delay)
    raise RuntimeError("unreachable retry loop exhausted")


def _retry_delay_seconds(
    *,
    headers: HTTPMessage | None,
    fallback_seconds: float,
    attempt: int,
) -> float:
    retry_after = headers.get("Retry-After") if headers is not None else None
    if retry_after:
        try:
            return min(60.0, max(0.0, float(retry_after)))
        except ValueError:
            pass
    return min(60.0, max(0.0, fallback_seconds) * (2 ** (attempt - 1)))


def _bucket_name(*, category: str, label: str, phase: str) -> str:
    safe_category = re.sub(r"[^A-Za-z0-9]+", "_", category).strip("_")
    safe_label = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    safe_phase = re.sub(r"[^A-Za-z0-9]+", "_", phase).strip("_")
    return f"arxiv_{safe_category}_{safe_label}_historical_citation_{safe_phase}"


def _normalize_arxiv_id(value: Any) -> str:
    text = str(value or "").strip()
    text = text.rsplit("/", 1)[-1]
    text = text.removeprefix("arXiv:")
    text = text.removeprefix("arxiv:")
    return text


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _normalize_space(value: str) -> str:
    return " ".join(value.split())


def _text(node: ET.Element | None) -> str:
    return "" if node is None or node.text is None else node.text.strip()


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
