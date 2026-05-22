from __future__ import annotations

import json
import urllib.error
import urllib.parse
from datetime import date
from email.message import Message

import pytest

from scripts.build_arxiv_historical_manifest import (
    ARXIV_API_URL,
    ArxivMetadataRateLimitError,
    ArxivPacingConfig,
    ArxivPaper,
    ArxivRequestPacer,
    BucketSpec,
    CitationMatch,
    DateBucket,
    HuldraMetadataError,
    assign_citation_labels,
    build_manifest_from_records,
    combine_manifests,
    fetch_arxiv_papers,
    load_reusable_part_manifest,
    parse_bucket_spec,
    parse_arxiv_feed,
    resolve_date_bucket,
)
from sestina.backtest_runner import load_dataset_manifest


def _paper(
    arxiv_id: str,
    title: str,
    citation_count: int,
) -> tuple[ArxivPaper, CitationMatch]:
    return (
        ArxivPaper(
            arxiv_id=arxiv_id,
            versioned_arxiv_id=f"{arxiv_id}v1",
            title=title,
            abstract=f"Abstract for {title}",
            primary_category="cs.LG",
            categories=["cs.LG"],
            published_at="2023-01-01T00:00:00Z",
            updated_at="2023-01-02T00:00:00Z",
        ),
        CitationMatch(
            provider="semantic_scholar",
            matched=True,
            cited_by_count=citation_count,
            work_id=f"s2-{arxiv_id}",
            title=title,
            method="arxiv_lookup",
            match_score=1.0,
        ),
    )


class _FakeBytesResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "_FakeBytesResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def _fake_json_response(payload: dict) -> _FakeBytesResponse:
    return _FakeBytesResponse(json.dumps(payload).encode("utf-8"))


def _request_json(request: object) -> dict:
    return json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]


def _request_path(request: object) -> str:
    return urllib.parse.urlparse(request.full_url).path  # type: ignore[attr-defined]


def _huldra_paper_payload() -> dict:
    return {
        "arxiv_id": "2304.00185v2",
        "title": " Huldra Test Paper ",
        "abstract": None,
        "primary_category": "cs.LG",
        "categories": ["cs.LG", "stat.ML"],
        "published_at": "2023-04-05T00:00:00+00:00",
        "updated_at": "2023-04-06T00:00:00+00:00",
        "doi": "10.1234/huldra",
        "authors": ["Ada Lovelace", "Grace Hopper"],
    }


def _run_successful_huldra_fetch() -> tuple[list[object], list[ArxivPaper]]:
    requests: list[object] = []
    responses = [
        _fake_json_response(
            {
                "requested_total": 1,
                "completed_windows_total": 1,
                "requests": [
                    {
                        "cache_key": "huldra:v1:test",
                        "raw_cache_status": "completed",
                        "serving_status": "ready",
                        "papers_total": 1,
                    }
                ],
            }
        ),
        _fake_json_response(
            {
                "status": "ready",
                "cache_key": "huldra:v1:test",
                "papers": [_huldra_paper_payload()],
                "papers_total": 1,
                "analysis_ready": True,
            }
        ),
    ]

    def fake_urlopen(request: object, **kwargs: object) -> _FakeBytesResponse:
        requests.append(request)
        return responses.pop(0)

    papers = fetch_arxiv_papers(
        category="cs.LG",
        start=date(2023, 4, 1),
        end=date(2023, 4, 30),
        limit=80,
        metadata_source="huldra",
        huldra_base_url="http://huldra.local",
        huldra_wait_timeout_seconds=600,
        huldra_client_id="test-client",
        urlopen=fake_urlopen,
    )

    assert not responses
    return requests, papers


def test_citation_labels_select_top_k_without_reordering_bucket_membership() -> None:
    usable = [
        _paper("2301.00001", "First chronological paper", 2),
        _paper("2301.00002", "Second chronological paper", 90),
        _paper("2301.00003", "Third chronological paper", 40),
    ]

    labeled, positive_count = assign_citation_labels(usable, k=2, top_alpha=None)

    assert positive_count == 2
    assert [item.paper.arxiv_id for item in labeled] == [
        "2301.00001",
        "2301.00002",
        "2301.00003",
    ]
    assert [item.good_paper for item in labeled] == [False, True, True]
    assert [item.citation_rank for item in labeled] == [3, 1, 2]


def test_citation_labels_support_top_alpha_cutoff() -> None:
    usable = [
        _paper("2301.00001", "A", 5),
        _paper("2301.00002", "B", 4),
        _paper("2301.00003", "C", 3),
        _paper("2301.00004", "D", 2),
        _paper("2301.00005", "E", 1),
    ]

    labeled, positive_count = assign_citation_labels(usable, k=5, top_alpha=0.4)

    assert positive_count == 2
    assert [item.good_paper for item in labeled] == [True, True, False, False, False]


def test_manifest_is_runner_compatible_and_hides_citation_data_from_metadata(
    tmp_path,
) -> None:
    papers_and_matches = [
        _paper("2301.00001", "Chronological low citation paper", 1),
        _paper("2301.00002", "Chronological high citation paper", 25),
        _paper("2301.00003", "Chronological middle citation paper", 10),
    ]
    papers = [paper for paper, _match in papers_and_matches]
    matches = {paper.arxiv_id: match for paper, match in papers_and_matches}

    manifest = build_manifest_from_records(
        papers=papers,
        matches=matches,
        category="cs.LG",
        date_bucket=DateBucket(
            label="2023-01",
            start=date(2023, 1, 1),
            end=date(2023, 1, 31),
        ),
        requested_limit=3,
        k=2,
        top_alpha=None,
        phase="smoke",
        metadata_provider="semantic_scholar",
        unmatched_policy="drop",
    )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    loaded = load_dataset_manifest(manifest_path)
    bucket = loaded.buckets[0]

    assert bucket.k == 2
    assert len(bucket.relevant_ids) == 2
    assert len(bucket.papers) == 3
    assert [paper.metadata["published_at"] for paper in bucket.papers] == [
        "2023-01-01T00:00:00Z",
        "2023-01-01T00:00:00Z",
        "2023-01-01T00:00:00Z",
    ]
    assert all(score == 0.5 for score in bucket.baseline_scores.values())
    for paper in bucket.papers:
        assert "citation_count" not in paper.metadata
        assert "citation_rank" not in paper.metadata
        assert "arxiv_id" not in paper.metadata
        assert "doi" not in paper.metadata
    assert manifest["buckets"][0]["papers"][1]["labels"]["citation_count"] == 25
    assert manifest["buckets"][0]["source"]["diagnostics"]["positive_labels"] == 2


def test_pilot_manifest_combines_multiple_buckets_with_phase_scoped_names(
    tmp_path,
) -> None:
    first_records = [
        _paper("2301.00001", "First LG paper", 5),
        _paper("2301.00002", "Second LG paper", 20),
    ]
    second_records = [
        _paper("2302.00001", "First CL paper", 7),
        _paper("2302.00002", "Second CL paper", 30),
    ]

    first_manifest = build_manifest_from_records(
        papers=[paper for paper, _match in first_records],
        matches={paper.arxiv_id: match for paper, match in first_records},
        category="cs.LG",
        date_bucket=DateBucket(
            label="2023-01",
            start=date(2023, 1, 1),
            end=date(2023, 1, 31),
        ),
        requested_limit=2,
        k=1,
        top_alpha=None,
        phase="pilot",
        metadata_provider="semantic_scholar",
        unmatched_policy="drop",
    )
    second_manifest = build_manifest_from_records(
        papers=[paper for paper, _match in second_records],
        matches={paper.arxiv_id: match for paper, match in second_records},
        category="cs.CL",
        date_bucket=DateBucket(
            label="2023-02",
            start=date(2023, 2, 1),
            end=date(2023, 2, 28),
        ),
        requested_limit=2,
        k=1,
        top_alpha=None,
        phase="pilot",
        metadata_provider="semantic_scholar",
        unmatched_policy="drop",
    )

    manifest = combine_manifests([first_manifest, second_manifest])
    manifest_path = tmp_path / "pilot-manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    loaded = load_dataset_manifest(manifest_path)

    assert manifest["label_definition"]["bucket_count"] == 2
    assert [bucket.phase for bucket in loaded.buckets] == ["pilot", "pilot"]
    assert [bucket.name for bucket in loaded.buckets] == [
        "arxiv_cs_LG_2023_01_historical_citation_pilot",
        "arxiv_cs_CL_2023_02_historical_citation_pilot",
    ]
    assert all(len(bucket.relevant_ids) == 1 for bucket in loaded.buckets)


def test_manifest_rejects_bucket_with_fewer_than_k_matched_citation_labels() -> None:
    paper, _match = _paper("2301.00001", "Only paper", 4)

    try:
        build_manifest_from_records(
            papers=[paper],
            matches={},
            category="cs.LG",
            date_bucket=DateBucket(
                label="2023-01",
                start=date(2023, 1, 1),
                end=date(2023, 1, 31),
            ),
            requested_limit=1,
            k=2,
            top_alpha=None,
            phase="smoke",
            metadata_provider="semantic_scholar",
            unmatched_policy="drop",
        )
    except RuntimeError as exc:
        assert "usable citation metadata" in str(exc)
    else:
        raise AssertionError("manifest build unexpectedly succeeded")


def test_arxiv_feed_parser_extracts_title_abstract_categories_and_dates() -> None:
    feed = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2301.00001v2</id>
        <updated>2023-01-03T00:00:00Z</updated>
        <published>2023-01-02T00:00:00Z</published>
        <title> A Historical Test Paper </title>
        <summary> First line.
        Second line. </summary>
        <author><name>Example Author</name></author>
        <arxiv:primary_category term="cs.LG"/>
        <category term="cs.LG"/>
        <category term="stat.ML"/>
        <arxiv:doi>10.1234/example</arxiv:doi>
      </entry>
    </feed>"""

    papers = parse_arxiv_feed(feed)

    assert len(papers) == 1
    assert papers[0].arxiv_id == "2301.00001"
    assert papers[0].versioned_arxiv_id == "2301.00001v2"
    assert papers[0].title == "A Historical Test Paper"
    assert papers[0].abstract == "First line. Second line."
    assert papers[0].primary_category == "cs.LG"
    assert papers[0].categories == ["cs.LG", "stat.ML"]
    assert papers[0].published_at == "2023-01-02T00:00:00Z"
    assert papers[0].doi == "10.1234/example"


def test_huldra_fetch_posts_sync_then_cache_only_request() -> None:
    requests, _papers = _run_successful_huldra_fetch()

    assert [_request_path(request) for request in requests] == [
        "/v1/sync",
        "/v1/requests",
    ]
    sync_body = _request_json(requests[0])
    read_body = _request_json(requests[1])

    assert sync_body["wait"] is True
    assert sync_body["wait_timeout_seconds"] == 600
    assert sync_body["requests"][0]["cache_policy"] == "cache_or_enqueue"
    assert read_body["cache_policy"] == "cache_only"
    assert read_body["readiness"] == "analysis_ready"


def test_huldra_fetch_uses_category_query_and_exclusive_submitted_window() -> None:
    requests, _papers = _run_successful_huldra_fetch()

    huldra_request = _request_json(requests[0])["requests"][0]

    assert huldra_request["search_query"] == "cat:cs.LG"
    assert "submittedDate" not in huldra_request["search_query"]
    assert huldra_request["submitted_start"] == "2023-04-01T00:00:00+00:00"
    assert huldra_request["submitted_end"] == "2023-05-01T00:00:00+00:00"
    assert huldra_request["sort_by"] == "submittedDate"
    assert huldra_request["sort_order"] == "ascending"
    assert huldra_request["max_results"] == 80


def test_huldra_fetch_converts_paper_json_to_local_arxiv_paper() -> None:
    _requests, papers = _run_successful_huldra_fetch()

    assert len(papers) == 1
    paper = papers[0]
    assert paper.arxiv_id == "2304.00185"
    assert paper.versioned_arxiv_id == "2304.00185v2"
    assert paper.title == "Huldra Test Paper"
    assert paper.abstract == ""
    assert paper.primary_category == "cs.LG"
    assert paper.categories == ["cs.LG", "stat.ML"]
    assert paper.published_at == "2023-04-05T00:00:00+00:00"
    assert paper.updated_at == "2023-04-06T00:00:00+00:00"
    assert paper.doi == "10.1234/huldra"
    assert paper.authors == ("Ada Lovelace", "Grace Hopper")


@pytest.mark.parametrize(
    "sync_request",
    [
        {
            "cache_key": "huldra:v1:cooldown",
            "raw_cache_status": "skipped",
            "serving_status": "queued",
            "error_category": "cooldown",
            "cooldown_until": "2026-05-22T10:00:00+00:00",
        },
        {
            "cache_key": "huldra:v1:rate-limited",
            "raw_cache_status": "rate_limited",
            "serving_status": "rate_limited",
        },
        {
            "cache_key": "huldra:v1:upstream-429",
            "raw_cache_status": "failed",
            "serving_status": "failed",
            "upstream_status": 429,
        },
    ],
)
def test_huldra_rate_limit_cooldown_and_upstream_429_are_rate_limit_failures(
    sync_request: dict,
) -> None:
    def fake_urlopen(request: object, **kwargs: object) -> _FakeBytesResponse:
        return _fake_json_response(
            {
                "requested_total": 1,
                "upstream_429_total": 1
                if sync_request.get("upstream_status") == 429
                else 0,
                "cooldown_active_total": 1
                if sync_request.get("error_category") == "cooldown"
                else 0,
                "rate_limited_windows_total": 1
                if sync_request.get("raw_cache_status") == "rate_limited"
                else 0,
                "requests": [sync_request],
            }
        )

    with pytest.raises(ArxivMetadataRateLimitError) as exc_info:
        fetch_arxiv_papers(
            category="cs.LG",
            start=date(2023, 4, 1),
            end=date(2023, 4, 30),
            limit=80,
            metadata_source="huldra",
            urlopen=fake_urlopen,
        )

    message = str(exc_info.value)
    assert "rate-limit/cooldown" in message
    assert f"cache_key={sync_request['cache_key']}" in message


def test_huldra_cache_miss_reports_status_cache_key_and_blocked_reason() -> None:
    responses = [
        _fake_json_response(
            {
                "requested_total": 1,
                "requests": [
                    {
                        "cache_key": "huldra:v1:missing",
                        "raw_cache_status": "queued",
                        "serving_status": "queued",
                    }
                ],
            }
        ),
        _fake_json_response(
            {
                "status": "cache_miss",
                "cache_key": "huldra:v1:missing",
                "blocked_reason": "cache_miss",
                "papers": [],
            }
        ),
    ]

    def fake_urlopen(request: object, **kwargs: object) -> _FakeBytesResponse:
        return responses.pop(0)

    with pytest.raises(HuldraMetadataError) as exc_info:
        fetch_arxiv_papers(
            category="cs.LG",
            start=date(2023, 4, 1),
            end=date(2023, 4, 30),
            limit=80,
            metadata_source="huldra",
            urlopen=fake_urlopen,
        )

    message = str(exc_info.value)
    assert "status=cache_miss" in message
    assert "cache_key=huldra:v1:missing" in message
    assert "blocked_reason=cache_miss" in message


def test_arxiv_fetch_uses_arxiv_org_api_endpoint_by_default() -> None:
    feed = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2304.00185v1</id>
        <updated>2023-04-01T00:41:51Z</updated>
        <published>2023-04-01T00:41:51Z</published>
        <title> Endpoint Test Paper </title>
        <summary> Example. </summary>
      </entry>
    </feed>"""
    requested_urls: list[str] = []

    def fake_urlopen(request: object, **kwargs: object) -> _FakeBytesResponse:
        requested_urls.append(request.full_url)  # type: ignore[attr-defined]
        return _FakeBytesResponse(feed)

    papers = fetch_arxiv_papers(
        category="cs.LG",
        start=date(2023, 4, 1),
        end=date(2023, 4, 30),
        limit=1,
        urlopen=fake_urlopen,
    )

    assert ARXIV_API_URL == "https://arxiv.org/api/query"
    assert requested_urls
    assert requested_urls[0].startswith("https://arxiv.org/api/query?")
    assert "submittedDate%3A%5B202304010000+TO+202304302359%5D" in requested_urls[0]
    assert [paper.arxiv_id for paper in papers] == ["2304.00185"]


def test_arxiv_fetch_pages_conservatively_when_page_size_is_smaller_than_limit() -> None:
    first_page = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2304.00001v1</id>
        <updated>2023-04-01T00:00:00Z</updated>
        <published>2023-04-01T00:00:00Z</published>
        <title> First Page Paper </title>
        <summary> Example. </summary>
      </entry>
    </feed>"""
    second_page = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2304.00002v1</id>
        <updated>2023-04-02T00:00:00Z</updated>
        <published>2023-04-02T00:00:00Z</published>
        <title> Second Page Paper </title>
        <summary> Example. </summary>
      </entry>
    </feed>"""
    requested_urls: list[str] = []
    responses = [_FakeBytesResponse(first_page), _FakeBytesResponse(second_page)]

    def fake_urlopen(request: object, **kwargs: object) -> _FakeBytesResponse:
        requested_urls.append(request.full_url)  # type: ignore[attr-defined]
        return responses.pop(0)

    papers = fetch_arxiv_papers(
        category="cs.LG",
        start=date(2023, 4, 1),
        end=date(2023, 4, 30),
        limit=2,
        page_size=1,
        urlopen=fake_urlopen,
    )

    assert len(requested_urls) == 2
    assert "start=0" in requested_urls[0]
    assert "max_results=1" in requested_urls[0]
    assert "start=1" in requested_urls[1]
    assert "max_results=1" in requested_urls[1]
    assert [paper.arxiv_id for paper in papers] == ["2304.00001", "2304.00002"]


def test_arxiv_fetch_retries_transient_server_error() -> None:
    feed = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2301.00001v1</id>
        <updated>2023-01-03T00:00:00Z</updated>
        <published>2023-01-02T00:00:00Z</published>
        <title> Retry Test Paper </title>
        <summary> Example. </summary>
        <arxiv:primary_category term="cs.LG"/>
        <category term="cs.LG"/>
      </entry>
    </feed>"""
    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(request: object, **kwargs: object) -> _FakeBytesResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            headers = Message()
            headers["Retry-After"] = "0.25"
            raise urllib.error.HTTPError(
                url="https://arxiv.org/api/query",
                code=503,
                msg="service unavailable",
                hdrs=headers,
                fp=None,
            )
        return _FakeBytesResponse(feed)

    papers = fetch_arxiv_papers(
        category="cs.LG",
        start=date(2023, 1, 1),
        end=date(2023, 1, 31),
        limit=1,
        urlopen=fake_urlopen,
        sleep=sleeps.append,
        retry_delay_seconds=0.1,
    )

    assert calls == 2
    assert sleeps == [0.25]
    assert [paper.arxiv_id for paper in papers] == ["2301.00001"]


def test_arxiv_fetch_stops_immediately_on_rate_limit() -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(request: object, **kwargs: object) -> _FakeBytesResponse:
        nonlocal calls
        calls += 1
        headers = Message()
        headers["Retry-After"] = "120"
        raise urllib.error.HTTPError(
            url="https://arxiv.org/api/query",
            code=429,
            msg="rate limited",
            hdrs=headers,
            fp=None,
        )

    try:
        fetch_arxiv_papers(
            category="cs.LG",
            start=date(2023, 1, 1),
            end=date(2023, 1, 31),
            limit=1,
            urlopen=fake_urlopen,
            sleep=sleeps.append,
            retry_delay_seconds=0.1,
        )
    except urllib.error.HTTPError as exc:
        assert exc.code == 429
    else:
        raise AssertionError("arXiv 429 unexpectedly retried or succeeded")

    assert calls == 1
    assert sleeps == []


def test_arxiv_fetch_retries_transient_timeout() -> None:
    feed = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2301.00002v1</id>
        <updated>2023-01-03T00:00:00Z</updated>
        <published>2023-01-02T00:00:00Z</published>
        <title> Timeout Retry Test Paper </title>
        <summary> Example. </summary>
      </entry>
    </feed>"""
    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(request: object, **kwargs: object) -> _FakeBytesResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("timed out")
        return _FakeBytesResponse(feed)

    papers = fetch_arxiv_papers(
        category="cs.LG",
        start=date(2023, 1, 1),
        end=date(2023, 1, 31),
        limit=1,
        urlopen=fake_urlopen,
        sleep=sleeps.append,
        retry_delay_seconds=0.1,
    )

    assert calls == 2
    assert sleeps == [0.1]
    assert [paper.arxiv_id for paper in papers] == ["2301.00002"]


def test_arxiv_pacer_starts_conservatively_and_ramps_after_successes() -> None:
    now = 0.0
    sleeps: list[float] = []

    def fake_clock() -> float:
        return now

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(round(seconds, 3))
        now += seconds

    pacer = ArxivRequestPacer(
        ArxivPacingConfig(
            delays_seconds=(15.0, 10.0, 6.0, 3.5),
            successes_per_step=2,
        ),
        sleep=fake_sleep,
        monotonic=fake_clock,
    )

    pacer.before_request()
    pacer.record_success()
    now += 1.0
    pacer.before_request()
    pacer.record_success()
    now += 1.0
    pacer.before_request()

    assert sleeps == [14.0, 9.0]
    assert pacer.current_delay_seconds == 10.0


def test_arxiv_pacer_rejects_delays_below_official_floor() -> None:
    try:
        ArxivPacingConfig(delays_seconds=(3.0,))
    except ValueError as exc:
        assert "at least 3.5 seconds" in str(exc)
    else:
        raise AssertionError("pacer accepted an arXiv delay below the safety floor")


def test_load_reusable_part_manifest_validates_bucket_identity(tmp_path) -> None:
    records = [
        _paper("2301.00001", "First LG paper", 5),
        _paper("2301.00002", "Second LG paper", 20),
    ]
    manifest = build_manifest_from_records(
        papers=[paper for paper, _match in records],
        matches={paper.arxiv_id: match for paper, match in records},
        category="cs.LG",
        date_bucket=DateBucket(
            label="2023-01",
            start=date(2023, 1, 1),
            end=date(2023, 1, 31),
        ),
        requested_limit=2,
        k=1,
        top_alpha=None,
        phase="pilot",
        metadata_provider="semantic_scholar",
        unmatched_policy="drop",
    )
    part_path = tmp_path / "cs_LG_2023-01.json"
    part_path.write_text(json.dumps(manifest))

    loaded = load_reusable_part_manifest(
        part_path,
        spec=BucketSpec(
            category="cs.LG",
            date_bucket=DateBucket(
                label="2023-01",
                start=date(2023, 1, 1),
                end=date(2023, 1, 31),
            ),
        ),
        phase="pilot",
        k=1,
    )

    assert loaded["buckets"][0]["name"] == (
        "arxiv_cs_LG_2023_01_historical_citation_pilot"
    )

    try:
        load_reusable_part_manifest(
            part_path,
            spec=BucketSpec(
                category="cs.CL",
                date_bucket=DateBucket(
                    label="2023-01",
                    start=date(2023, 1, 1),
                    end=date(2023, 1, 31),
                ),
            ),
            phase="pilot",
            k=1,
        )
    except ValueError as exc:
        assert "does not match requested bucket" in str(exc)
    else:
        raise AssertionError("mismatched reusable part was accepted")


def test_resolve_date_bucket_supports_month_and_quarter() -> None:
    january = resolve_date_bucket(month="2023-01", quarter=None, start=None, end=None)
    first_quarter = resolve_date_bucket(
        month=None,
        quarter="2023-Q1",
        start=None,
        end=None,
    )

    assert january.start == date(2023, 1, 1)
    assert january.end == date(2023, 1, 31)
    assert first_quarter.start == date(2023, 1, 1)
    assert first_quarter.end == date(2023, 3, 31)


def test_parse_bucket_spec_supports_category_month_and_quarter() -> None:
    monthly = parse_bucket_spec("cs.LG:2023-01")
    quarterly = parse_bucket_spec("cs.CL:2023-Q1")

    assert monthly.category == "cs.LG"
    assert monthly.date_bucket.label == "2023-01"
    assert quarterly.category == "cs.CL"
    assert quarterly.date_bucket.start == date(2023, 1, 1)
