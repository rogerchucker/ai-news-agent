import json
import sys
import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from main import (  # noqa: E402
    FeedItem,
    _strip_html,
    cluster_items,
    ensure_db,
    fetch_hn_front_page,
    heuristic_score,
    parse_rss,
)


def test_strip_html_unescapes_and_removes_tags():
    raw = "<p>Hello &amp; <b>world</b></p>"
    assert _strip_html(raw) == "Hello & world"


def test_heuristic_score_prefers_ai_terms():
    item = FeedItem(
        title="OpenAI launches new GPT model",
        link="http://example.com",
        description="Large language model update",
        published=None,
    )
    ai_score = heuristic_score(item)

    non_ai = FeedItem(
        title="Apple releases iPhone",
        link="http://example.com/phone",
        description="New smartphone announcement",
        published=None,
    )
    non_ai_score = heuristic_score(non_ai)

    assert ai_score > non_ai_score
    assert ai_score > 0
    assert non_ai_score >= 0


def test_parse_rss_extracts_items():
    xml = b"""
    <rss><channel>
      <item>
        <title>AI story</title>
        <link>http://example.com/a</link>
        <description>Something about models</description>
        <pubDate>Tue, 19 Nov 2024 10:00:00 GMT</pubDate>
      </item>
      <item>
        <title>Non-AI story</title>
        <link>http://example.com/b</link>
        <description>General news</description>
      </item>
    </channel></rss>
    """
    items = parse_rss(xml)
    assert len(items) == 2
    assert items[0].title == "AI story"
    assert items[1].link == "http://example.com/b"


def test_cluster_items_uses_heuristic_fast_path():
    item = FeedItem(
        title="NVIDIA GPU for AI",
        link="http://example.com/gpu",
        description="CUDA improvements",
        published=datetime.now(timezone.utc),
    )
    args = SimpleNamespace(
        include_seen=True,
        borderline_high=0.1,  # very low to force heuristic include
        borderline_low=0.0,
        heuristic_only=False,
        min_confidence=0.1,
        model="dummy-model",
    )
    with tempfile.NamedTemporaryFile() as dbfile:
        conn = ensure_db(dbfile.name)
        clusters = cluster_items([item], args, conn, client=SimpleNamespace())
    assert sum(len(v) for v in clusters.values()) == 1


class DummyResp:
    def __init__(self, payload: dict):
        self.output_text = json.dumps(payload)


class DummyResponses:
    def __init__(self, payload: dict):
        self._payload = payload
        self.calls = 0

    def create(self, *_, **__):
        self.calls += 1
        return DummyResp(self._payload)


class DummyOpenAI:
    def __init__(self, payload: dict):
        self.responses = DummyResponses(payload)


def test_cluster_items_invokes_llm_on_borderline(monkeypatch):
    item = FeedItem(
        title="New agent framework released",
        link="http://example.com/agent",
        description="A toolkit for AI agents",
        published=None,
    )
    verdict_payload = {
        "is_ai": True,
        "confidence": 0.9,
        "topic": "Products",
        "tags": ["agents"],
        "takeaway": "New agent tooling",
    }
    args = SimpleNamespace(
        include_seen=True,
        borderline_high=1.0,  # force LLM path
        borderline_low=0.0,
        heuristic_only=False,
        min_confidence=0.2,
        model="dummy-model",
    )
    dummy_client = DummyOpenAI(verdict_payload)
    with tempfile.NamedTemporaryFile() as dbfile:
        conn = ensure_db(dbfile.name)
        clusters = cluster_items([item], args, conn, client=dummy_client)
    assert dummy_client.responses.calls == 1
    assert clusters["Products"][0][0].title == item.title


def test_fetch_hn_front_page_uses_monkeypatched_client(monkeypatch):
    captured_ids = []
    captured_requested = []

    def fake_top():
        captured_ids.append(True)
        return [123, 456]

    def fake_get_item(iid: int):
        captured_requested.append(iid)
        return {
            "id": iid,
            "title": f"Story {iid}",
            "url": "" if iid == 456 else f"http://example.com/{iid}",
            "time": 1732000000,
            "text": "<b>AI content</b>",
        }

    import importlib

    client_mod = importlib.import_module("hn_sdk.client.v0.client")
    monkeypatch.setattr(client_mod, "get_top_stories", fake_top)
    monkeypatch.setattr(client_mod, "get_item_by_id", fake_get_item)

    items = fetch_hn_front_page(limit=2)
    assert captured_ids  # ensured get_top_stories called
    assert set(captured_requested) == {123, 456}
    assert items[0].title == "Story 123"
    # second item lacked URL, so link should point to HN item page
    assert "news.ycombinator.com/item" in items[1].link
