#!/usr/bin/env python3
"""
Agentic AI-only news reader that combines:
1) Techmeme RSS (existing)
2) Hacker News front page via hn-sdk
With LLM topic tagging, clustering, and a SQLite memory cache to avoid repeats.

Install:
  uv sync   # or pip install -r requirements from pyproject

Run:
  export OPENAI_API_KEY="..."
  python main.py --limit 60 --hn-limit 40
  python main.py --limit 120 --min-confidence 0.65
  python main.py --reset-db   # clears memory

Notes:
- Uses heuristic prefilter to save tokens, then LLM for borderline items + topic tagging.
- Only prints items NOT seen in previous runs (unless --include-seen).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import html
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

from openai import OpenAI

DEFAULT_FEED_URL = "https://www.techmeme.com/feed.xml"
DEFAULT_DB_PATH = os.path.expanduser("~/.cache/techmeme_ai_cache.sqlite3")

# ----------------------------
# RSS parsing
# ----------------------------
@dataclass
class FeedItem:
    title: str
    link: str
    description: str
    published: Optional[datetime]


def _safe_text(elem: Optional[ET.Element]) -> str:
    if elem is None or elem.text is None:
        return ""
    return elem.text.strip()


def _parse_pubdate(text: str) -> Optional[datetime]:
    if not text:
        return None
    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def fetch_rss(url: str, timeout_s: int = 15) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": "techmeme-ai-agent/2.0",
            "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
        },
        method="GET",
    )
    with urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def parse_rss(xml_bytes: bytes) -> List[FeedItem]:
    root = ET.fromstring(xml_bytes)
    channel = root.find("channel")
    if channel is None:
        channel = root.find(".//channel")
    if channel is None:
        raise ValueError("Could not find <channel> in RSS feed.")

    items: List[FeedItem] = []
    for item in channel.findall("item"):
        title = _safe_text(item.find("title"))
        link = _safe_text(item.find("link"))
        desc = _safe_text(item.find("description"))
        pub = _parse_pubdate(_safe_text(item.find("pubDate")))
        if title and link:
            items.append(FeedItem(title=title, link=link, description=desc, published=pub))
    return items


TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    if not text:
        return ""
    cleaned = TAG_RE.sub(" ", text)
    collapsed = re.sub(r"\s+", " ", cleaned).strip()
    return html.unescape(collapsed)


# ----------------------------
# Hacker News (via hn-sdk)
# ----------------------------
def _hn_item_url(item_id: int) -> str:
    return f"https://news.ycombinator.com/item?id={item_id}"


def fetch_hn_front_page(limit: int = 40) -> List[FeedItem]:
    try:
        from hn_sdk.client.v0 import client as hn_client
    except Exception as e:
        raise RuntimeError(f"hn-sdk not available: {e}") from e

    try:
        ids = hn_client.get_top_stories()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch Hacker News top stories: {e}") from e

    items: List[FeedItem] = []
    for raw_id in ids[: max(limit, 0)]:
        try:
            iid = int(raw_id)
        except (TypeError, ValueError):
            continue
        try:
            data = hn_client.get_item_by_id(iid)
        except Exception:
            continue
        if not data:
            continue
        title = data.get("title") or ""
        link = data.get("url") or _hn_item_url(iid)
        description = _strip_html(data.get("text") or "")
        ts = data.get("time")
        pub = None
        if ts:
            try:
                pub = datetime.fromtimestamp(ts, tz=timezone.utc)
            except Exception:
                pub = None
        if title and link:
            items.append(FeedItem(title=title, link=link, description=description, published=pub))
    return items


# ----------------------------
# Cheap heuristics (fast)
# ----------------------------
AI_KEYWORDS_STRONG = {
    "openai", "chatgpt", "gpt", "anthropic", "claude", "gemini", "deepmind",
    "llm", "large language model", "generative ai", "genai", "foundation model",
    "transformer", "diffusion", "stable diffusion", "midjourney", "dall-e", "dalle",
    "hugging face", "nvidia", "cuda", "inference", "fine-tuning", "finetuning",
    "rag", "vector database", "qdrant", "milvus", "pinecone", "langchain", "langgraph",
    "agent", "agents", "copilot", "model", "benchmark",
    "ai safety", "ai regulation", "alignment"
}

AI_KEYWORDS_NEGATIVE = {"iphone", "android", "tesla", "crypto", "bitcoin", "stocks", "earnings"}


def heuristic_score(item: FeedItem) -> float:
    text = f"{item.title}\n{item.description}".lower()
    score = 0.0
    for kw in AI_KEYWORDS_STRONG:
        if kw in text:
            score += 1.0
    for kw in AI_KEYWORDS_NEGATIVE:
        if kw in text:
            score -= 0.25
    return max(0.0, min(1.0, score / 4.0))


def heuristic_topic(item: FeedItem) -> str:
    # Lightweight topic guesser for the high-confidence shortcut path.
    text = f"{item.title} {item.description}".lower()
    def has(words: Tuple[str, ...]) -> bool:
        return any(w in text for w in words)

    if has(("model", "models", "llm", "gpt", "transformer", "moe", "mixture-of-experts", "mixture of experts", "mamba", "diffusion")):
        return "Models"
    if has(("chip", "chips", "gpu", "cuda", "accelerator", "semiconductor", "hardware", "h100", "gh200", "b200", "npu")):
        return "Chips"
    if has(("policy", "regulation", "regulator", "law", "eu ai act", "white house", "safety", "alignment", "compliance", "oversight", "ban", "licensing")):
        return "Policy"
    if has(("startup", "start-up", "seed", "series a", "series b", "series c", "raises", "funding", "valuation", "unicorn")):
        return "Startups"
    if has(("research", "paper", "arxiv", "preprint", "benchmark", "sota", "dataset", "conference")):
        return "Research"
    if has(("product", "feature", "update", "launch", "preview", "beta", "app", "tool", "plugin", "integration")):
        return "Products"
    if has(("security", "vulnerability", "exploit", "backdoor", "prompt injection", "jailbreak", "breach", "attack", "supply chain")):
        return "Security"
    if has(("revenue", "earnings", "stock", "partnership", "deal", "acquisition", "ipo", "merger", "profit", "loss")):
        return "Business"
    return "Other"


# ----------------------------
# SQLite memory cache
# ----------------------------
def ensure_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_items (
            id TEXT PRIMARY KEY,
            link TEXT,
            title TEXT,
            first_seen_utc TEXT NOT NULL,
            published_utc TEXT,
            topic TEXT,
            confidence REAL
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_seen_first_seen ON seen_items(first_seen_utc);")
    conn.commit()
    return conn


def item_id(item: FeedItem) -> str:
    # Prefer stable dedupe by link; fallback to content hash.
    base = item.link.strip().lower()
    if base:
        return "link:" + hashlib.sha256(base.encode("utf-8")).hexdigest()
    blob = (item.title + "\n" + item.description).strip().lower()
    return "hash:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def is_seen(conn: sqlite3.Connection, iid: str) -> bool:
    cur = conn.execute("SELECT 1 FROM seen_items WHERE id = ? LIMIT 1", (iid,))
    return cur.fetchone() is not None


def mark_seen(
    conn: sqlite3.Connection,
    iid: str,
    item: FeedItem,
    topic: str,
    confidence: float,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    pub = item.published.isoformat() if item.published else None
    conn.execute(
        """
        INSERT OR IGNORE INTO seen_items (id, link, title, first_seen_utc, published_utc, topic, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (iid, item.link, item.title, now, pub, topic, confidence),
    )
    conn.commit()


def reset_db(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM seen_items;")
    conn.commit()


# ----------------------------
# LLM: classify + topic tagging
# ----------------------------
TOPICS = [
    "Models",
    "Chips",
    "Policy",
    "Startups",
    "Research",
    "Products",
    "Security",
    "Business",
    "Other",
]

AI_TAG_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "is_ai": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "topic": {"type": "string", "enum": TOPICS},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "takeaway": {"type": "string"},
    },
    "required": ["is_ai", "confidence", "topic", "tags", "takeaway"],
}


def classify_and_tag(client: OpenAI, model: str, item: FeedItem) -> Dict[str, Any]:
    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are an AI-news classifier and topic tagger.\n"
                    "- Decide if the item is primarily about AI (models, chips for AI, AI policy/safety, "
                    "AI companies, agent systems, RAG, training/inference, AI research, AI products).\n"
                    "- Pick ONE topic from the allowed list.\n"
                    "- Provide a short 1-sentence takeaway.\n"
                    "Return ONLY valid JSON matching the schema."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Title: {item.title}\n"
                    f"Description: {item.description}\n"
                    f"Link: {item.link}\n"
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "ai_news_item",
                "schema": AI_TAG_SCHEMA,
                "strict": True,
            }
        },
    )

    raw = resp.output_text
    return json.loads(raw)


def cluster_items(
    items: List[FeedItem],
    args: argparse.Namespace,
    conn: sqlite3.Connection,
    client: OpenAI,
) -> Dict[str, List[Tuple[FeedItem, Dict[str, Any]]]]:
    clusters: Dict[str, List[Tuple[FeedItem, Dict[str, Any]]]] = {t: [] for t in TOPICS}

    for it in items:
        iid = item_id(it)
        if (not args.include_seen) and is_seen(conn, iid):
            continue

        hs = heuristic_score(it)

        if hs >= args.borderline_high:
            topic_guess = heuristic_topic(it)
            verdict = {
                "is_ai": True,
                "confidence": max(args.min_confidence, hs),
                "topic": topic_guess,
                "tags": ["heuristic", f"topic:{topic_guess}"],
                "takeaway": "Matched AI-related keywords (heuristic).",
            }
            if verdict["confidence"] >= args.min_confidence:
                clusters[verdict["topic"]].append((it, verdict))
                mark_seen(conn, iid, it, verdict["topic"], float(verdict["confidence"]))
            continue

        if hs <= args.borderline_low:
            continue

        if args.heuristic_only:
            continue

        try:
            verdict = classify_and_tag(client, args.model, it)
        except Exception as e:
            print(f"LLM error (skipping {it.link}): {e}", file=sys.stderr)
            continue

        if verdict.get("is_ai") and float(verdict.get("confidence", 0.0)) >= args.min_confidence:
            topic = verdict.get("topic", "Other")
            if topic not in clusters:
                topic = "Other"
                verdict["topic"] = "Other"
            clusters[topic].append((it, verdict))
            mark_seen(conn, iid, it, topic, float(verdict["confidence"]))

        time.sleep(0.05)

    return clusters


def render_clusters(source_name: str, clusters: Dict[str, List[Tuple[FeedItem, Dict[str, Any]]]]) -> None:
    total = sum(len(v) for v in clusters.values())
    print(f"\n### {source_name}")
    if total == 0:
        print("No new AI items found (or all were previously seen).")
        return

    ordered_topics = sorted(TOPICS, key=lambda t: len(clusters[t]), reverse=True)
    for topic in ordered_topics:
        group = clusters[topic]
        if not group:
            continue
        print(f"\n## {topic} ({len(group)})")
        for idx, (it, v) in enumerate(group, 1):
            ts = it.published.isoformat() if it.published else "unknown-time"
            tags = ", ".join(v.get("tags", []))
            conf = float(v.get("confidence", 0.0))
            takeaway = (v.get("takeaway") or "").strip()
            print(f"{idx:02d}. {it.title}")
            print(f"    {it.link}")
            print(f"    published: {ts} | confidence: {conf:.2f} | tags: {tags}")
            if takeaway:
                print(f"    takeaway: {takeaway[:220]}")
            print()


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_FEED_URL, help="Techmeme RSS feed URL")
    ap.add_argument("--limit", type=int, default=60, help="Max Techmeme RSS items to consider")
    ap.add_argument("--hn-limit", type=int, default=40, help="Max Hacker News front-page items to consider")
    ap.add_argument("--timeout", type=int, default=15, help="HTTP timeout seconds for RSS")
    ap.add_argument("--model", default="gpt-5.2-chat-latest", help="OpenAI model for classification/tagging")
    ap.add_argument("--min-confidence", type=float, default=0.55, help="Min confidence to include")
    ap.add_argument("--borderline-low", type=float, default=0.15, help="Heuristic score below = ignore")
    ap.add_argument("--borderline-high", type=float, default=0.70, help="Heuristic score above = include (no LLM)")
    ap.add_argument("--heuristic-only", action="store_true", help="Skip LLM; only keyword filter (no topic clustering)")
    ap.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite db path for memory cache")
    ap.add_argument("--reset-db", action="store_true", help="Clear memory cache and exit")
    ap.add_argument("--include-seen", action="store_true", help="Include items even if seen before")
    ap.add_argument("--since-minutes", type=int, default=0, help="Only consider items within last N minutes (0=off)")
    ap.add_argument("--no-hn", action="store_true", help="Skip Hacker News")
    ap.add_argument("--no-techmeme", action="store_true", help="Skip Techmeme RSS")
    args = ap.parse_args()

    conn = ensure_db(args.db)
    if args.reset_db:
        reset_db(conn)
        print(f"Cleared cache: {args.db}")
        return 0

    sources: List[Tuple[str, Dict[str, List[Tuple[FeedItem, Dict[str, Any]]]]]] = []
    now = datetime.now(timezone.utc).isoformat()
    print(f"AI-only feeds (clustered) @ {now}")
    print(f"Cache: {args.db}")
    print("-" * 90)

    if args.no_hn and args.no_techmeme:
        print("Nothing to do: both Hacker News and Techmeme were disabled.")
        return 0

    client = OpenAI()

    if not args.no_techmeme:
        try:
            xml_bytes = fetch_rss(args.url, timeout_s=args.timeout)
            items = parse_rss(xml_bytes)[: max(args.limit, 0)]
        except Exception as e:
            print(f"Error fetching/parsing RSS: {e}", file=sys.stderr)
            items = []

        if args.since_minutes and args.since_minutes > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=args.since_minutes)
            items = [it for it in items if it.published is None or it.published >= cutoff]

        clusters = cluster_items(items, args, conn, client) if items else {t: [] for t in TOPICS}
        sources.append(("Techmeme", clusters))

    if not args.no_hn:
        try:
            hn_items = fetch_hn_front_page(args.hn_limit)
        except Exception as e:
            print(f"Error fetching Hacker News: {e}", file=sys.stderr)
            hn_items = []

        if args.since_minutes and args.since_minutes > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=args.since_minutes)
            hn_items = [it for it in hn_items if it.published is None or it.published >= cutoff]

        hn_clusters = cluster_items(hn_items, args, conn, client) if hn_items else {t: [] for t in TOPICS}
        sources.append(("Hacker News", hn_clusters))

    for name, clusters in sources:
        render_clusters(name, clusters)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
