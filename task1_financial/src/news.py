"""Headline retrieval with source fallback.

A single free news source is unreliable enough that one bad response would empty the
sentiment stage, so three are tried in order until the quota is met.
"""

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Iterable

import requests
import yfinance as yf

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
TIMEOUT = 12
YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
GOOGLE_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


STOP_ALIASES = {"inc", "corp", "corporation", "company", "co", "ltd", "plc", "group", "holdings", "the"}


def company_aliases(ticker: str, company_name: str | None = None) -> set[str]:
    """Tokens that mark a headline as actually about this company."""
    aliases = {ticker.lower()}
    if company_name:
        cleaned = re.sub(r"[^a-zA-Z0-9 ]", " ", company_name).lower()
        for token in cleaned.split():
            if len(token) > 2 and token not in STOP_ALIASES:
                aliases.add(token)
    return aliases


def _relevance(title: str, aliases: set[str]) -> int:
    words = set(_normalise(title).split())
    return sum(1 for alias in aliases if alias in words)


def _normalise(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _item(title: str, publisher: str, link: str, published: str, source: str) -> dict:
    return {
        "headline": title.strip(),
        "publisher": (publisher or "unknown").strip(),
        "link": (link or "").strip(),
        "published": published,
        "source": source,
    }


def _epoch_to_iso(value) -> str:
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return ""


def _from_yfinance(ticker: str) -> list[dict]:
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception as exc:
        log.warning("yfinance news failed for %s: %s", ticker, exc)
        return []

    items = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        # yfinance moved the payload under "content" in 0.2.5x; older builds are flat.
        body = entry.get("content") if isinstance(entry.get("content"), dict) else entry
        title = body.get("title") or entry.get("title") or ""
        if not title:
            continue
        provider = body.get("provider")
        publisher = (
            provider.get("displayName") if isinstance(provider, dict) else None
        ) or entry.get("publisher") or ""
        url = body.get("canonicalUrl")
        link = (url.get("url") if isinstance(url, dict) else None) or entry.get("link") or ""
        published = body.get("pubDate") or _epoch_to_iso(entry.get("providerPublishTime"))
        items.append(_item(title, publisher, link, published, "yfinance"))
    return items


def _from_rss(url: str, source: str) -> list[dict]:
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except (requests.RequestException, ET.ParseError) as exc:
        log.warning("%s feed failed: %s", source, exc)
        return []

    items = []
    for node in root.iter("item"):
        title = (node.findtext("title") or "").strip()
        if not title:
            continue
        items.append(
            _item(
                title,
                node.findtext("source") or source,
                node.findtext("link") or "",
                node.findtext("pubDate") or "",
                source,
            )
        )
    return items


def _dedupe(batches: Iterable[list[dict]]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for batch in batches:
        for entry in batch:
            key = _normalise(entry["headline"])
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(entry)
    return out


def fetch_news(ticker: str, n: int = 10, company_name: str | None = None) -> list[dict]:
    """Return up to `n` headlines, most relevant first.

    Yahoo's per-ticker feed mixes in general market stories, so a raw take of the first
    n items can hand the sentiment stage headlines about an unrelated company. Every
    source is drained first, then results are ranked by whether the headline actually
    names this company, and only then truncated.
    """
    ticker = ticker.strip().upper()
    query = f"{company_name} stock" if company_name else f"{ticker} stock"
    aliases = company_aliases(ticker, company_name)

    sources = [
        lambda: _from_yfinance(ticker),
        lambda: _from_rss(YAHOO_RSS.format(ticker=ticker), "yahoo_rss"),
        lambda: _from_rss(GOOGLE_RSS.format(query=query.replace(" ", "+")), "google_rss"),
    ]

    collected: list[list[dict]] = []
    for get in sources:
        try:
            collected.append(get())
        except Exception as exc:
            log.warning("news source raised: %s", exc)
            collected.append([])

    candidates = _dedupe(collected)
    for entry in candidates:
        entry["relevance"] = _relevance(entry["headline"], aliases)

    # Rank on-topic vs off-topic only, never on how many times the name appears.
    # Aggregator headlines pad in "(NASDAQ:NVDA)" and would otherwise outrank every
    # editorial headline, collapsing three sources back down to one.
    ranked = sorted(candidates, key=lambda e: e["relevance"] == 0)
    headlines = ranked[:n]

    on_topic = sum(1 for e in headlines if e["relevance"] > 0)
    if on_topic < n:
        log.warning(
            "%s: %d of %d headlines do not name the company; they are kept but flagged",
            ticker, n - on_topic, len(headlines),
        )
    if len(headlines) < n:
        log.warning("%s: only %d headlines retrieved, wanted %d", ticker, len(headlines), n)
    return headlines
