"""The five agent tools, Task 3A.

Tools never raise into the agent loop. A failure comes back as {"ok": false, "error": ...}
so the model observes it and can choose another route. An exception would end the run,
which is the opposite of the required behaviour.

Price and news logic is imported from Task 1 rather than reimplemented, so the agent and
the Task 1 brief cannot disagree about the same ticker.
"""

import json
import math
from typing import Any, Callable

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from common.llm import LLMError, SchemaValidationError, get_client
from task1_financial.src import analysis as t1_analysis
from task1_financial.src import data as t1_data
from task1_financial.src import indicators as ind
from task1_financial.src import news as t1_news
from task1_financial.src import summary as t1_summary
from task3_agentic.src import trace as tracing

TRADING_DAYS = 252
DEFAULT_VOL_WINDOW = 30
PERIOD_YEARS = {"1mo": 1, "3mo": 1, "6mo": 1, "1y": 1, "2y": 2, "5y": 5}

_tracer: tracing.ToolTracer | None = None


def set_tracer(tracer: tracing.ToolTracer | None) -> None:
    global _tracer
    _tracer = tracer


def _traced(name: str, inputs: dict, fn: Callable[[], dict]) -> dict:
    if _tracer is None:
        return fn()
    with _tracer.span(name, inputs) as box:
        box["output"] = fn()
    return box["output"]


def _fail(reason: str, **extra) -> dict:
    return {"ok": False, "error": reason, **extra}


def _clean(value):
    """Strip NaN and numpy scalars so the payload survives json.dumps."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def price_data(ticker: str, period: str = "1y") -> dict:
    def run() -> dict:
        years = PERIOD_YEARS.get(period, 1)
        try:
            history = t1_data.fetch_ohlcv(ticker, years=max(years, 1))
        except (t1_data.DataFetchError, ValueError) as exc:
            return _fail(str(exc), ticker=ticker)

        enriched = ind.add_all(history.frame)
        window = enriched.loc[enriched.index >= history.analysis_start]
        summary = t1_summary.build_summary(ticker.upper(), enriched, window)
        recent = enriched.tail(5)[["Open", "High", "Low", "Close", "Volume"]]
        return _clean({
            "ok": True,
            "ticker": summary["ticker"],
            "company_name": summary["company_name"],
            "as_of": summary["as_of"],
            "current_price": summary["current_price"],
            "week52_high": summary["week52_high"],
            "week52_low": summary["week52_low"],
            "pe_ratio": summary["pe_ratio"],
            "ytd_return_pct": summary["ytd_return_pct"],
            "indicators": summary["indicators"],
            "momentum": summary["momentum"],
            "bars": summary["bars_analysed"],
            "recent_ohlcv": [
                {"date": str(i.date()), **{c: float(r[c]) for c in recent.columns}}
                for i, r in recent.iterrows()
            ],
        })

    return _traced("get_price_data", {"ticker": ticker, "period": period}, run)


def news(ticker: str, n: int = 10) -> dict:
    def run() -> dict:
        try:
            headlines = t1_news.fetch_news(ticker, n=n)
        except Exception as exc:
            return _fail(f"news retrieval failed: {exc}", ticker=ticker)
        if not headlines:
            return _fail("no headlines found from any source", ticker=ticker)
        return {
            "ok": True,
            "ticker": ticker.upper(),
            "count": len(headlines),
            "headlines": [
                {"headline": h["headline"], "publisher": h["publisher"],
                 "published": h.get("published", ""), "source": h["source"]}
                for h in headlines
            ],
        }

    return _traced("get_news", {"ticker": ticker, "n": n}, run)


def volatility(ticker: str, window: int = DEFAULT_VOL_WINDOW) -> dict:
    def run() -> dict:
        try:
            history = t1_data.fetch_ohlcv(ticker, years=1)
        except (t1_data.DataFetchError, ValueError) as exc:
            return _fail(str(exc), ticker=ticker)

        close = history.frame["Close"].astype(float)
        # Log returns, so compounding does not bias the deviation upward.
        log_returns = np.log(close / close.shift(1)).dropna()
        if len(log_returns) < window:
            return _fail(
                f"only {len(log_returns)} return observations, need {window}", ticker=ticker
            )

        recent = log_returns.tail(window)
        annualised = float(recent.std(ddof=1) * math.sqrt(TRADING_DAYS) * 100)
        full_year = float(log_returns.tail(TRADING_DAYS).std(ddof=1) * math.sqrt(TRADING_DAYS) * 100)
        daily_move = float(recent.abs().mean() * 100)

        regime = "elevated" if annualised > full_year * 1.15 else (
            "subdued" if annualised < full_year * 0.85 else "in line with the past year"
        )
        return _clean({
            "ok": True,
            "ticker": ticker.upper(),
            "window_days": window,
            "annualised_volatility_pct": round(annualised, 2),
            "annualised_volatility_1y_pct": round(full_year, 2),
            "mean_abs_daily_move_pct": round(daily_move, 3),
            "regime": regime,
            "observations": len(recent),
        })

    return _traced("calculate_volatility", {"ticker": ticker, "window": window}, run)


def llm_sentiment(headlines: list[Any], ticker: str = "", company: str = "") -> dict:
    def run() -> dict:
        items = []
        for h in headlines or []:
            if isinstance(h, str):
                items.append({"headline": h, "publisher": "unknown"})
            elif isinstance(h, dict) and h.get("headline"):
                items.append(h)
        if not items:
            return _fail("no headlines supplied")

        try:
            outcome = t1_analysis.analyse_headlines(
                items, ticker.upper() or "the company", company or ticker.upper() or "the company",
                client=get_client(),
            )
        except (LLMError, SchemaValidationError) as exc:
            return _fail(f"sentiment model unavailable: {exc}")

        if outcome.aggregate is None:
            return _fail("every headline failed classification",
                         failures=len(outcome.failures))
        agg = outcome.aggregate
        return _clean({
            "ok": True,
            "score": agg.score,
            "label": agg.label,
            "positive": agg.positive,
            "negative": agg.negative,
            "neutral": agg.neutral,
            "mean_confidence": agg.mean_confidence,
            "classified": len(outcome.results),
            "failed": len(outcome.failures),
            "per_headline": [
                {"headline": r.headline, "sentiment": r.sentiment,
                 "confidence": r.confidence, "brief_reason": r.brief_reason}
                for r in outcome.results
            ],
        })

    return _traced("llm_sentiment", {"headline_count": len(headlines or []), "ticker": ticker}, run)


def web_search(query: str, max_results: int = 5) -> dict:
    def run() -> dict:
        try:
            from ddgs import DDGS
        except ImportError:
            return _fail("ddgs is not installed")

        try:
            with DDGS() as engine:
                hits = list(engine.text(query, max_results=max_results))
        except Exception as exc:
            return _fail(f"search failed: {type(exc).__name__}: {exc}", query=query)

        if not hits:
            return _fail("search returned no results", query=query)
        return {
            "ok": True,
            "query": query,
            "count": len(hits),
            "results": [
                {"title": h.get("title", ""), "snippet": (h.get("body") or "")[:400],
                 "url": h.get("href", "")}
                for h in hits
            ],
        }

    return _traced("web_search", {"query": query, "max_results": max_results}, run)


def _dumps(payload: dict) -> str:
    return json.dumps(payload, default=str)


@tool
def get_price_data(ticker: str, period: str = "1y") -> str:
    """Fetch OHLCV history for a ticker with computed technical indicators.

    Returns current price, 52-week range, P/E, year-to-date return, SMA50, SMA200,
    RSI(14), MACD, Bollinger bands and a rule-based momentum reading.
    """
    return _dumps(price_data(ticker, period))


@tool
def get_news(ticker: str, n: int = 10) -> str:
    """Retrieve recent news headlines for a ticker as a structured list.

    Each item carries the headline, publisher, publication time and which source it
    came from. Use this for what is being said about the company right now.
    """
    return _dumps(news(ticker, n))


@tool
def calculate_volatility(ticker: str, window: int = 30) -> str:
    """Compute annualised historical volatility from daily log returns.

    Returns volatility over the requested window, the one-year figure for comparison,
    the mean absolute daily move, and whether the current regime is elevated or subdued.
    """
    return _dumps(volatility(ticker, window))


@tool
def llm_sentiment_tool(ticker: str = "", headlines: list[str] | None = None, n: int = 10) -> str:
    """Score recent news sentiment for a company.

    Pass only the ticker and this fetches the headlines itself. Do not paste headline
    text back in, it is unnecessary and long headlines get mangled in transit.

    Returns an aggregate score from -1 to 1, a label, the positive/negative/neutral
    breakdown, and a per-headline classification with confidence.
    """
    if not headlines and ticker:
        fetched = news(ticker, n)
        if not fetched.get("ok"):
            return _dumps(_fail(f"could not fetch headlines to score: {fetched.get('error')}"))
        headlines = [h["headline"] for h in fetched["headlines"]]
    return _dumps(llm_sentiment(headlines or [], ticker))


@tool
def web_search_tool(query: str, max_results: int = 5) -> str:
    """Search the web for analyst commentary, filings coverage and market context.

    Use this for qualitative information that price data cannot supply, such as analyst
    opinion, competitive developments or regulatory news.
    """
    return _dumps(web_search(query, max_results))


ALL_TOOLS = [get_price_data, get_news, calculate_volatility, llm_sentiment_tool, web_search_tool]
QUANT_TOOLS = [get_price_data, calculate_volatility, llm_sentiment_tool]
QUALITATIVE_TOOLS = [web_search_tool, get_news]
