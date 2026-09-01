"""Summary dictionary and the rule-based momentum signal.

The momentum signal is deterministic and computed here, not by the LLM. The LLM is
asked to reason about it in Task 1B, which keeps the arithmetic auditable and stops a
sampling failure from silently changing a number in the report.
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from task1_financial.src import indicators as ind

log = logging.getLogger(__name__)

TRADING_DAYS_52W = 252
RSI_OVERBOUGHT, RSI_OVERSOLD = 70.0, 30.0
RSI_BULL, RSI_BEAR = 55.0, 45.0

SIGNAL_BANDS = [
    (0.5, "Strong Bullish"),
    (0.2, "Bullish"),
    (-0.2, "Neutral"),
    (-0.5, "Bearish"),
]


def _num(value: Any) -> float | None:
    """Coerce to a JSON-safe float, or None for NaN, inf and non-numerics."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) or math.isinf(out) else round(out, 4)


def _last(series: pd.Series) -> float | None:
    clean = series.dropna()
    return _num(clean.iloc[-1]) if len(clean) else None


def _fetch_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception as exc:
        log.warning("%s: info lookup failed, %s", ticker, exc)
        return {}


def _ytd_return(close: pd.Series) -> float | None:
    if close.empty:
        return None
    year_start = pd.Timestamp(year=close.index[-1].year, month=1, day=1)
    ytd = close.loc[close.index >= year_start]
    if len(ytd) < 2 or not ytd.iloc[0]:
        return None
    return _num((ytd.iloc[-1] / ytd.iloc[0] - 1) * 100)


def momentum_signal(row: pd.Series, close: float | None) -> dict:
    """Four equally weighted components, each in {-1, 0, +1}."""
    sma50, sma200 = _num(row.get(f"sma_{ind.SMA_SHORT}")), _num(row.get(f"sma_{ind.SMA_LONG}"))
    rsi_val = _num(row.get(f"rsi_{ind.RSI_PERIOD}"))
    hist = _num(row.get("macd_hist"))
    pct_b = _num(row.get("bb_pct_b"))

    components: dict[str, int] = {}

    if sma50 is not None and sma200 is not None:
        components["trend_cross"] = 1 if sma50 > sma200 else -1
    if close is not None and sma50 is not None:
        components["price_vs_sma50"] = 1 if close > sma50 else -1
    if hist is not None:
        components["macd_histogram"] = 1 if hist > 0 else -1
    if rsi_val is not None:
        components["rsi_zone"] = 1 if rsi_val > RSI_BULL else (-1 if rsi_val < RSI_BEAR else 0)

    if not components:
        return {"signal": "Unknown", "score": None, "components": {}, "flags": ["insufficient_data"]}

    score = sum(components.values()) / len(components)
    label = "Strong Bearish"
    for threshold, name in SIGNAL_BANDS:
        if score >= threshold:
            label = name
            break

    flags = []
    if sma50 is not None and sma200 is not None:
        flags.append("golden_cross" if sma50 > sma200 else "death_cross")
    if rsi_val is not None:
        if rsi_val >= RSI_OVERBOUGHT:
            flags.append("overbought")
        elif rsi_val <= RSI_OVERSOLD:
            flags.append("oversold")
    if pct_b is not None:
        if pct_b > 1:
            flags.append("above_upper_band")
        elif pct_b < 0:
            flags.append("below_lower_band")

    return {
        "signal": label,
        "score": _num(score),
        "components": components,
        "flags": flags,
    }


def build_summary(ticker: str, enriched: pd.DataFrame, window: pd.DataFrame | None = None) -> dict:
    frame = window if window is not None and not window.empty else enriched
    close = frame["Close"].astype(float)
    latest = enriched.dropna(subset=["Close"]).iloc[-1]
    current = _num(latest["Close"])

    trailing = close.tail(TRADING_DAYS_52W)
    info = _fetch_info(ticker)
    pe = _num(info.get("trailingPE")) or _num(info.get("forwardPE"))

    indicator_snapshot = {
        "sma_50": _num(latest.get(f"sma_{ind.SMA_SHORT}")),
        "sma_200": _num(latest.get(f"sma_{ind.SMA_LONG}")),
        "rsi_14": _num(latest.get(f"rsi_{ind.RSI_PERIOD}")),
        "macd": _num(latest.get("macd")),
        "macd_signal": _num(latest.get("macd_signal")),
        "macd_hist": _num(latest.get("macd_hist")),
        "bb_upper": _num(latest.get("bb_upper")),
        "bb_mid": _num(latest.get("bb_mid")),
        "bb_lower": _num(latest.get("bb_lower")),
        "bb_pct_b": _num(latest.get("bb_pct_b")),
    }

    return {
        "ticker": ticker,
        "company_name": info.get("shortName") or info.get("longName") or ticker,
        "sector": info.get("sector"),
        "as_of": str(latest.name.date()),
        "current_price": current,
        "currency": info.get("currency", "USD"),
        "week52_high": _num(trailing.max()),
        "week52_low": _num(trailing.min()),
        "pe_ratio": pe,
        "market_cap": _num(info.get("marketCap")),
        "ytd_return_pct": _ytd_return(close),
        "volume_latest": _num(latest.get("Volume")),
        "avg_volume_30d": _num(frame["Volume"].tail(30).mean()),
        "indicators": indicator_snapshot,
        "momentum": momentum_signal(latest, current),
        "bars_analysed": int(len(frame)),
        "history_start": str(frame.index[0].date()),
    }
