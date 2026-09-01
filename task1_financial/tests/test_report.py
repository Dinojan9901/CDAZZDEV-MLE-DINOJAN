"""Offline tests for the research brief renderer. No network, no API key.

Run: python -m task1_financial.tests.test_report
"""

import re

import numpy as np
import pandas as pd

from task1_financial.src import charts, indicators as ind, report

SUMMARY = {
    "ticker": "TEST",
    "company_name": "Test Corporation",
    "sector": "Technology",
    "as_of": "2026-08-31",
    "current_price": 220.78,
    "currency": "USD",
    "week52_high": 235.47,
    "week52_low": 164.98,
    "pe_ratio": 27.88,
    "market_cap": 5.33e12,
    "ytd_return_pct": 17.05,
    "bars_analysed": 500,
    "history_start": "2024-09-03",
    "indicators": {
        "sma_50": 208.62, "sma_200": 195.80, "rsi_14": 54.56,
        "macd": 2.48, "macd_signal": 2.77, "macd_hist": -0.28,
        "bb_upper": 229.36, "bb_mid": 218.75, "bb_lower": 208.15, "bb_pct_b": 0.60,
    },
    "momentum": {
        "signal": "Bullish", "score": 0.25,
        "components": {"trend_cross": 1, "macd_histogram": -1},
        "flags": ["golden_cross"],
    },
}

ANALYSIS = {
    "ticker": "TEST",
    "headline_sentiment": [
        {"headline": "Test beats on earnings", "sentiment": "positive",
         "confidence": 0.91, "brief_reason": "Beat raises guidance credibility."},
        {"headline": "Test faces export limits", "sentiment": "negative",
         "confidence": 0.78, "brief_reason": "Restrictions narrow the addressable market."},
        {"headline": "Test names new director", "sentiment": "neutral",
         "confidence": 0.42, "brief_reason": "Routine governance with no earnings impact."},
    ],
    "sentiment_failures": [],
    "classification_success_rate": 1.0,
    "aggregate_sentiment": {
        "score": 0.1464, "label": "positive", "positive": 1,
        "negative": 1, "neutral": 1, "mean_confidence": 0.70,
    },
    "signal": {
        "signal": "Hold",
        "justification": (
            "The golden cross keeps the primary trend intact. A negative MACD histogram "
            "shows momentum fading before price has turned. RSI near the midpoint adds no "
            "confirmation. That divergence argues for patience."
        ),
        "key_drivers": ["golden cross against falling MACD histogram"],
    },
    "signal_error": None,
}


def _frame(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(11)
    close = 100 + np.cumsum(rng.normal(0.15, 1.4, n))
    df = pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99,
         "Close": close, "Volume": rng.integers(1e6, 5e6, n)},
        index=idx,
    )
    return ind.add_all(df)


def test_markdown_has_every_required_section():
    md = report.render_markdown(SUMMARY, ANALYSIS, [])
    for heading in ["## Company snapshot", "## Technical outlook", "## News sentiment",
                    "### Top three headlines", "## Recommendation", "## Risk disclaimer"]:
        assert heading in md, f"missing {heading}"
    assert md.startswith("# Test Corporation (TEST)")
    return 7


def test_disclaimer_is_present_and_unambiguous():
    md = report.render_markdown(SUMMARY, ANALYSIS, [])
    lowered = md.lower()
    for phrase in ["not investment advice", "at their own risk",
                   "past performance", "qualified financial adviser"]:
        assert phrase in lowered, f"disclaimer missing {phrase!r}"
    return 4


def test_top_three_headlines_are_ranked_and_capped():
    many = dict(ANALYSIS)
    many["headline_sentiment"] = ANALYSIS["headline_sentiment"] + [
        {"headline": f"Filler {i}", "sentiment": "neutral", "confidence": 0.2,
         "brief_reason": "Filler."} for i in range(8)
    ]
    md = report.render_markdown(SUMMARY, many, [])
    section = md.split("### Top three headlines")[1].split("## Recommendation")[0]
    numbered = re.findall(r"^\d+\. ", section, re.M)
    assert len(numbered) == 3, f"expected exactly 3 headlines, got {len(numbered)}"
    # Highest-confidence non-neutral must lead.
    assert "Test beats on earnings" in section
    assert "Filler" not in section, "low-confidence neutrals must not displace real ones"
    return 3


def test_missing_values_never_render_as_none():
    sparse = {"ticker": "TEST", "indicators": {}, "momentum": {}}
    empty = {"ticker": "TEST", "headline_sentiment": [], "sentiment_failures": [],
             "classification_success_rate": 0.0, "aggregate_sentiment": None,
             "signal": None, "signal_error": "provider unavailable"}
    md = report.render_markdown(sparse, empty, [])
    assert "None" not in md, "raw None leaked into the brief"
    assert "n/a" in md
    assert "No headline sentiment could be retrieved" in md
    assert "provider unavailable" in md
    assert "## Risk disclaimer" in md, "disclaimer is mandatory even on a degraded run"
    return 5


def test_html_embeds_chart_and_is_self_contained():
    frame = _frame()
    png = charts.build_chart(frame, "TEST", "Test Corporation")
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "not a valid PNG"

    md = report.render_markdown(SUMMARY, ANALYSIS, [])
    html = report.render_html(md, png, SUMMARY, ANALYSIS)

    assert "data:image/png;base64," in html
    assert 'class="verdict hold"' in html, "signal must render as a styled pill"
    assert "<table>" in html, "indicator tables must survive markdown conversion"
    assert html.count("<div") == html.count("</div>"), "unbalanced divs"
    # No external fetches: the page must render offline from one file.
    assert "http://" not in html and "https://" not in html
    assert "<script" not in html.lower(), "a static brief needs no scripts"
    return 7


def test_chart_handles_short_history():
    short = _frame(n=30)
    png = charts.build_chart(short, "TEST", "Test Corporation")
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "must still render when SMA200 is all NaN"
    return 1


def main():
    checks = [
        test_markdown_has_every_required_section,
        test_disclaimer_is_present_and_unambiguous,
        test_top_three_headlines_are_ranked_and_capped,
        test_missing_values_never_render_as_none,
        test_html_embeds_chart_and_is_self_contained,
        test_chart_handles_short_history,
    ]
    total = 0
    for check in checks:
        n = check()
        total += n
        print(f"  PASS  {check.__name__:46s} ({n} assertions)")
    print(f"\n{len(checks)} tests passed, {total} assertions.")


if __name__ == "__main__":
    main()
