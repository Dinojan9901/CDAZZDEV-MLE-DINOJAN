"""Correctness checks for the hand-rolled indicators.

RSI is verified against Wilder's published worked example rather than against another
library, so this proves the implementation independently of any TA package.

Run: python -m task1_financial.tests.test_indicators
"""

import numpy as np
import pandas as pd

from task1_financial.src import indicators as ind

# Wilder's 14-period RSI worked example, as reproduced by StockCharts.
WILDER_CLOSES = [
    44.3389, 44.0902, 44.1497, 43.6124, 44.3278, 44.8264, 45.0955, 45.4245,
    45.8433, 46.0826, 45.8931, 46.0328, 45.6140, 46.2820, 46.2820, 46.0028,
    46.0328, 46.4116, 46.2222, 45.6439, 46.2122, 46.2521, 45.7142, 46.4515,
    45.7840, 45.3548, 44.0288, 44.1783, 44.2181, 44.5672, 43.4205, 42.6628,
    43.1314,
]
WILDER_RSI = {
    14: 70.53, 15: 66.32, 16: 66.55, 17: 69.41, 18: 66.36, 19: 57.97,
    20: 62.93, 21: 63.26, 22: 56.06, 23: 62.38, 24: 54.71, 25: 50.42,
    26: 39.99, 27: 41.46, 28: 41.87, 29: 45.46, 30: 37.30, 31: 33.08,
    32: 37.77,
}


def _series(values):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


def test_rsi_matches_wilder():
    close = _series(WILDER_CLOSES)
    got = ind.rsi(close, period=14)

    assert got.iloc[:14].isna().all(), "RSI must be undefined before the seed period"
    for i, expected in WILDER_RSI.items():
        actual = got.iloc[i]
        assert abs(actual - expected) < 0.02, f"bar {i}: got {actual:.4f}, want {expected}"
    return len(WILDER_RSI)


def test_rsi_bounds_and_extremes():
    rising = _series(np.arange(1, 60, dtype=float))
    assert abs(ind.rsi(rising).iloc[-1] - 100.0) < 1e-9, "monotonic rise must give RSI 100"

    falling = _series(np.arange(60, 1, -1, dtype=float))
    assert abs(ind.rsi(falling).iloc[-1] - 0.0) < 1e-9, "monotonic fall must give RSI 0"

    noisy = _series(100 + np.sin(np.arange(300)) * 5)
    values = ind.rsi(noisy).dropna()
    assert values.between(0, 100).all(), "RSI must stay within [0, 100]"
    return 3


def test_sma():
    close = _series(np.arange(1, 101, dtype=float))
    got = ind.sma(close, 50)
    assert got.iloc[:49].isna().all()
    # Mean of 1..50
    assert abs(got.iloc[49] - 25.5) < 1e-9
    assert abs(got.iloc[99] - 75.5) < 1e-9
    return 3


def test_macd_against_manual_ema():
    close = _series(100 + np.cumsum(np.random.default_rng(7).normal(0, 1, 400)))
    got = ind.macd(close)

    fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    expected_line = fast - slow
    expected_signal = expected_line.ewm(span=9, adjust=False, min_periods=9).mean()

    assert np.allclose(got["macd"].dropna(), expected_line.dropna(), atol=1e-9)
    assert np.allclose(got["macd_signal"].dropna(), expected_signal.dropna(), atol=1e-9)
    hist = got["macd_hist"].dropna()
    assert np.allclose(hist, (got["macd"] - got["macd_signal"]).dropna(), atol=1e-9)
    return 3


def test_bollinger_population_std():
    close = _series(100 + np.cumsum(np.random.default_rng(3).normal(0, 1, 200)))
    got = ind.bollinger(close, window=20, num_std=2.0)

    manual_mid = close.rolling(20).mean()
    manual_sd = close.rolling(20).std(ddof=0)
    assert np.allclose(got["bb_mid"].dropna(), manual_mid.dropna(), atol=1e-9)
    assert np.allclose(
        got["bb_upper"].dropna(), (manual_mid + 2 * manual_sd).dropna(), atol=1e-9
    )
    # Sample std would inflate the band; assert we are not accidentally using ddof=1.
    sample_sd = close.rolling(20).std(ddof=1)
    assert not np.allclose(manual_sd.dropna(), sample_sd.dropna(), atol=1e-6)

    band = got.dropna()
    assert (band["bb_upper"] >= band["bb_mid"]).all()
    assert (band["bb_mid"] >= band["bb_lower"]).all()
    return 4


def test_handles_gaps_and_short_series():
    values = [10.0, 11.0, np.nan, 12.0, np.nan, 13.0] + list(np.linspace(13, 30, 60))
    close = _series(values)
    out = ind.add_all(pd.DataFrame({"Close": close}))
    assert "rsi_14" in out.columns and "bb_upper" in out.columns

    tiny = pd.DataFrame({"Close": _series([1.0, 2.0, 3.0])})
    short = ind.add_all(tiny)
    assert short["rsi_14"].isna().all(), "too-short input must yield NaN, not raise"
    return 3


def main():
    checks = [
        test_rsi_matches_wilder,
        test_rsi_bounds_and_extremes,
        test_sma,
        test_macd_against_manual_ema,
        test_bollinger_population_std,
        test_handles_gaps_and_short_series,
    ]
    total = 0
    for check in checks:
        n = check()
        total += n
        print(f"  PASS  {check.__name__:38s} ({n} assertions)")
    print(f"\n{len(checks)} tests passed, {total} assertions.")


if __name__ == "__main__":
    main()
