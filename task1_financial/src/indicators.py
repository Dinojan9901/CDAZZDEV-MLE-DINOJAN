"""Technical indicators computed from first principles. No TA-Lib.

# AI-ASSISTED: Claude (claude-opus-5), Prompt: 'Implement SMA, Wilder RSI, MACD and
# Bollinger Bands from first principles in pandas without TA-Lib', Date: 2026-09-01
"""

import numpy as np
import pandas as pd

RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_WINDOW, BB_STD = 20, 2.0
SMA_SHORT, SMA_LONG = 50, 200


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing, seeded with the SMA of the first `period` observations.

    Not the same as an EWM with alpha=1/period, which uses the first value as its
    seed and drifts from published RSI figures over the first few dozen bars.
    """
    values = series.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    if len(values) <= period:
        return pd.Series(out, index=series.index)

    first = np.nanmean(values[1:period + 1])
    out[period] = first
    for i in range(period + 1, len(values)):
        prev = out[i - 1]
        current = values[i]
        if np.isnan(current):
            out[i] = prev
        else:
            out[i] = (prev * (period - 1) + current) / period
    return pd.Series(out, index=series.index)


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(avg_gain != 0.0, 0.0)
    return out.where(avg_gain.notna())


def macd(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {
            "macd": line,
            "macd_signal": signal_line,
            "macd_hist": line - signal_line,
        }
    )


def bollinger(
    close: pd.Series, window: int = BB_WINDOW, num_std: float = BB_STD
) -> pd.DataFrame:
    mid = sma(close, window)
    # ddof=0 to match the population standard deviation used by charting platforms.
    sd = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    width = (upper - lower) / mid.replace(0.0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_width": width,
            "bb_pct_b": pct_b,
        }
    )


def add_all(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    if price_col not in df.columns:
        raise KeyError(f"{price_col!r} not in frame, got {list(df.columns)}")

    out = df.copy()
    close = out[price_col].astype(float)

    out[f"sma_{SMA_SHORT}"] = sma(close, SMA_SHORT)
    out[f"sma_{SMA_LONG}"] = sma(close, SMA_LONG)
    out[f"rsi_{RSI_PERIOD}"] = rsi(close)
    out = out.join(macd(close)).join(bollinger(close))
    return out
