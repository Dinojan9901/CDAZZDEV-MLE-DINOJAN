"""OHLCV retrieval via yfinance."""

import logging
from dataclasses import dataclass

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
TRADING_DAYS_PER_YEAR = 252

# SMA200 needs 200 prior bars, so fetch a warm-up buffer before the analysis window
# or the first year of the requested range has no long-average value at all.
WARMUP_TRADING_DAYS = 260


class DataFetchError(RuntimeError):
    pass


@dataclass
class PriceHistory:
    ticker: str
    frame: pd.DataFrame
    analysis_start: pd.Timestamp

    @property
    def window(self) -> pd.DataFrame:
        return self.frame.loc[self.frame.index >= self.analysis_start]

    def __len__(self) -> int:
        return len(self.window)


def _flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    if ticker in df.columns.get_level_values(-1):
        df = df.xs(ticker, axis=1, level=-1)
    else:
        df.columns = df.columns.get_level_values(0)
    return df


def _clean(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = _flatten_columns(df, ticker)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataFetchError(f"{ticker}: response missing columns {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    dropped = int(df["Close"].isna().sum())
    if dropped:
        log.warning("%s: dropping %d rows with null Close", ticker, dropped)
        df = df[df["Close"].notna()]

    # Holidays and halts leave gaps in volume rather than price; zero is the honest value.
    df["Volume"] = df["Volume"].fillna(0)
    df[["Open", "High", "Low"]] = df[["Open", "High", "Low"]].ffill()
    return df


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=15), reraise=True)
def _download(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # auto_adjust=True is required, not cosmetic: an unadjusted series breaks every
    # moving average across a split (NVDA split 10:1 in June 2024).
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise DataFetchError(f"{ticker}: empty response for {start.date()} to {end.date()}")
    return df


def fetch_ohlcv(ticker: str, years: int = 2) -> PriceHistory:
    """Fetch at least `years` of daily bars plus a warm-up buffer for SMA200.

    All dates are derived from today, never hardcoded.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("ticker must not be empty")

    today = pd.Timestamp.today().normalize()
    analysis_start = today - pd.DateOffset(years=years)
    fetch_start = analysis_start - pd.Timedelta(days=WARMUP_TRADING_DAYS * 7 // 5)

    try:
        raw = _download(ticker, fetch_start, today + pd.Timedelta(days=1))
    except DataFetchError:
        raise
    except Exception as exc:
        raise DataFetchError(f"{ticker}: download failed, {exc}") from exc

    df = _clean(raw, ticker)

    expected = years * TRADING_DAYS_PER_YEAR
    in_window = int((df.index >= analysis_start).sum())
    if in_window < expected * 0.8:
        raise DataFetchError(
            f"{ticker}: only {in_window} bars in the {years}-year window, expected near {expected}. "
            "Ticker may be delisted or recently listed."
        )

    log.info("%s: %d bars total, %d in analysis window", ticker, len(df), in_window)
    return PriceHistory(ticker=ticker, frame=df, analysis_start=analysis_start)
