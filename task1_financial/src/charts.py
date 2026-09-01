"""Matplotlib chart for the equity research brief.

Three stacked panels sharing one time axis rather than a dual-axis overlay. Price,
RSI and MACD live on incompatible scales, and putting them on two y-axes would let the
reader infer crossings that are an artefact of the scaling.

Palette is the validated categorical set. Aqua sits below 3:1 against the surface, so
every line carries a direct label at the right edge, which is the required relief.
"""

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e5e4e0"

CLOSE = "#2a78d6"
SMA_50 = "#eb6834"
SMA_200 = "#1baf7a"
MACD_LINE = "#4a3aa7"
MACD_SIGNAL = "#eb6834"
POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"
BAND = "#c9c8c3"

RSI_HIGH, RSI_LOW = 70, 30


def _style_axis(ax, label: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.set_ylabel(label, color=INK_MUTED, fontsize=9.5)


def _label_group(ax, x, items: list[tuple[float, str, str]], min_gap_frac: float = 0.09) -> None:
    """Place right-edge labels, nudging them apart when values sit close together.

    MACD and its signal line converge near a crossover, which is exactly when the panel
    matters most and exactly when two labels would overprint each other.
    """
    points = [(y, text, colour) for y, text, colour in items if not pd.isna(y)]
    if not points:
        return

    low, high = ax.get_ylim()
    min_gap = (high - low) * min_gap_frac

    points.sort(key=lambda p: p[0])
    placed = []
    for y, text, colour in points:
        if placed and y - placed[-1][0] < min_gap:
            y = placed[-1][0] + min_gap
        placed.append((y, text, colour))

    overflow = placed[-1][0] - high
    if overflow > 0:
        placed = [(y - overflow, t, c) for y, t, c in placed]

    for (y_at, _, _), (y_label, text, colour) in zip(points, placed):
        ax.annotate(
            text,
            xy=(x, y_at),
            xytext=(8, (y_label - y_at) / (high - low) * ax.bbox.height),
            textcoords="offset points",
            color=colour,
            fontsize=8.5,
            fontweight="semibold",
            va="center",
            clip_on=False,
        )


def build_chart(df: pd.DataFrame, ticker: str, company: str, months: int = 12) -> bytes:
    """Render the brief's chart to PNG bytes.

    Only the trailing `months` are drawn. The indicators were computed on the full
    history, so the SMA200 line is complete from the first plotted bar.
    """
    cutoff = df.index[-1] - pd.DateOffset(months=months)
    view = df.loc[df.index >= cutoff]
    if view.empty:
        view = df

    fig, (ax_price, ax_rsi, ax_macd) = plt.subplots(
        3, 1, figsize=(11, 8.6), sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1.0, 1.35], "hspace": 0.16},
    )
    fig.patch.set_facecolor(SURFACE)

    x_end = view.index[-1]

    if {"bb_upper", "bb_lower"} <= set(view.columns):
        ax_price.fill_between(
            view.index, view["bb_lower"], view["bb_upper"],
            color=BAND, alpha=0.30, linewidth=0, zorder=1,
        )
    ax_price.plot(view.index, view["Close"], color=CLOSE, linewidth=2.0, zorder=4)
    price_labels = [(view["Close"].iloc[-1], "Close", CLOSE)]

    for col, colour, name in (("sma_50", SMA_50, "SMA50"), ("sma_200", SMA_200, "SMA200")):
        if col in view.columns:
            ax_price.plot(view.index, view[col], color=colour, linewidth=2.0, zorder=3)
            price_labels.append((view[col].iloc[-1], name, colour))
    _label_group(ax_price, x_end, price_labels, min_gap_frac=0.05)

    _style_axis(ax_price, "Price")
    ax_price.set_title(
        f"{company} ({ticker})   last {months} months",
        color=INK, fontsize=13.5, fontweight="bold", loc="left", pad=14,
    )

    if "rsi_14" in view.columns:
        ax_rsi.axhspan(RSI_HIGH, 100, color=NEGATIVE, alpha=0.06, linewidth=0)
        ax_rsi.axhspan(0, RSI_LOW, color=POSITIVE, alpha=0.06, linewidth=0)
        for level in (RSI_LOW, RSI_HIGH):
            ax_rsi.axhline(level, color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 3)), alpha=0.55)
        ax_rsi.plot(view.index, view["rsi_14"], color=CLOSE, linewidth=1.8)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_yticks([0, 30, 50, 70, 100])
        _label_group(ax_rsi, x_end, [(view["rsi_14"].iloc[-1], "RSI", CLOSE)])
    _style_axis(ax_rsi, "RSI(14)")

    if {"macd", "macd_signal", "macd_hist"} <= set(view.columns):
        hist = view["macd_hist"]
        ax_macd.bar(
            view.index, hist, width=1.0, linewidth=0,
            color=[NEGATIVE if v < 0 else POSITIVE for v in hist.fillna(0)],
            alpha=0.30, zorder=1,
        )
        ax_macd.axhline(0, color=INK_MUTED, linewidth=0.9, alpha=0.6)
        ax_macd.plot(view.index, view["macd"], color=MACD_LINE, linewidth=1.8, zorder=3)
        ax_macd.plot(
            view.index, view["macd_signal"], color=MACD_SIGNAL,
            linewidth=1.6, linestyle=(0, (5, 2)), zorder=3,
        )
        _label_group(ax_macd, x_end, [
            (view["macd"].iloc[-1], "MACD", MACD_LINE),
            (view["macd_signal"].iloc[-1], "Signal", MACD_SIGNAL),
        ], min_gap_frac=0.13)
    _style_axis(ax_macd, "MACD(12,26,9)")

    ax_macd.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_macd.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=0, ha="center")

    # Room on the right for the direct labels, which sit outside the axes.
    fig.subplots_adjust(left=0.07, right=0.90, top=0.93, bottom=0.07)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return buffer.getvalue()
