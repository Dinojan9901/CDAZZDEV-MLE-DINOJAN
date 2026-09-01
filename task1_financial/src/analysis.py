"""Task 1B, LLM sentiment classification and signal reasoning.

Every LLM response is validated against a Pydantic model before it reaches the report.
A headline whose classification fails validation is recorded as a failure and excluded
from the aggregate rather than being silently defaulted to neutral, which would drag the
score toward zero and hide the problem.
"""

import logging
from dataclasses import dataclass, field

from common import config
from common.llm import LLMError, SchemaValidationError, get_client
from common.schemas import AggregateSentiment, HeadlineSentiment, TradingSignal
from task1_financial.src import prompts

log = logging.getLogger(__name__)

POLARITY = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
LABEL_BANDS = [(0.15, "positive"), (-0.15, "neutral")]

# Below this many classified headlines the aggregate is too noisy to lean on, so the
# prompt says so and tells the model to weight technicals instead. It is not a hard gate:
# the call rests on the indicators, so losing news degrades it rather than cancelling it.
THIN_COVERAGE = 5


@dataclass
class SentimentOutcome:
    results: list[HeadlineSentiment] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    aggregate: AggregateSentiment | None = None

    @property
    def success_rate(self) -> float:
        total = len(self.results) + len(self.failures)
        return len(self.results) / total if total else 0.0


def _fmt(value, suffix: str = "") -> str:
    return "not available" if value is None else f"{value}{suffix}"


def classify_headline(entry: dict, ticker: str, company: str, client) -> HeadlineSentiment:
    user = prompts.SENTIMENT_USER.substitute(
        company=company,
        ticker=ticker,
        publisher=entry.get("publisher") or "unknown",
        headline=entry["headline"],
    )
    result = client.structured(prompts.SENTIMENT_SYSTEM, user, HeadlineSentiment)
    # Models paraphrase the headline back. Pin it to the original so the report and the
    # source list cannot disagree.
    result.headline = entry["headline"]
    return result


def analyse_headlines(
    headlines: list[dict], ticker: str, company: str, client=None
) -> SentimentOutcome:
    # One call per headline is what the brief asks for, and it is also the single
    # largest consumer of the daily token budget, so it runs on the smaller model.
    client = client or get_client(model=config.GROQ_MODEL_FAST)
    outcome = SentimentOutcome()

    for entry in headlines:
        try:
            outcome.results.append(classify_headline(entry, ticker, company, client))
        except (SchemaValidationError, LLMError) as exc:
            log.warning("classification failed for %r: %s", entry["headline"][:60], exc)
            outcome.failures.append({"headline": entry["headline"], "error": str(exc)})
        except Exception as exc:
            log.error("unexpected error on %r: %s", entry["headline"][:60], exc)
            outcome.failures.append({"headline": entry["headline"], "error": repr(exc)})

    outcome.aggregate = aggregate(outcome.results)
    if outcome.failures:
        log.warning(
            "%d of %d headlines failed classification",
            len(outcome.failures), len(headlines),
        )
    return outcome


def aggregate(results: list[HeadlineSentiment]) -> AggregateSentiment | None:
    """Confidence-weighted mean polarity.

    Weighting by confidence stops a hedged 0.35 call from carrying the same weight as an
    unambiguous 0.95 one, which an unweighted count would do.
    """
    if not results:
        return None

    weight_total = sum(r.confidence for r in results)
    if weight_total <= 0:
        score = 0.0
    else:
        score = sum(POLARITY[r.sentiment] * r.confidence for r in results) / weight_total

    label = "negative"
    for threshold, name in LABEL_BANDS:
        if score >= threshold:
            label = name
            break

    return AggregateSentiment(
        score=round(score, 4),
        label=label,
        positive=sum(1 for r in results if r.sentiment == "positive"),
        negative=sum(1 for r in results if r.sentiment == "negative"),
        neutral=sum(1 for r in results if r.sentiment == "neutral"),
        mean_confidence=round(weight_total / len(results), 4),
    )


def _cross_state(sma_50, sma_200) -> str:
    if sma_50 is None or sma_200 is None:
        return "not available"
    gap = (sma_50 / sma_200 - 1) * 100 if sma_200 else 0.0
    direction = "above" if sma_50 > sma_200 else "below"
    return f"SMA50 {direction} SMA200 by {abs(gap):.2f} percent"


def _price_vs_sma(price, sma_50) -> str:
    if price is None or sma_50 is None:
        return "not available"
    gap = (price / sma_50 - 1) * 100 if sma_50 else 0.0
    side = "above" if price > sma_50 else "below"
    return f"price {side} SMA50 by {abs(gap):.2f} percent"


def _top_headlines_block(outcome: SentimentOutcome, limit: int = 3) -> str:
    ranked = sorted(
        outcome.results,
        key=lambda r: (r.sentiment != "neutral", r.confidence),
        reverse=True,
    )[:limit]
    if not ranked:
        return "    none available"
    return "\n".join(
        f"    [{r.sentiment}, {r.confidence:.2f}] {r.headline}" for r in ranked
    )


def _sentiment_block(outcome: SentimentOutcome, attempted: int) -> str:
    agg = outcome.aggregate
    if agg is None:
        return (
            "  No headline sentiment could be retrieved for this run.\n"
            "  Base the call on the technical evidence alone and state that limitation\n"
            "  explicitly in the justification."
        )

    lines = [
        f"  Headlines classified: {len(outcome.results)} of {attempted} retrieved",
        f"  Aggregate score: {agg.score} on a scale of -1 to 1, read as {agg.label}",
        f"  Breakdown: {agg.positive} positive, {agg.negative} negative, {agg.neutral} neutral",
        f"  Mean confidence: {agg.mean_confidence}",
    ]
    if len(outcome.results) < THIN_COVERAGE:
        lines.append(
            "  Coverage is thin, so treat this as weak evidence and weight the technical\n"
            "  picture more heavily."
        )
    lines.append("  Most notable headlines:")
    lines.append(_top_headlines_block(outcome))
    return "\n".join(lines)


def generate_signal(
    summary: dict, outcome: SentimentOutcome, attempted: int = 0, client=None
) -> TradingSignal:
    client = client or get_client()
    ind = summary.get("indicators", {})
    momentum = summary.get("momentum", {})

    user = prompts.SIGNAL_USER.substitute(
        company=summary.get("company_name", summary["ticker"]),
        ticker=summary["ticker"],
        as_of=summary.get("as_of", "unknown"),
        price=_fmt(summary.get("current_price")),
        currency=summary.get("currency", "USD"),
        sma_50=_fmt(ind.get("sma_50")),
        sma_200=_fmt(ind.get("sma_200")),
        price_vs_sma50=_price_vs_sma(summary.get("current_price"), ind.get("sma_50")),
        cross_state=_cross_state(ind.get("sma_50"), ind.get("sma_200")),
        rsi_14=_fmt(ind.get("rsi_14")),
        macd=_fmt(ind.get("macd")),
        macd_signal=_fmt(ind.get("macd_signal")),
        macd_hist=_fmt(ind.get("macd_hist")),
        bb_upper=_fmt(ind.get("bb_upper")),
        bb_mid=_fmt(ind.get("bb_mid")),
        bb_lower=_fmt(ind.get("bb_lower")),
        bb_pct_b=_fmt(ind.get("bb_pct_b")),
        week52_high=_fmt(summary.get("week52_high")),
        week52_low=_fmt(summary.get("week52_low")),
        pe_ratio=_fmt(summary.get("pe_ratio")),
        ytd_return_pct=_fmt(summary.get("ytd_return_pct")),
        momentum_signal=momentum.get("signal", "not available"),
        momentum_score=_fmt(momentum.get("score")),
        momentum_flags=", ".join(momentum.get("flags", [])) or "none",
        sentiment_block=_sentiment_block(outcome, attempted or len(outcome.results)),
    )
    return client.structured(prompts.SIGNAL_SYSTEM, user, TradingSignal)


def run_analysis(summary: dict, headlines: list[dict], client=None) -> dict:
    client = client or get_client()
    ticker = summary["ticker"]
    company = summary.get("company_name", ticker)

    outcome = analyse_headlines(headlines, ticker, company, client)

    signal, signal_error = None, None
    try:
        signal = generate_signal(summary, outcome, attempted=len(headlines), client=client)
    except (SchemaValidationError, LLMError) as exc:
        signal_error = str(exc)
        log.error("signal generation failed: %s", exc)

    return {
        "ticker": ticker,
        "headline_sentiment": [r.model_dump() for r in outcome.results],
        "sentiment_failures": outcome.failures,
        "classification_success_rate": round(outcome.success_rate, 4),
        "aggregate_sentiment": outcome.aggregate.model_dump() if outcome.aggregate else None,
        "signal": signal.model_dump() if signal else None,
        "signal_error": signal_error,
    }
