"""Task 1B tests against a scripted client. No API key and no network required.

Run: python -m task1_financial.tests.test_analysis
"""

from common.llm import LLMError, SchemaValidationError
from common.schemas import AggregateSentiment, HeadlineSentiment, TradingSignal
from task1_financial.src import analysis, prompts

SUMMARY = {
    "ticker": "NVDA",
    "company_name": "NVIDIA Corporation",
    "as_of": "2026-08-31",
    "current_price": 220.78,
    "currency": "USD",
    "week52_high": 235.47,
    "week52_low": 164.98,
    "pe_ratio": 27.88,
    "ytd_return_pct": 17.05,
    "indicators": {
        "sma_50": 208.62, "sma_200": 195.80, "rsi_14": 54.56,
        "macd": 2.48, "macd_signal": 2.77, "macd_hist": -0.28,
        "bb_upper": 229.36, "bb_mid": 218.75, "bb_lower": 208.15, "bb_pct_b": 0.60,
    },
    "momentum": {"signal": "Bullish", "score": 0.25, "flags": ["golden_cross"]},
}

HEADLINES = [
    {"headline": "Nvidia beats on Q2 earnings", "publisher": "Reuters"},
    {"headline": "Nvidia faces new export restrictions", "publisher": "Bloomberg"},
    {"headline": "Nvidia announces routine board change", "publisher": "PR"},
]


class MockClient:
    """Stands in for LLMClient. Scripts one response per structured() call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def structured(self, system, user, schema, repair=True):
        self.calls.append({"system": system, "user": user, "schema": schema.__name__})
        if not self.script:
            raise AssertionError("mock ran out of scripted responses")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return schema.model_validate(item)

    def chat(self, system, user, json_mode=False):
        raise AssertionError("analysis must go through structured(), not chat()")


def _sent(headline, sentiment, confidence, reason="because"):
    return {
        "headline": headline,
        "sentiment": sentiment,
        "confidence": confidence,
        "brief_reason": reason,
    }


SIGNAL_OK = {
    "signal": "Hold",
    "justification": (
        "The golden cross keeps the primary trend intact. A negative MACD histogram "
        "against that cross shows momentum fading before price has turned. RSI near the "
        "midpoint offers no confirmation either way. That divergence argues for patience."
    ),
    "key_drivers": ["golden cross against falling MACD histogram", "RSI neutral in an uptrend"],
}


def test_happy_path_end_to_end():
    client = MockClient([
        _sent("Nvidia beats on Q2 earnings", "positive", 0.9),
        _sent("Nvidia faces new export restrictions", "negative", 0.6),
        _sent("Nvidia announces routine board change", "neutral", 0.5),
        SIGNAL_OK,
    ])
    out = analysis.run_analysis(SUMMARY, HEADLINES, client=client)

    assert len(out["headline_sentiment"]) == 3
    assert out["sentiment_failures"] == []
    assert out["classification_success_rate"] == 1.0
    assert out["signal"]["signal"] == "Hold"
    assert out["signal_error"] is None
    assert len(client.calls) == 4, "three classifications plus one signal call"
    return 6


def test_aggregation_is_confidence_weighted():
    results = [
        HeadlineSentiment(**_sent("a", "positive", 0.9)),
        HeadlineSentiment(**_sent("b", "negative", 0.6)),
        HeadlineSentiment(**_sent("c", "neutral", 0.5)),
    ]
    agg = analysis.aggregate(results)
    # (1*0.9 + -1*0.6 + 0*0.5) / (0.9+0.6+0.5) = 0.3 / 2.0 = 0.15
    assert abs(agg.score - 0.15) < 1e-9, f"got {agg.score}"
    assert agg.label == "positive"
    assert (agg.positive, agg.negative, agg.neutral) == (1, 1, 1)
    assert abs(agg.mean_confidence - 0.6667) < 1e-3

    # An unweighted count would call this net zero. Confidence weighting must not.
    naive = (1 - 1 + 0) / 3
    assert abs(naive) < 1e-9 and agg.score > naive
    return 6


def test_aggregate_of_empty_is_none():
    assert analysis.aggregate([]) is None
    return 1


def test_validation_failure_is_recorded_not_defaulted():
    client = MockClient([
        _sent("Nvidia beats on Q2 earnings", "positive", 0.9),
        SchemaValidationError("confidence must be <= 1"),
        _sent("Nvidia announces routine board change", "neutral", 0.5),
        SIGNAL_OK,
    ])
    out = analysis.run_analysis(SUMMARY, HEADLINES, client=client)

    assert len(out["headline_sentiment"]) == 2
    assert len(out["sentiment_failures"]) == 1
    assert "confidence" in out["sentiment_failures"][0]["error"]
    assert abs(out["classification_success_rate"] - 0.6667) < 1e-3

    # The failed headline must be absent from the aggregate, not folded in as neutral.
    agg = out["aggregate_sentiment"]
    assert agg["positive"] + agg["negative"] + agg["neutral"] == 2
    return 5


def test_transport_error_does_not_abort_the_batch():
    client = MockClient([
        LLMError("all providers failed"),
        _sent("Nvidia faces new export restrictions", "negative", 0.8),
        _sent("Nvidia announces routine board change", "neutral", 0.5),
        SIGNAL_OK,
    ])
    out = analysis.run_analysis(SUMMARY, HEADLINES, client=client)
    assert len(out["headline_sentiment"]) == 2
    assert len(out["sentiment_failures"]) == 1
    assert out["signal"] is not None, "one bad headline must not block the signal"
    return 3


def test_signal_failure_is_caught():
    client = MockClient([
        _sent("Nvidia beats on Q2 earnings", "positive", 0.9),
        _sent("Nvidia faces new export restrictions", "negative", 0.6),
        _sent("Nvidia announces routine board change", "neutral", 0.5),
        SchemaValidationError("justification must be 3 to 5 sentences, found 1"),
    ])
    out = analysis.run_analysis(SUMMARY, HEADLINES, client=client)
    assert out["signal"] is None
    assert "sentences" in out["signal_error"]
    assert len(out["headline_sentiment"]) == 3, "sentiment survives a signal failure"
    return 3


def test_thin_coverage_is_flagged_not_blocked():
    client = MockClient([_sent("only one", "positive", 0.9), SIGNAL_OK])
    out = analysis.run_analysis(SUMMARY, [{"headline": "only one", "publisher": "x"}], client=client)

    # The call rests on the indicators, so one headline degrades it, never cancels it.
    assert out["signal"] is not None
    assert out["signal_error"] is None
    prompt = client.calls[-1]["user"]
    assert "Coverage is thin" in prompt
    assert "Headlines classified: 1 of 1" in prompt
    return 4


def test_total_sentiment_failure_still_produces_a_signal():
    client = MockClient([
        LLMError("provider down"),
        LLMError("provider down"),
        LLMError("provider down"),
        SIGNAL_OK,
    ])
    out = analysis.run_analysis(SUMMARY, HEADLINES, client=client)

    assert out["aggregate_sentiment"] is None
    assert len(out["sentiment_failures"]) == 3
    assert out["classification_success_rate"] == 0.0
    assert out["signal"] is not None, "technicals alone must still yield a call"

    prompt = client.calls[-1]["user"]
    assert "No headline sentiment could be retrieved" in prompt
    assert "54.56" in prompt, "indicators must still reach the model"
    return 6


def test_original_headline_is_pinned():
    original = "Nvidia beats on Q2 earnings"
    client = MockClient([_sent("NVIDIA exceeded second quarter estimates", "positive", 0.9)])
    outcome = analysis.analyse_headlines(
        [{"headline": original, "publisher": "Reuters"}], "NVDA", "NVIDIA Corporation", client
    )
    assert outcome.results[0].headline == original, "paraphrase must not overwrite the source"
    return 1


def test_prompt_separation():
    client = MockClient([
        _sent("Nvidia beats on Q2 earnings", "positive", 0.9),
        _sent("Nvidia faces new export restrictions", "negative", 0.6),
        _sent("Nvidia announces routine board change", "neutral", 0.5),
        SIGNAL_OK,
    ])
    analysis.run_analysis(SUMMARY, HEADLINES, client=client)

    for call in client.calls:
        # Per-request data belongs in the user turn. A system prompt naming the ticker
        # would mean the role definition changes per call.
        assert "NVDA" not in call["system"], f"ticker leaked into system prompt"
        assert "220.78" not in call["system"], "price leaked into system prompt"
        assert len(call["system"]) > 100, "system prompt must actually define the role"

    signal_call = client.calls[-1]
    assert signal_call["schema"] == "TradingSignal"
    for token in ["208.62", "195.8", "54.56", "-0.28", "Bullish", "golden_cross"]:
        assert token in signal_call["user"], f"{token} missing from signal prompt"
    assert "restating their values back is a failure" in signal_call["system"].lower()
    return 12


def test_schema_rejects_bad_llm_output():
    import pydantic

    bad = [
        ({"headline": "h", "sentiment": "bullish", "confidence": 0.5, "brief_reason": "r"},
         "sentiment outside the allowed literal"),
        ({"headline": "h", "sentiment": "positive", "confidence": 1.4, "brief_reason": "r"},
         "ambiguous 1.4, neither a valid score nor clearly a percentage"),
        ({"headline": "h", "sentiment": "positive", "confidence": 140, "brief_reason": "r"},
         "percentage above 100"),
        ({"headline": "h", "sentiment": "positive", "confidence": -0.2, "brief_reason": "r"},
         "negative confidence"),
        ({"headline": "h", "sentiment": "positive", "confidence": 0.5, "brief_reason": ""},
         "empty brief_reason"),
    ]
    for payload, why in bad:
        try:
            HeadlineSentiment(**payload)
            raise AssertionError(f"should have rejected: {why}")
        except pydantic.ValidationError:
            pass

    # Percentages are coerced rather than rejected, models emit them often.
    assert abs(HeadlineSentiment(**_sent("h", "positive", 85)).confidence - 0.85) < 1e-9
    assert abs(HeadlineSentiment(**_sent("h", "positive", "0.7")).confidence - 0.7) < 1e-9
    assert abs(HeadlineSentiment(**_sent("h", "positive", "85%")).confidence - 0.85) < 1e-9
    # Exactly 1.0 is a legitimate score and must not be divided by 100.
    assert HeadlineSentiment(**_sent("h", "positive", 1.0)).confidence == 1.0
    assert HeadlineSentiment(**_sent("h", "positive", 0.0)).confidence == 0.0

    try:
        TradingSignal(signal="Buy", justification="Too short.", key_drivers=[])
        raise AssertionError("should have rejected a one-sentence justification")
    except pydantic.ValidationError:
        pass
    return 12


def main():
    checks = [
        test_happy_path_end_to_end,
        test_aggregation_is_confidence_weighted,
        test_aggregate_of_empty_is_none,
        test_validation_failure_is_recorded_not_defaulted,
        test_transport_error_does_not_abort_the_batch,
        test_signal_failure_is_caught,
        test_thin_coverage_is_flagged_not_blocked,
        test_total_sentiment_failure_still_produces_a_signal,
        test_original_headline_is_pinned,
        test_prompt_separation,
        test_schema_rejects_bad_llm_output,
    ]
    total = 0
    for check in checks:
        n = check()
        total += n
        print(f"  PASS  {check.__name__:48s} ({n} assertions)")
    print(f"\n{len(checks)} tests passed, {total} assertions.")


if __name__ == "__main__":
    main()
