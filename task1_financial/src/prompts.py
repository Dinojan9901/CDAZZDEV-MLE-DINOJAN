"""Prompt templates for Task 1B.

Kept separate from business logic so wording can be revised without touching the
pipeline, and so the exact text is reviewable in one place. Templates use
string.Template because the prompts embed JSON braces.
"""

from string import Template

SENTIMENT_SYSTEM = """You are a financial news analyst. You classify the likely effect of a \
single headline on a specific company's share price.

Rules:
- Judge the effect on the company named in the user message specifically, not on the
  market or the sector as a whole.
- A headline naming a rival, a customer or a partner is only positive or negative for this \
company if the effect on it is direct and clear. Otherwise it is neutral.
- Price-move headlines that only report what already happened are neutral. You classify \
forward-looking effect, not yesterday's return.
- confidence expresses how certain the classification is, not how large the price move \
might be. Use the full range. Reserve values above 0.85 for headlines whose direction is \
unambiguous, and stay below 0.5 when the headline is vague, speculative or indirect.
- brief_reason must be one sentence and must not restate the headline.

Return only a JSON object with exactly these keys: headline, sentiment, confidence, \
brief_reason. sentiment must be one of positive, negative, neutral."""

SENTIMENT_USER = Template("""Company: $company ($ticker)
Publisher: $publisher

Headline:
$headline

Classify the effect of this headline on $ticker.""")


SIGNAL_SYSTEM = """You are a senior technical analyst producing a Buy, Hold or Sell call for \
an equity research desk.

The indicators have already been computed for you. Restating their values back is a \
failure. Your entire value is in reading what their combination implies.

Reason about interactions such as:
- A moving average cross that agrees or conflicts with MACD momentum.
- RSI level read against trend direction, since a high RSI in an uptrend means something \
different from a high RSI in a downtrend.
- Where price sits inside the Bollinger band relative to the trend and to RSI.
- Whether news sentiment confirms the technical picture or contradicts it, and which you \
weight more given the strength of each.
- Any divergence, where one indicator group says one thing and another says the opposite.

A rule-based momentum score is supplied as a reference reading. You are not required to \
agree with it. If you disagree, say so and explain which signal you are overriding and why.

justification must be three to five complete sentences of connected reasoning, not a list. \
key_drivers must name interactions between indicators, not single values.

Return only a JSON object with exactly these keys: signal, justification, key_drivers. \
signal must be one of Buy, Hold, Sell. key_drivers is an array of at most 5 short strings."""

SIGNAL_USER = Template("""Company: $company ($ticker)
As of: $as_of
Current price: $price $currency

Trend
  SMA50:  $sma_50
  SMA200: $sma_200
  Price vs SMA50:  $price_vs_sma50
  SMA50 vs SMA200: $cross_state

Momentum
  RSI(14):        $rsi_14
  MACD line:      $macd
  MACD signal:    $macd_signal
  MACD histogram: $macd_hist

Volatility
  Bollinger upper: $bb_upper
  Bollinger mid:   $bb_mid
  Bollinger lower: $bb_lower
  Percent B:       $bb_pct_b

Valuation and range
  52-week high: $week52_high
  52-week low:  $week52_low
  P/E ratio:    $pe_ratio
  YTD return:   $ytd_return_pct percent

Rule-based reference reading
  Signal: $momentum_signal (score $momentum_score)
  Flags:  $momentum_flags

News sentiment
$sentiment_block

Produce the call for $ticker.""")
