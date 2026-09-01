import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Sentiment = Literal["positive", "negative", "neutral"]
Signal = Literal["Buy", "Hold", "Sell"]

TYPOGRAPHY = {
    "‑": "-", "‐": "-", "‒": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    " ": " ", "…": "...",
}


def normalise_prose(text: str) -> str:
    """Fold model typography down to plain ASCII punctuation.

    Applied to generated prose only, never to a source headline: those are quotations
    and rewriting their punctuation would misquote the publisher.
    """
    for src, dst in TYPOGRAPHY.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s*[–—]\s*", ", ", text)
    text = re.sub(r",\s*,", ",", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


class HeadlineSentiment(BaseModel):
    headline: str
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    brief_reason: str = Field(min_length=1, max_length=400)

    @field_validator("brief_reason")
    @classmethod
    def _clean_prose(cls, v):
        return normalise_prose(v)

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v):
        """Accept "0.85" and 85, reject anything ambiguous.

        Only values of 2 or more are treated as percentages. A bare 1.4 could be a
        percentage or a broken 0-to-1 score, and guessing turns it into 0.014, which
        would then be silently weighted into the aggregate. Ambiguous input fails
        validation instead, where the caller logs it and drops the headline.
        """
        if isinstance(v, str):
            v = float(v.strip().rstrip("%"))
        if isinstance(v, (int, float)) and 2 <= v <= 100:
            v = v / 100
        return v


class SentimentBatch(BaseModel):
    results: list[HeadlineSentiment]


class AggregateSentiment(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    label: Sentiment
    positive: int
    negative: int
    neutral: int
    mean_confidence: float = Field(ge=0.0, le=1.0)


class TradingSignal(BaseModel):
    signal: Signal
    justification: str = Field(min_length=1)
    key_drivers: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("key_drivers")
    @classmethod
    def _clean_drivers(cls, v):
        return [normalise_prose(d) for d in v]

    @field_validator("justification")
    @classmethod
    def _min_sentences(cls, v):
        v = normalise_prose(v)
        # Count sentence terminators followed by a boundary, so decimals in
        # "RSI at 54.5" are not mistaken for three sentences.
        n = len(re.findall(r"[.!?](?:\s|$)", v.strip()))
        if n < 3:
            raise ValueError(f"justification must be 3 to 5 sentences, found {n}")
        return v
