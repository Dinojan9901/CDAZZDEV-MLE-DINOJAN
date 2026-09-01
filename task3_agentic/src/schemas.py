"""Typed contracts between the agents, Task 3B.

Agent A hands Agent B a validated model, never a formatted string. A string handoff
would let a malformed number reach the report as prose, where nothing can catch it.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from common.schemas import normalise_prose

Severity = Literal["high", "medium", "low"]


class PriceSnapshot(BaseModel):
    current_price: float | None = None
    week52_high: float | None = None
    week52_low: float | None = None
    pe_ratio: float | None = None
    ytd_return_pct: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    rsi_14: float | None = None
    macd_hist: float | None = None
    momentum_signal: str = "Unknown"
    momentum_flags: list[str] = Field(default_factory=list)


class VolatilityProfile(BaseModel):
    window_days: int
    annualised_volatility_pct: float
    annualised_volatility_1y_pct: float | None = None
    mean_abs_daily_move_pct: float | None = None
    regime: str = "unknown"


class SentimentSummary(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    label: str
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    mean_confidence: float | None = None
    classified: int = 0


class DataBrief(BaseModel):
    """Agent A's deliverable. Everything quantitative Agent B is allowed to rely on."""

    ticker: str
    company_name: str = ""
    as_of: str = ""
    price: PriceSnapshot = Field(default_factory=PriceSnapshot)
    volatility: VolatilityProfile | None = None
    sentiment: SentimentSummary | None = None
    quant_observations: list[str] = Field(default_factory=list, max_length=8)
    data_gaps: list[str] = Field(default_factory=list)

    @field_validator("quant_observations", "data_gaps")
    @classmethod
    def _clean(cls, v):
        return [normalise_prose(x) for x in v if x and x.strip()]


class ClarificationRequest(BaseModel):
    """Agent B may raise exactly one of these back to Agent A."""

    question: str = Field(min_length=5)
    reason: str = ""

    @field_validator("question", "reason")
    @classmethod
    def _clean(cls, v):
        return normalise_prose(v)


class ClarificationResponse(BaseModel):
    question: str
    answer: str = Field(min_length=1)
    supporting_data: dict = Field(default_factory=dict)

    @field_validator("answer")
    @classmethod
    def _clean(cls, v):
        return normalise_prose(v)


class RiskItem(BaseModel):
    risk: str = Field(min_length=5)
    evidence: str = Field(min_length=5)
    severity: Severity = "medium"

    @field_validator("risk", "evidence")
    @classmethod
    def _clean(cls, v):
        return normalise_prose(v)


class HedgeStrategy(BaseModel):
    strategy: str = Field(min_length=5)
    rationale: str = Field(min_length=5)
    instruments: list[str] = Field(default_factory=list, max_length=5)
    data_basis: str = ""

    @field_validator("strategy", "rationale", "data_basis")
    @classmethod
    def _clean(cls, v):
        return normalise_prose(v)

    @field_validator("instruments")
    @classmethod
    def _clean_list(cls, v):
        return [normalise_prose(x) for x in v]


class ResearchReport(BaseModel):
    """The deliverable. Three sections, exactly as the brief specifies."""

    ticker: str
    financial_health_summary: str = Field(min_length=40)
    top_risks: list[RiskItem] = Field(min_length=3, max_length=3)
    hedge_strategy: HedgeStrategy

    @field_validator("financial_health_summary")
    @classmethod
    def _clean(cls, v):
        return normalise_prose(v)

    def to_markdown(self) -> str:
        lines = [
            f"# Research report: {self.ticker}",
            "",
            "## Financial Health Summary",
            "",
            self.financial_health_summary,
            "",
            "## Top Three Risks",
            "",
        ]
        for i, item in enumerate(self.top_risks, 1):
            lines += [
                f"{i}. **{item.risk}** ({item.severity} severity)",
                f"   Evidence: {item.evidence}",
                "",
            ]
        lines += [
            "## Hedge Strategy Recommendation",
            "",
            f"**{self.hedge_strategy.strategy}**",
            "",
            self.hedge_strategy.rationale,
            "",
        ]
        if self.hedge_strategy.instruments:
            lines += ["Instruments: " + ", ".join(self.hedge_strategy.instruments), ""]
        if self.hedge_strategy.data_basis:
            lines += [f"Data basis: {self.hedge_strategy.data_basis}", ""]
        return "\n".join(lines)
