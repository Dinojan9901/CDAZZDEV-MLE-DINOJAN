"""Task definition for financial risk clause extraction.

The taxonomy is closed on purpose. An open-ended category field would make the task
unscoreable: two labels meaning the same thing could not be told apart from a genuine
disagreement, and the model could never be wrong about a category it invented.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from common.schemas import normalise_prose

RISK_CATEGORIES = [
    "market_risk",
    "credit_risk",
    "liquidity_risk",
    "operational_risk",
    "regulatory_risk",
    "cybersecurity_risk",
    "supply_chain_risk",
    "concentration_risk",
    "geopolitical_risk",
    "technology_risk",
    "litigation_risk",
    "environmental_risk",
]

CATEGORY_DEFINITIONS = {
    "market_risk": "losses from moves in prices, rates, spreads or foreign exchange",
    "credit_risk": "a counterparty, borrower or customer failing to pay",
    "liquidity_risk": "inability to meet obligations or fund operations when due",
    "operational_risk": "failed internal processes, people, systems or execution",
    "regulatory_risk": "new or changed rules, licences, tax treatment or enforcement",
    "cybersecurity_risk": "breach, ransomware, data loss or attack on systems",
    "supply_chain_risk": "supplier failure, input shortage, logistics or capacity limits",
    "concentration_risk": "dependence on a few customers, products, suppliers or regions",
    "geopolitical_risk": "conflict, sanctions, trade restrictions or political instability",
    "technology_risk": "obsolescence, failed adoption or a competing technology shift",
    "litigation_risk": "lawsuits, claims, investigations or contractual disputes",
    "environmental_risk": "climate, weather, emissions liability or resource scarcity",
}

Severity = Literal["high", "medium", "low"]
Category = Literal[tuple(RISK_CATEGORIES)]  # type: ignore[valid-type]

SECTORS = [
    "semiconductor manufacturing", "regional banking", "pharmaceuticals",
    "oil and gas exploration", "specialty retail", "third-party logistics",
    "property and casualty insurance", "commercial REIT", "regional airline",
    "copper mining", "telecommunications", "agricultural processing",
    "automotive components", "enterprise software", "regulated water utility",
]

DOC_STYLES = {
    "10k_risk_factors": "a Risk Factors section of an annual report, formal and hedged",
    "mdna": "Management Discussion and Analysis, explanatory and forward-looking",
    "earnings_call": "prepared remarks from an earnings call, conversational but precise",
    "bond_prospectus": "a debt prospectus risk summary, dense and legalistic",
}


class RiskItem(BaseModel):
    category: Category
    summary: str = Field(min_length=10, max_length=200)
    trigger: str = Field(min_length=10, max_length=300)
    potential_impact: str = Field(min_length=10, max_length=300)
    severity: Severity

    @field_validator("summary", "trigger", "potential_impact")
    @classmethod
    def _clean(cls, v):
        return normalise_prose(v)


class RiskExtraction(BaseModel):
    risks: list[RiskItem] = Field(min_length=1, max_length=5)


class GeneratedExample(BaseModel):
    """One teacher-produced training pair, plus the seed that shaped it.

    `sector` and `doc_style` are filled in from the generation seed rather than asked
    for, so the teacher spends no tokens echoing back what we already told it.

    `extraction` accepts either the wrapped object or a bare list. Models return the
    bare list most of the time, and rejecting it forced a repair call that doubled the
    token cost of the run for no gain in data quality.
    """

    sector: str = ""
    doc_style: str = ""
    teacher_provider: str = ""
    passage: str = Field(min_length=200)
    extraction: RiskExtraction

    @field_validator("extraction", mode="before")
    @classmethod
    def _accept_bare_list(cls, v):
        if isinstance(v, list):
            return {"risks": v}
        return v

    @field_validator("passage")
    @classmethod
    def _clean(cls, v):
        return normalise_prose(v)


class GenerationBatch(BaseModel):
    examples: list[GeneratedExample] = Field(min_length=1, max_length=6)
