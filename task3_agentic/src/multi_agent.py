"""Two-agent research pipeline, Task 3B.

Agent A owns the numbers, Agent B owns the narrative, and neither can do the other's
job. The restriction is enforced by which tool objects each agent is constructed with,
not by asking the model nicely in a prompt: Agent B has no reference to a price tool, so
there is no path by which it can call one.

Handoff is a validated Pydantic model. Agent B may raise exactly one clarification back
to Agent A, which A answers from its own data before B writes the report.
"""

import json
import time
from dataclasses import dataclass, field

from common.llm import LLMError, SchemaValidationError, get_client
from task3_agentic.src import prompts, tools as toolkit, trace as tracing
from task3_agentic.src.agent import ResearchAgent, _shorten
from task3_agentic.src.schemas import (
    BriefNarrative, ClarificationRequest, ClarificationResponse, DataBrief,
    PriceSnapshot, ResearchReport, SentimentSummary, VolatilityProfile,
)

AGENT_A = "agent_a_data_analyst"
AGENT_B = "agent_b_research_writer"


@dataclass
class Handoff:
    sender: str
    recipient: str
    kind: str
    payload: dict
    at: float = field(default_factory=time.time)


@dataclass
class PipelineResult:
    ticker: str
    brief: DataBrief | None = None
    clarification: ClarificationRequest | None = None
    clarification_answer: ClarificationResponse | None = None
    report: ResearchReport | None = None
    transcript: list[Handoff] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def render_transcript(self) -> str:
        lines = []
        for i, h in enumerate(self.transcript, 1):
            lines.append(f"[{i}] {h.sender}  ->  {h.recipient}   ({h.kind})")
            body = json.dumps(h.payload, indent=2, default=str)
            for line in body.splitlines()[:14]:
                lines.append("      " + line)
            if len(body.splitlines()) > 14:
                lines.append("      ...")
            lines.append("")
        return "\n".join(lines)


class TwoAgentPipeline:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.agent_a = ResearchAgent(
            name=AGENT_A,
            tool_list=toolkit.QUANT_TOOLS,
            system_prompt=prompts.AGENT_A_SYSTEM,
            verbose=verbose,
        )
        self.agent_b = ResearchAgent(
            name=AGENT_B,
            tool_list=toolkit.QUALITATIVE_TOOLS,
            system_prompt=prompts.AGENT_B_SYSTEM,
            verbose=verbose,
        )

    def _say(self, text: str) -> None:
        if self.verbose:
            print(text)

    def tool_access(self) -> dict[str, list[str]]:
        return {
            AGENT_A: sorted(self.agent_a.by_name),
            AGENT_B: sorted(self.agent_b.by_name),
        }

    @staticmethod
    def _parse_observations(observations: dict) -> tuple[dict, list[str]]:
        """Pull the figures out of raw tool JSON in Python, not through the model."""
        parsed, failures = {}, []
        for name, raw in observations.items():
            try:
                payload = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                failures.append(f"{name}: output was not valid JSON")
                continue
            if not isinstance(payload, dict) or payload.get("ok") is False:
                reason = payload.get("error", "unknown") if isinstance(payload, dict) else "unknown"
                failures.append(f"{name}: {reason}")
                continue
            parsed[name] = payload
        return parsed, failures

    @staticmethod
    def _derive_observations(price, volatility, sentiment) -> list[str]:
        """Factual observations computed from the figures.

        Used when the model returns none. The handoff should never reach Agent B empty
        just because a narrative call came back thin, and these statements are checkable
        against the numbers rather than generated prose.
        """
        notes = []
        if price.current_price and price.sma_50 and price.sma_200:
            side = "above" if price.current_price > price.sma_50 else "below"
            cross = "above" if price.sma_50 > price.sma_200 else "below"
            notes.append(
                f"Price sits {side} the 50-day SMA, and the 50-day sits {cross} the "
                f"200-day, a {'golden cross' if cross == 'above' else 'death cross'} structure."
            )
        if price.rsi_14 is not None:
            zone = ("overbought" if price.rsi_14 >= 70 else
                    "oversold" if price.rsi_14 <= 30 else "neutral")
            notes.append(f"RSI(14) at {price.rsi_14:.1f} is {zone}.")
        if price.macd_hist is not None:
            drift = "negative" if price.macd_hist < 0 else "positive"
            notes.append(
                f"The MACD histogram is {drift} at {price.macd_hist:.2f}, so short-term "
                f"momentum is {'fading against' if price.macd_hist < 0 else 'confirming'} the trend."
            )
        if volatility and volatility.annualised_volatility_1y_pct:
            ratio = volatility.annualised_volatility_pct / volatility.annualised_volatility_1y_pct
            notes.append(
                f"{volatility.window_days}-day annualised volatility of "
                f"{volatility.annualised_volatility_pct:.2f}% is {ratio:.2f}x the one-year "
                f"figure of {volatility.annualised_volatility_1y_pct:.2f}%, regime "
                f"{volatility.regime}."
            )
        if sentiment:
            notes.append(
                f"Headline sentiment is {sentiment.label} at {sentiment.score:+.3f} across "
                f"{sentiment.classified} classified items "
                f"({sentiment.positive}/{sentiment.negative}/{sentiment.neutral} pos/neg/neutral)."
            )
        if price.current_price and price.week52_high and price.week52_low:
            span = price.week52_high - price.week52_low
            if span:
                pos = (price.current_price - price.week52_low) / span * 100
                notes.append(f"Price sits at {pos:.0f}% of the 52-week range.")
        return notes[:6]

    def _compile_brief(self, ticker: str, run, result: "PipelineResult") -> DataBrief:
        """Assemble the handoff.

        Figures are read straight out of the tool payloads. An earlier version asked the
        model to transcribe them from truncated JSON and it returned an entirely null
        brief, which validated cleanly and told Agent B nothing.
        """
        parsed, failures = self._parse_observations(run.memory.observations)

        price_payload = parsed.get("get_price_data", {})
        indicators = price_payload.get("indicators") or {}
        momentum = price_payload.get("momentum") or {}
        price = PriceSnapshot(
            current_price=price_payload.get("current_price"),
            week52_high=price_payload.get("week52_high"),
            week52_low=price_payload.get("week52_low"),
            pe_ratio=price_payload.get("pe_ratio"),
            ytd_return_pct=price_payload.get("ytd_return_pct"),
            sma_50=indicators.get("sma_50"),
            sma_200=indicators.get("sma_200"),
            rsi_14=indicators.get("rsi_14"),
            macd_hist=indicators.get("macd_hist"),
            momentum_signal=momentum.get("signal", "Unknown"),
            momentum_flags=momentum.get("flags") or [],
        )

        vol_payload = parsed.get("calculate_volatility")
        volatility = None
        if vol_payload:
            volatility = VolatilityProfile(
                window_days=vol_payload.get("window_days", 30),
                annualised_volatility_pct=vol_payload.get("annualised_volatility_pct", 0.0),
                annualised_volatility_1y_pct=vol_payload.get("annualised_volatility_1y_pct"),
                mean_abs_daily_move_pct=vol_payload.get("mean_abs_daily_move_pct"),
                regime=vol_payload.get("regime", "unknown"),
            )

        sent_payload = parsed.get("llm_sentiment_tool") or parsed.get("llm_sentiment")
        sentiment = None
        if sent_payload:
            sentiment = SentimentSummary(
                score=sent_payload.get("score", 0.0),
                label=sent_payload.get("label", "neutral"),
                positive=sent_payload.get("positive", 0),
                negative=sent_payload.get("negative", 0),
                neutral=sent_payload.get("neutral", 0),
                mean_confidence=sent_payload.get("mean_confidence"),
                classified=sent_payload.get("classified", 0),
            )

        figures = {
            "price": price.model_dump(),
            "volatility": volatility.model_dump() if volatility else None,
            "sentiment": sentiment.model_dump() if sentiment else None,
        }
        narrative = BriefNarrative()
        try:
            with tracing.acting_as(AGENT_A):
                narrative = get_client().structured(
                    prompts.AGENT_A_BRIEF_SYSTEM,
                    prompts.AGENT_A_BRIEF_USER.substitute(
                        ticker=ticker,
                        figures=json.dumps(figures, indent=2)[:3500],
                        failures=chr(10).join(failures) or "none",
                    ),
                    BriefNarrative,
                )
        except (SchemaValidationError, LLMError) as exc:
            # The figures are already correct, so a failed narrative degrades the brief
            # rather than voiding it.
            result.errors.append(f"brief narrative failed: {exc}")

        observations = narrative.quant_observations or self._derive_observations(
            price, volatility, sentiment
        )
        gaps = list(narrative.data_gaps) + failures
        if price.current_price is None:
            gaps.append("current price unavailable")
        if volatility is None:
            gaps.append("volatility not measured")
        if sentiment is None:
            gaps.append("news sentiment not measured")

        return DataBrief(
            ticker=ticker,
            company_name=price_payload.get("company_name", "") or ticker,
            as_of=price_payload.get("as_of", ""),
            price=price,
            volatility=volatility,
            sentiment=sentiment,
            quant_observations=observations,
            data_gaps=gaps,
        )

    def run(self, ticker: str) -> PipelineResult:
        started = time.perf_counter()
        result = PipelineResult(ticker=ticker.upper())

        self._say("=" * 78)
        self._say(f"TWO-AGENT PIPELINE  |  {result.ticker}")
        self._say(f"  {AGENT_A}: {sorted(self.agent_a.by_name)}")
        self._say(f"  {AGENT_B}: {sorted(self.agent_b.by_name)}")
        self._say("=" * 78)

        # --- Stage 1, Agent A gathers and compiles -------------------------------
        self._say(f"\n----- STAGE 1: {AGENT_A} gathers quantitative evidence -----")
        run_a = self.agent_a.run(
            f"Gather the quantitative picture for {result.ticker}: price and technical "
            f"indicators, volatility, and news sentiment. Report what the numbers show."
        )
        brief = self._compile_brief(result.ticker, run_a, result)
        if brief is None:
            result.errors.append("Agent A could not compile a data brief")
            result.elapsed_s = time.perf_counter() - started
            return result

        result.brief = brief
        result.transcript.append(
            Handoff(AGENT_A, AGENT_B, "data_brief", brief.model_dump())
        )
        self._say(f"\n>>> HANDOFF {AGENT_A} -> {AGENT_B}: DataBrief "
                  f"({len(brief.quant_observations)} observations, "
                  f"{len(brief.data_gaps)} gaps)")

        # --- Stage 2, critique loop ---------------------------------------------
        self._say(f"\n----- STAGE 2: critique loop -----")
        try:
            with tracing.acting_as(AGENT_B):
                request = get_client().structured(
                    prompts.AGENT_B_CLARIFY_SYSTEM,
                    prompts.AGENT_B_CLARIFY_USER.substitute(
                        brief=brief.model_dump_json(indent=2)[:3500]
                    ),
                    ClarificationRequest,
                )
            result.clarification = request
            result.transcript.append(
                Handoff(AGENT_B, AGENT_A, "clarification_request", request.model_dump())
            )
            self._say(f">>> {AGENT_B} asks: {request.question}")

            observations = {k: _shorten(v, 700) for k, v in run_a.memory.observations.items()}
            with tracing.acting_as(AGENT_A):
                answer = get_client().structured(
                    prompts.AGENT_A_CLARIFY_SYSTEM,
                    prompts.AGENT_A_CLARIFY_USER.substitute(
                        brief=brief.model_dump_json(indent=2)[:2500],
                        observations=json.dumps(observations, indent=2)[:3000],
                        question=request.question,
                    ),
                    ClarificationResponse,
                )
            result.clarification_answer = answer
            result.transcript.append(
                Handoff(AGENT_A, AGENT_B, "clarification_response", answer.model_dump())
            )
            self._say(f">>> {AGENT_A} answers: {answer.answer[:220]}")
        except (SchemaValidationError, LLMError) as exc:
            # The critique loop is an enhancement, not a precondition for the report.
            result.errors.append(f"critique loop failed: {exc}")
            self._say(f"[warn] critique loop failed, continuing: {exc}")

        # --- Stage 3, Agent B gathers qualitative context ------------------------
        self._say(f"\n----- STAGE 3: {AGENT_B} gathers qualitative context -----")
        company = brief.company_name or result.ticker
        run_b = self.agent_b.run(
            f"You are writing a 90-day risk assessment for {company} ({result.ticker}). "
            f"Agent A's data shows: {'; '.join(brief.quant_observations[:4]) or 'see brief'}. "
            f"Search for analyst commentary, competitive and regulatory developments that "
            f"would explain or challenge that picture."
        )

        # --- Stage 4, Agent B writes the report ----------------------------------
        self._say(f"\n----- STAGE 4: {AGENT_B} writes the final report -----")
        clarification_text = "none was exchanged"
        if result.clarification and result.clarification_answer:
            clarification_text = (
                f"Q: {result.clarification.question}\n"
                f"A: {result.clarification_answer.answer}"
            )
        context = {k: _shorten(v, 900) for k, v in run_b.memory.observations.items()}
        try:
            with tracing.acting_as(AGENT_B):
                report = get_client().structured(
                    prompts.AGENT_B_REPORT_SYSTEM,
                    prompts.AGENT_B_REPORT_USER.substitute(
                        ticker=result.ticker,
                        brief=brief.model_dump_json(indent=2)[:3500],
                        clarification=clarification_text,
                        context=json.dumps(context, indent=2)[:4000],
                    ),
                    ResearchReport,
                )
            result.report = report
            result.transcript.append(
                Handoff(AGENT_B, "output", "research_report", report.model_dump())
            )
            self._say(f">>> report complete: {len(report.top_risks)} risks, "
                      f"hedge = {report.hedge_strategy.strategy[:70]}")
        except (SchemaValidationError, LLMError) as exc:
            result.errors.append(f"report generation failed: {exc}")
            self._say(f"[error] report generation failed: {exc}")

        result.elapsed_s = time.perf_counter() - started
        return result
