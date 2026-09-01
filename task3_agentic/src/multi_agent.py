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
    ClarificationRequest, ClarificationResponse, DataBrief, ResearchReport,
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

    def _compile_brief(self, ticker: str, run, result: "PipelineResult") -> DataBrief | None:
        observations = {k: _shorten(v, 900) for k, v in run.memory.observations.items()}
        user = prompts.AGENT_A_BRIEF_USER.substitute(
            ticker=ticker, observations=json.dumps(observations, indent=2)[:3000]
        )
        try:
            with tracing.acting_as(AGENT_A):
                return get_client().structured(
                    prompts.AGENT_A_BRIEF_SYSTEM, user, DataBrief
                )
        except (SchemaValidationError, LLMError) as exc:
            result.errors.append(f"brief compilation failed: {exc}")
            return None

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
