"""Tool-using research agent, Task 3A.

The reasoning loop is written out rather than taken from a prebuilt helper. The brief
requires a visible cycle of call, observe, then decide the next action from what was
observed, and an explicit loop is what makes that legible in a notebook.

Tool order is never hardcoded. The agent is handed the query and the toolset and picks
what to call, including whether to call anything at all.
"""

import json
import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from common import config
from common.llm import LLMError, SchemaValidationError, get_client
from task3_agentic.src import prompts, tools as toolkit, trace as tracing
from task3_agentic.src.memory import SessionMemory
from task3_agentic.src.schemas import ResearchReport

MAX_ITERATIONS = 10

# Groq's free tier allows 8000 tokens per minute and rejects any single request above
# it. Full tool payloads accumulate fast: eight calls at the old 3500-char limit pushed
# one request to 9139 tokens and the run died. Budget the context instead of hoping.
TOOL_OUTPUT_LIMIT = 1200
CONTEXT_CHAR_BUDGET = 20000
RECENT_TOOL_MESSAGES_KEPT_FULL = 3
CHARS_PER_TOKEN = 4

# Left to itself the model will keep searching until the ceiling stops it, then return
# nothing at all. Telling it what budget remains turns a hard cutoff into a decision it
# makes, which is the behaviour worth having.
BUDGET_WARNING_AT = 2
MAX_PROVIDER_ERRORS = 3


@dataclass
class Step:
    index: int
    thought: str
    tool_calls: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)


@dataclass
class AgentRun:
    query: str
    steps: list[Step] = field(default_factory=list)
    final_text: str = ""
    report: ResearchReport | None = None
    report_error: str | None = None
    memory: SessionMemory = field(default_factory=SessionMemory)
    provider_errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def tool_call_count(self) -> int:
        return sum(len(s.tool_calls) for s in self.steps)

    @property
    def tools_used(self) -> list[str]:
        seen = []
        for step in self.steps:
            for call in step.tool_calls:
                if call["name"] not in seen:
                    seen.append(call["name"])
        return seen


def _shorten(text: str, limit: int = TOOL_OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f'... [truncated, {len(text)} chars total]'


def _observation_gist(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw[:120]
    if not isinstance(payload, dict):
        return str(payload)[:120]
    if payload.get("ok") is False:
        return f"FAILED: {payload.get('error', 'unknown error')}"
    bits = []
    for key in ("current_price", "annualised_volatility_pct", "regime", "count",
                "score", "label", "classified"):
        if key in payload:
            bits.append(f"{key}={payload[key]}")
    if "momentum" in payload and isinstance(payload["momentum"], dict):
        bits.append(f"momentum={payload['momentum'].get('signal')}")
    return "ok, " + ", ".join(bits) if bits else "ok"


def estimate_tokens(messages) -> int:
    return sum(len(str(getattr(m, "content", ""))) for m in messages) // CHARS_PER_TOKEN


def compact_context(messages, budget: int = CONTEXT_CHAR_BUDGET,
                    keep_full: int = RECENT_TOOL_MESSAGES_KEPT_FULL):
    """Shrink the history by collapsing older tool payloads to their gist.

    The agent needs the substance of what it saw earlier, not the raw JSON. Recent
    results stay intact because those are what it is actively reasoning about.
    """
    total = sum(len(str(getattr(m, "content", ""))) for m in messages)
    if total <= budget:
        return messages

    tool_positions = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    protected = set(tool_positions[-keep_full:])

    out = []
    for i, message in enumerate(messages):
        if isinstance(message, ToolMessage) and i not in protected:
            gist = _observation_gist(str(message.content))
            out.append(ToolMessage(content=f"[earlier result] {gist}",
                                   tool_call_id=message.tool_call_id))
        else:
            out.append(message)
    return out


def _is_oversized(exc: Exception) -> bool:
    """Request exceeded the per-request size ceiling. Sending less actually helps."""
    text = str(exc).lower()
    return "too large" in text or "413" in text or "reduce your message size" in text


def _is_quota_exhausted(exc: Exception) -> bool:
    """Daily or per-minute allowance is spent.

    Compacting the payload does nothing here, which is the distinction that matters:
    an earlier version treated both cases the same and kept shrinking a request that
    was never too big, reporting "context too large" at 707 tokens.
    """
    text = str(exc).lower()
    if _is_oversized(exc):
        return False
    return "429" in text or "tokens per day" in text or "tpd" in text or (
        "rate limit" in text or "rate_limit" in text
    )


def _is_rate_limit(exc: Exception) -> bool:
    return _is_oversized(exc) or _is_quota_exhausted(exc)


class ResearchAgent:
    def __init__(self, name: str = "research_agent", tool_list=None,
                 system_prompt: str = prompts.SINGLE_AGENT_SYSTEM,
                 model: str | None = None, temperature: float = 0.2,
                 max_iterations: int = MAX_ITERATIONS, verbose: bool = True, llm=None):
        self.name = name
        self.tools = tool_list if tool_list is not None else toolkit.ALL_TOOLS
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.by_name = {t.name: t for t in self.tools}
        self.llm_model = model or config.GROQ_MODEL
        self.temperature = temperature
        self.used_fallback = False
        # llm is injectable so the loop can be tested without a key or a network call.
        self.llm = llm or ChatGroq(
            model=self.llm_model,
            api_key=config.GROQ_API_KEY,
            temperature=temperature,
            max_tokens=2048,
        ).bind_tools(self.tools)

    def _plain_llm(self):
        return ChatGroq(model=self.llm_model, api_key=config.GROQ_API_KEY,
                        temperature=self.temperature, max_tokens=2048)

    def _say(self, text: str) -> None:
        if self.verbose:
            print(text)

    def _fallback_llm(self, bind_tools: bool):
        """A second provider for the agent loop.

        common/llm.py already fails over for structured calls, but the reasoning loop
        talks to ChatGroq directly, so exhausting the Groq quota used to kill the agent
        outright while a perfectly good fallback sat unused.
        """
        if not config.OPENROUTER_API_KEY:
            return None
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=config.OPENROUTER_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            temperature=self.temperature,
            max_tokens=2048,
        )
        return llm.bind_tools(self.tools) if bind_tools else llm

    def _call_model(self, llm, messages, run=None, bind_tools: bool = True):
        """Invoke, then respond to failure according to what actually failed."""
        payload = compact_context(messages)
        try:
            return llm.invoke(payload)
        except Exception as exc:
            oversized = _is_oversized(exc)
            exhausted = _is_quota_exhausted(exc)
            if not (oversized or exhausted):
                raise
            if run is not None:
                run.provider_errors.append(f"{'oversized' if oversized else 'quota'}: {str(exc)[:160]}")

            if oversized:
                self._say(f"          [request too large at ~{estimate_tokens(payload)} "
                          f"tokens, compacting and retrying]")
                time.sleep(1)
                return llm.invoke(compact_context(messages, budget=0, keep_full=1))

            fallback = self._fallback_llm(bind_tools)
            if fallback is None:
                self._say("          [provider quota exhausted and no fallback configured]")
                raise
            self._say(f"          [Groq quota exhausted, switching to "
                      f"{config.OPENROUTER_MODEL}]")
            self.used_fallback = True
            return fallback.invoke(payload)

    def _invoke_tool(self, call: dict) -> str:
        name = call["name"]
        if name not in self.by_name:
            return json.dumps({"ok": False, "error": f"unknown tool {name!r}"})
        try:
            return self.by_name[name].invoke(call["args"])
        except Exception as exc:
            # A tool that raises must still come back as an observation, otherwise the
            # run ends where the brief requires the agent to try another route.
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def run(self, query: str, memory: SessionMemory | None = None) -> AgentRun:
        started = time.perf_counter()
        run = AgentRun(query=query, memory=memory or SessionMemory())
        if not run.memory.messages:
            run.memory.messages = [SystemMessage(self.system_prompt), HumanMessage(query)]
        else:
            run.memory.add_message(HumanMessage(query))

        self._say(f"\n=== {self.name}: {query[:90]} ===")

        with tracing.acting_as(self.name):
            for index in range(1, self.max_iterations + 1):
                remaining = self.max_iterations - index
                if 0 < remaining <= BUDGET_WARNING_AT:
                    run.memory.add_message(HumanMessage(
                        f"You have {remaining} tool call(s) left in this session. If the "
                        f"evidence you already hold is enough, stop calling tools and "
                        f"write your findings now."
                    ))

                try:
                    response: AIMessage = self._call_model(self.llm, run.memory.messages, run)
                except Exception as exc:
                    # The provider itself can reject a turn, most often when the model
                    # emits tool arguments that are not valid JSON. That is a recoverable
                    # planning error, not a reason to abandon the research task.
                    detail = str(exc)[:300]
                    self._say(f"\n[step {index}] provider rejected the turn: {detail}")
                    run.provider_errors.append(detail)
                    if len(run.provider_errors) > MAX_PROVIDER_ERRORS:
                        self._say("[stopped] too many provider errors, concluding early")
                        break
                    run.memory.add_message(HumanMessage(
                        "Your previous turn was rejected because the tool arguments were "
                        "not valid JSON. Keep arguments short and simple: pass a ticker "
                        "symbol rather than pasting long text. Try a different tool or a "
                        "smaller argument."
                    ))
                    continue

                run.memory.add_message(response)

                thought = (response.content or "").strip()
                step = Step(index=index, thought=thought)

                if not response.tool_calls:
                    run.final_text = thought
                    run.steps.append(step)
                    self._say(f"\n[step {index}] no further tools needed, drafting answer")
                    break

                self._say(f"\n[step {index}] decided to call "
                          f"{', '.join(c['name'] for c in response.tool_calls)}")
                if thought:
                    self._say(f"          reasoning: {thought[:200]}")

                for call in response.tool_calls:
                    raw = self._invoke_tool(call)
                    gist = _observation_gist(raw)
                    step.tool_calls.append({"name": call["name"], "args": call["args"]})
                    step.observations.append({"tool": call["name"], "gist": gist})
                    run.memory.remember(call["name"], raw)

                    self._say(f"          -> {call['name']}({json.dumps(call['args'])})")
                    self._say(f"             observed: {gist}")

                    run.memory.add_message(
                        ToolMessage(content=_shorten(raw), tool_call_id=call["id"])
                    )
                run.steps.append(step)
            else:
                # Ceiling reached mid-investigation. Ask once more without tools bound so
                # the session still yields a conclusion rather than an empty draft.
                self._say(f"\n[stopped] hit the {self.max_iterations} iteration ceiling, "
                          f"forcing a conclusion from what was gathered")
                run.memory.add_message(HumanMessage(
                    "Tool budget is spent. Write your findings now using only what you "
                    "have already gathered, and name anything you could not verify."
                ))
                try:
                    closing = self._call_model(self._plain_llm(), run.memory.messages, run,
                                               bind_tools=False)
                    run.memory.add_message(closing)
                    run.final_text = (closing.content or "").strip()
                except Exception as exc:
                    run.provider_errors.append(str(exc)[:300])
                    self._say(f"[warn] closing synthesis failed: {exc}")

        run.elapsed_s = time.perf_counter() - started
        return run

    def ask(self, run: AgentRun, question: str) -> tuple[str, int]:
        """Follow-up inside the same session.

        Returns the answer and how many new tool calls it required. Answering a question
        the session has already gathered the data for should cost zero.
        """
        run.memory.add_message(HumanMessage(question))
        self._say(f"\n=== follow-up: {question} ===")
        with tracing.acting_as(f"{self.name}:followup"):
            try:
                response: AIMessage = self._call_model(self.llm, run.memory.messages, run)
            except Exception as exc:
                self._say(f"follow-up failed: {exc}")
                run.provider_errors.append(str(exc)[:300])
                return "", -1
            run.memory.add_message(response)
            new_calls = len(response.tool_calls or [])
            for call in response.tool_calls or []:
                raw = self._invoke_tool(call)
                run.memory.add_message(ToolMessage(content=_shorten(raw), tool_call_id=call["id"]))
            if response.tool_calls:
                response = self._call_model(self.llm, run.memory.messages, run)
                run.memory.add_message(response)
        answer = (response.content or "").strip()
        self._say(f"answer ({new_calls} new tool calls): {answer[:300]}")
        return answer, new_calls

    def synthesise(self, run: AgentRun, ticker: str) -> ResearchReport | None:
        """Turn the gathered observations into the validated three-section report."""
        gathered = {
            name: _shorten(raw, 900) for name, raw in run.memory.observations.items()
        }
        user = prompts.REPORT_USER.substitute(
            ticker=ticker,
            draft=run.final_text or "(the agent produced no free-text draft)",
            observations=json.dumps(gathered, indent=2)[:5000],
        )
        try:
            report = get_client().structured(prompts.REPORT_SYSTEM, user, ResearchReport)
        except (SchemaValidationError, LLMError) as exc:
            run.report_error = str(exc)
            self._say(f"[report] synthesis failed: {exc}")
            return None
        run.report = report
        return report
