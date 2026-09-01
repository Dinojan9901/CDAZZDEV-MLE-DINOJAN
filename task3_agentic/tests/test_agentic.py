"""Offline tests for Task 3. No network, no API key, no LLM.

Run: python -m task3_agentic.tests.test_agentic
"""

import json
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage

from task3_agentic.src import multi_agent, tools as toolkit, trace as tracing
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from task3_agentic.src.agent import (
    ResearchAgent, _is_oversized, _is_quota_exhausted, _is_rate_limit,
    _observation_gist, compact_context, estimate_tokens,
)
from common.llm import LLMError
from task3_agentic.src.memory import BriefCache, SessionMemory
from task3_agentic.src.schemas import DataBrief, ResearchReport

# Referenced by codepoint so the literal character never appears in the source.
EM_DASH = chr(0x2014)


class FakeLLM:
    """Scripts AIMessage turns so the reasoning loop runs with no provider."""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def invoke(self, messages):
        self.seen.append(list(messages))
        if not self.script:
            return AIMessage(content="Out of script, concluding.")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return AIMessage(content=item)
        return AIMessage(content=item.get("content", ""), tool_calls=item.get("tool_calls", []))


def _call(name, args, cid="c1"):
    return {"name": name, "args": args, "id": cid, "type": "tool_call"}


# --- trace ------------------------------------------------------------------

def test_trace_records_inputs_output_and_duration():
    with tempfile.TemporaryDirectory() as tmp:
        tr = tracing.ToolTracer(Path(tmp) / "t.jsonl")
        with tr.span("get_price_data", {"ticker": "NVDA"}) as box:
            box["output"] = {"ok": True, "current_price": 220.0}
        rows = tr.read()
        assert len(rows) == 1
        row = rows[0]
        for field in ("tool", "inputs", "output", "duration_ms", "agent", "status"):
            assert field in row, f"trace row missing {field}"
        assert row["tool"] == "get_price_data"
        assert row["inputs"] == {"ticker": "NVDA"}
        assert row["duration_ms"] >= 0
        assert row["status"] == "ok"
    return 8


def test_trace_truncates_output_to_200_chars():
    with tempfile.TemporaryDirectory() as tmp:
        tr = tracing.ToolTracer(Path(tmp) / "t.jsonl")
        with tr.span("web_search", {"query": "x"}) as box:
            box["output"] = {"ok": True, "blob": "y" * 5000}
        row = tr.read()[0]
        assert len(row["output"]) == tracing.OUTPUT_TRUNCATE == 200
        assert row["output_truncated"] is True
        assert row["output_chars"] > 200, "the full length must still be recorded"
    return 4


def test_handled_failure_is_logged_as_error_not_ok():
    with tempfile.TemporaryDirectory() as tmp:
        tr = tracing.ToolTracer(Path(tmp) / "t.jsonl")
        with tr.span("get_price_data", {"ticker": "BAD"}) as box:
            box["output"] = {"ok": False, "error": "delisted"}
        row = tr.read()[0]
        # A tool that returns a handled failure must not read as a clean call.
        assert row["status"] == "error", "ok=False payload must mark the row as an error"
        assert "delisted" in row["error"]
    return 2


def test_trace_records_raised_exceptions_then_reraises():
    with tempfile.TemporaryDirectory() as tmp:
        tr = tracing.ToolTracer(Path(tmp) / "t.jsonl")
        try:
            with tr.span("boom", {"a": 1}):
                raise ValueError("kaboom")
        except ValueError:
            pass
        else:
            raise AssertionError("span must re-raise")
        row = tr.read()[0]
        assert row["status"] == "error" and "kaboom" in row["error"]
    return 2


def test_trace_attributes_calls_to_the_acting_agent():
    with tempfile.TemporaryDirectory() as tmp:
        tr = tracing.ToolTracer(Path(tmp) / "t.jsonl")
        with tracing.acting_as("agent_a"):
            with tr.span("get_price_data", {}) as box:
                box["output"] = {"ok": True}
        with tracing.acting_as("agent_b"):
            with tr.span("web_search", {}) as box:
                box["output"] = {"ok": True}
        agents = [r["agent"] for r in tr.read()]
        assert agents == ["agent_a", "agent_b"]
        assert tracing.current_agent() == "root", "context must be restored"
    return 3


def test_trace_is_jsonl_and_summarises():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.jsonl"
        tr = tracing.ToolTracer(path)
        for i in range(3):
            with tr.span("get_news", {"n": i}) as box:
                box["output"] = {"ok": i > 0}
        for line in path.read_text(encoding="utf-8").splitlines():
            json.loads(line)
        s = tr.summary()
        assert s["total_calls"] == 3 and s["errors"] == 1
        assert s["by_tool"]["get_news"]["calls"] == 3
    return 4


# --- memory -----------------------------------------------------------------

def test_session_memory_recall():
    m = SessionMemory()
    assert not m.has("get_price_data")
    m.remember("get_price_data", {"ok": True, "current_price": 220})
    assert m.has("get_price_data")
    assert m.recall("get_price_data")["current_price"] == 220
    assert m.recall("missing") is None
    return 4


def test_cache_round_trip_and_date_keying():
    with tempfile.TemporaryDirectory() as tmp:
        cache = BriefCache(Path(tmp))
        assert cache.load("NVDA") is None, "cold cache must miss"

        cache.save("NVDA", {"report": {"ticker": "NVDA"}})
        hit = cache.load("NVDA")
        assert hit is not None and hit["ticker"] == "NVDA"
        assert "cached_on" in hit

        # Yesterday's brief is stale by definition and must not satisfy today.
        assert cache.load("NVDA", "2020-01-01") is None
        assert cache.load("MSFT") is None, "another ticker must not hit"
        assert len(cache.list_cached()) == 1
    return 6


def test_cache_rejects_foreign_and_corrupt_files():
    with tempfile.TemporaryDirectory() as tmp:
        cache = BriefCache(Path(tmp))
        cache.path_for("NVDA").write_text('{"cache_version": 999}', encoding="utf-8")
        assert cache.load("NVDA") is None, "a newer layout must not be trusted"
        cache.path_for("MSFT").write_text("{not json", encoding="utf-8")
        assert cache.load("MSFT") is None, "corrupt cache must miss, not raise"
    return 2


# --- tool restriction -------------------------------------------------------

def test_agents_have_disjoint_enforced_tool_access():
    pipeline = multi_agent.TwoAgentPipeline.__new__(multi_agent.TwoAgentPipeline)
    a_names = {t.name for t in toolkit.QUANT_TOOLS}
    b_names = {t.name for t in toolkit.QUALITATIVE_TOOLS}

    assert "web_search_tool" not in a_names, "Agent A must not reach web search"
    assert "get_price_data" not in b_names, "Agent B must not reach price data"
    assert "calculate_volatility" not in b_names
    assert not (a_names & b_names), "tool sets must be disjoint"
    assert a_names | b_names == {t.name for t in toolkit.ALL_TOOLS}
    return 5


def test_restriction_holds_even_if_the_model_asks_for_a_forbidden_tool():
    agent_b = ResearchAgent(
        name="agent_b", tool_list=toolkit.QUALITATIVE_TOOLS,
        verbose=False, llm=FakeLLM([]),
    )
    # Enforcement is structural: the tool is simply not reachable by name.
    out = json.loads(agent_b._invoke_tool(_call("get_price_data", {"ticker": "NVDA"})))
    assert out["ok"] is False
    assert "unknown tool" in out["error"]
    return 2


# --- agent loop -------------------------------------------------------------

def test_agent_selects_tools_autonomously_and_stops():
    llm = FakeLLM([
        {"tool_calls": [_call("get_price_data", {"ticker": "NVDA"}, "a")]},
        {"tool_calls": [_call("calculate_volatility", {"ticker": "NVDA"}, "b")]},
        "Volatility is elevated against a bullish trend, so I have what I need.",
    ])
    agent = ResearchAgent(verbose=False, llm=llm)
    agent.by_name = {
        "get_price_data": _StubTool("get_price_data", {"ok": True, "current_price": 220}),
        "calculate_volatility": _StubTool("calculate_volatility",
                                          {"ok": True, "annualised_volatility_pct": 44.7}),
    }
    run = agent.run("Analyse NVDA")

    assert run.tools_used == ["get_price_data", "calculate_volatility"]
    assert run.tool_call_count == 2
    assert "elevated" in run.final_text
    assert len(run.steps) == 3, "two tool steps plus the concluding step"
    return 4


def test_agent_routes_around_a_failing_tool():
    llm = FakeLLM([
        {"tool_calls": [_call("get_price_data", {"ticker": "BAD"}, "a")]},
        {"tool_calls": [_call("get_news", {"ticker": "BAD"}, "b")]},
        "Price data was unavailable, so the assessment rests on news alone.",
    ])
    agent = ResearchAgent(verbose=False, llm=llm)
    agent.by_name = {
        "get_price_data": _StubTool("get_price_data", {"ok": False, "error": "delisted"}),
        "get_news": _StubTool("get_news", {"ok": True, "count": 5}),
    }
    run = agent.run("Analyse BAD")

    assert run.tool_call_count == 2, "a failed tool must not end the run"
    assert "FAILED" in run.steps[0].observations[0]["gist"]
    assert run.final_text, "the agent must still conclude"
    return 3


def test_agent_survives_a_raising_tool():
    llm = FakeLLM([
        {"tool_calls": [_call("get_price_data", {"ticker": "X"}, "a")]},
        "Concluded despite the exception.",
    ])
    agent = ResearchAgent(verbose=False, llm=llm)
    agent.by_name = {"get_price_data": _RaisingTool("get_price_data")}
    run = agent.run("Analyse X")
    assert "FAILED" in run.steps[0].observations[0]["gist"]
    assert run.final_text == "Concluded despite the exception."
    return 2


def test_agent_recovers_from_a_provider_rejection():
    llm = FakeLLM([
        RuntimeError("400 tool_use_failed: Failed to parse tool call arguments as JSON"),
        {"tool_calls": [_call("get_price_data", {"ticker": "NVDA"}, "a")]},
        "Recovered and concluded.",
    ])
    agent = ResearchAgent(verbose=False, llm=llm)
    agent.by_name = {"get_price_data": _StubTool("get_price_data", {"ok": True})}
    run = agent.run("Analyse NVDA")

    assert len(run.provider_errors) == 1
    assert run.tool_call_count == 1, "the run must continue past a rejected turn"
    assert run.final_text == "Recovered and concluded."
    return 3


def test_agent_stops_after_repeated_provider_failures():
    llm = FakeLLM([RuntimeError("400 bad request")] * 8)
    agent = ResearchAgent(verbose=False, llm=llm, max_iterations=8)
    run = agent.run("Analyse NVDA")
    assert len(run.provider_errors) <= 4, "must give up rather than burn every iteration"
    return 1


def test_observation_gist_flags_failures():
    assert "FAILED" in _observation_gist(json.dumps({"ok": False, "error": "delisted"}))
    assert "current_price=220" in _observation_gist(json.dumps({"ok": True, "current_price": 220}))
    assert _observation_gist("not json at all").startswith("not json")
    return 3


# --- schemas ----------------------------------------------------------------

def test_report_requires_exactly_three_risks():
    import pydantic

    base = {
        "ticker": "NVDA",
        "financial_health_summary": "A" * 60,
        "hedge_strategy": {"strategy": "Buy protective puts",
                           "rationale": "Volatility is elevated versus the one-year mean."},
    }
    risk = {"risk": "Concentration risk", "evidence": "Top customers dominate revenue",
            "severity": "high"}
    for count in (0, 2, 4):
        try:
            ResearchReport(**base, top_risks=[risk] * count)
            raise AssertionError(f"{count} risks should not validate")
        except pydantic.ValidationError:
            pass
    ok = ResearchReport(**base, top_risks=[risk] * 3)
    assert len(ok.top_risks) == 3
    return 4


def test_report_markdown_has_the_three_required_sections():
    report = ResearchReport(
        ticker="NVDA",
        financial_health_summary="B" * 60,
        top_risks=[{"risk": f"Risk {i}", "evidence": f"Evidence {i}", "severity": "medium"}
                   for i in range(3)],
        hedge_strategy={"strategy": "Collar", "rationale": "Caps downside cheaply.",
                        "instruments": ["puts", "calls"]},
    )
    md = report.to_markdown()
    for heading in ["## Financial Health Summary", "## Top Three Risks",
                    "## Hedge Strategy Recommendation"]:
        assert heading in md, f"missing {heading}"
    return 3


def test_handoff_schema_normalises_prose():
    brief = DataBrief(
        ticker="NVDA",
        quant_observations=[f"30-day vol is elevated {EM_DASH} well above the 1-year mean"],
        data_gaps=["P/E unavailable"],
    )
    joined = " ".join(brief.quant_observations)
    assert EM_DASH not in joined, "typography must be folded in the handoff too"
    assert brief.quant_observations and brief.data_gaps
    return 3


# --- context budgeting ------------------------------------------------------

def _history(n_tools: int, payload_chars: int):
    msgs = [SystemMessage("system prompt"), HumanMessage("analyse NVDA")]
    for i in range(n_tools):
        msgs.append(AIMessage(content="", tool_calls=[_call("get_news", {"n": i}, f"id{i}")]))
        msgs.append(ToolMessage(content=json.dumps({"ok": True, "count": i, "pad": "z" * payload_chars}),
                                tool_call_id=f"id{i}"))
    return msgs


def test_small_history_is_left_untouched():
    msgs = _history(2, 50)
    assert compact_context(msgs) is msgs, "no compaction below the budget"
    return 1


def test_large_history_collapses_older_tool_payloads():
    msgs = _history(8, 4000)
    before = estimate_tokens(msgs)
    out = compact_context(msgs)
    after = estimate_tokens(out)

    assert after < before / 2, f"expected a large reduction, {before} -> {after}"
    tool_msgs = [m for m in out if isinstance(m, ToolMessage)]
    collapsed = [m for m in tool_msgs if str(m.content).startswith("[earlier result]")]
    assert len(collapsed) == 5, f"expected 5 collapsed, got {len(collapsed)}"
    assert len(tool_msgs) - len(collapsed) == 3, "the 3 most recent must stay intact"
    return 4


def test_compaction_preserves_tool_call_id_pairing():
    msgs = _history(6, 4000)
    out = compact_context(msgs)
    # Every tool_call must keep a matching response or the provider rejects the turn.
    call_ids = [c["id"] for m in out if isinstance(m, AIMessage) for c in (m.tool_calls or [])]
    response_ids = [m.tool_call_id for m in out if isinstance(m, ToolMessage)]
    assert call_ids == response_ids, "compaction must not orphan a tool call"
    assert len(out) == len(msgs), "no message may be dropped, only shrunk"
    return 2


def test_collapsed_payload_keeps_the_substance():
    msgs = _history(6, 4000)
    out = compact_context(msgs)
    collapsed = [str(m.content) for m in out
                 if isinstance(m, ToolMessage) and str(m.content).startswith("[earlier")]
    assert any("count=" in c for c in collapsed), "the gist must survive compaction"
    return 1


def test_oversized_and_quota_are_told_apart():
    oversized = RuntimeError(
        "413 - Request too large for model, please reduce your message size")
    quota = RuntimeError(
        "429 - Rate limit reached, tokens per day (TPD): Limit 200000, Used 198947")

    # These need different responses. Compacting a request that was never too big just
    # reported "context too large" at 707 tokens and then failed again.
    assert _is_oversized(oversized) and not _is_quota_exhausted(oversized)
    assert _is_quota_exhausted(quota) and not _is_oversized(quota)
    assert _is_rate_limit(oversized) and _is_rate_limit(quota)
    assert not _is_rate_limit(RuntimeError("connection reset"))
    return 6


def test_quota_exhaustion_switches_provider_rather_than_compacting():
    class Exhausted:
        def invoke(self, messages):
            raise RuntimeError("429 - Rate limit reached, tokens per day (TPD): Limit 200000")

    class Fallback:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return AIMessage(content="answered by the fallback provider")

    fallback = Fallback()
    agent = ResearchAgent(verbose=False, llm=FakeLLM([]))
    agent._fallback_llm = lambda bind_tools: fallback

    out = agent._call_model(Exhausted(), _history(2, 50))
    assert out.content == "answered by the fallback provider"
    assert fallback.calls == 1
    assert agent.used_fallback is True
    return 3


def test_quota_exhaustion_reraises_when_no_fallback_exists():
    class Exhausted:
        def invoke(self, messages):
            raise RuntimeError("429 - tokens per day (TPD) limit reached")

    agent = ResearchAgent(verbose=False, llm=FakeLLM([]))
    agent._fallback_llm = lambda bind_tools: None
    try:
        agent._call_model(Exhausted(), _history(2, 50))
        raise AssertionError("must not swallow the failure when nothing can serve it")
    except RuntimeError:
        pass
    return 1


def test_call_model_recovers_from_a_rate_limit_by_sending_less():
    class RateLimitedOnce:
        def __init__(self):
            self.payload_sizes = []

        def invoke(self, messages):
            size = sum(len(str(m.content)) for m in messages)
            self.payload_sizes.append(size)
            if len(self.payload_sizes) == 1:
                raise RuntimeError("Error code: 413 - rate_limit_exceeded, request too large")
            return AIMessage(content="ok on the smaller payload")

    llm = RateLimitedOnce()
    agent = ResearchAgent(verbose=False, llm=FakeLLM([]))
    out = agent._call_model(llm, _history(8, 4000))

    assert out.content == "ok on the smaller payload"
    assert len(llm.payload_sizes) == 2
    assert llm.payload_sizes[1] < llm.payload_sizes[0], "the retry must send strictly less"
    return 3


# --- deterministic handoff --------------------------------------------------

PRICE_JSON = json.dumps({
    "ok": True, "ticker": "NVDA", "company_name": "NVIDIA Corporation",
    "as_of": "2026-09-02", "current_price": 216.7, "week52_high": 235.47,
    "week52_low": 164.98, "pe_ratio": 27.6, "ytd_return_pct": 17.05,
    "indicators": {"sma_50": 208.6, "sma_200": 195.8, "rsi_14": 54.6, "macd_hist": -0.28},
    "momentum": {"signal": "Bullish", "flags": ["golden_cross"]},
})
VOL_JSON = json.dumps({
    "ok": True, "window_days": 30, "annualised_volatility_pct": 44.96,
    "annualised_volatility_1y_pct": 37.89, "mean_abs_daily_move_pct": 2.1,
    "regime": "elevated",
})
SENT_JSON = json.dumps({
    "ok": True, "score": 0.5707, "label": "positive", "positive": 6,
    "negative": 1, "neutral": 3, "mean_confidence": 0.71, "classified": 10,
})


class _NarrativeClient:
    def structured(self, system, user, schema, repair=True):
        self.user = user
        return schema.model_validate({
            "quant_observations": ["Volatility is elevated against the one-year figure."],
            "data_gaps": [],
        })


def test_parse_observations_separates_failures():
    parsed, failures = multi_agent.TwoAgentPipeline._parse_observations({
        "get_price_data": PRICE_JSON,
        "calculate_volatility": json.dumps({"ok": False, "error": "delisted"}),
        "broken": "not json at all",
    })
    assert set(parsed) == {"get_price_data"}
    assert len(failures) == 2
    assert any("delisted" in f for f in failures)
    assert any("not valid JSON" in f for f in failures)
    return 4


def test_brief_figures_are_parsed_not_transcribed(monkey=None):
    pipeline = multi_agent.TwoAgentPipeline.__new__(multi_agent.TwoAgentPipeline)
    run = type("R", (), {"memory": SessionMemory()})()
    run.memory.observations = {
        "get_price_data": PRICE_JSON,
        "calculate_volatility": VOL_JSON,
        "llm_sentiment_tool": SENT_JSON,
    }
    result = multi_agent.PipelineResult(ticker="NVDA")

    original = multi_agent.get_client
    multi_agent.get_client = lambda: _NarrativeClient()
    try:
        brief = pipeline._compile_brief("NVDA", run, result)
    finally:
        multi_agent.get_client = original

    # Every figure must match the tool payload exactly. A null here is the bug this
    # test exists for: an LLM-transcribed brief once came back entirely empty.
    assert brief.price.current_price == 216.7
    assert brief.price.sma_50 == 208.6 and brief.price.rsi_14 == 54.6
    assert brief.price.momentum_signal == "Bullish"
    assert brief.volatility.annualised_volatility_pct == 44.96
    assert brief.volatility.regime == "elevated"
    assert brief.sentiment.score == 0.5707 and brief.sentiment.classified == 10
    assert brief.company_name == "NVIDIA Corporation"
    assert brief.quant_observations, "the narrative must still be attached"
    return 9


def test_brief_records_gaps_when_tools_fail():
    pipeline = multi_agent.TwoAgentPipeline.__new__(multi_agent.TwoAgentPipeline)
    run = type("R", (), {"memory": SessionMemory()})()
    run.memory.observations = {"get_price_data": PRICE_JSON}
    result = multi_agent.PipelineResult(ticker="NVDA")

    original = multi_agent.get_client
    multi_agent.get_client = lambda: _NarrativeClient()
    try:
        brief = pipeline._compile_brief("NVDA", run, result)
    finally:
        multi_agent.get_client = original

    assert brief.volatility is None and brief.sentiment is None
    gaps = " ".join(brief.data_gaps)
    assert "volatility not measured" in gaps
    assert "news sentiment not measured" in gaps
    assert brief.price.current_price == 216.7, "what was gathered must survive"
    return 5


def test_brief_survives_a_failed_narrative_call():
    class Failing:
        def structured(self, *a, **k):
            raise LLMError("provider down")

    pipeline = multi_agent.TwoAgentPipeline.__new__(multi_agent.TwoAgentPipeline)
    run = type("R", (), {"memory": SessionMemory()})()
    run.memory.observations = {"get_price_data": PRICE_JSON, "calculate_volatility": VOL_JSON}
    result = multi_agent.PipelineResult(ticker="NVDA")

    original = multi_agent.get_client
    multi_agent.get_client = lambda: Failing()
    try:
        brief = pipeline._compile_brief("NVDA", run, result)
    finally:
        multi_agent.get_client = original

    # The numbers do not depend on the model, so they must survive its failure, and
    # the observations fall back to facts derived from those same numbers.
    assert brief.price.current_price == 216.7
    assert brief.volatility.annualised_volatility_pct == 44.96
    assert brief.quant_observations, "a failed narrative must not empty the handoff"
    assert "44.96" in " ".join(brief.quant_observations)
    assert any("narrative failed" in e for e in result.errors)
    return 5


class _EmptyNarrativeClient:
    def structured(self, system, user, schema, repair=True):
        return schema.model_validate({"quant_observations": [], "data_gaps": ["options data absent"]})


def test_handoff_is_never_empty_even_if_the_model_returns_nothing():
    pipeline = multi_agent.TwoAgentPipeline.__new__(multi_agent.TwoAgentPipeline)
    run = type("R", (), {"memory": SessionMemory()})()
    run.memory.observations = {
        "get_price_data": PRICE_JSON,
        "calculate_volatility": VOL_JSON,
        "llm_sentiment_tool": SENT_JSON,
    }
    result = multi_agent.PipelineResult(ticker="NVDA")

    original = multi_agent.get_client
    multi_agent.get_client = lambda: _EmptyNarrativeClient()
    try:
        brief = pipeline._compile_brief("NVDA", run, result)
    finally:
        multi_agent.get_client = original

    # An empty narrative must not produce an empty handoff.
    assert brief.quant_observations, "observations must fall back to derived facts"
    joined = " ".join(brief.quant_observations)
    assert "golden cross" in joined, "the cross structure should be stated"
    assert "44.96" in joined, "the measured volatility must appear"
    assert "elevated" in joined
    assert "positive" in joined, "sentiment label should be stated"
    assert "options data absent" in " ".join(brief.data_gaps), "model gaps still kept"
    return 6


def test_derived_observations_are_checkable_against_the_figures():
    from task3_agentic.src.schemas import PriceSnapshot, SentimentSummary, VolatilityProfile

    price = PriceSnapshot(current_price=220.0, sma_50=210.0, sma_200=200.0,
                          rsi_14=75.0, macd_hist=0.5, week52_high=240.0, week52_low=160.0)
    vol = VolatilityProfile(window_days=30, annualised_volatility_pct=40.0,
                            annualised_volatility_1y_pct=20.0, regime="elevated")
    sent = SentimentSummary(score=-0.4, label="negative", positive=1, negative=6,
                            neutral=3, classified=10)
    notes = multi_agent.TwoAgentPipeline._derive_observations(price, vol, sent)
    joined = " ".join(notes)

    assert "overbought" in joined, "RSI 75 must read as overbought"
    assert "2.00x" in joined, "the volatility ratio must be computed, not asserted"
    assert "negative" in joined
    assert "75%" in joined, "220 of a 160-240 range is 75%"
    assert len(notes) <= 6
    return 5


class _StubTool:
    def __init__(self, name, payload):
        self.name = name
        self.payload = payload

    def invoke(self, args):
        return json.dumps(self.payload)


class _RaisingTool:
    def __init__(self, name):
        self.name = name

    def invoke(self, args):
        raise ConnectionError("network down")


def main():
    checks = [
        test_trace_records_inputs_output_and_duration,
        test_trace_truncates_output_to_200_chars,
        test_handled_failure_is_logged_as_error_not_ok,
        test_trace_records_raised_exceptions_then_reraises,
        test_trace_attributes_calls_to_the_acting_agent,
        test_trace_is_jsonl_and_summarises,
        test_session_memory_recall,
        test_cache_round_trip_and_date_keying,
        test_cache_rejects_foreign_and_corrupt_files,
        test_agents_have_disjoint_enforced_tool_access,
        test_restriction_holds_even_if_the_model_asks_for_a_forbidden_tool,
        test_agent_selects_tools_autonomously_and_stops,
        test_agent_routes_around_a_failing_tool,
        test_agent_survives_a_raising_tool,
        test_agent_recovers_from_a_provider_rejection,
        test_agent_stops_after_repeated_provider_failures,
        test_observation_gist_flags_failures,
        test_report_requires_exactly_three_risks,
        test_report_markdown_has_the_three_required_sections,
        test_handoff_schema_normalises_prose,
        test_small_history_is_left_untouched,
        test_large_history_collapses_older_tool_payloads,
        test_compaction_preserves_tool_call_id_pairing,
        test_collapsed_payload_keeps_the_substance,
        test_oversized_and_quota_are_told_apart,
        test_quota_exhaustion_switches_provider_rather_than_compacting,
        test_quota_exhaustion_reraises_when_no_fallback_exists,
        test_call_model_recovers_from_a_rate_limit_by_sending_less,
        test_parse_observations_separates_failures,
        test_brief_figures_are_parsed_not_transcribed,
        test_brief_records_gaps_when_tools_fail,
        test_brief_survives_a_failed_narrative_call,
        test_handoff_is_never_empty_even_if_the_model_returns_nothing,
        test_derived_observations_are_checkable_against_the_figures,
    ]
    total = 0
    for check in checks:
        n = check()
        total += n
        print(f"  PASS  {check.__name__:56s} ({n} assertions)")
    print(f"\n{len(checks)} tests passed, {total} assertions.")


if __name__ == "__main__":
    main()
