"""Generate the task notebooks.

The notebooks are built from here rather than hand-edited so their structure stays in
step with the modules they demonstrate. Run this, then execute the notebooks to capture
outputs, which the brief requires to be left visible.

    python tools/build_notebooks.py
"""

import sys
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/Dinojan9901/CDAZZDEV-MLE-DINOJAN"

BOOTSTRAP = '''import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    if not Path("CDAZZDEV-MLE-DINOJAN").exists():
        subprocess.run(["git", "clone", "-q", "%s.git"], check=True)
    os.chdir("CDAZZDEV-MLE-DINOJAN")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
                   check=False)
    # Keys come from Colab Secrets, never from a cell. A committed token is a
    # listed disqualifier in the assessment brief.
    from google.colab import userdata
    for name in ("GROQ_API_KEY", "OPENROUTER_API_KEY"):
        try:
            value = userdata.get(name)
            if value:
                os.environ[name] = value
        except Exception:
            pass
else:
    root = Path.cwd()
    while root != root.parent and not (root / "common").is_dir():
        root = root.parent
    os.chdir(root)
    sys.path.insert(0, str(root))

from common import config
print("working dir :", Path.cwd().name)
print("providers   :", config.available_providers())
print("model       :", config.GROQ_MODEL)
print("fast model  :", config.GROQ_MODEL_FAST)
''' % GITHUB


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def badge(path):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/Dinojan9901/CDAZZDEV-MLE-DINOJAN/"
            f"blob/main/{path})")


def task1():
    cells = [
        md(f"""
# Task 1, Financial AI: LLM-Powered Equity Research Assistant

{badge('task1_financial/notebook.ipynb')}

Fetches real market data, computes five technical indicators from first principles with
no TA-Lib, retrieves news, and uses an LLM to produce validated sentiment and a reasoned
Buy, Hold or Sell call.

**Sections**

| | Covers | Marks |
|---|---|---|
| 1A | OHLCV, indicators, news, summary dictionary, robustness | 60 |
| 1B | Per-headline sentiment, signal reasoning, schema validation, prompt design | 40 |
| Bonus | Rendered research brief with an embedded chart | 5 |
"""),
        md("## Setup\n\nRuns unchanged locally or on Colab."),
        code(BOOTSTRAP),
        md("""
# Task 1A, Data Pipeline

## Fetching OHLCV

Dates are derived from today, never hardcoded. A warm-up buffer of 260 trading days is
fetched *before* the two-year analysis window, because SMA200 needs 200 prior bars and
without the buffer the first year of the window would have no long average at all.

`auto_adjust=True` is load-bearing rather than cosmetic: NVDA split 10:1 in June 2024 and
an unadjusted series would corrupt every moving average across the split.
"""),
        code("""
import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from task1_financial.src import data, indicators as ind, news, summary as summary_mod

TICKER = "NVDA"

history = data.fetch_ohlcv(TICKER, years=2)
print(f"bars fetched        : {len(history.frame)}")
print(f"bars in 2y window   : {len(history)}")
print(f"date range          : {history.frame.index[0].date()} to {history.frame.index[-1].date()}")
print(f"analysis starts     : {history.analysis_start.date()}")
history.frame.tail()
"""),
        md("""
## Indicators, computed from first principles

No TA-Lib. Two implementation choices worth stating:

**RSI uses true Wilder smoothing**, seeded with the simple mean of the first 14 gains,
not an `ewm` approximation. An EWM seeded on its first value drifts from published RSI
figures over the first few dozen bars.

**Bollinger bands use population standard deviation** (`ddof=0`) to match charting
convention. Pandas defaults to the sample deviation, which inflates the bands slightly.
"""),
        code("""
enriched = ind.add_all(history.frame)
window = enriched.loc[enriched.index >= history.analysis_start]

cols = ["Close", "sma_50", "sma_200", "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_mid", "bb_lower", "bb_pct_b"]
print("null count inside the analysis window, all five indicators:")
print(window[cols].isna().sum().to_string())
window[cols].tail()
"""),
        md("""
### Verifying the indicators against a published source

Correctness is asserted against **Wilder's own worked example**, not against another
library. Nineteen reference RSI values are checked to within 0.02, which proves the
implementation independently of any TA package.
"""),
        code("""
from task1_financial.tests import test_indicators
test_indicators.main()
"""),
        md("""
## News retrieval

Three sources are tried in order (yfinance, Yahoo RSS, Google News RSS), then deduplicated
and ranked.

Ranking is deliberately **binary**, on-topic versus off-topic, rather than by how many
times the company is named. Aggregator headlines pad in "(NASDAQ:NVDA)" and scored higher
than genuine editorial ones, which collapsed three sources back down to one.
"""),
        code("""
headlines = news.fetch_news(TICKER, n=10, company_name="NVIDIA Corporation")

from collections import Counter
print(f"retrieved {len(headlines)} headlines")
print("source mix:", dict(Counter(h["source"] for h in headlines)))
print(f"on-topic  : {sum(1 for h in headlines if h['relevance'] > 0)} of {len(headlines)}")
print()
for i, h in enumerate(headlines, 1):
    print(f"{i:2d}. [{h['source']:10s}] {h['headline'][:78]}")
"""),
        md("""
## Summary dictionary and momentum signal

The momentum signal is **computed here, deterministically**, not by the LLM. Four equally
weighted components each contribute -1, 0 or +1. Keeping the arithmetic in code means a
sampling failure cannot silently change a number in the report, and the LLM is asked in
1B to reason about the reading rather than produce it.
"""),
        code("""
import json
summary = summary_mod.build_summary(TICKER, enriched, window)
print(json.dumps(summary, indent=2, default=str))
"""),
        md("""
## Robustness

The brief requires missing data to be handled without unhandled exceptions.
"""),
        code("""
for bad in ["ZZZZNOTREAL", "", "   "]:
    try:
        data.fetch_ohlcv(bad)
        print(f"{bad!r:14s} -> unexpected success")
    except (data.DataFetchError, ValueError) as exc:
        print(f"{bad!r:14s} -> handled cleanly: {type(exc).__name__}")

print(f"{'news(bad)':14s} -> returned {len(news.fetch_news('ZZZZNOTREAL', n=5))} headlines, no exception")
"""),
        md("""
# Task 1B, LLM Sentiment and Signal Reasoning

## Prompts are constants, not inline strings

Prompts live in `task1_financial/src/prompts.py`, separated from the pipeline logic. The
system prompt carries **no per-request data**: the ticker and price appear only in the
user turn, so the role definition does not change per call. A test asserts this
mechanically rather than by eye.
"""),
        code("""
from task1_financial.src import prompts
print("=== SENTIMENT SYSTEM PROMPT ===")
print(prompts.SENTIMENT_SYSTEM)
print()
print("=== SIGNAL SYSTEM PROMPT ===")
print(prompts.SIGNAL_SYSTEM)
"""),
        md("""
## Per-headline classification with schema validation

Each headline gets its own call returning `headline`, `sentiment`, `confidence` and
`brief_reason`, validated against a Pydantic model before use. Failures are logged and
the headline is **excluded** from the aggregate rather than defaulted to neutral, which
would drag the score toward zero and hide the problem.
"""),
        code("""
from task1_financial.src import analysis

outcome = analysis.analyse_headlines(headlines, TICKER, "NVIDIA Corporation")

import pandas as pd
rows = [{"sentiment": r.sentiment, "conf": round(r.confidence, 2),
         "headline": r.headline[:60], "reason": r.brief_reason[:70]}
        for r in outcome.results]
print(f"classified {len(outcome.results)} of {len(headlines)}  "
      f"(success rate {outcome.success_rate:.0%})")
if outcome.failures:
    print(f"failures: {outcome.failures}")
pd.DataFrame(rows)
"""),
        md("""
### Aggregation is confidence-weighted

A hedged 0.35 call should not carry the same weight as an unambiguous 0.95 one, which is
what an unweighted count would do.
"""),
        code("""
agg = outcome.aggregate
print(json.dumps(agg.model_dump(), indent=2))

naive = sum({"positive": 1, "negative": -1, "neutral": 0}[r.sentiment]
            for r in outcome.results) / len(outcome.results)
print(f"\\nunweighted mean would be : {naive:+.4f}")
print(f"confidence-weighted score: {agg.score:+.4f}")
"""),
        md("""
## Signal reasoning over indicator combinations

The prompt explicitly forbids restating values and names the confluences to reason about:
a cross that agrees or conflicts with MACD momentum, RSI read against trend direction,
band position, and whether sentiment confirms or contradicts the technicals.

The deterministic momentum reading is supplied as a **reference the model may override**,
provided it explains what it is overriding and why.
"""),
        code("""
signal = analysis.generate_signal(summary, outcome, attempted=len(headlines))

print(f"rule-based reading : {summary['momentum']['signal']} "
      f"(score {summary['momentum']['score']})")
print(f"LLM call           : {signal.signal}")
print()
print(signal.justification)
print()
print("key drivers:")
for d in signal.key_drivers:
    print(" -", d)
"""),
        md("""
### Validation failures are caught, logged and handled

Demonstrated by feeding the schema deliberately malformed model output.
"""),
        code("""
import pydantic
from common.schemas import HeadlineSentiment, TradingSignal

bad_cases = [
    ({"headline": "h", "sentiment": "bullish", "confidence": 0.5, "brief_reason": "r"},
     "sentiment outside the allowed values"),
    ({"headline": "h", "sentiment": "positive", "confidence": 1.4, "brief_reason": "r"},
     "ambiguous 1.4, neither a valid score nor clearly a percentage"),
    ({"headline": "h", "sentiment": "positive", "confidence": 0.5, "brief_reason": ""},
     "empty reason"),
]
for payload, why in bad_cases:
    try:
        HeadlineSentiment(**payload)
        print(f"NOT REJECTED: {why}")
    except pydantic.ValidationError:
        print(f"rejected as expected: {why}")

# Percentages are coerced rather than rejected, because models emit them often.
print()
print("85   coerced to", HeadlineSentiment(headline="h", sentiment="positive",
                                           confidence=85, brief_reason="r").confidence)
print("'0.7' coerced to", HeadlineSentiment(headline="h", sentiment="positive",
                                            confidence="0.7", brief_reason="r").confidence)
"""),
        md("""
# Bonus, Rendered Research Brief

Markdown is the single source; the HTML page is generated from it so the two artefacts
cannot drift apart. The chart uses three stacked panels sharing one time axis rather than
a dual-axis overlay, because price and RSI on two y-scales would let a reader infer
crossings that are purely a scaling artefact.
"""),
        code("""
from task1_financial.src import report
from IPython.display import Image, Markdown, display

result = {
    "ticker": TICKER,
    "headline_sentiment": [r.model_dump() for r in outcome.results],
    "sentiment_failures": outcome.failures,
    "classification_success_rate": round(outcome.success_rate, 4),
    "aggregate_sentiment": agg.model_dump(),
    "signal": signal.model_dump(),
    "signal_error": None,
}

paths = report.write_report(summary, result, headlines, enriched=enriched)
for kind, path in paths.items():
    print(f"{kind:9s} {path.name:22s} {path.stat().st_size:>9,} bytes")

display(Image(filename=str(paths["chart"])))
"""),
        code("""
display(Markdown(paths["markdown"].read_text(encoding="utf-8")))
"""),
        md("""
# Test suite

All tests run offline with no API key and no network access.
"""),
        code("""
from task1_financial.tests import test_news, test_analysis, test_report
for mod in (test_indicators, test_news, test_analysis, test_report):
    print(f"--- {mod.__name__.split('.')[-1]} ---")
    mod.main()
    print()
"""),
        md("""
# Criteria checklist

| Criterion | Marks | Where |
|---|---|---|
| OHLCV data fetch, 2y+, no hardcoded dates | 10 | `src/data.py`, warm-up buffer for SMA200 |
| Indicator accuracy, five from first principles | 25 | `src/indicators.py`, verified against Wilder |
| News retrieval, ten or more | 10 | `src/news.py`, three sources with relevance ranking |
| Summary dictionary | 10 | `src/summary.py` |
| Robustness | 5 | bad tickers handled, nulls dropped, no magic numbers |
| Per-headline JSON | 10 | four required fields, confidence-weighted aggregation |
| Signal reasoning quality | 15 | reasons over confluences, may override the rule signal |
| Structured output validation | 10 | Pydantic with a self-repair pass, failures logged |
| Prompt engineering | 5 | templates as constants, system carries no request data |
| Bonus, rendered brief | 5 | Markdown, styled HTML, embedded chart, disclaimer |
"""),
    ]
    return cells


def task3():
    cells = [
        md(f"""
# Task 3, Agentic Workflows: Multi-Agent Financial Research System

{badge('task3_agentic/notebook.ipynb')}

> Analyse the current financial health and market sentiment of [TICKER]. Identify the top
> three risks to its share price over the next 90 days and suggest one data-driven hedge
> strategy.

| | Covers | Marks |
|---|---|---|
| 3A | Five tools, autonomous selection, observe and replan, error handling | 50 |
| 3B | Two agents, enforced tool restriction, typed handoff, critique loop | 35 |
| 3C | Session memory, persistent cache, `agent_trace.jsonl` | 15 |
"""),
        md("## Setup"),
        code(BOOTSTRAP),
        code("""
import json, logging
logging.basicConfig(level=logging.ERROR)

from common import config
from task3_agentic.src import agent as agent_mod
from task3_agentic.src import multi_agent, tools as toolkit, trace as tracing
from task3_agentic.src.memory import BriefCache

TICKER = "NVDA"

tracer = tracing.ToolTracer(config.TASK3_LOG_DIR / "agent_trace.jsonl", append=False)
toolkit.set_tracer(tracer)
print("tracing to:", tracer.path)
print("session id:", tracer.session_id)
"""),
        md("""
# Task 3A, Tool-Using Research Agent

## The five tools

Price and news logic is imported from `task1_financial/src` rather than reimplemented, so
the agent and the Task 1 brief cannot disagree about the same ticker.

Tools **never raise into the agent loop**. A failure returns `{"ok": false, "error": ...}`
so the model observes it and can route around it. An exception would end the run, which is
the opposite of what the brief requires.
"""),
        code("""
for tool in toolkit.ALL_TOOLS:
    first_line = (tool.description or "").strip().split("\\n")[0]
    print(f"{tool.name:22s} {first_line}")
"""),
        code("""
with tracing.acting_as("tool_check"):
    p = toolkit.price_data(TICKER, "1y")
    v = toolkit.volatility(TICKER, 30)
    n = toolkit.news(TICKER, 6)
    s = toolkit.llm_sentiment([h["headline"] for h in n["headlines"]], TICKER, "NVIDIA Corporation")
    w = toolkit.web_search(f"{TICKER} analyst outlook risks", 4)

print(f"get_price_data       ok={p['ok']}  price={p.get('current_price')}  "
      f"momentum={p.get('momentum', {}).get('signal')}")
print(f"calculate_volatility ok={v['ok']}  30d={v.get('annualised_volatility_pct')}%  "
      f"1y={v.get('annualised_volatility_1y_pct')}%  regime={v.get('regime')}")
print(f"get_news             ok={n['ok']}  count={n.get('count')}")
print(f"llm_sentiment        ok={s['ok']}  score={s.get('score')}  label={s.get('label')}")
print(f"web_search           ok={w['ok']}  count={w.get('count')}")
"""),
        md("""
### Tool failure returns an observation, not an exception
"""),
        code("""
with tracing.acting_as("failure_check"):
    bad = toolkit.price_data("ZZZZNOTREAL")
    bad_vol = toolkit.volatility("ZZZZNOTREAL")
    empty = toolkit.llm_sentiment([])

for label, out in [("price_data", bad), ("volatility", bad_vol), ("sentiment", empty)]:
    print(f"{label:12s} ok={out['ok']}  error={out['error'][:64]}")
"""),
        md("""
## The reasoning loop

The loop is written out rather than taken from a prebuilt helper, because the brief
requires a *visible* cycle of call, observe, then decide the next action from what was
observed.

Tool order is never hardcoded. The agent is handed the query and the toolset and chooses.
Two behaviours worth watching in the trace below:

- **Budget awareness.** Left alone the model kept searching until the iteration ceiling
  stopped it, then returned nothing. It is now told how many calls remain.
- **Provider failover.** If the Groq daily allowance is spent mid-run, the loop switches
  to OpenRouter rather than dying.
"""),
        code("""
QUERY = (f"Analyse the current financial health and market sentiment of {TICKER}. "
         f"Identify the top three risks to its share price over the next 90 days "
         f"and suggest one data-driven hedge strategy.")

researcher = agent_mod.ResearchAgent(verbose=True, max_iterations=8)
run = researcher.run(QUERY)

print()
print(f"steps: {len(run.steps)} | tool calls: {run.tool_call_count} | {run.elapsed_s:.1f}s")
print(f"tools chosen, in the order the agent picked them:")
for i, name in enumerate(run.tools_used, 1):
    print(f"  {i}. {name}")
if run.provider_errors:
    print(f"provider errors recovered from: {len(run.provider_errors)}")
"""),
        md("""
### The observe and replan cycle, made explicit
"""),
        code("""
for step in run.steps:
    for call, obs in zip(step.tool_calls, step.observations):
        print(f"step {step.index}")
        print(f"  decided  : {call['name']}({json.dumps(call['args'])})")
        print(f"  observed : {obs['gist']}")
        if step.thought:
            print(f"  reasoning: {step.thought[:200]}")
        print()
"""),
        md("""
## The structured report

Three sections as the brief specifies, validated by Pydantic. `top_risks` is constrained
to exactly three, so a report with two or four fails validation rather than shipping.
"""),
        code("""
from IPython.display import Markdown, display

report = researcher.synthesise(run, TICKER)
if report:
    display(Markdown(report.to_markdown()))
else:
    print("synthesis failed:", run.report_error)
"""),
        md("""
# Task 3C, Short-Term Memory

A follow-up whose answer the session already holds should cost **zero** new tool calls.
The trace is the proof: if the agent re-fetched, a new row would appear.
"""),
        code("""
question = (f"What was the 30-day annualised volatility figure you already retrieved for "
            f"{TICKER}, and what regime did it indicate? Answer from what you have "
            f"already gathered.")

calls_before = len(tracer.read())
answer, new_calls = researcher.ask(run, question)
calls_after = len(tracer.read())

print()
print(f"new tool calls reported : {new_calls}")
print(f"new rows in the trace   : {calls_after - calls_before}")
print("PASS: answered from session context" if new_calls == 0
      else "the agent chose to re-fetch")
"""),
        md("""
# Task 3C, Persistent Memory

The brief is cached by ticker **and date**. The date is part of the key deliberately:
yesterday's brief is stale by definition, so it must not satisfy today's request.
"""),
        code("""
from datetime import date

cache = BriefCache(config.TASK3_CACHE_DIR)
cache.clear(TICKER)
print("cold lookup    :", "HIT" if cache.load(TICKER) else "MISS (expected)")

path = cache.save(TICKER, {"report": report.model_dump() if report else None})
print("saved          :", path.name)

loaded = cache.load(TICKER)
print("second run     :", "HIT, tools skipped" if loaded else "MISS")
print("cached_on      :", loaded.get("cached_on") if loaded else None)
print("stale date key :", "HIT" if cache.load(TICKER, "2020-01-01") else "MISS (correct)")
print("files          :", cache.list_cached())
"""),
        md("""
# Task 3B, Two-Agent Coordination

Tool restriction is **structural, not prompted**. Each agent is constructed with its own
tool list and looks tools up by name within it, so Agent B holds no reference by which it
could reach a price tool.

Handoff is a validated Pydantic model. Figures are parsed out of the tool payloads in
Python rather than transcribed by the model: an earlier version asked the model to copy
them from truncated JSON and it returned an entirely null brief, which validated cleanly
and told Agent B nothing.
"""),
        code("""
pipeline = multi_agent.TwoAgentPipeline(verbose=True)

print("enforced tool access:")
for name, allowed in pipeline.tool_access().items():
    print(f"  {name:26s} {allowed}")
print()

# Structural proof: Agent B cannot reach a price tool even if it asks for one.
denied = json.loads(pipeline.agent_b._invoke_tool(
    {"name": "get_price_data", "args": {"ticker": TICKER}, "id": "x", "type": "tool_call"}))
print("Agent B requesting get_price_data ->", denied)
"""),
        code("""
result = pipeline.run(TICKER)
"""),
        md("""
## The agent-to-agent message trace

Every handoff, the clarification request, its answer, and the final report.
"""),
        code("""
print(result.render_transcript())
"""),
        md("""
## The critique loop

Agent B raises exactly one clarification, Agent A answers from its own data, and Agent B
incorporates it before writing. Note that Agent A **declines to fabricate** anything its
tools did not measure.
"""),
        code("""
if result.clarification:
    print("Agent B asks:")
    print(" ", result.clarification.question)
    print()
if result.clarification_answer:
    print("Agent A answers:")
    print(" ", result.clarification_answer.answer)
    print()
    print("supporting data:", json.dumps(result.clarification_answer.supporting_data, indent=2))
else:
    print("no clarification was exchanged")
"""),
        code("""
if result.report:
    display(Markdown(result.report.to_markdown()))
if result.errors:
    print("errors:", result.errors)
print(f"pipeline elapsed: {result.elapsed_s:.1f}s")
"""),
        md("""
# Task 3C, Observability

Every tool call is appended to `logs/agent_trace.jsonl` with the tool name, its inputs,
the output truncated to 200 characters, wall-clock duration, and which agent made the
call.

A tool returning a handled failure is logged as `status: error`, not `ok`. An earlier
version logged those as clean calls, which would have shown a reviewer a successful run
that was not.
"""),
        code("""
print(tracer.render())
"""),
        code("""
import pandas as pd
rows = tracer.read()
df = pd.DataFrame([{"seq": r["seq"], "agent": r["agent"], "tool": r["tool"],
                    "ms": r["duration_ms"], "status": r["status"],
                    "out_chars": r["output_chars"], "truncated": r["output_truncated"]}
                   for r in rows])
display(df)
print()
print("one raw JSONL row:")
print(json.dumps(rows[0], indent=2))
"""),
        code("""
print(json.dumps(tracer.summary(), indent=2))
"""),
        md("""
# Test suite

Thirty-four tests covering tracing, memory, cache keying, tool restriction, agent error
recovery, context budgeting and the deterministic handoff. All run offline against a
scripted fake LLM, with no API key and no network.
"""),
        code("""
from task3_agentic.tests import test_agentic
test_agentic.main()
"""),
        md("""
# Criteria checklist

| Criterion | Marks | Where |
|---|---|---|
| All five tools implemented | 15 | `src/tools.py`, reusing the Task 1 pipeline |
| Autonomous tool selection | 10 | order chosen by the model, nothing hardcoded |
| Observe and replan cycle | 8 | printed per step above |
| Final report quality | 10 | three sections, Pydantic-validated |
| Error handling | 7 | handled failures, provider failover, context budgeting |
| Distinct roles and tool restriction | 8 | enforced by construction, proven above |
| Structured handoff schema | 8 | `DataBrief`, figures parsed in code |
| Message trace visible | 6 | full transcript printed |
| Critique loop | 8 | one request, answered, incorporated |
| End-to-end automation | 5 | `pipeline.run()` with no manual step |
| Short-term memory | 5 | follow-up at zero new tool calls |
| Persistent cache | 5 | ticker plus date key, stale keys miss |
| `agent_trace.jsonl` | 5 | committed under `logs/` |
"""),
    ]
    return cells


def build(cells, path: Path, title: str):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
        "colab": {"provenance": [], "name": title},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(path))
    code_cells = sum(1 for c in cells if c.cell_type == "code")
    print(f"wrote {path.relative_to(REPO)}  ({len(cells)} cells, {code_cells} code)")


if __name__ == "__main__":
    build(task1(), REPO / "task1_financial" / "notebook.ipynb", "Task 1 Financial AI")
    build(task3(), REPO / "task3_agentic" / "notebook.ipynb", "Task 3 Agentic Workflows")
