"""End-to-end demonstration of Task 3.

    python -m task3_agentic.src.run_demo --ticker NVDA

Covers 3A (single agent), 3B (two-agent pipeline with a critique loop) and 3C (session
memory, persistent cache, trace file).
"""

import argparse
import json
import logging
import sys
import time
from datetime import date

from common import config
from task3_agentic.src import agent as agent_mod
from task3_agentic.src import multi_agent, tools as toolkit, trace as tracing
from task3_agentic.src.memory import BriefCache

QUERY = ("Analyse the current financial health and market sentiment of {ticker}. "
         "Identify the top three risks to its share price over the next 90 days "
         "and suggest one data-driven hedge strategy.")

RULE = "=" * 78


def banner(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def run_3a(ticker: str, tracer) -> tuple:
    banner(f"TASK 3A  single tool-using research agent  |  {ticker}")
    researcher = agent_mod.ResearchAgent(verbose=True)
    run = researcher.run(QUERY.format(ticker=ticker))

    print(f"\n--- loop summary ---")
    print(f"steps: {len(run.steps)} | tool calls: {run.tool_call_count} "
          f"| elapsed: {run.elapsed_s:.1f}s")
    print(f"tools chosen, in the order the agent picked them: {run.tools_used}")

    print("\n--- observe and replan cycle ---")
    for step in run.steps:
        for call, obs in zip(step.tool_calls, step.observations):
            print(f"  step {step.index}: called {call['name']}{tuple(call['args'].values())}")
            print(f"           observed -> {obs['gist']}")
        if step.thought and step.tool_calls:
            print(f"           reasoning -> {step.thought[:160]}")

    report = researcher.synthesise(run, ticker)
    if report:
        print("\n--- structured report ---")
        print(report.to_markdown())
    else:
        print(f"\n[report synthesis failed] {run.report_error}")
    return researcher, run, report


def run_3c_short_term(researcher, run, ticker: str) -> int:
    banner("TASK 3C  short-term memory: follow-up answered from session context")
    question = (f"What was the 30-day annualised volatility figure you already retrieved "
                f"for {ticker}, and what regime did it indicate? Answer from what you "
                f"have already gathered.")
    answer, new_calls = researcher.ask(run, question)
    if new_calls < 0:
        print()
        print("INCONCLUSIVE: the follow-up call itself failed, memory not exercised")
    elif new_calls == 0:
        print()
        print("new tool calls required: 0")
        print("PASS: answered from session context without re-calling any tool")
    else:
        print()
        print(f"new tool calls required: {new_calls}")
        print("NOTE: the agent chose to re-fetch rather than use context")
    return new_calls


def run_3c_cache(ticker: str, report, cache: BriefCache) -> None:
    banner("TASK 3C  persistent memory: cache write then detected reload")
    path = cache.save(ticker, {"report": report.model_dump() if report else None,
                               "generated_at": time.time()})
    print(f"saved  -> {path.name}")

    loaded = cache.load(ticker)
    print(f"reload -> {'HIT' if loaded else 'MISS'} for {ticker} on {date.today().isoformat()}")
    if loaded:
        risks = (loaded.get("report") or {}).get("top_risks") or []
        print(f"         cached brief carries {len(risks)} risks, "
              f"cached_on={loaded.get('cached_on')}")
    print(f"stale key check -> "
          f"{'HIT' if cache.load(ticker, '2020-01-01') else 'MISS (correct, yesterday is stale)'}")
    print(f"files in cache: {cache.list_cached()}")


def run_3b(ticker: str) -> None:
    banner(f"TASK 3B  two-agent pipeline with critique loop  |  {ticker}")
    pipeline = multi_agent.TwoAgentPipeline(verbose=True)
    result = pipeline.run(ticker)

    print("\n--- enforced tool access ---")
    for name, allowed in pipeline.tool_access().items():
        print(f"  {name}: {allowed}")

    print("\n--- message trace, agent to agent ---")
    print(result.render_transcript())

    if result.report:
        print("--- final report ---")
        print(result.report.to_markdown())
    if result.errors:
        print(f"\nerrors: {result.errors}")
    print(f"\npipeline elapsed: {result.elapsed_s:.1f}s")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Task 3 demonstration")
    parser.add_argument("--ticker", default=config.DEFAULT_TICKER)
    parser.add_argument("--skip-3b", action="store_true")
    parser.add_argument("--fresh-trace", action="store_true",
                        help="truncate agent_trace.jsonl before running")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    if not config.available_providers():
        print("No LLM key found. Set GROQ_API_KEY in .env.", file=sys.stderr)
        return 2

    tracer = tracing.ToolTracer(
        config.TASK3_LOG_DIR / "agent_trace.jsonl", append=not args.fresh_trace
    )
    toolkit.set_tracer(tracer)
    cache = BriefCache(config.TASK3_CACHE_DIR)
    ticker = args.ticker.upper()

    researcher, run, report = run_3a(ticker, tracer)
    run_3c_short_term(researcher, run, ticker)
    run_3c_cache(ticker, report, cache)
    if not args.skip_3b:
        run_3b(ticker)

    banner("TASK 3C  observability: agent_trace.jsonl")
    print(tracer.render())
    print("\n--- summary ---")
    print(json.dumps(tracer.summary(), indent=2))
    print(f"\ntrace file: {tracer.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
