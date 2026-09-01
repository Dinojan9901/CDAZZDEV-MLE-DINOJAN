"""Task 1 end-to-end runner.

    python -m task1_financial.src.pipeline --ticker NVDA
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from common import config
from task1_financial.src import analysis, data, indicators as ind, news, report
from task1_financial.src import summary as summary_mod

log = logging.getLogger("task1")


def run(ticker: str, years: int = 2, n_headlines: int = 10, skip_llm: bool = False,
        output_dir: Path | None = None) -> dict:
    print(f"[1/5] fetching {years}y of OHLCV for {ticker}")
    history = data.fetch_ohlcv(ticker, years=years)
    print(f"      {len(history.frame)} bars fetched, {len(history)} in the analysis window")

    print("[2/5] computing indicators")
    enriched = ind.add_all(history.frame)
    window = enriched.loc[enriched.index >= history.analysis_start]

    print("[3/5] building summary")
    summary = summary_mod.build_summary(ticker, enriched, window)
    momentum = summary["momentum"]
    print(f"      price {summary['current_price']}, momentum {momentum['signal']} "
          f"(score {momentum['score']})")

    print(f"[4/5] retrieving up to {n_headlines} headlines")
    headlines = news.fetch_news(ticker, n=n_headlines, company_name=summary["company_name"])
    sources = {}
    for h in headlines:
        sources[h["source"]] = sources.get(h["source"], 0) + 1
    print(f"      {len(headlines)} headlines from {sources}")

    if skip_llm:
        print("[5/5] skipping LLM stage (--no-llm)")
        result = {
            "ticker": ticker, "headline_sentiment": [], "sentiment_failures": [],
            "classification_success_rate": 0.0, "aggregate_sentiment": None,
            "signal": None, "signal_error": "LLM stage skipped via --no-llm",
        }
    else:
        print("[5/5] classifying headlines and generating the signal")
        result = analysis.run_analysis(summary, headlines)
        agg = result["aggregate_sentiment"]
        if agg:
            print(f"      sentiment {agg['label']} at {agg['score']}, "
                  f"{result['classification_success_rate'] * 100:.0f}% classified")
        if result["signal"]:
            print(f"      signal: {result['signal']['signal']}")
        else:
            print(f"      no signal: {result['signal_error']}")

    paths = report.write_report(summary, result, headlines, enriched=enriched,
                                output_dir=output_dir)

    payload = {"summary": summary, "analysis": result,
               "headlines": headlines, "outputs": {k: str(v) for k, v in paths.items()}}
    json_path = Path(paths["markdown"]).with_name(f"{ticker}_analysis.json")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    paths["json"] = json_path

    print("\nwritten:")
    for kind, path in paths.items():
        print(f"  {kind:9s} {path}")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="LLM-powered equity research brief")
    parser.add_argument("--ticker", default=config.DEFAULT_TICKER)
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--headlines", type=int, default=10)
    parser.add_argument("--no-llm", action="store_true",
                        help="run the 1A pipeline only, no API calls")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.no_llm and not config.available_providers():
        print("No LLM key found. Set GROQ_API_KEY in .env, or pass --no-llm to run "
              "the data pipeline alone.", file=sys.stderr)
        return 2

    try:
        run(args.ticker, args.years, args.headlines, args.no_llm, args.output_dir)
    except data.DataFetchError as exc:
        print(f"data error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
