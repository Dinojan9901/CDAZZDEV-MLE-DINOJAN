# Task 3, Agentic Workflows

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dinojan9901/CDAZZDEV-MLE-DINOJAN/blob/main/task3_agentic/notebook.ipynb)

Multi-agent financial research system. 100 points plus 5 bonus.

## Query the system answers

> Analyse the current financial health and market sentiment of [TICKER]. Identify the top three risks to its share price over the next 90 days and suggest one data-driven hedge strategy.

## Run it

```powershell
.\.venv\Scripts\python.exe -m task3_agentic.src.run_demo --ticker NVDA --fresh-trace
```

Offline tests, no key or network needed:

```powershell
.\.venv\Scripts\python.exe -m task3_agentic.tests.test_agentic
```

## Modules

| File | Covers |
|---|---|
| `src/tools.py` | the five tools, each returning handled failures rather than raising |
| `src/agent.py` | 3A, explicit reasoning loop with autonomous tool selection |
| `src/multi_agent.py` | 3B, two-agent pipeline with typed handoff and critique loop |
| `src/memory.py` | 3C, session memory and the persistent brief cache |
| `src/trace.py` | 3C, JSONL tool tracing |
| `src/schemas.py` | Pydantic contracts between the agents |
| `src/prompts.py` | system prompts, kept out of the reasoning logic |
| `src/run_demo.py` | end-to-end demonstration of 3A, 3B and 3C |

## Tools

| Tool | Purpose |
|---|---|
| `get_price_data(ticker, period)` | yfinance OHLCV plus computed indicators |
| `get_news(ticker, n)` | structured recent headlines |
| `calculate_volatility(ticker, window)` | annualised volatility from daily log returns |
| `llm_sentiment(headlines)` | structured sentiment score |
| `web_search(query)` | analyst commentary via DuckDuckGo |

Price and news logic is imported from `task1_financial/src` rather than reimplemented, so the agent and the Task 1 brief cannot disagree about the same ticker.

Tools never raise into the agent loop. A failure returns `{"ok": false, "error": ...}` so the model observes it and can route around it. An exception would end the run, which is the opposite of what the brief requires.

## Agents

| Agent | Role | Tool access | Output |
|---|---|---|---|
| A, Data Analyst | quantitative | `get_price_data`, `calculate_volatility`, `llm_sentiment_tool` | `DataBrief` |
| B, Research Writer | qualitative | `web_search_tool`, `get_news` | `ResearchReport` |

Tool restriction is structural, not a prompt instruction. Each agent is constructed with its own tool list and looks tools up by name in that list alone, so Agent B has no reference by which it could reach a price tool. A test asserts that asking for a forbidden tool returns `unknown tool` rather than data.

Handoff is a validated Pydantic model. A string handoff would let a malformed number reach the report as prose, where nothing can catch it.

## Design notes

**Budget-aware planning.** Left alone the model kept searching until the iteration ceiling stopped it, then returned nothing. It is now told how many calls remain, which turns a hard cutoff into a decision it makes. If the ceiling is still reached, one final call without tools bound forces a conclusion from what was gathered.

**Recovery from provider rejections.** Groq rejects a turn when the model emits tool arguments that are not valid JSON, which happens with long string arguments. The loop catches that, tells the model to use shorter arguments, and continues. It gives up after three such failures rather than burning the whole budget.

**Sentiment by ticker, not by pasted text.** `llm_sentiment_tool` accepts a ticker and fetches the headlines itself. Making the model re-emit ten headlines as JSON arguments was the direct cause of the rejection above.

## Memory and observability

- Short-term: context persists across tool calls within a session, demonstrated by a follow-up answered with zero new tool calls.
- Persistent: the brief is cached to `cache/<TICKER>_<DATE>.json` and detected on a repeat run. The date is part of the key on purpose, since yesterday's brief is stale by definition.
- Tracing: every tool call appends to `logs/agent_trace.jsonl` with tool name, inputs, output truncated to 200 characters, wall-clock duration, and which agent made the call. A tool returning a handled failure is logged as `status: error`, not `ok`, so the trace does not show a clean run that was not.

## Mark allocation

| Section | Criterion | Marks |
|---|---|---|
| 3A | All five tools implemented | 15 |
| 3A | Autonomous tool selection | 10 |
| 3A | Observe and replan cycle | 8 |
| 3A | Final report quality | 10 |
| 3A | Error handling | 7 |
| 3B | Distinct roles and tool restriction | 8 |
| 3B | Structured handoff schema | 8 |
| 3B | Message trace visible | 6 |
| 3B | Critique loop | 8 |
| 3B | End-to-end automation | 5 |
| 3C | Short-term memory | 5 |
| 3C | Persistent cache | 5 |
| 3C | agent_trace.jsonl present | 5 |

Bonus, LangSmith or a Streamlit trace dashboard: up to 5.
