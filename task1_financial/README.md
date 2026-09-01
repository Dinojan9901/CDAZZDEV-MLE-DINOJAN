# Task 1, Financial AI

LLM-powered equity research assistant. 100 points plus 5 bonus.

## Notebook

`notebook.ipynb` runs the full pipeline end to end with outputs left visible.

## Modules

| File | Covers |
|---|---|
| `src/data.py` | OHLCV fetch, two years minimum, relative dates only |
| `src/indicators.py` | SMA50, SMA200, RSI-14, MACD 12/26/9, Bollinger 20/2, all from first principles |
| `src/news.py` | Headline retrieval with fallback sources |
| `src/summary.py` | Summary dictionary and momentum signal |
| `src/prompts.py` | Prompt templates, kept out of business logic |
| `src/analysis.py` | Per-headline sentiment, aggregation, Buy/Hold/Sell reasoning |
| `src/report.py` | Bonus, Markdown and HTML brief with a matplotlib chart |

No TA-Lib. Every indicator is implemented directly, RSI using Wilder smoothing rather than a simple mean.

## Mark allocation

### Task 1A, data pipeline, 60 marks

| Criterion | Marks |
|---|---|
| OHLCV data fetch | 10 |
| Indicator accuracy | 25 |
| News retrieval | 10 |
| Summary dictionary | 10 |
| Robustness | 5 |

### Task 1B, LLM reasoning, 40 marks

| Criterion | Marks |
|---|---|
| Per-headline JSON | 10 |
| Signal reasoning quality | 15 |
| Structured output validation | 10 |
| Prompt engineering | 5 |

Bonus, rendered research brief: up to 5.

## Run

```powershell
python -m task1_financial.src.pipeline --ticker NVDA
```
