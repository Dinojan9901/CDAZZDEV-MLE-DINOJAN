# Task 2, Generative AI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dinojan9901/CDAZZDEV-MLE-DINOJAN/blob/main/task2_genai/notebook.ipynb)

QLoRA fine-tuning pipeline for financial risk clause extraction. 100 points plus 5 bonus.

## Use case

**Input:** a passage of filing-style prose from a company disclosure or risk section.
**Output:** structured JSON listing each distinct risk factor with its category, the trigger described, and the stated or implied impact.
**Correct:** every risk present in the passage is extracted once, categorised from a fixed taxonomy, and no risk is invented.
**Incorrect:** a hallucinated risk, a missed risk, a category outside the taxonomy, or malformed JSON.

This is deliberately not a general chatbot task. The brief caps generic chatbot and creative writing use cases at 5 of 30 marks.

## Teacher and student

| Role | Model | Reason |
|---|---|---|
| Teacher, data generation | `minimax/minimax-m3:free` via OpenRouter | Strong at structured extraction, 1M context |
| Student, fine-tuned | `microsoft/Phi-3-mini-4k-instruct` | 3.8B fits a T4 under 4-bit, ungated so no licence wait |

The teacher was originally planned as Groq `openai/gpt-oss-120b`. Groq's free tier caps at
200,000 tokens per day and was exhausted when generation ran, so the client failed over to
OpenRouter and MiniMax-M3 produced all 120 examples. Each record carries a
`teacher_provider` field, so the provenance is in the data rather than only in this note.

Teacher and student are different model families, as the brief requires.

The full teacher system prompt is at [prompts/teacher_system_prompt.md](prompts/teacher_system_prompt.md)
and is loaded from that file at generation time, so the submitted copy cannot drift from
the one actually used.

## Dataset

Generated with `python -m task2_genai.src.generate_dataset --target 120`.

| | Value |
|---|---|
| Examples | 120 |
| Teacher calls | 30, batched 4 per call |
| Failed calls | 0 |
| Near-duplicates rejected | 0 |
| Risk items | 289, mean 2.41 per example |

Batching matters on a free tier: one call per example would have cost 120 calls, and
generation would not have finished inside a daily allowance.

### Diversity

Run `python -m task2_genai.src.diversity`. All nine checks pass.

| Measure | Value |
|---|---|
| Passage length | mean 153.7 words, std 30.6, range 70 to 246 |
| Pairwise similarity | mean 0.0001, p95 0.0, max 0.018 |
| Pairs above 0.45 overlap | 0 |
| Vocabulary | 11,320 tokens, 3,261 unique, type-token ratio 0.288 |
| Categories present | 12 of 12, largest share 13.8 percent |
| Sectors | 15 |
| Document styles | 4, evenly distributed |

Diversity is engineered rather than hoped for. Seeds traverse the sector-by-style grid
diagonally, which covers every sector and every style within 30 calls without repeating a
pair, and a Jaccard shingle check rejects near-duplicate passages before they enter the
dataset.

### Split

`python -m task2_genai.src.format_split`, stratified by document style.

| Split | Examples | Risk items | Categories | Ratio |
|---|---|---|---|---|
| train | 96 | 230 | 12 of 12 | 80% |
| val | 12 | 28 | 9 of 12 | 10% |
| test | 12 | 31 | 11 of 12 | 10% |

Stratifying by document style matters at this size: a random split of 120 can leave a
style out of the 12-example test set entirely, which would make the held-out score a
measurement of luck rather than of the model.

Examples are stored as a `messages` list rather than a pre-rendered string. The chat
template belongs to the tokenizer, so rendering it into the file would bake one model's
special tokens into the dataset.

## Runs on Colab

Fine-tuning needs a CUDA GPU. See [../docs/SETUP.md](../docs/SETUP.md).

## Mark allocation

| Section | Criterion | Marks |
|---|---|---|
| 2A | Use case quality | 10 |
| 2A | Dataset size, 100+ examples | 5 |
| 2A | Dataset diversity metrics | 10 |
| 2A | JSONL format and 80/10/10 split | 5 |
| 2B | QLoRA 4-bit NF4 implementation | 10 |
| 2B | Hyperparameter justification | 15 |
| 2B | Train and val loss per epoch | 10 |
| 2B | Merged model saved and linked | 5 |
| 2C | ROUGE-L base vs fine-tuned | 8 |
| 2C | Additional metric | 7 |
| 2C | Hallucination rate, 10+ reviewed | 7 |
| 2C | Qualitative analysis | 8 |

Bonus, ChromaDB RAG fallback on low confidence: up to 5.

## Model link

Hugging Face: TODO after training.
