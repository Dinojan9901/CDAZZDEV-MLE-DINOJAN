# Task 2, Generative AI

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
| Teacher, data generation | Groq `openai/gpt-oss-120b` | Free, strong at structured extraction |
| Student, fine-tuned | a smaller base model, chosen in the notebook | Must differ from the teacher, per the brief |

The full teacher system prompt is at `prompts/teacher_system_prompt.md`.

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
