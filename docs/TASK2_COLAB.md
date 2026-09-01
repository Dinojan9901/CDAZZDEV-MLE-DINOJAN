# Task 2 runbook, start this early

Fine-tuning is the only part of the assessment that needs a GPU you do not have, so it
is the only part with a queue in front of it. Everything below is ordered so nothing
waits on anything it does not have to.

## The key scheduling decision

Dataset generation is Groq API calls, not GPU work. It runs on your laptop. Moving it
off the Colab session means the GPU session covers training and evaluation only, which
cuts the session from roughly 3 to 5 hours down to roughly 2 to 3.

| Phase | Where | Wall clock | Blocked by |
|---|---|---|---|
| 0. Accounts and keys | browser | 15 min | nothing, do it now |
| 1. Generate 120 examples | your laptop | 20 to 30 min | Groq key, generation script |
| 2. Diversity report and split | your laptop | 5 min | phase 1 |
| 3. QLoRA training | Colab T4 | 40 to 70 min | phases 1 and 2, HF token |
| 4. Merge, save, push to Hub | Colab T4 | 20 to 30 min | phase 3 |
| 5. Evaluation, 2C | Colab T4 | 40 to 60 min | phase 4 |

Phase 0 has no dependency on any code. Start it now.

## Phase 0, accounts and keys

### Groq, required

1. Open https://console.groq.com and sign in with Google.
2. API Keys, then Create API Key. Copy it once, it is not shown again.
3. Paste into `.env` as `GROQ_API_KEY=`.

Free tier is rate limited per minute, not meaningfully per day at this volume. Building
120 examples costs a few hundred thousand tokens in total.

### Hugging Face, required for the model deliverable

1. Create an account at https://huggingface.co/join.
2. Settings, Access Tokens, Create new token.
3. Token type must be **Write**. A Read token cannot push a model and you will only find
   out after training finishes.
4. Create the model repo now at https://huggingface.co/new, named
   `financial-risk-extractor-phi3-qlora`. Public is preferred by the brief.

### Google Colab, required

1. Open https://colab.research.google.com.
2. Runtime, Change runtime type, T4 GPU, Save.
3. New cell, run `!nvidia-smi`. You want to see `Tesla T4` and about 15 GB.

If Colab refuses a GPU, it is busy. Retry in an hour or at a quieter time of day. This
is exactly why phase 0 happens now rather than on the last afternoon.

### Weights and Biases, optional

https://wandb.ai/authorize gives you a key for `WANDB_API_KEY`. Skip it if you prefer,
the brief accepts manual loss logging, and the notebook prints a per-epoch loss table
either way.

## Storing keys inside Colab

Never paste a key into a notebook cell. A committed token is a listed disqualifier and
the notebook is submitted with outputs visible.

Use Colab Secrets, the key icon in the left sidebar. Add `HF_TOKEN` and `GROQ_API_KEY`,
switch on Notebook access for each, then read them:

```python
from google.colab import userdata
hf_token = userdata.get("HF_TOKEN")
```

## Student model

| | Choice | Why |
|---|---|---|
| Teacher | Groq `openai/gpt-oss-120b` | Free, strong at structured extraction |
| Student | `microsoft/Phi-3-mini-4k-instruct` | 3.8B fits a T4 comfortably under 4-bit, ungated so no licence wait, and genuinely different from the teacher as the brief requires |

`Qwen/Qwen2.5-3B-Instruct` is an equally good ungated fallback. Avoid Llama and Mistral
base models here: they are gated, and waiting on an access approval can cost you a day.

## Colab free tier limits worth planning around

- Idle disconnect at roughly 90 minutes. Keep the tab open and visible.
- Session ceiling around 12 hours, and shorter if usage has been heavy recently.
- Disk is around 78 GB, which is ample. Base model plus merged model is roughly 16 GB.
- A disconnect loses everything not written to Drive. Mount Drive and checkpoint to it:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Point `output_dir` at a Drive path so a dropped session costs minutes, not the whole run.

## Do not clear the outputs

The brief disqualifies notebooks submitted with cleared cell outputs. After the run,
download the notebook with outputs intact and place it at
`task2_genai/notebook.ipynb`.
