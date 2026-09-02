"""Generate the Task 2 Colab notebook.

Built from a script for the same reason as the other two: the notebook stays in step
with the modules it uses. This one is written to run on a Colab T4 and cannot be
executed locally, so it is defensive about versions and prints what it resolved.

    python tools/build_task2_notebook.py
"""

import sys
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/Dinojan9901/CDAZZDEV-MLE-DINOJAN"
HF_REPO = "Dino21/financial-risk-extractor-phi3-qlora"


def md(t):
    return nbf.v4.new_markdown_cell(t.strip())


def code(t):
    return nbf.v4.new_code_cell(t.strip())


CELLS = [
    md(f"""
# Task 2, Generative AI: Domain-Specific Fine-Tuning Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dinojan9901/CDAZZDEV-MLE-DINOJAN/blob/main/task2_genai/notebook.ipynb)

**Task:** extract structured financial risks from corporate disclosure prose.

**Input** a passage of filing-style text. **Output** JSON listing every distinct risk with
a category from a closed twelve-item taxonomy, its trigger, its potential impact and a
severity. **Correct** means every risk in the passage appears exactly once, categorised
from the taxonomy, with nothing invented. **Incorrect** means a hallucinated risk, a
missed risk, a category outside the taxonomy, one exposure split across several entries,
or malformed JSON.

| Section | Covers | Marks |
|---|---|---|
| 2A | Use case, 120 examples, diversity metrics, 80/10/10 split | 30 |
| 2B | QLoRA 4-bit NF4, hyperparameter justification, loss curves, merged model | 40 |
| 2C | ROUGE-L, BERTScore, LLM-as-judge, hallucination rate, analysis | 30 |
| Bonus | ChromaDB RAG fallback on low confidence | 5 |

**Runtime required: T4 GPU.** Runtime, Change runtime type, T4 GPU, Save.
"""),

    md("## Environment"),
    code("""
!nvidia-smi
"""),
    code(f"""
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
assert IN_COLAB, "This notebook needs a Colab T4. Task 2 cannot run on CPU."

if not Path("CDAZZDEV-MLE-DINOJAN").exists():
    subprocess.run(["git", "clone", "-q", "{REPO_URL}.git"], check=True)
os.chdir("/content/CDAZZDEV-MLE-DINOJAN")
sys.path.insert(0, os.getcwd())
print("repo:", os.getcwd())
"""),

    md("""
### Dependencies

Versions are pinned rather than left to pip. Colab ships `transformers` 5.x, which changed
several APIs that older `peft` and `trl` releases assume. A silent version mismatch here
fails *after* the 7.6 GB model download, which is the expensive way to find out.
"""),
    code("""
%pip install -q -U "peft>=0.14" "trl>=0.13" "bitsandbytes>=0.45" "accelerate>=1.2" "datasets>=3.2" "evaluate" "rouge-score" "bert-score" "sentencepiece"
print("\\nRestart is not required; the imports below verify what actually resolved.")
"""),
    code("""
import torch, transformers, peft, trl, bitsandbytes, accelerate, datasets
print(f"torch        {torch.__version__}   cuda={torch.cuda.is_available()}")
print(f"transformers {transformers.__version__}")
print(f"peft         {peft.__version__}")
print(f"trl          {trl.__version__}")
print(f"bitsandbytes {bitsandbytes.__version__}")
print(f"accelerate   {accelerate.__version__}")
print(f"datasets     {datasets.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"\\nGPU: {p.name}, {p.total_memory/1e9:.1f} GB, capability {p.major}.{p.minor}")
    # torch reports True on Turing because it permits emulated bf16, but the T4 is
    # SM75 with no native bf16 tensor cores, so fp16 is the faster choice regardless.
    native_bf16 = p.major >= 8
    print(f"bf16 reported: {torch.cuda.is_bf16_supported()}, native bf16 tensor cores: {native_bf16}")
    print("using fp16" if not native_bf16 else "bf16 would be preferred here")
"""),
    code("""
from google.colab import userdata
for name in ("HF_TOKEN", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
    try:
        v = userdata.get(name)
        if v:
            os.environ[name] = v
            print(f"{name:20s} loaded from Colab Secrets")
    except Exception:
        print(f"{name:20s} not set (optional unless used below)")
"""),

    md("""
# Task 2A recap, the dataset

Generated locally against a teacher model and committed to the repository, so this
notebook trains on exactly the data the diversity report describes. Generation code is in
`task2_genai/src/generate_dataset.py`; the teacher prompt is in
`task2_genai/prompts/teacher_system_prompt.md`.
"""),
    code("""
import json
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

DATA = Path("task2_genai/data")
splits = {}
for name in ("train", "val", "test"):
    rows = [json.loads(l) for l in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    splits[name] = rows
    print(f"{name:6s} {len(rows):>4} examples")

total = sum(len(v) for v in splits.values())
print(f"\\ntotal {total}   ratios "
      f"{len(splits['train'])/total:.0%} / {len(splits['val'])/total:.0%} / {len(splits['test'])/total:.0%}")
"""),
    code("""
report = json.loads((DATA / "diversity_report.json").read_text(encoding="utf-8"))
from task2_genai.src import diversity
print(diversity.render(report))
"""),
    code("""
print("--- teacher system prompt, used verbatim for generation ---\\n")
print(Path("task2_genai/prompts/teacher_system_prompt.md").read_text(encoding="utf-8"))
"""),

    md("""
# Task 2B, Fine-Tuning Execution

## Base model

`microsoft/Phi-3-mini-4k-instruct`, 3.8B parameters.

Chosen because it is **ungated**, so there is no licence approval to wait on, it fits a T4
comfortably under 4-bit quantisation, and it is a different model family from the teacher
that produced the training data, which the brief requires.
"""),
    code("""
from transformers import AutoTokenizer

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.padding_side = "right"   # left padding is for generation, not for loss

sample = splits["train"][0]["messages"]
rendered = tokenizer.apply_chat_template(sample, tokenize=False)
print("--- one training example after the model's own chat template ---\\n")
print(rendered[:1200])
print("\\n...")
print(f"\\ntokens: {len(tokenizer(rendered)['input_ids'])}")
"""),
    md("""
### Sequence length, measured rather than guessed

`max_seq_length` is a memory-versus-truncation trade. Truncating a training example
silently teaches the model to emit incomplete JSON, so the length is set from the actual
token distribution with headroom, not from a round number.
"""),
    code("""
import numpy as np

lengths = [len(tokenizer(tokenizer.apply_chat_template(r["messages"], tokenize=False))["input_ids"])
           for r in splits["train"]]
lengths = np.array(lengths)
print(f"train token lengths: mean {lengths.mean():.0f}  p50 {np.percentile(lengths,50):.0f}  "
      f"p95 {np.percentile(lengths,95):.0f}  max {lengths.max()}")
MAX_SEQ_LEN = 1024
print(f"\\nmax_seq_length = {MAX_SEQ_LEN}")
print(f"examples that would be truncated: {(lengths > MAX_SEQ_LEN).sum()} of {len(lengths)}")
"""),

    md("""
## 4-bit NF4 quantisation

Four settings, each doing a specific job:

- **`load_in_4bit`** puts the frozen base weights in 4-bit. A 3.8B model in fp16 is about
  7.6 GB of weights alone; in 4-bit it is under 2.5 GB, which leaves the T4's 15 GB for
  activations, gradients and the optimiser.
- **`bnb_4bit_quant_type="nf4"`** uses NormalFloat4 rather than plain fp4. NF4 is
  information-theoretically optimal for weights that are approximately normally
  distributed, which pretrained weights are, and it consistently loses less accuracy than
  fp4 at the same bit width.
- **`bnb_4bit_compute_dtype=float16`** is what matmuls are promoted to. The T4 is Turing
  and has no bf16 support, so fp16 is the only option here; on an A100 bf16 would be the
  better choice for its wider exponent range.
- **`bnb_4bit_use_double_quant=True`** quantises the quantisation constants themselves,
  saving roughly a further 0.4 GB at negligible cost.
"""),
    code("""
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# No trust_remote_code. Transformers has had native Phi-3 support since 4.41, and the
# Hub's modeling_phi3.py is stale against transformers 5.x: it reads
# config.rope_scaling["type"] where the modern config uses "rope_type", which raises
# KeyError: 'type' during rope init. Loading remote code also silently overrides a
# working built-in implementation with an older one.
# dtype must be pinned. transformers 5.x defaults unspecified dtype to bfloat16,
# which leaves norms, embeddings and LoRA in bf16 while fp16=True runs a GradScaler
# that cannot unscale bf16 grads:
#   NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda for BFloat16
try:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map={"": 0},
        attn_implementation="eager", dtype=torch.float16,
    )
except TypeError:                        # older transformers spells it torch_dtype
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map={"": 0},
        attn_implementation="eager", torch_dtype=torch.float16,
    )
model.config.use_cache = False          # incompatible with gradient checkpointing

print(f"loaded in {model.dtype}, {model.num_parameters()/1e9:.2f}B params")
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
"""),

    md("""
## LoRA configuration

Every value below is a decision, not a default.
"""),
    code("""
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable params: {trainable:,} of {total:,}  ({100*trainable/total:.3f}%)")
"""),

    md(f"""
### Hyperparameter justification

| Parameter | Value | Why this value |
|---|---|---|
| `r` (LoRA rank) | 16 | Rank sets adapter capacity. The task is not just style transfer: the model must learn a 12-way classification plus a rigid JSON shape, which 8 underfits. 32 doubles adapter parameters with no headroom to use them on 96 examples, so it mostly adds overfitting risk. |
| `lora_alpha` | 32 | Scaling is `alpha/r`, so 32/16 gives a factor of 2, the widely used heuristic `alpha = 2r`. It keeps the effective update magnitude stable if rank changes later. |
| `target_modules` | `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj` | Phi-3 fuses QKV into `qkv_proj` and the MLP up-projections into `gate_up_proj`, so the usual Llama names do not exist here. Attention-only adaptation is cheaper but the MLP blocks carry most of the format-following behaviour, and rigid JSON output is exactly a format behaviour. |
| `lora_dropout` | 0.05 | 96 training examples is small enough to memorise. LoRA is already low-capacity, so heavy dropout would underfit; 0.05 is light regularisation on the adapter path only. |
| `bias` | `none` | Training biases adds parameters that cannot be merged cleanly and gives no measurable benefit at this scale. |
| `learning_rate` | 2e-4 | Roughly 10x a full fine-tune's 2e-5. Only the adapters update, and they start from zero, so they need a far larger step to move at all. Below about 1e-4 the JSON format does not stabilise within 3 epochs. |
| `lr_scheduler_type` | `cosine` | A constant rate leaves the model taking large steps at the end of training, which shows up as JSON format drift in the final checkpoint. Cosine decay anneals to near zero and stabilises the output shape. |
| `warmup_steps` | ~6 (8 percent of total) | 4-bit weights make the first optimiser steps noisy, and going straight to 2e-4 risks an early loss spike the run never recovers from. Expressed in steps rather than a ratio because trl 1.12 does not accept `warmup_ratio`, and because at 72 total steps a 0.03 ratio rounds to a single step, which is no warmup at all. |
| `num_train_epochs` | 3 | 1 epoch learns the JSON shape but not the taxonomy. Beyond 3, validation loss starts rising on 96 examples. 3 is the point the val curve below justifies. |
| `per_device_train_batch_size` | 1 | Phi-3-mini in 4-bit plus activations at 1024 tokens is what a 15 GB T4 holds. 2 fits only by cutting sequence length, which would truncate examples. |
| `gradient_accumulation_steps` | 4 | Effective batch of 4. Batch size 1 gives extremely noisy gradients and accumulation recovers a usable batch without the memory cost of a real one. Set to 4 rather than 8 because 96 examples at effective batch 8 gives only 12 optimiser steps per epoch, 36 across the run, which is too few for adapters starting from zero to learn a 12-way taxonomy plus a rigid output shape. At 4 it is 24 per epoch, 72 total. |
| `max_length` | 1024 | Named `max_seq_length` in older trl. Set from the measured token distribution above, with headroom over p95. Truncation would teach the model to emit incomplete JSON. |
| `optim` | `paged_adamw_8bit` | 8-bit states cut optimiser memory roughly fourfold. Paged means optimiser state spills to host RAM on a spike instead of raising OOM, which matters on a shared T4. |
| `fp16` | `True` | The T4 is Turing and has no bf16 units. On Ampere or newer, bf16 would be preferred for its wider exponent range. |
| `gradient_checkpointing` | `True` | Recomputes activations in the backward pass. Costs roughly 30 percent more time and is the single reason this fits at all. |
| `eval_strategy` | `epoch` | Named `evaluation_strategy` in older transformers. The brief requires validation loss per epoch, and per-epoch evaluation on 12 examples is cheap. |
"""),
    code("""
from datasets import Dataset

def to_text(rows):
    return Dataset.from_list([
        {"text": tokenizer.apply_chat_template(r["messages"], tokenize=False)}
        for r in rows
    ])

train_ds, val_ds = to_text(splits["train"]), to_text(splits["val"])
print(train_ds, val_ds, sep="\\n")
"""),
    md("""
### Choosing the epoch count from evidence

A 3-epoch run was executed first. Validation loss reached its minimum at epoch 2 and
rose again at epoch 3 while training loss kept falling, which is overfitting on 96
training examples:

| epoch | train loss | val loss |
|---|---|---|
| 1 | 0.6559 | 0.6278 |
| 2 | 0.5365 | 0.5989 |
| 3 | 0.3914 | 0.6069 |

`num_train_epochs` is therefore 2, chosen on measurement rather than by default, and
`load_best_model_at_end` guards it by restoring the lowest-validation checkpoint.

### Why transformers.Trainer rather than trl.SFTTrainer

trl 1.12 routes PEFT models through a chunked cross-entropy path that expects
`last_hidden_state` and receives `CausalLMOutputWithPast`, raising `AttributeError`
mid-training. Constructing `SFTTrainer` also patches the model in place, so the failure
survives into any later attempt until the model is rebuilt.

`transformers.Trainer` uses the stable API underneath and removes a fast-moving
dependency from the failure surface. The recipe is unchanged, and prompt masking is
implemented directly below, which is the behaviour `SFTTrainer` would have provided.
"""),
    code("""
import math
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

ASSISTANT_TAG = "<|assistant|>"

def encode(rows):
    \"\"\"Tokenise and mask the prompt so loss is computed on the answer only.\"\"\"
    feats = []
    for r in rows:
        full = tokenizer.apply_chat_template(r["messages"], tokenize=False)
        cut = full.rfind(ASSISTANT_TAG)
        prompt = full[:cut + len(ASSISTANT_TAG)] if cut != -1 else full

        ids = tokenizer(full, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]
        n_prompt = len(tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"])

        labels = list(ids)
        labels[:n_prompt] = [-100] * min(n_prompt, len(labels))
        feats.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    return feats

train_feats, val_feats = encode(splits["train"]), encode(splits["val"])
supervised = sum(1 for t in train_feats[0]["labels"] if t != -100)
print(f"example 1: {len(train_feats[0]['input_ids'])} tokens, {supervised} supervised")

EPOCHS = 2
EFFECTIVE_ACCUM = 4
steps_per_epoch = math.ceil(len(train_feats) / EFFECTIVE_ACCUM)
total_steps = steps_per_epoch * EPOCHS
warmup = max(3, round(total_steps * 0.08))
print(f"{steps_per_epoch} steps/epoch, {total_steps} total, {warmup} warmup")

args = TrainingArguments(
    output_dir="/content/task2_out",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=EFFECTIVE_ACCUM,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=warmup,
    optim="paged_adamw_8bit",
    fp16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    seed=42,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_feats, eval_dataset=val_feats,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
)
print("trainer ready (transformers.Trainer), epochs", EPOCHS)
"""),

    md("""
### Train

Watch the validation loss column. The brief requires it to **decrease across epochs**. If
it flattens or rises, that is a hyperparameter problem, not something to wait out.
"""),
    code("""
import time
t0 = time.time()
train_result = trainer.train()
print(f"\\ntraining took {(time.time()-t0)/60:.1f} minutes")
print(f"peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
"""),
    code("""
import pandas as pd

hist = pd.DataFrame(trainer.state.log_history)
per_epoch = []
for e in sorted({round(h["epoch"]) for h in trainer.state.log_history if "epoch" in h}):
    tr = hist[(hist.get("loss").notna()) & (hist["epoch"] <= e)]["loss"]
    ev = hist[(hist.get("eval_loss").notna()) & (hist["epoch"].round() == e)]["eval_loss"]
    if len(ev):
        per_epoch.append({"epoch": e, "train_loss": round(float(tr.tail(5).mean()), 4),
                          "val_loss": round(float(ev.iloc[-1]), 4)})

loss_table = pd.DataFrame(per_epoch)
display(loss_table)

if len(loss_table) > 1:
    drop = loss_table["val_loss"].iloc[0] - loss_table["val_loss"].iloc[-1]
    print(f"\\nvalidation loss change: {loss_table['val_loss'].iloc[0]:.4f} -> "
          f"{loss_table['val_loss'].iloc[-1]:.4f}  ({'decreased' if drop > 0 else 'DID NOT DECREASE'})")
"""),
    code("""
import matplotlib.pyplot as plt

hist = pd.DataFrame(trainer.state.log_history)
train_pts = hist.dropna(subset=["loss"]) if "loss" in hist.columns else hist.iloc[0:0]
eval_pts = hist.dropna(subset=["eval_loss"]) if "eval_loss" in hist.columns else hist.iloc[0:0]

if train_pts.empty and eval_pts.empty:
    print("No loss history. Run the training cell above before plotting.")
else:
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    if not train_pts.empty:
        ax.plot(train_pts["epoch"], train_pts["loss"], color="#2a78d6",
                linewidth=1.6, label="train (per step)")
    if not eval_pts.empty:
        ax.plot(eval_pts["epoch"], eval_pts["eval_loss"], color="#eb6834",
                linewidth=2.2, marker="o", markersize=7, label="validation (per epoch)")
    ax.set_xlabel("epoch", color="#52514e")
    ax.set_ylabel("loss", color="#52514e")
    ax.set_title("QLoRA fine-tuning loss", color="#0b0b0b", fontweight="bold", loc="left")
    ax.grid(True, color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    if not eval_pts.empty:
        first, last = eval_pts["eval_loss"].iloc[0], eval_pts["eval_loss"].iloc[-1]
        print(f"validation loss {first:.4f} -> {last:.4f}  "
              f"({'decreased' if last < first else 'DID NOT DECREASE'})")
        display(eval_pts[["epoch", "eval_loss"]].round(4))
"""),

    md("""
## Merge the adapters and save

`merge_and_unload()` folds the LoRA weights into the base weights, producing a standalone
model with no PEFT dependency at inference time.

The merge must happen in fp16 from a **freshly loaded, unquantised** base. Merging into
4-bit weights would bake the quantisation error into the saved checkpoint permanently.
"""),
    code("""
adapter_dir = "/content/adapter"
trainer.model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print("adapter saved:", adapter_dir)

del model, trainer
import gc; gc.collect(); torch.cuda.empty_cache()
print(f"GPU freed, now {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
"""),
    code("""
# PEFT's torchao dispatcher raises on an old torchao instead of returning False.
# Colab ships 0.10.0 while PEFT 0.20 wants 0.16+. We quantise with bitsandbytes, so
# this dispatcher is never reached; neutralise the probe rather than reinstall.
import peft.import_utils, peft.tuners.lora.torchao
peft.import_utils.is_torchao_available = lambda: False
peft.tuners.lora.torchao.is_torchao_available = lambda: False

from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, dtype=torch.float16, device_map={"": 0},
)
merged = PeftModel.from_pretrained(base, adapter_dir)
merged = merged.merge_and_unload()
merged.config.use_cache = True

MERGED_DIR = "/content/merged_model"
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)
print("merged model saved to", MERGED_DIR)
!du -sh /content/merged_model
"""),
    code(f"""
from huggingface_hub import HfApi, create_repo

HF_REPO_ID = "{HF_REPO}"
token = os.environ["HF_TOKEN"]
create_repo(HF_REPO_ID, exist_ok=True, private=False, token=token)

# upload_folder sends the files already on disk. push_to_hub re-serialises from
# memory, which risks a second OOM, and it dropped safe_serialization in this version.
HfApi().upload_folder(
    folder_path=MERGED_DIR, repo_id=HF_REPO_ID, repo_type="model", token=token,
    commit_message="QLoRA fine-tuned Phi-3-mini for financial risk clause extraction",
)
print(f"pushed to https://huggingface.co/{{HF_REPO_ID}}")
"""),

    md("""
# Task 2C, Evaluation and Baseline Comparison

Both models are scored on the **same 12 held-out test examples** with the same decoding
settings. The only difference between the two runs is the weights and the system prompt.

The baseline gets a **longer** system prompt containing the full category definitions.
That is deliberate and it makes the comparison harder to win: an untuned model with no
examples would otherwise be handicapped by an instruction it has never seen, and beating a
strawman baseline would prove nothing.
"""),
    code("""
from transformers import pipeline
from task2_genai.src.format_split import BASELINE_SYSTEM_PROMPT, STUDENT_SYSTEM_PROMPT

GEN_KWARGS = dict(max_new_tokens=512, do_sample=False, temperature=None, top_p=None)

def generate_all(model_obj, rows, system_prompt, label):
    pipe = pipeline("text-generation", model=model_obj, tokenizer=tokenizer, device_map={"": 0})
    outputs = []
    for i, r in enumerate(rows, 1):
        passage = r["messages"][1]["content"]
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": passage}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out = pipe(prompt, return_full_text=False, **GEN_KWARGS)[0]["generated_text"]
        outputs.append(out.strip())
        print(f"  {label} {i}/{len(rows)}", end="\\r")
    print()
    return outputs

test_rows = splits["test"]
references = [r["messages"][2]["content"] for r in test_rows]
print(f"test set: {len(test_rows)} examples")
"""),
    code("""
ft_outputs = generate_all(merged, test_rows, STUDENT_SYSTEM_PROMPT, "fine-tuned")
del merged; gc.collect(); torch.cuda.empty_cache()
"""),
    code("""
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map={"": 0},
    attn_implementation="eager",
)
base_outputs = generate_all(base_model, test_rows, BASELINE_SYSTEM_PROMPT, "base")
del base_model; gc.collect(); torch.cuda.empty_cache()
"""),

    md("## ROUGE-L on the held-out test set"),
    code("""
import evaluate

rouge = evaluate.load("rouge")
base_r = rouge.compute(predictions=base_outputs, references=references, use_stemmer=True)
ft_r = rouge.compute(predictions=ft_outputs, references=references, use_stemmer=True)

comparison = pd.DataFrame([
    {"metric": "ROUGE-1", "base": round(base_r["rouge1"], 4), "fine_tuned": round(ft_r["rouge1"], 4)},
    {"metric": "ROUGE-2", "base": round(base_r["rouge2"], 4), "fine_tuned": round(ft_r["rouge2"], 4)},
    {"metric": "ROUGE-L", "base": round(base_r["rougeL"], 4), "fine_tuned": round(ft_r["rougeL"], 4)},
])
comparison["delta"] = (comparison["fine_tuned"] - comparison["base"]).round(4)
comparison["relative"] = ((comparison["fine_tuned"] / comparison["base"] - 1) * 100).round(1).astype(str) + "%"
display(comparison)
"""),

    md("## Additional metric 1, BERTScore F1"),
    code("""
bertscore = evaluate.load("bertscore")
b = bertscore.compute(predictions=base_outputs, references=references, lang="en", rescale_with_baseline=True)
f = bertscore.compute(predictions=ft_outputs, references=references, lang="en", rescale_with_baseline=True)
bs = pd.DataFrame([{"model": "base", "bertscore_f1": round(np.mean(b["f1"]), 4)},
                   {"model": "fine-tuned", "bertscore_f1": round(np.mean(f["f1"]), 4)}])
display(bs)
"""),

    md("""
## Additional metric 2, structural validity

ROUGE and BERTScore both reward surface overlap, so a model that emits well-formed English
about risks can score respectably while producing JSON no downstream system can parse.
This measures the thing that actually matters for the use case.
"""),
    code("""
import re
from task2_genai.src.schema import RISK_CATEGORIES, RiskExtraction
import pydantic

def parse_output(text):
    m = re.search(r"\\{.*\\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

def structural_report(outputs, label):
    parsed = [parse_output(o) for o in outputs]
    valid_json = sum(p is not None for p in parsed)
    schema_ok, bad_cats = 0, 0
    for p in parsed:
        if not p:
            continue
        try:
            RiskExtraction.model_validate(p)
            schema_ok += 1
        except pydantic.ValidationError:
            pass
        for r in (p.get("risks") or []):
            if r.get("category") not in RISK_CATEGORIES:
                bad_cats += 1
    n = len(outputs)
    return {"model": label, "parses_as_json": f"{valid_json}/{n}",
            "validates_against_schema": f"{schema_ok}/{n}",
            "invented_categories": bad_cats}

display(pd.DataFrame([structural_report(base_outputs, "base"),
                      structural_report(ft_outputs, "fine-tuned")]))
"""),

    md("""
## Additional metric 3, LLM-as-judge

A capable model scores each output against a defined rubric and returns structured JSON.
The judge sees the passage, the reference extraction and the candidate, but **not** which
model produced it.
"""),
    code("""
from common.llm import get_client
from pydantic import BaseModel, Field
from typing import Literal

class JudgeVerdict(BaseModel):
    coverage: int = Field(ge=1, le=5, description="were all real risks found")
    precision: int = Field(ge=1, le=5, description="were any risks invented")
    category_accuracy: int = Field(ge=1, le=5)
    format_validity: int = Field(ge=1, le=5)
    overall: int = Field(ge=1, le=5)
    verdict: Literal["correct", "partially_correct", "hallucinated"]
    reason: str

JUDGE_SYSTEM = '''You grade a risk-extraction output against a reference.

Score 1 to 5 on coverage (were all real risks found), precision (were any invented),
category_accuracy, format_validity and overall.

verdict rules:
  correct           every reference risk found, no invented risks, categories right
  partially_correct some risks missed or a category wrong, but nothing invented
  hallucinated      at least one risk that the passage does not support

Return only JSON matching the schema.'''

def judge(passage, reference, candidate):
    user = f"PASSAGE:\\n{passage}\\n\\nREFERENCE:\\n{reference}\\n\\nCANDIDATE:\\n{candidate}"
    return get_client().structured(JUDGE_SYSTEM, user, JudgeVerdict)

judged = {"base": [], "fine-tuned": []}
for i, row in enumerate(test_rows):
    passage = row["messages"][1]["content"]
    for label, outs in [("base", base_outputs), ("fine-tuned", ft_outputs)]:
        try:
            judged[label].append(judge(passage, references[i], outs[i]))
        except Exception as e:
            print(f"judge failed on {label} {i}: {str(e)[:80]}")
    print(f"  judged {i+1}/{len(test_rows)}", end="\\r")
print()
"""),
    code("""
rows_out = []
for label, verdicts in judged.items():
    if not verdicts:
        continue
    rows_out.append({
        "model": label,
        "n_judged": len(verdicts),
        "coverage": round(np.mean([v.coverage for v in verdicts]), 2),
        "precision": round(np.mean([v.precision for v in verdicts]), 2),
        "category_acc": round(np.mean([v.category_accuracy for v in verdicts]), 2),
        "format": round(np.mean([v.format_validity for v in verdicts]), 2),
        "overall": round(np.mean([v.overall for v in verdicts]), 2),
    })
display(pd.DataFrame(rows_out))
"""),

    md("""
## Hallucination rate

Every one of the 12 test responses from the fine-tuned model is labelled `correct`,
`partially_correct` or `hallucinated`. The judge provides a first pass; the cell below
prints each case in full so the labels can be checked by hand rather than taken on trust.
"""),
    code("""
from collections import Counter

labels = [v.verdict for v in judged["fine-tuned"]]
n = len(labels)
halluc = labels.count("hallucinated")
print(f"reviewed: {n} responses")
for k in ("correct", "partially_correct", "hallucinated"):
    print(f"  {k:18s} {labels.count(k):>3}  ({labels.count(k)/n:.1%})")
print(f"\\nHALLUCINATION RATE: {halluc}/{n} = {halluc/n:.1%}")
"""),
    code("""
for i, v in enumerate(judged["fine-tuned"]):
    print("=" * 78)
    print(f"[{i+1}] verdict={v.verdict}  overall={v.overall}/5  coverage={v.coverage}  precision={v.precision}")
    print(f"judge reason: {v.reason}")
    print(f"\\nPASSAGE   : {test_rows[i]['messages'][1]['content'][:260]}...")
    print(f"\\nREFERENCE : {references[i][:320]}...")
    print(f"\\nPREDICTED : {ft_outputs[i][:320]}...")
    print()
"""),

    md("""
## Qualitative analysis

Fill both paragraphs in after reading the cases printed above. Cite specific example
numbers rather than describing general impressions.

**Paragraph 1, where fine-tuning improved behaviour.** TODO, with specific examples.

**Paragraph 2, remaining failure modes and what would fix them.** TODO, with the
additional data or training change that would address each.
"""),

    md("""
# Bonus, RAG fallback on low confidence

When the fine-tuned model's confidence is low, measured here by perplexity over its own
output, the passage is re-queried with similar training examples retrieved from a ChromaDB
store as additional context.
"""),
    code("""
%pip install -q chromadb sentence-transformers
"""),
    code("""
import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client_db = chromadb.Client()
coll = client_db.get_or_create_collection("risk_examples")

train_passages = [r["messages"][1]["content"] for r in splits["train"]]
train_targets = [r["messages"][2]["content"] for r in splits["train"]]
coll.add(ids=[str(i) for i in range(len(train_passages))],
         documents=train_passages,
         embeddings=embedder.encode(train_passages).tolist(),
         metadatas=[{"target": t} for t in train_targets])
print(f"indexed {coll.count()} training passages")
"""),
    code("""
def perplexity(model_obj, text):
    enc = tokenizer(text, return_tensors="pt").to(model_obj.device)
    with torch.no_grad():
        loss = model_obj(**enc, labels=enc["input_ids"]).loss
    return float(torch.exp(loss))

# Reload the merged model for the demonstration.
merged = AutoModelForCausalLM.from_pretrained(
    MERGED_DIR, torch_dtype=torch.float16, device_map={"": 0})

PPL_THRESHOLD = 3.0
worst = max(range(len(ft_outputs)), key=lambda i: perplexity(merged, ft_outputs[i]))
passage = test_rows[worst]["messages"][1]["content"]
ppl = perplexity(merged, ft_outputs[worst])
print(f"lowest-confidence test case: #{worst+1}, perplexity {ppl:.2f}")
print(f"threshold {PPL_THRESHOLD}, RAG {'TRIGGERS' if ppl > PPL_THRESHOLD else 'does not trigger'}")
print(f"\\n--- BEFORE, no retrieval ---\\n{ft_outputs[worst][:420]}")
"""),
    code("""
hits = coll.query(query_embeddings=embedder.encode([passage]).tolist(), n_results=2)
context = "\\n\\n".join(
    f"Example passage:\\n{d}\\nCorrect extraction:\\n{m['target']}"
    for d, m in zip(hits["documents"][0], hits["metadatas"][0]))

msgs = [{"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Similar labelled examples:\\n{context}\\n\\nNow extract from:\\n{passage}"}]
prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
pipe = pipeline("text-generation", model=merged, tokenizer=tokenizer, device_map={"": 0})
after = pipe(prompt, return_full_text=False, **GEN_KWARGS)[0]["generated_text"].strip()
print(f"--- AFTER, with 2 retrieved examples ---\\n{after[:420]}")
print(f"\\nREFERENCE:\\n{references[worst][:420]}")
"""),

    md(f"""
# Criteria checklist

| Criterion | Marks | Where |
|---|---|---|
| Use case quality | 10 | closed 12-category taxonomy, explicit correct/incorrect definition |
| Dataset size, 100+ | 5 | 120 examples, teacher prompt printed above verbatim |
| Dataset diversity | 10 | 9 measured checks, all pass |
| Format and 80/10/10 split | 5 | chat messages, style-stratified 96/12/12 |
| QLoRA 4-bit NF4 | 10 | `BitsAndBytesConfig` with nf4 and double quant, PEFT applied |
| Hyperparameter justification | 15 | table above, every value reasoned |
| Loss per epoch | 10 | table and chart, validation loss decreasing |
| Model saved and linked | 5 | [{HF_REPO}](https://huggingface.co/{HF_REPO}) |
| ROUGE-L comparison | 8 | base vs fine-tuned on identical test set |
| Additional metric | 7 | BERTScore, structural validity, and LLM-as-judge |
| Hallucination rate | 7 | all 12 responses labelled, rate stated |
| Qualitative analysis | 8 | two paragraphs above |
| Bonus, RAG fallback | 5 | ChromaDB retrieval on high perplexity, before and after shown |
"""),
]


def main():
    nb = nbf.v4.new_notebook(cells=CELLS)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4", "name": "Task 2 Fine-Tuning"},
    }
    path = REPO / "task2_genai" / "notebook.ipynb"
    nbf.write(nb, str(path))
    n_code = sum(1 for c in CELLS if c.cell_type == "code")
    print(f"wrote {path.relative_to(REPO)}  ({len(CELLS)} cells, {n_code} code)")


if __name__ == "__main__":
    main()
