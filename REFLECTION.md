# Reflection

## Architectural decisions

**Keep numbers out of the model.** The recurring decision across all three tasks was to
compute figures in code and ask the LLM only to reason about them. Task 1's momentum
signal is a deterministic four-component rule, so a sampling failure cannot silently
change a number in the report. Task 3's agent handoff parses figures from tool payloads
in Python. I learned that one the hard way: my first version asked the model to transcribe
them and it returned an entirely null brief that validated cleanly and told the receiving
agent nothing.

**Enforce constraints structurally, not by prompt.** Task 3B restricts each agent's tools
by constructing it with its own tool list, so Agent B holds no reference by which it could
reach a price tool. A test asserts that requesting a forbidden tool returns `unknown tool`
rather than data. A prompt saying "do not use web search" is a request; this is a
guarantee.

**Share the pipeline across tasks.** Task 3's tools import Task 1's indicators and news
code rather than reimplementing them, so the agent and the research brief cannot disagree
about the same ticker.

**Validate every model output.** Pydantic schemas with a self-repair pass sit on every
LLM call. This looked like overhead until it repeatedly saved runs: Groq once emitted
`"confidence": 0. nine`, the fallback provider returned a differently shaped payload, and
the Task 2C judge kept omitting a required field. All three recovered automatically.

## What I would improve with more time

**A larger test set.** With twelve held-out examples, one response changing category moves
the hallucination rate by 8.3 points. Every Task 2C number carries wide confidence
intervals, and that is the first thing I would fix.

**Balance the training data by category.** `concentration_risk` appears 40 times against
11 each for `liquidity_risk` and `environmental_risk`. Category accuracy scored 3.50 of 5
against a perfect 5.00 for format, which points directly at that imbalance rather than at
insufficient data overall.

**Validate against real filings.** All 120 training examples come from one teacher model,
so the student may be fitting that teacher's phrasing rather than the underlying task.

**Batch the sentiment calls.** One call per headline is what the brief specifies, and also
the largest consumer of the token budget.

## Limitations encountered

**No local GPU.** An Intel Celeron with 8 GB RAM and no CUDA meant Task 2 could only run on
Colab, which shaped the plan: dataset generation locally against an API, training remotely.

**Free-tier token limits changed the design.** Groq's 200,000 daily and 8,000 per-minute
caps were hit repeatedly. That forced agent context budgeting, batched dataset generation
at four examples per call rather than one, and a second provider as fallback. Groq had also
retired every Llama model, so the planned teacher was unavailable and MiniMax-M3 generated
all 120 examples through the fallback. Each record carries a `teacher_provider` field
rather than the documentation claiming a teacher that never ran.

**Transformers 5.x ecosystem drift cost most of the debugging time.** Four separate
failures: the Hub's `modeling_phi3.py` reading a config key that 5.x renamed; `trl 1.12`
renaming three config fields; an unspecified model dtype defaulting to bfloat16 that the
fp16 gradient scaler cannot unscale; and `SFTTrainer` patching the model in place, so its
failure survived into a plain `Trainer` until the model was rebuilt. I replaced TRL with
`transformers.Trainer` rather than pin an old version and inherit its other bugs.

**Two out-of-memory errors**, both from an orphaned model left bound after a failed
operation rather than from the training itself, which peaked at 3.85 GB of 15 GB.
