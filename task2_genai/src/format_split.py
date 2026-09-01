"""Task 2A, chat formatting and the 80/10/10 split.

Examples are stored as a `messages` list rather than a pre-rendered string. The chat
template belongs to the tokenizer, so rendering it here would bake one model's special
tokens into the dataset and make it unusable if the base model changes.

The split is stratified by document style. A random split of 120 examples can leave a
style out of the 12-example test set entirely, which would make the held-out score a
measurement of luck rather than of the model.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from common import config
from task2_genai.src.schema import CATEGORY_DEFINITIONS, RISK_CATEGORIES

DATA_DIR = config.REPO_ROOT / "task2_genai" / "data"
SPLIT_RATIOS = (0.8, 0.1, 0.1)
SEED = 13

# The prompt the fine-tuned model is served with. Deliberately short: the point of
# fine-tuning is to move the taxonomy and output shape into the weights, so a long
# instruction at inference time would hide whether that actually happened.
STUDENT_SYSTEM_PROMPT = (
    "You extract financial risks from corporate disclosure text. "
    "Return only a JSON object with a \"risks\" array. Each risk has: category, summary, "
    "trigger, potential_impact, severity. "
    "category must be one of: " + ", ".join(RISK_CATEGORIES) + ". "
    "severity must be one of: high, medium, low."
)

BASELINE_SYSTEM_PROMPT = (
    STUDENT_SYSTEM_PROMPT
    + "\n\nCategory definitions:\n"
    + "\n".join(f"- {k}: {v}" for k, v in CATEGORY_DEFINITIONS.items())
    + "\n\nExtract every distinct risk the passage raises and nothing else. Merge "
      "sentences describing the same exposure into a single entry."
)


def to_messages(row: dict, system: str = STUDENT_SYSTEM_PROMPT) -> list[dict]:
    target = {"risks": [r for r in row["extraction"]["risks"]]}
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": row["passage"]},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
    ]


def stratified_split(rows: list[dict], ratios=SPLIT_RATIOS, seed: int = SEED):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for row in rows:
        buckets[row.get("doc_style", "unknown")].append(row)

    train, val, test = [], [], []
    for style in sorted(buckets):
        group = buckets[style][:]
        rng.shuffle(group)
        n = len(group)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        train += group[:n_train]
        val += group[n_train:n_train + n_val]
        test += group[n_train + n_val:]

    for part in (train, val, test):
        rng.shuffle(part)
    return train, val, test


def write_split(rows: list[dict], path: Path, system: str = STUDENT_SYSTEM_PROMPT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            record = {
                "messages": to_messages(row, system),
                "meta": {"sector": row.get("sector", ""),
                         "doc_style": row.get("doc_style", ""),
                         "n_risks": len(row["extraction"]["risks"])},
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def describe(name: str, rows: list[dict]) -> str:
    styles = Counter(r.get("doc_style", "?") for r in rows)
    cats = Counter(x["category"] for r in rows for x in r["extraction"]["risks"])
    risks = sum(len(r["extraction"]["risks"]) for r in rows)
    return (f"{name:6s} {len(rows):>4} examples  {risks:>4} risk items  "
            f"{len(cats):>2}/{len(RISK_CATEGORIES)} categories  "
            f"styles={dict(sorted(styles.items()))}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Format and split the Task 2 dataset")
    parser.add_argument("--src", type=Path, default=DATA_DIR / "raw_examples.jsonl")
    args = parser.parse_args(argv)

    rows = [json.loads(l) for l in args.src.read_text(encoding="utf-8").splitlines() if l.strip()]
    train, val, test = stratified_split(rows)

    total = len(train) + len(val) + len(test)
    assert total == len(rows), f"split lost examples: {total} vs {len(rows)}"
    assert not ({id(r) for r in train} & {id(r) for r in val} & {id(r) for r in test})

    for name, part in [("train", train), ("val", val), ("test", test)]:
        write_split(part, DATA_DIR / f"{name}.jsonl")

    print(f"source: {len(rows)} examples\n")
    print(describe("train", train))
    print(describe("val", val))
    print(describe("test", test))
    print(f"\nratios: {len(train)/total:.0%} / {len(val)/total:.0%} / {len(test)/total:.0%}")
    print(f"\nwritten to {DATA_DIR.relative_to(config.REPO_ROOT)}/"
          + "{train,val,test}.jsonl")

    sample = json.loads((DATA_DIR / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    print("\n--- one training record ---")
    for m in sample["messages"]:
        body = m["content"]
        print(f"[{m['role']}] {body[:220]}{'...' if len(body) > 220 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
