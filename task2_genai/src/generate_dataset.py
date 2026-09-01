"""Task 2A, synthetic training data generation via a teacher model.

    python -m task2_genai.src.generate_dataset --target 120

Generation is batched. One call per seed produces several examples, so 120 examples cost
roughly 30 calls rather than 120. On a free tier with a 200k daily token allowance that
difference decides whether the dataset finishes in one sitting.

Diversity is engineered rather than hoped for. Seeds sweep a grid of sector, document
style and target risk count, and near-duplicate passages are rejected on a shingle
overlap check before they can reach the dataset.
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from common import config
from common.llm import LLMError, SchemaValidationError, get_client
from task2_genai.src.schema import (
    CATEGORY_DEFINITIONS, DOC_STYLES, GeneratedExample, GenerationBatch,
    RISK_CATEGORIES, SECTORS,
)

log = logging.getLogger(__name__)

PROMPT_PATH = config.REPO_ROOT / "task2_genai" / "prompts" / "teacher_system_prompt.md"
DATA_DIR = config.REPO_ROOT / "task2_genai" / "data"

EXAMPLES_PER_CALL = 4
NEAR_DUPLICATE_THRESHOLD = 0.45
SHINGLE_SIZE = 5
PACE_SECONDS = 2.0


def load_teacher_prompt() -> str:
    """The prompt lives in one file so the submitted copy cannot drift from the used one."""
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_user_prompt(sector: str, style_key: str, risk_counts: list[int],
                      focus: list[str], n: int) -> str:
    taxonomy = "\n".join(f"  {c}: {CATEGORY_DEFINITIONS[c]}" for c in RISK_CATEGORIES)
    counts = ", ".join(str(c) for c in risk_counts)
    return f"""Sector: {sector}
Document style: {style_key}, {DOC_STYLES[style_key]}

Permitted risk categories:
{taxonomy}

Produce {n} examples for this sector and style.

Give the examples these risk counts, in order: {counts}.
Between them, make sure these categories appear at least once: {', '.join(focus)}.
Beyond those, choose whatever categories the passages genuinely support.

Each passage must describe a different business situation from the others in this batch."""


def _shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < size:
        return {" ".join(words)}
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}


def similarity(a: str, b: str) -> float:
    """Jaccard overlap on word shingles. Catches paraphrase, not just exact repeats."""
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class GenerationStats:
    calls: int = 0
    failed_calls: int = 0
    accepted: int = 0
    rejected_duplicate: int = 0
    rejected_invalid: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)
    # Which provider actually served each call. The client fails over silently, so
    # without this the dataset card could not say truthfully which teacher produced what.
    by_provider: dict = field(default_factory=dict)


def build_seeds(target: int, seed: int = 7) -> list[dict]:
    """Sweep the sector and style grid rather than sampling it.

    Random sampling at this size leaves whole sectors unrepresented, which is exactly
    the homogeneity the brief penalises.
    """
    rng = random.Random(seed)
    styles = list(DOC_STYLES)
    # Diagonal traversal of the sector-by-style grid. 15 sectors and 4 styles are
    # coprime, so stepping both indices together covers every sector and every style
    # early without repeating a pair until the grid is exhausted. Shuffling missed
    # sectors entirely at 30 calls; sector-major ordering missed half the styles.
    pairs = [(SECTORS[i % len(SECTORS)], styles[i % len(styles)])
             for i in range(len(SECTORS) * len(styles))]

    seeds, produced, i = [], 0, 0
    while produced < target:
        sector, style = pairs[i % len(pairs)]
        n = min(EXAMPLES_PER_CALL, target - produced)
        counts = [rng.choice([1, 2, 2, 3, 3, 4]) for _ in range(n)]
        # Rotate the focus window so every category is requested across the run.
        start = (i * 3) % len(RISK_CATEGORIES)
        focus = [RISK_CATEGORIES[(start + k) % len(RISK_CATEGORIES)] for k in range(3)]
        seeds.append({"sector": sector, "style": style, "counts": counts,
                      "focus": focus, "n": n})
        produced += n
        i += 1
    return seeds


def generate(target: int = 120, out_path: Path | None = None,
             pace: float = PACE_SECONDS) -> tuple[list[GeneratedExample], GenerationStats]:
    client = get_client(max_tokens=4096, temperature=0.9)
    system = load_teacher_prompt()
    stats = GenerationStats()
    started = time.perf_counter()

    accepted: list[GeneratedExample] = []
    passages: list[str] = []

    for index, seed in enumerate(build_seeds(target), 1):
        user = build_user_prompt(seed["sector"], seed["style"], seed["counts"],
                                 seed["focus"], seed["n"])
        stats.calls += 1
        try:
            batch = client.structured(system, user, GenerationBatch)
        except (SchemaValidationError, LLMError) as exc:
            stats.failed_calls += 1
            stats.errors.append(f"seed {index} ({seed['sector']}): {str(exc)[:120]}")
            log.warning("seed %d failed: %s", index, str(exc)[:160])
            continue

        provider = getattr(client, "last_provider", "unknown")
        stats.by_provider[provider] = stats.by_provider.get(provider, 0) + len(batch.examples)

        for example in batch.examples:
            example.sector = seed["sector"]
            example.doc_style = seed["style"]
            example.teacher_provider = provider

            dup = max((similarity(example.passage, p) for p in passages), default=0.0)
            if dup >= NEAR_DUPLICATE_THRESHOLD:
                stats.rejected_duplicate += 1
                log.info("rejected near-duplicate (overlap %.2f)", dup)
                continue

            accepted.append(example)
            passages.append(example.passage)
            stats.accepted += 1

        print(f"  [{index:>2}] {seed['sector'][:26]:26s} {seed['style']:16s} "
              f"accepted {stats.accepted:>3}/{target}", flush=True)

        if stats.accepted >= target:
            break
        if pace:
            time.sleep(pace)

    stats.elapsed_s = time.perf_counter() - started

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for example in accepted:
                fh.write(json.dumps(example.model_dump(), ensure_ascii=False) + "\n")
    return accepted, stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate the Task 2 training dataset")
    parser.add_argument("--target", type=int, default=120)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "raw_examples.jsonl")
    parser.add_argument("--pace", type=float, default=PACE_SECONDS)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if not config.available_providers():
        print("No LLM key found. Set GROQ_API_KEY in .env.", file=sys.stderr)
        return 2

    print(f"teacher model : {config.GROQ_MODEL}")
    print(f"target        : {args.target} examples, {EXAMPLES_PER_CALL} per call")
    print(f"prompt        : {PROMPT_PATH.relative_to(config.REPO_ROOT)}")
    print()

    examples, stats = generate(args.target, args.out, args.pace)

    print()
    print(f"accepted           : {stats.accepted}")
    print(f"calls made         : {stats.calls} ({stats.failed_calls} failed)")
    print(f"near-duplicates    : {stats.rejected_duplicate} rejected")
    print(f"teacher providers  : {stats.by_provider}")
    print(f"elapsed            : {stats.elapsed_s:.1f}s")
    print(f"written to         : {args.out}")
    if stats.errors:
        print(f"errors             : {len(stats.errors)}")
        for e in stats.errors[:5]:
            print(f"  {e}")
    return 0 if examples else 1


if __name__ == "__main__":
    raise SystemExit(main())
