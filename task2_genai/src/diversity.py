"""Task 2A, dataset diversity analysis.

The brief awards zero for diversity if most examples are minor variations of one
scenario, so this measures it rather than asserting it. Pairwise passage similarity is
reported as a distribution, not just a maximum, because a dataset can have no exact
duplicates and still be homogeneous.
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from common import config
from task2_genai.src.generate_dataset import similarity
from task2_genai.src.schema import RISK_CATEGORIES

DATA_DIR = config.REPO_ROOT / "task2_genai" / "data"

STOPWORDS = {
    "the", "and", "for", "that", "with", "our", "are", "may", "not", "which", "from",
    "this", "have", "has", "was", "were", "will", "would", "could", "any", "all", "its",
    "we", "us", "of", "to", "in", "on", "a", "an", "is", "be", "as", "by", "or", "at",
    "if", "it", "their", "these", "those", "such", "than", "then", "there", "been",
    "company", "group", "also", "other", "more", "most", "some", "over", "into", "but",
    "during", "under", "where", "when", "while", "can", "including", "further", "no",
}


def load(path: Path | None = None) -> list[dict]:
    path = path or DATA_DIR / "raw_examples.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOPWORDS]


def length_stats(rows: list[dict]) -> dict:
    lengths = np.array([len(r["passage"].split()) for r in rows])
    return {
        "count": int(len(lengths)),
        "mean": round(float(lengths.mean()), 1),
        "std": round(float(lengths.std(ddof=1)), 1),
        "min": int(lengths.min()),
        "p25": int(np.percentile(lengths, 25)),
        "median": int(np.median(lengths)),
        "p75": int(np.percentile(lengths, 75)),
        "max": int(lengths.max()),
    }


def similarity_stats(rows: list[dict], sample: int = 120) -> dict:
    """Distribution of pairwise passage overlap.

    A dataset with no exact duplicates can still be homogeneous, so the mean and the
    upper tail matter more than the maximum alone.
    """
    passages = [r["passage"] for r in rows][:sample]
    scores = [similarity(passages[i], passages[j])
              for i in range(len(passages)) for j in range(i + 1, len(passages))]
    arr = np.array(scores) if scores else np.array([0.0])
    return {
        "pairs_compared": int(len(scores)),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "max": round(float(arr.max()), 4),
        "above_0.30": int((arr > 0.30).sum()),
        "above_0.45": int((arr > 0.45).sum()),
    }


def vocabulary_stats(rows: list[dict]) -> dict:
    all_tokens = [t for r in rows for t in _tokens(r["passage"])]
    counts = Counter(all_tokens)
    total = sum(counts.values())
    return {
        "total_tokens": total,
        "unique_tokens": len(counts),
        "type_token_ratio": round(len(counts) / total, 4) if total else 0.0,
        "hapax_legomena": sum(1 for c in counts.values() if c == 1),
        "top_terms": counts.most_common(25),
    }


def category_stats(rows: list[dict]) -> dict:
    cats = Counter(risk["category"] for r in rows for risk in r["extraction"]["risks"])
    per_example = [len(r["extraction"]["risks"]) for r in rows]
    missing = [c for c in RISK_CATEGORIES if c not in cats]
    total = sum(cats.values())
    return {
        "total_risk_items": total,
        "categories_present": len(cats),
        "categories_missing": missing,
        "distribution": dict(cats.most_common()),
        "share_of_largest": round(max(cats.values()) / total, 4) if total else 0.0,
        "risks_per_example": dict(sorted(Counter(per_example).items())),
        "mean_risks_per_example": round(float(np.mean(per_example)), 2),
    }


def coverage_stats(rows: list[dict]) -> dict:
    return {
        "sectors": dict(Counter(r["sector"] for r in rows).most_common()),
        "doc_styles": dict(Counter(r["doc_style"] for r in rows).most_common()),
        "teacher_providers": dict(Counter(r.get("teacher_provider", "unknown") for r in rows)),
    }


def report(rows: list[dict]) -> dict:
    return {
        "examples": len(rows),
        "passage_length_words": length_stats(rows),
        "pairwise_similarity": similarity_stats(rows),
        "vocabulary": vocabulary_stats(rows),
        "risk_categories": category_stats(rows),
        "coverage": coverage_stats(rows),
    }


def verdict(rep: dict) -> list[str]:
    """Hard checks a reviewer would apply, stated as pass or fail rather than prose."""
    out = []
    sim = rep["pairwise_similarity"]
    cat = rep["risk_categories"]
    vocab = rep["vocabulary"]
    length = rep["passage_length_words"]
    cov = rep["coverage"]

    def line(ok, text):
        out.append(f"{'PASS' if ok else 'FAIL'}  {text}")

    line(rep["examples"] >= 100, f"at least 100 examples ({rep['examples']})")
    line(sim["mean"] < 0.10, f"mean pairwise similarity below 0.10 ({sim['mean']})")
    line(sim["above_0.45"] == 0, f"no near-duplicate pairs above 0.45 ({sim['above_0.45']})")
    line(not cat["categories_missing"],
         f"all {len(RISK_CATEGORIES)} categories present ({cat['categories_present']})")
    line(cat["share_of_largest"] < 0.30,
         f"no category exceeds 30 percent of items ({cat['share_of_largest']:.1%})")
    line(vocab["type_token_ratio"] > 0.15,
         f"type-token ratio above 0.15 ({vocab['type_token_ratio']})")
    line(length["std"] > 10, f"passage length varies (std {length['std']} words)")
    line(len(cov["sectors"]) >= 12, f"at least 12 sectors represented ({len(cov['sectors'])})")
    line(len(cov["doc_styles"]) == 4, f"all 4 document styles present ({len(cov['doc_styles'])})")
    return out


def render(rep: dict) -> str:
    lines = [f"Dataset diversity report: {rep['examples']} examples", ""]
    L = rep["passage_length_words"]
    lines += [
        "Passage length in words",
        f"  mean {L['mean']}  std {L['std']}  min {L['min']}  "
        f"p25 {L['p25']}  median {L['median']}  p75 {L['p75']}  max {L['max']}",
        "",
    ]
    S = rep["pairwise_similarity"]
    lines += [
        f"Pairwise passage similarity over {S['pairs_compared']} pairs",
        f"  mean {S['mean']}  median {S['median']}  p95 {S['p95']}  max {S['max']}",
        f"  pairs above 0.30: {S['above_0.30']}   above 0.45: {S['above_0.45']}",
        "",
    ]
    V = rep["vocabulary"]
    lines += [
        "Vocabulary",
        f"  {V['total_tokens']} tokens, {V['unique_tokens']} unique, "
        f"type-token ratio {V['type_token_ratio']}, {V['hapax_legomena']} appear once",
        "  top terms: " + ", ".join(f"{w}({c})" for w, c in V["top_terms"][:15]),
        "",
    ]
    C = rep["risk_categories"]
    lines += [f"Risk categories, {C['total_risk_items']} items across {rep['examples']} examples"]
    for name, count in C["distribution"].items():
        bar = "#" * max(1, round(count / max(C["distribution"].values()) * 34))
        lines.append(f"  {name:20s} {count:>4}  {bar}")
    if C["categories_missing"]:
        lines.append(f"  MISSING: {C['categories_missing']}")
    lines += ["", f"  risks per example: {C['risks_per_example']} "
                  f"(mean {C['mean_risks_per_example']})", ""]

    cov = rep["coverage"]
    lines += ["Coverage",
              f"  sectors     : {len(cov['sectors'])} distinct",
              f"  doc styles  : {cov['doc_styles']}",
              f"  teacher     : {cov['teacher_providers']}", ""]
    lines += ["Diversity checks"] + ["  " + v for v in verdict(rep)]
    return "\n".join(lines)


def main() -> int:
    rows = load()
    rep = report(rows)
    print(render(rep))
    out = DATA_DIR / "diversity_report.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nwritten to {out.relative_to(config.REPO_ROOT)}")
    return 0 if all(v.startswith("PASS") for v in verdict(rep)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
