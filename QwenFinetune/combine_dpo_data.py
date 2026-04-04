"""
Combine format-preference and quality-preference DPO data into a single training set.

Reads:
  data/dpo_format_train_cot.jsonl   (format preference pairs)
  data/dpo_format_eval_cot.jsonl
  data/dpo_refine_train_cot.jsonl   (quality preference pairs)
  data/dpo_refine_eval_cot.jsonl

Writes:
  data/dpo_combined_train_cot.jsonl
  data/dpo_combined_eval_cot.jsonl
  data/dataset_stats_dpo_combined.json

Usage:
  python combine_dpo_data.py
  python combine_dpo_data.py --seed 42
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def write_jsonl(samples: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(samples)} -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = SCRIPT_DIR / args.data_dir

    # Load all sources
    format_train = load_jsonl(data_dir / "dpo_format_train_cot.jsonl")
    format_eval = load_jsonl(data_dir / "dpo_format_eval_cot.jsonl")
    quality_train = load_jsonl(data_dir / "dpo_refine_train_cot.jsonl")
    quality_eval = load_jsonl(data_dir / "dpo_refine_eval_cot.jsonl")

    print(f"Format  train={len(format_train)}, eval={len(format_eval)}")
    print(f"Quality train={len(quality_train)}, eval={len(quality_eval)}")

    # Tag source type
    for s in format_train + format_eval:
        s["dpo_source"] = "format"
    for s in quality_train + quality_eval:
        s["dpo_source"] = "quality"

    # Combine and shuffle
    combined_train = format_train + quality_train
    combined_eval = format_eval + quality_eval
    random.shuffle(combined_train)
    random.shuffle(combined_eval)

    print(f"\nCombined train={len(combined_train)}, eval={len(combined_eval)}")

    # Write
    write_jsonl(combined_train, data_dir / "dpo_combined_train_cot.jsonl")
    write_jsonl(combined_eval, data_dir / "dpo_combined_eval_cot.jsonl")

    # Stats
    source_dist_train = Counter(s.get("dpo_source", "unknown") for s in combined_train)
    source_dist_eval = Counter(s.get("dpo_source", "unknown") for s in combined_eval)
    corruption_dist = Counter(
        s.get("corruption_type", "quality_preference")
        for s in combined_train + combined_eval
    )

    stats = {
        "combined_train": len(combined_train),
        "combined_eval": len(combined_eval),
        "source_dist_train": dict(sorted(source_dist_train.items())),
        "source_dist_eval": dict(sorted(source_dist_eval.items())),
        "corruption_type_dist": dict(sorted(corruption_dist.items())),
    }
    stats_path = data_dir / "dataset_stats_dpo_combined.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Stats -> {stats_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
