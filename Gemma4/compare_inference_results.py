"""
Compare Gemma 4 inference results across different configurations.

Compares 2+ JSONL result files (matched by record id) and reports:
  1. Format compliance rate
  2. Word-level statistics (avg word count, vocabulary richness)
  3. N-gram overlap (bigram Jaccard between corresponding prompts)
  4. Intra-group diversity (how different the 5 prompts are within each group)
  5. Structural patterns (camera, lighting, composition, style terms)
  6. Quality constraints (forbidden phrases, quality hints)
  7. Keyword coverage (LP-relevant keywords in prompts)

Usage:
  python Gemma4/compare_inference_results.py \\
      --files path/to/cot.jsonl "CoT" \\
             path/to/nocot.jsonl "NoCot" \\
             path/to/nocot_trunc.jsonl "NoCot+Trunc" \\
      --output_report Gemma4/results/comparison_report.json \\
      --per_sample Gemma4/results/comparison_per_sample.jsonl
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path



# ---------------------------------------------------------------------------
# Helpers (replace numpy dependency)
# ---------------------------------------------------------------------------

def _mean(values):
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOPWORDS = {
    "this", "that", "with", "from", "have", "been", "will", "your",
    "they", "their", "them", "what", "when", "where", "which", "while",
    "also", "about", "more", "into", "than", "then", "some", "such",
    "each", "must", "very", "just", "like", "over", "only", "other",
    "would", "could", "should", "those", "these", "does", "done",
    "being", "were", "here", "there", "through", "during", "before",
    "after", "under", "between", "without",
}

FORBIDDEN_PHRASES = [
    "agi", "watermark", "logo", "advertisement", "promo",
    "promotion", "stock photo", "studio backdrop",
]

QUALITY_HINTS = [
    "sharp focus", "clean composition", "correct anatomy",
    "natural hands", "no extra text", "no logos",
]

STRUCTURAL_PATTERNS = {
    "camera_terms": [
        "cinematic", "wide-angle", "close-up", "macro", "aerial",
        "overhead", "bird's-eye", "eye-level", "low-angle", "high-angle",
        "telephoto", "panoramic", "pov", "medium shot", "full-body",
    ],
    "lighting_terms": [
        "natural light", "golden hour", "soft light", "backlit",
        "ambient", "warm light", "diffused", "studio lighting",
        "morning light", "sunset", "overcast", "soft shadow",
        "dramatic lighting", "rim light",
    ],
    "composition_terms": [
        "depth of field", "shallow depth", "bokeh", "foreground",
        "background", "symmetry", "framing", "negative space",
        "rule of thirds", "leading lines",
    ],
    "style_terms": [
        "photorealistic", "editorial", "lifestyle", "documentary",
        "candid", "portrait", "still life", "flat lay", "minimalist",
        "cinematic", "hyper-realistic", "8k", "4k", "high-end",
    ],
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def tokenize(text):
    return text.lower().split()


def get_bigrams(text):
    words = tokenize(text)
    if len(words) < 2:
        return set()
    return set(zip(words[:-1], words[1:]))


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


# ---------------------------------------------------------------------------
# Per-sample metric functions
# ---------------------------------------------------------------------------

def compute_word_stats(prompts):
    word_counts = [len(tokenize(p)) for p in prompts]
    all_words = tokenize(" ".join(prompts))
    unique_words = set(all_words)
    return {
        "avg_word_count": _mean(word_counts),
        "min_word_count": min(word_counts),
        "max_word_count": max(word_counts),
        "total_words": sum(word_counts),
        "vocab_richness": len(unique_words) / len(all_words) if all_words else 0,
    }


def compute_ngram_overlap_aligned(prompts_a, prompts_b):
    """Bigram Jaccard between corresponding prompts (Prompt1 vs Prompt1, etc.)."""
    n = min(len(prompts_a), len(prompts_b))
    scores = []
    for i in range(n):
        bg_a = get_bigrams(prompts_a[i])
        bg_b = get_bigrams(prompts_b[i])
        scores.append(jaccard(bg_a, bg_b))
    return _mean(scores) if scores else 0.0


def compute_ngram_overlap_best_match(prompts_a, prompts_b):
    """For each prompt in A, find the most similar prompt in B (by bigram Jaccard)."""
    bigrams_a = [get_bigrams(p) for p in prompts_a]
    bigrams_b = [get_bigrams(p) for p in prompts_b]
    scores = []
    for bg_a in bigrams_a:
        best = max(jaccard(bg_a, bg_b) for bg_b in bigrams_b) if bigrams_b else 0.0
        scores.append(best)
    return _mean(scores) if scores else 0.0


def compute_self_diversity(prompts):
    """How diverse are the 5 prompts within the group (1 = all different, 0 = all same)."""
    bigrams = [get_bigrams(p) for p in prompts]
    pairs = list(combinations(range(len(bigrams)), 2))
    if not pairs:
        return 0.0
    sims = [jaccard(bigrams[i], bigrams[j]) for i, j in pairs]
    return 1.0 - _mean(sims)


def compute_structural_patterns(prompts):
    combined = " ".join(prompts).lower()
    result = {}
    for category, terms in STRUCTURAL_PATTERNS.items():
        count = sum(1 for t in terms if t in combined)
        result[category] = count
    return result


def compute_quality_constraints(prompts):
    n_forbidden = 0
    n_quality = 0
    for p in prompts:
        p_lower = p.lower()
        if any(f in p_lower for f in FORBIDDEN_PHRASES):
            n_forbidden += 1
        if any(h in p_lower for h in QUALITY_HINTS):
            n_quality += 1
    return {
        "prompts_with_forbidden": n_forbidden,
        "prompts_with_quality_hints": n_quality,
    }


def extract_keywords_from_prompts(all_prompts_lists):
    """Extract LP-relevant keywords from the union of all prompt sets."""
    combined = " ".join(
        p for prompts in all_prompts_lists for p in prompts
    )
    words = set(
        w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", combined)
        if w.lower() not in STOPWORDS
    )
    return words


def compute_keyword_coverage(prompts, keywords):
    if not keywords:
        return 0.0
    combined = " ".join(prompts).lower()
    covered = sum(1 for kw in keywords if kw in combined)
    return covered / len(keywords)


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def run_comparison(file_label_pairs):
    labels = [label for _, label in file_label_pairs]

    # Load all files
    data = {}
    for path, label in file_label_pairs:
        records = load_jsonl(path)
        data[label] = {r.get("id", ""): r for r in records if r.get("id")}
        print(f"  Loaded {label}: {len(records)} records from {path}")

    # Find matched ids (present in ALL files)
    all_ids = [set(data[l].keys()) for l in labels]
    matched_ids = sorted(set.intersection(*all_ids)) if all_ids else []
    total_per_file = {l: len(data[l]) for l in labels}
    print(f"  Matched samples: {len(matched_ids)}")

    if not matched_ids:
        print("[ERROR] No matched IDs across files.")
        return None, None

    # Per-sample computation
    per_sample_results = []
    agg = {
        "format_compliance": {l: [] for l in labels},
        "word_stats": {l: defaultdict(list) for l in labels},
        "self_diversity": {l: [] for l in labels},
        "structural_patterns": {l: defaultdict(list) for l in labels},
        "quality_constraints": {l: defaultdict(list) for l in labels},
        "keyword_coverage": {l: [] for l in labels},
        "ngram_aligned": defaultdict(list),
        "ngram_best_match": defaultdict(list),
    }

    for rid in matched_ids:
        sample_result = {"id": rid}
        records = {l: data[l][rid] for l in labels}

        # Get prompts per label (skip if not exactly 5)
        prompts_map = {}
        skip = False
        for l in labels:
            ps = records[l].get("generated_prompts", [])
            if len(ps) != 5:
                skip = True
                break
            prompts_map[l] = ps

        if skip:
            # Still count format compliance
            for l in labels:
                agg["format_compliance"][l].append(
                    records[l].get("format_compliant", False)
                )
            continue

        # 1. Format compliance
        for l in labels:
            agg["format_compliance"][l].append(
                records[l].get("format_compliant", False)
            )

        # 2. Word stats
        ws = {}
        for l in labels:
            ws[l] = compute_word_stats(prompts_map[l])
            for k, v in ws[l].items():
                agg["word_stats"][l][k].append(v)
        sample_result["word_stats"] = ws

        # 3. N-gram overlap (pairwise between labels)
        ngram = {}
        for i, l1 in enumerate(labels):
            for l2 in labels[i + 1:]:
                key = f"{l1}_vs_{l2}"
                aligned = compute_ngram_overlap_aligned(prompts_map[l1], prompts_map[l2])
                best = compute_ngram_overlap_best_match(prompts_map[l1], prompts_map[l2])
                ngram[key] = {"aligned": round(aligned, 4), "best_match": round(best, 4)}
                agg["ngram_aligned"][key].append(aligned)
                agg["ngram_best_match"][key].append(best)
        sample_result["ngram_overlap"] = ngram

        # 4. Self-diversity
        sd = {}
        for l in labels:
            d = compute_self_diversity(prompts_map[l])
            sd[l] = round(d, 4)
            agg["self_diversity"][l].append(d)
        sample_result["self_diversity"] = sd

        # 5. Structural patterns
        sp = {}
        for l in labels:
            sp[l] = compute_structural_patterns(prompts_map[l])
            for k, v in sp[l].items():
                agg["structural_patterns"][l][k].append(v)
        sample_result["structural_patterns"] = sp

        # 6. Quality constraints
        qc = {}
        for l in labels:
            qc[l] = compute_quality_constraints(prompts_map[l])
            for k, v in qc[l].items():
                agg["quality_constraints"][l][k].append(v)
        sample_result["quality_constraints"] = qc

        # 7. Keyword coverage (extract from union of all prompts)
        all_prompts = [prompts_map[l] for l in labels]
        keywords = extract_keywords_from_prompts(all_prompts)
        kc = {}
        for l in labels:
            cov = compute_keyword_coverage(prompts_map[l], keywords)
            kc[l] = round(cov, 4)
            agg["keyword_coverage"][l].append(cov)
        sample_result["keyword_coverage"] = kc

        per_sample_results.append(sample_result)

    # Aggregate
    report = {
        "meta": {
            "files": {l: p for p, l in file_label_pairs},
            "matched_samples": len(matched_ids),
            "compared_samples": len(per_sample_results),
            "total_per_file": total_per_file,
        },
        "format_compliance": {},
        "word_stats": {},
        "self_diversity": {},
        "structural_patterns": {},
        "quality_constraints": {},
        "keyword_coverage": {},
        "ngram_overlap": {"aligned": {}, "best_match": {}},
    }

    for l in labels:
        fc = agg["format_compliance"][l]
        report["format_compliance"][l] = {
            "rate": round(sum(fc) / len(fc) * 100, 1) if fc else 0,
            "count": sum(fc),
            "total": len(fc),
        }

        ws = agg["word_stats"][l]
        report["word_stats"][l] = {
            k: round(float(_mean(v)), 1) for k, v in ws.items()
        }

        sd = agg["self_diversity"][l]
        report["self_diversity"][l] = round(float(_mean(sd)), 4) if sd else 0

        sp = agg["structural_patterns"][l]
        report["structural_patterns"][l] = {
            k: round(float(_mean(v)), 2) for k, v in sp.items()
        }

        qc = agg["quality_constraints"][l]
        report["quality_constraints"][l] = {
            k: round(float(_mean(v)), 2) for k, v in qc.items()
        }

        kc = agg["keyword_coverage"][l]
        report["keyword_coverage"][l] = round(float(_mean(kc)), 4) if kc else 0

    for key in agg["ngram_aligned"]:
        report["ngram_overlap"]["aligned"][key] = round(
            float(_mean(agg["ngram_aligned"][key])), 4
        )
        report["ngram_overlap"]["best_match"][key] = round(
            float(_mean(agg["ngram_best_match"][key])), 4
        )

    return report, per_sample_results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_comparison_table(report):
    labels = list(report["meta"]["files"].keys())
    col_w = max(len(l) for l in labels) + 2
    col_w = max(col_w, 14)
    header = " " * 28 + "".join(l.rjust(col_w) for l in labels)

    print(f"\n{'=' * 70}")
    print(f"Comparison Report")
    print(f"{'=' * 70}")
    print(f"Matched samples: {report['meta']['matched_samples']}")
    print(f"Compared samples: {report['meta']['compared_samples']}")
    print()
    print(header)
    print(" " * 28 + "-" * (col_w * len(labels)))

    # Format compliance
    vals = [f"{report['format_compliance'][l]['rate']}%" for l in labels]
    print(f"{'Format compliant rate':<28}" + "".join(v.rjust(col_w) for v in vals))

    # Word stats
    print()
    for key, desc in [
        ("avg_word_count", "Avg words/prompt"),
        ("vocab_richness", "Vocabulary richness"),
        ("total_words", "Total words/sample"),
    ]:
        vals = [f"{report['word_stats'][l].get(key, 0):.1f}" for l in labels]
        print(f"{desc:<28}" + "".join(v.rjust(col_w) for v in vals))

    # Self-diversity
    print()
    vals = [f"{report['self_diversity'][l]:.3f}" for l in labels]
    print(f"{'Intra-group diversity':<28}" + "".join(v.rjust(col_w) for v in vals))

    # Structural patterns
    print()
    for cat, cat_label in [
        ("camera_terms", "Camera terms"),
        ("lighting_terms", "Lighting terms"),
        ("composition_terms", "Composition terms"),
        ("style_terms", "Style terms"),
    ]:
        vals = [f"{report['structural_patterns'][l].get(cat, 0):.1f}" for l in labels]
        print(f"{cat_label:<28}" + "".join(v.rjust(col_w) for v in vals))

    # Quality constraints
    print()
    for key, desc in [
        ("prompts_with_quality_hints", "Quality hints (/5)"),
        ("prompts_with_forbidden", "Forbidden phrases (/5)"),
    ]:
        vals = [f"{report['quality_constraints'][l].get(key, 0):.2f}" for l in labels]
        print(f"{desc:<28}" + "".join(v.rjust(col_w) for v in vals))

    # Keyword coverage
    print()
    vals = [f"{report['keyword_coverage'][l]:.3f}" for l in labels]
    print(f"{'Keyword coverage':<28}" + "".join(v.rjust(col_w) for v in vals))

    # N-gram overlap matrix
    if report["ngram_overlap"]["aligned"]:
        print()
        print("Cross-file Bigram Jaccard (aligned / best-match):")
        pair_col_w = max(col_w, 20)
        pair_header = " " * 12
        pairs = list(report["ngram_overlap"]["aligned"].keys())
        for pair in pairs:
            pair_header += pair.rjust(pair_col_w)
        print(pair_header)
        row1 = "  aligned:  "
        row2 = "  best-match:"
        for pair in pairs:
            row1 += f"{report['ngram_overlap']['aligned'][pair]:.4f}".rjust(pair_col_w)
            row2 += f"{report['ngram_overlap']['best_match'][pair]:.4f}".rjust(pair_col_w)
        print(row1)
        print(row2)

    print(f"\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Gemma 4 inference results across configurations"
    )
    parser.add_argument(
        "--files", nargs="+", required=True,
        help="Alternating pairs: <path> <label> <path> <label> ..."
    )
    parser.add_argument("--output_report", default="", help="Save JSON report to this path")
    parser.add_argument("--per_sample", default="", help="Save per-sample JSONL to this path")
    args = parser.parse_args()

    # Parse file-label pairs
    if len(args.files) % 2 != 0:
        print("[ERROR] --files requires alternating <path> <label> pairs")
        sys.exit(1)

    file_label_pairs = []
    for i in range(0, len(args.files), 2):
        path = args.files[i]
        label = args.files[i + 1]
        if not Path(path).exists():
            print(f"[ERROR] File not found: {path}")
            sys.exit(1)
        file_label_pairs.append((path, label))

    if len(file_label_pairs) < 2:
        print("[ERROR] Need at least 2 files to compare")
        sys.exit(1)

    print(f"Comparing {len(file_label_pairs)} result files ...")
    report, per_sample = run_comparison(file_label_pairs)

    if report is None:
        sys.exit(1)

    print_comparison_table(report)

    if args.output_report:
        Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nJSON report saved to {args.output_report}")

    if args.per_sample and per_sample:
        Path(args.per_sample).parent.mkdir(parents=True, exist_ok=True)
        with open(args.per_sample, "w", encoding="utf-8") as f:
            for r in per_sample:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Per-sample JSONL saved to {args.per_sample}")


if __name__ == "__main__":
    main()
