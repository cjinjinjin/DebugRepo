"""
Offline evaluation of the fine-tuned model's generated prompts.

Metrics:
  1. Format compliance  — all 5 <PromptN> tags present and within 150 words
  2. CoT compliance     — <think> block present with all required fields
                          (ProductType / SpecificProduct / Category /
                           VisualAnchors / LifestyleVibe / CoreValueSignals)
  3. Keyword coverage   — LP title/heading keywords appear (semantic alignment)
  4. Quality classifier — re-use the VLM-based reward from the existing pipeline
                          (optional; requires ZImage or cached generated images)
  5. CLIP/SigLIP score  — prompt-image semantic similarity (if images are available)

Usage:
  # Evaluate generated prompts (text-only, no images needed)
  python evaluate.py \
      --generated_file results/generated_prompts.jsonl \
      --report_file    results/eval_report.json

  # Evaluate swift infer output directly (auto-detected by "response" field)
  python evaluate.py \
      --generated_file results/eval_swift_output.jsonl \
      --report_file    results/eval_report.json

  # Also run VLM image quality check (requires image paths in generated_file)
  python evaluate.py \
      --generated_file results/generated_prompts_with_images.jsonl \
      --report_file    results/eval_report.json \
      --eval_images
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Swift infer output adapter
# ---------------------------------------------------------------------------

def extract_raw_output(record: dict) -> tuple[str, list[str], dict]:
    """
    Normalise a record to (raw_output, generated_prompts, lp_fields).

    Supports two formats:
      - inference.py output: keys "raw_output", "generated_prompts", "lp_fields"
      - swift infer output:  keys "response" (and optionally "messages" for lp_fields)
    """
    if "raw_output" in record:
        raw = record.get("raw_output", "")
        prompts = record.get("generated_prompts", [])
        lp_fields = record.get("lp_fields", {})
        return raw, prompts, lp_fields

    # swift infer format
    raw = record.get("response", "")
    prompts = []
    for i in range(1, 6):
        m = re.search(rf"<Prompt{i}>(.*?)</Prompt{i}>", raw, re.DOTALL)
        if m:
            prompts.append(m.group(1).strip())

    # Try to recover lp_fields from the user message in "messages"
    lp_fields: dict = {}
    messages = record.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # Extract simple key: value lines like "- Document Title: ..."
            for line in content.splitlines():
                line = line.lstrip("- ").strip()
                if ": " in line:
                    k, v = line.split(": ", 1)
                    lp_fields[k.strip()] = v.strip()
            break

    return raw, prompts, lp_fields


# ---------------------------------------------------------------------------
# Text-based metrics
# ---------------------------------------------------------------------------

def check_cot_compliance(raw_output: str) -> dict:
    """
    Check that the <think> block is present and contains all required CoT fields.
    Required fields: ProductType, SpecificProduct, Category,
                     VisualAnchors, LifestyleVibe, CoreValueSignals
    """
    COT_FIELDS = [
        "ProductType",
        "SpecificProduct",
        "Category",
        "VisualAnchors",
        "LifestyleVibe",
        "CoreValueSignals",
    ]
    think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
    if not think_match:
        return {
            "think_block_present": False,
            "fields_present": {f: False for f in COT_FIELDS},
            "all_fields_present": False,
        }

    think_content = think_match.group(1)
    fields_present = {f: (f + ":") in think_content for f in COT_FIELDS}
    return {
        "think_block_present": True,
        "fields_present": fields_present,
        "all_fields_present": all(fields_present.values()),
    }


def check_format_compliance(raw_output: str) -> dict:
    """Check that the model output contains all 5 <PromptN>...</PromptN> tags."""
    results = {}
    all_present = True
    word_counts = []

    for i in range(1, 6):
        pattern = rf"<Prompt{i}>(.*?)</Prompt{i}>"
        match = re.search(pattern, raw_output, re.DOTALL)
        if match:
            content = match.group(1).strip()
            wc = len(content.split())
            word_counts.append(wc)
            results[f"prompt_{i}"] = {"present": True, "word_count": wc, "within_150": wc <= 150}
        else:
            results[f"prompt_{i}"] = {"present": False, "word_count": 0, "within_150": False}
            all_present = False

    return {
        "all_tags_present": all_present,
        "prompts_within_150_words": sum(1 for wc in word_counts if wc <= 150),
        "avg_word_count": sum(word_counts) / len(word_counts) if word_counts else 0,
        "detail": results,
    }


def check_quality_constraints(prompts: list[str]) -> dict:
    """
    Heuristic checks for quality constraint adherence.
    These were specified in the ImagePromptCreator system prompt.
    """
    forbidden_phrases = [
        "agi", "watermark", "logo", "advertisement", "promo",
        "promotion", "stock photo", "studio backdrop",
    ]
    required_hints = [
        # at least one of these quality anchors should be present per prompt
        ["sharp focus", "clean composition", "correct anatomy",
         "natural hands", "no extra text", "no logos"],
    ]

    results = []
    for p in prompts:
        p_lower = p.lower()
        forbidden_found = [f for f in forbidden_phrases if f in p_lower]
        quality_hint_present = any(hint in p_lower for hints in required_hints for hint in hints)
        results.append({
            "forbidden_phrases_found": forbidden_found,
            "has_quality_constraints": quality_hint_present,
        })

    return {
        "prompts_with_forbidden_phrases": sum(1 for r in results if r["forbidden_phrases_found"]),
        "prompts_with_quality_constraints": sum(1 for r in results if r["has_quality_constraints"]),
        "detail": results,
    }


def check_keyword_coverage(prompts: list[str], lp_fields: dict) -> dict:
    """
    Compute what fraction of LP keywords (title, heading) appear in the prompts.
    This is a loose proxy for semantic grounding.
    """
    source_texts = " ".join([
        lp_fields.get("DocumentTitle", ""),
        lp_fields.get("VisualTitle", ""),
        lp_fields.get("Heading", ""),
        lp_fields.get("Title_CB", ""),
    ])
    # Simple tokenization: alphanumeric words, len >= 4
    keywords = set(
        w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", source_texts)
        if w.lower() not in STOPWORDS
    )

    if not keywords:
        return {"keyword_coverage": None, "keywords": []}

    combined_prompts = " ".join(prompts).lower()
    covered = {kw for kw in keywords if kw in combined_prompts}

    return {
        "keyword_coverage": len(covered) / len(keywords),
        "keywords_total": len(keywords),
        "keywords_covered": len(covered),
    }


STOPWORDS = {
    "this", "that", "with", "from", "have", "been", "will", "your",
    "they", "their", "them", "what", "when", "where", "which", "while",
    "also", "about", "more", "into", "than", "then", "some", "such",
}


# ---------------------------------------------------------------------------
# Image-based evaluation (optional)
# ---------------------------------------------------------------------------

def run_vlm_evaluation(sample: dict) -> Optional[float]:
    """
    Run VLM-based quality scoring if image paths are available.
    Integrates with the existing PromptFollowingEvalution pipeline.
    Returns a score in [0, 1] or None if unavailable.
    """
    image_paths = sample.get("image_paths", [])
    if not image_paths:
        return None

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from PromptFollowingEvalution.Qwen3_vl import evaluate_prompt_image_pair

        prompts = sample.get("generated_prompts", [])
        scores = []
        for prompt, img_path in zip(prompts, image_paths):
            if Path(img_path).exists():
                score = evaluate_prompt_image_pair(prompt, img_path)
                scores.append(score)

        return sum(scores) / len(scores) if scores else None
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_file(generated_file: str, eval_images: bool = False) -> dict:
    records = []
    with open(generated_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    format_scores = []
    cot_scores = []
    quality_scores = []
    coverage_scores = []
    vlm_scores = []

    for record in records:
        raw_output, prompts, lp_fields = extract_raw_output(record)

        fmt = check_format_compliance(raw_output)
        cot = check_cot_compliance(raw_output)
        qual = check_quality_constraints(prompts)
        cov = check_keyword_coverage(prompts, lp_fields)

        format_scores.append(fmt)
        cot_scores.append(cot)
        quality_scores.append(qual)
        coverage_scores.append(cov)

        if eval_images:
            vlm_score = run_vlm_evaluation(record)
            if vlm_score is not None:
                vlm_scores.append(vlm_score)

    n = len(records)
    report = {
        "total_samples": n,
        "format": {
            "all_5_tags_present_rate": sum(f["all_tags_present"] for f in format_scores) / n,
            "avg_prompts_within_150_words": sum(f["prompts_within_150_words"] for f in format_scores) / n,
            "avg_word_count": sum(f["avg_word_count"] for f in format_scores) / n,
        },
        "cot": {
            "think_block_present_rate": sum(c["think_block_present"] for c in cot_scores) / n,
            "all_fields_present_rate": sum(c["all_fields_present"] for c in cot_scores) / n,
        },
        "quality_constraints": {
            "avg_prompts_with_constraints": sum(q["prompts_with_quality_constraints"] for q in quality_scores) / n,
            "avg_prompts_with_forbidden": sum(q["prompts_with_forbidden_phrases"] for q in quality_scores) / n,
        },
        "keyword_coverage": {
            "avg_coverage": sum(
                c["keyword_coverage"] for c in coverage_scores if c.get("keyword_coverage") is not None
            ) / max(1, sum(1 for c in coverage_scores if c.get("keyword_coverage") is not None)),
        },
    }

    if vlm_scores:
        report["vlm_image_quality"] = {
            "avg_score": sum(vlm_scores) / len(vlm_scores),
            "evaluated_samples": len(vlm_scores),
        }

    return report


def print_report(report: dict) -> None:
    print("\n" + "=" * 60)
    print("Evaluation Report")
    print("=" * 60)
    print(f"Total samples: {report['total_samples']}")

    print("\n[Format Compliance]")
    fmt = report["format"]
    print(f"  All 5 tags present:       {fmt['all_5_tags_present_rate']:.1%}")
    print(f"  Prompts within 150 words: {fmt['avg_prompts_within_150_words']:.1f} / 5")
    print(f"  Avg word count per prompt: {fmt['avg_word_count']:.1f}")

    print("\n[CoT Compliance]")
    cot = report["cot"]
    print(f"  <think> block present:    {cot['think_block_present_rate']:.1%}")
    print(f"  All 6 CoT fields present: {cot['all_fields_present_rate']:.1%}")

    print("\n[Quality Constraints]")
    qc = report["quality_constraints"]
    print(f"  Prompts with quality hints:    {qc['avg_prompts_with_constraints']:.1f} / 5")
    print(f"  Prompts with forbidden words:  {qc['avg_prompts_with_forbidden']:.1f} / 5")

    print("\n[Keyword Coverage]")
    print(f"  Avg LP keyword coverage: {report['keyword_coverage']['avg_coverage']:.1%}")

    if "vlm_image_quality" in report:
        print("\n[VLM Image Quality]")
        v = report["vlm_image_quality"]
        print(f"  Avg VLM score: {v['avg_score']:.3f} (n={v['evaluated_samples']})")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--generated_file", required=True,
                   help="JSONL file with generated prompts (output of inference.py)")
    p.add_argument("--report_file", default="",
                   help="Path to save JSON report (optional)")
    p.add_argument("--eval_images", action="store_true", default=False,
                   help="Also evaluate generated images via VLM (requires image_paths in data)")
    return p.parse_args()


def main():
    args = parse_args()
    report = evaluate_file(args.generated_file, eval_images=args.eval_images)
    print_report(report)

    if args.report_file:
        Path(args.report_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {args.report_file}")


if __name__ == "__main__":
    main()
