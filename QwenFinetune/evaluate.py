"""
Offline evaluation of the fine-tuned model's generated prompts.

Metrics:
  1. Format compliance  — all 5 <PromptN> tags present and within 150 words
  2. CoT compliance     — <think> block present with all required fields
                          (ProductType / SpecificProduct / Category /
                           VisualAnchors / LifestyleVibe / CoreValueSignals)
  3. Keyword coverage   — LP title/heading keywords appear (semantic alignment)
  4. LLM-as-Judge       — GPT scores each prompt on 4 dimensions (1-5 scale)
                          driven by OPENAI_API_KEY / OPENAI_API_BASE env vars
  5. Ground-truth agreement — predicted good_count vs annotated good_prompt_count
  6. Quality classifier — re-use the VLM-based reward from the existing pipeline
                          (optional; requires ZImage or cached generated images)

Usage:
  # Text-only evaluation (no API key needed)
  python evaluate.py \
      --generated_file results/generated_prompts.jsonl \
      --report_file    results/eval_report.json

  # Add LLM-as-Judge scoring (requires OPENAI_API_KEY)
  python evaluate.py \
      --generated_file results/eval_swift_output.jsonl \
      --report_file    results/eval_report.json \
      --llm_judge

  # Also compare against ground-truth annotations in the eval dataset
  python evaluate.py \
      --generated_file results/eval_swift_output.jsonl \
      --gt_file        data/sft_eval_cot.jsonl \
      --report_file    results/eval_report.json \
      --llm_judge

  # Evaluate swift infer output (auto-detected by "response" field)
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
import os
import re
import sys
import time
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
# LLM-as-Judge  (Direction A)
# ---------------------------------------------------------------------------

_LLM_JUDGE_SYSTEM = """\
You are an expert evaluator for Native Ad image generation prompts.
Score each prompt on these 4 dimensions using integers 1-5:

1. LP Relevance    — Does the prompt visually represent the landing page product/service?
2. Native Feel     — Does it look non-promotional, lifestyle-oriented, not stock-photo?
3. Visual Clarity  — Is the scene specific, concrete, and unambiguous for an image model?
4. Constraint Adherence — Does it avoid text/logos, ensure anatomy/hands quality, stay ≤150 words?

Return ONLY a JSON array with one object per prompt:
[
  {"prompt_index": 1, "lp_relevance": 4, "native_feel": 3, "visual_clarity": 5, "constraint_adherence": 4, "overall": 4},
  ...
]
Do not include any explanation outside the JSON."""

_LLM_JUDGE_USER_TMPL = """\
Landing Page Summary:
{lp_summary}

Generated Prompts:
{prompt_block}"""


def _build_lp_summary(lp_fields: dict) -> str:
    keys = ["DocumentTitle", "VisualTitle", "Heading", "Title_CB", "BestSnippet_CB"]
    lines = []
    for k in keys:
        v = lp_fields.get(k, "").strip()
        if v:
            lines.append(f"- {k}: {v[:200]}")
    return "\n".join(lines) if lines else "(no LP fields available)"


def llm_judge_score(
    prompts: list[str],
    lp_fields: dict,
    api_key: str,
    api_base: str,
    model: str = "gpt-4o",
) -> Optional[list[dict]]:
    """
    Call an OpenAI-compatible LLM to score all 5 prompts for one sample.
    Returns a list of score dicts or None on failure.
    """
    try:
        import openai
    except ImportError:
        print("[LLM Judge] openai package not installed. Skipping.")
        return None

    if not prompts:
        return None

    lp_summary = _build_lp_summary(lp_fields)
    prompt_block = "\n\n".join(
        f"[Prompt {i+1}]\n{p}" for i, p in enumerate(prompts)
    )
    user_msg = _LLM_JUDGE_USER_TMPL.format(
        lp_summary=lp_summary, prompt_block=prompt_block
    )

    client = openai.OpenAI(api_key=api_key, base_url=api_base)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _LLM_JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content or ""
            # Extract JSON array from response
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            wait = 2 ** attempt
            print(f"[LLM Judge] Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return None


def _aggregate_judge_scores(scores_list: list[Optional[list[dict]]]) -> dict:
    """Aggregate per-sample judge scores into dataset-level statistics."""
    dims = ["lp_relevance", "native_feel", "visual_clarity", "constraint_adherence", "overall"]
    totals = defaultdict(list)
    evaluated = 0

    for sample_scores in scores_list:
        if not sample_scores:
            continue
        evaluated += 1
        for prompt_score in sample_scores:
            for dim in dims:
                if dim in prompt_score:
                    totals[dim].append(prompt_score[dim])

    if not evaluated:
        return {"evaluated_samples": 0}

    result = {"evaluated_samples": evaluated}
    for dim in dims:
        vals = totals[dim]
        result[f"avg_{dim}"] = sum(vals) / len(vals) if vals else 0.0
    return result


# ---------------------------------------------------------------------------
# Ground-truth agreement  (Direction B)
# ---------------------------------------------------------------------------

def _judge_predict_good_count(
    judge_scores: Optional[list[dict]],
    threshold: float = 3.5,
) -> Optional[int]:
    """
    Convert LLM judge scores to a predicted good_prompt_count.
    A prompt is considered 'good' if its overall score >= threshold.
    """
    if not judge_scores:
        return None
    return sum(1 for s in judge_scores if s.get("overall", 0) >= threshold)


def check_ground_truth_agreement(
    judge_scores_list: list[Optional[list[dict]]],
    gt_records: list[dict],
    threshold: float = 3.5,
) -> dict:
    """
    Compare predicted good_prompt_count (from LLM judge) against
    ground-truth good_prompt_count annotations in the eval dataset.

    Returns MAE, exact-match rate, and Pearson correlation.
    """
    predicted = []
    actual = []

    for judge_scores, gt in zip(judge_scores_list, gt_records):
        pred = _judge_predict_good_count(judge_scores, threshold)
        gt_count = gt.get("good_prompt_count")
        if pred is not None and gt_count is not None:
            predicted.append(pred)
            actual.append(gt_count)

    n = len(predicted)
    if n == 0:
        return {"compared_samples": 0}

    mae = sum(abs(p - a) for p, a in zip(predicted, actual)) / n
    exact_match = sum(1 for p, a in zip(predicted, actual) if p == a) / n

    # Pearson correlation
    mean_p = sum(predicted) / n
    mean_a = sum(actual) / n
    cov = sum((p - mean_p) * (a - mean_a) for p, a in zip(predicted, actual))
    std_p = (sum((p - mean_p) ** 2 for p in predicted) / n) ** 0.5
    std_a = (sum((a - mean_a) ** 2 for a in actual) / n) ** 0.5
    pearson = cov / (n * std_p * std_a) if (std_p > 0 and std_a > 0) else 0.0

    return {
        "compared_samples": n,
        "mae": round(mae, 3),
        "exact_match_rate": round(exact_match, 3),
        "pearson_r": round(pearson, 3),
        "judge_threshold": threshold,
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

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_file(
    generated_file: str,
    eval_images: bool = False,
    llm_judge: bool = False,
    gt_file: str = "",
    llm_model: str = "gpt-4o",
) -> dict:
    records = load_jsonl(generated_file)
    gt_records = load_jsonl(gt_file) if gt_file else []

    # Build id→gt lookup for alignment
    gt_by_id: dict[str, dict] = {r["id"]: r for r in gt_records if "id" in r}

    format_scores = []
    cot_scores = []
    quality_scores = []
    coverage_scores = []
    vlm_scores = []
    judge_scores_list: list[Optional[list[dict]]] = []
    aligned_gt: list[dict] = []

    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

    for i, record in enumerate(records):
        raw_output, prompts, lp_fields = extract_raw_output(record)

        format_scores.append(check_format_compliance(raw_output))
        cot_scores.append(check_cot_compliance(raw_output))
        quality_scores.append(check_quality_constraints(prompts))
        coverage_scores.append(check_keyword_coverage(prompts, lp_fields))

        if eval_images:
            vlm_score = run_vlm_evaluation(record)
            if vlm_score is not None:
                vlm_scores.append(vlm_score)

        if llm_judge:
            if not api_key:
                print("[LLM Judge] OPENAI_API_KEY not set. Skipping judge scoring.")
                llm_judge = False  # disable for remaining records
                judge_scores_list.append(None)
            else:
                print(f"[LLM Judge] Scoring sample {i+1}/{len(records)} ...", end="\r")
                scores = llm_judge_score(prompts, lp_fields, api_key, api_base, llm_judge if isinstance(llm_judge, str) else llm_model)
                judge_scores_list.append(scores)

            # Align with ground truth
            rec_id = record.get("id", "")
            if rec_id in gt_by_id:
                aligned_gt.append(gt_by_id[rec_id])
            elif gt_records and i < len(gt_records):
                aligned_gt.append(gt_records[i])
            else:
                aligned_gt.append({})
        else:
            judge_scores_list.append(None)

    n = len(records)
    report: dict = {
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

    # LLM judge aggregation
    if any(s is not None for s in judge_scores_list):
        report["llm_judge"] = _aggregate_judge_scores(judge_scores_list)

        # Ground-truth agreement
        if aligned_gt:
            report["ground_truth_agreement"] = check_ground_truth_agreement(
                judge_scores_list, aligned_gt
            )

    if vlm_scores:
        report["vlm_image_quality"] = {
            "avg_score": sum(vlm_scores) / len(vlm_scores),
            "evaluated_samples": len(vlm_scores),
        }

    return report


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(report: dict) -> None:
    print("\n" + "=" * 60)
    print("Evaluation Report")
    print("=" * 60)
    print(f"Total samples: {report['total_samples']}")

    print("\n[Format Compliance]")
    fmt = report["format"]
    print(f"  All 5 tags present:        {fmt['all_5_tags_present_rate']:.1%}")
    print(f"  Prompts within 150 words:  {fmt['avg_prompts_within_150_words']:.1f} / 5")
    print(f"  Avg word count per prompt: {fmt['avg_word_count']:.1f}")

    print("\n[CoT Compliance]")
    cot = report["cot"]
    print(f"  <think> block present:     {cot['think_block_present_rate']:.1%}")
    print(f"  All 6 CoT fields present:  {cot['all_fields_present_rate']:.1%}")

    print("\n[Quality Constraints]")
    qc = report["quality_constraints"]
    print(f"  Prompts with quality hints:   {qc['avg_prompts_with_constraints']:.1f} / 5")
    print(f"  Prompts with forbidden words: {qc['avg_prompts_with_forbidden']:.1f} / 5")

    print("\n[Keyword Coverage]")
    print(f"  Avg LP keyword coverage: {report['keyword_coverage']['avg_coverage']:.1%}")

    if "llm_judge" in report:
        print("\n[LLM-as-Judge]")
        j = report["llm_judge"]
        print(f"  Evaluated samples:      {j['evaluated_samples']}")
        for dim in ["lp_relevance", "native_feel", "visual_clarity", "constraint_adherence", "overall"]:
            key = f"avg_{dim}"
            if key in j:
                print(f"  {dim:<26} {j[key]:.2f} / 5")

    if "ground_truth_agreement" in report:
        print("\n[Ground-Truth Agreement]")
        g = report["ground_truth_agreement"]
        print(f"  Compared samples:    {g['compared_samples']}")
        print(f"  MAE (good_count):    {g.get('mae', 'N/A')}")
        print(f"  Exact match rate:    {g.get('exact_match_rate', 0):.1%}")
        print(f"  Pearson r:           {g.get('pearson_r', 'N/A')}")

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
                   help="JSONL file with generated prompts (output of inference.py or swift infer)")
    p.add_argument("--gt_file", default="",
                   help="Ground-truth eval JSONL (e.g. data/sft_eval_cot.jsonl) for agreement metrics")
    p.add_argument("--report_file", default="",
                   help="Path to save JSON report (optional)")
    p.add_argument("--llm_judge", action="store_true", default=False,
                   help="Enable LLM-as-Judge scoring (requires OPENAI_API_KEY env var)")
    p.add_argument("--llm_model", default="gpt-4o",
                   help="Model to use for LLM judge (default: gpt-4o)")
    p.add_argument("--eval_images", action="store_true", default=False,
                   help="Also evaluate generated images via VLM (requires image_paths in data)")
    return p.parse_args()


def main():
    args = parse_args()
    report = evaluate_file(
        generated_file=args.generated_file,
        eval_images=args.eval_images,
        llm_judge=args.llm_judge,
        gt_file=args.gt_file,
        llm_model=args.llm_model,
    )
    print_report(report)

    if args.report_file:
        Path(args.report_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {args.report_file}")


if __name__ == "__main__":
    main()
