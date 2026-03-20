"""
Prepare SFT data from the new Random1000 batch, then merge with the existing
training set and deduplicate (drop any sample whose 5 prompts are not all unique).

Input files (RawData/):
  UHRS2K_SD.tsv                                          -- LP fields (no header, 12 cols)
  UHRS2K_SD_Sample1000.GPT5PromptsCreator.0131-2-latest.tsv  -- prompts + CoT fields
  UHRS_Task_lp_labeling_0306_random1K_Quality.tsv        -- UHRS annotations (3 judges)

Existing data:
  data/sft_train_cot.jsonl
  data/sft_eval_cot.jsonl

Output:
  data/sft_train_cot.jsonl   (overwritten with merged + deduped)
  data/sft_eval_cot.jsonl    (overwritten with merged + deduped)
  data/dataset_stats_random1k.json

Usage:
  python prepare_data_random1k.py
  python prepare_data_random1k.py --include_fair_as_bad
  python prepare_data_random1k.py --eval_ratio 0.1 --seed 42
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RAW_DIR    = SCRIPT_DIR / "RawData"

# UHRS2K_SD.tsv has no header and no RowId (12 cols)
SD_COLS = [
    "LPURL", "ImageURL", "label",
    "DocumentTitle", "VisualTitle", "Heading",
    "Title_CB", "VisualTitle_CB", "Heading_CB",
    "BestSnippet_CB", "MetaDescription_CB",
    "PrimaryContentNoTitleNoHeading",
]

# Reuse system prompts from prepare_data.py
SYSTEM_PROMPT_COT = (
    "You are an expert Ad Creative Director and Senior AI Image Prompt Engineer, "
    "specialized in high-performing Native Advertisement visuals.\n\n"
    "Given a landing page URL and its extracted content fields, your task is to "
    "generate five (5) high-quality English image generation prompts for Native Ads.\n\n"
    "First, reason about the product inside <think>...</think> tags. "
    "Extract the following from the landing page:\n"
    "- ProductType: Physical Product / Digital Product / Service\n"
    "- SpecificProduct: concise noun phrase\n"
    "- Category: broad product/service category\n"
    "- VisualAnchors: 2-3 specific physical elements implied by the page\n"
    "- LifestyleVibe: emotional tone of the experience\n"
    "- CoreValueSignals: up to 3 from [professional, premium, affordable, "
    "efficient, reliable, simple]\n\n"
    "Then output exactly 5 prompts. Each prompt must:\n"
    "- Be <=150 words\n"
    "- Embed all safety, realism, quality, and exclusion constraints\n"
    "- Feel native and non-promotional\n"
    "- Show the product outcome or value naturally in context\n"
    "- Avoid stereotypes, text/logos in image, and stock-photo aesthetics\n"
    "- Ensure correct anatomy, natural hands, sharp focus, clean composition\n\n"
    "Output format:\n"
    "<think>\n"
    "ProductType: ...\n"
    "SpecificProduct: ...\n"
    "Category: ...\n"
    "VisualAnchors: ...\n"
    "LifestyleVibe: ...\n"
    "CoreValueSignals: ...\n"
    "</think>\n"
    "<Prompt1>...</Prompt1>\n"
    "<Prompt2>...</Prompt2>\n"
    "<Prompt3>...</Prompt3>\n"
    "<Prompt4>...</Prompt4>\n"
    "<Prompt5>...</Prompt5>"
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_sd_file(path: Path) -> dict:
    """Load UHRS2K_SD.tsv (no header) -> {LPURL: row_dict}"""
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            while len(parts) < len(SD_COLS):
                parts.append("")
            row = dict(zip(SD_COLS, parts))
            url = row["LPURL"].strip()
            if url:
                result[url] = row
    return result


def load_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(samples: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(samples)} -> {path}")


# ---------------------------------------------------------------------------
# Label aggregation  (same logic as prepare_data.py)
# ---------------------------------------------------------------------------

def normalize_decision(label: str) -> str:
    label = label.strip().lower()
    if label == "good":      return "good"
    if label in ("bad", "logo"): return "bad"
    if label == "fair":      return "fair"
    return "skip"


def aggregate_labels(label_rows: list[dict], include_fair_as_bad: bool = False) -> dict:
    """
    lp_id format: Random1000_468_Prompt2_1344x768.jpg
    img_key      : Random1000_468_Prompt2
    """
    groups = defaultdict(list)
    for row in label_rows:
        lp_id = row.get("lp_id", "")
        m = re.match(r"(Random1000_\d+_Prompt\d+)_", lp_id)
        key = m.group(1) if m else lp_id
        groups[key].append(row)

    result = {}
    for key, rows in groups.items():
        votes = Counter()
        bad_reasons = []
        for r in rows:
            norm = normalize_decision(r["FinalDecision"])
            votes[norm] += 1
            reason = r.get("MainSubjectClarityBadOpotion", "").strip()
            if reason:
                bad_reasons.append(reason)

        total_valid = sum(v for k, v in votes.items() if k != "skip")
        if total_valid == 0:
            final = "skip"
        else:
            g, b, fair = votes.get("good", 0), votes.get("bad", 0), votes.get("fair", 0)
            if g >= 2:
                final = "good"
            elif b >= 2:
                final = "bad"
            elif fair >= 2:
                final = "bad" if include_fair_as_bad else "fair"
            elif g == 1 and b == 1 and fair == 1:
                final = "bad" if include_fair_as_bad else "fair"
            elif g > b:
                final = "good"
            elif b > g:
                final = "bad"
            else:
                final = "fair"

        result[key] = {
            "label": final,
            "votes": dict(votes),
            "bad_reasons": list(set(r for r in bad_reasons if r)),
        }
    return result


# ---------------------------------------------------------------------------
# CoT block builder  (same logic as prepare_data.py)
# ---------------------------------------------------------------------------

def build_cot_block(anchor_row: dict) -> str:
    def clean(text):
        return text.strip().lstrip("-").strip()

    lines = []

    intent_raw = clean(anchor_row.get("ProductIntent", ""))
    if intent_raw:
        ptype, specific = "", ""
        for part in intent_raw.replace(" - ", "\n").splitlines():
            part = part.strip()
            if part.lower().startswith("producttype:"):
                ptype = part.split(":", 1)[1].strip()
            elif part.lower().startswith("specificproduct:"):
                specific = part.split(":", 1)[1].strip()
        if ptype:    lines.append(f"ProductType: {ptype}")
        if specific: lines.append(f"SpecificProduct: {specific}")

    cat_raw = clean(anchor_row.get("ProductCategory", ""))
    if cat_raw:
        cat = ""
        for part in cat_raw.splitlines():
            part = part.strip().lstrip("-").strip()
            if part.lower().startswith("category:"):
                cat = part.split(":", 1)[1].strip()
                break
        lines.append(f"Category: {cat or cat_raw}")

    visual_raw = clean(anchor_row.get("VisualContext", ""))
    if visual_raw:
        anchors, vibe = "", ""
        for part in visual_raw.replace(" - ", "\n").splitlines():
            part = part.strip().lstrip("-").strip()
            if part.lower().startswith("visualanchors:"):
                anchors = part.split(":", 1)[1].strip()
            elif part.lower().startswith("lifestylevibe:"):
                vibe = part.split(":", 1)[1].strip()
        if anchors: lines.append(f"VisualAnchors: {anchors}")
        if vibe:    lines.append(f"LifestyleVibe: {vibe}")

    val_raw = clean(anchor_row.get("ValueSignals", ""))
    if val_raw:
        signals = ""
        for part in val_raw.splitlines():
            part = part.strip().lstrip("-").strip()
            if part.lower().startswith("corevaluesignals:"):
                signals = part.split(":", 1)[1].strip()
                break
        lines.append(f"CoreValueSignals: {signals or val_raw}")

    if not lines:
        return ""
    return "<think>\n" + "\n".join(lines) + "\n</think>"


# ---------------------------------------------------------------------------
# User message builder  (same logic as prepare_data.py)
# ---------------------------------------------------------------------------

def build_user_message(lp: dict) -> str:
    field_labels = [
        ("LPURL",                          "Landing Page URL"),
        ("DocumentTitle",                  "Document Title"),
        ("VisualTitle",                    "Visual Title"),
        ("Heading",                        "Heading"),
        ("Title_CB",                       "Title (CB)"),
        ("VisualTitle_CB",                 "Visual Title (CB)"),
        ("Heading_CB",                     "Heading (CB)"),
        ("BestSnippet_CB",                 "Best Snippet (CB)"),
        ("MetaDescription_CB",             "Meta Description"),
        ("PrimaryContentNoTitleNoHeading", "Page Content"),
    ]
    parts = ["Generate 5 image generation prompts for a Native Ad based on the "
             "following landing page information:\n"]
    for key, label in field_labels:
        val = lp.get(key, "").strip()
        if val:
            parts.append(f"[{label}]\n{val}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# SFT sample builder
# ---------------------------------------------------------------------------

MIN_GOOD_FAIR = 3  # require at least this many unique good+fair prompts


def build_sft_sample(
    lp: dict,
    url_hash: str,
    prompt_rows_for_lp: list[dict],
    label_lookup: dict,
) -> dict | None:
    tag_order  = ["Prompt1", "Prompt2", "Prompt3", "Prompt4", "Prompt5"]
    row_by_tag = {r["Tag"]: r for r in prompt_rows_for_lp}
    anchor_row = prompt_rows_for_lp[0] if prompt_rows_for_lp else {}

    good_prompts = []
    fair_prompts = []
    bad_prompts  = []
    all_bad_reasons = []

    for tag in tag_order:
        row = row_by_tag.get(tag)
        if row is None:
            continue
        img_key     = f"{url_hash}_{tag}"
        prompt_text = row.get("Prompt", "").strip()
        ann         = label_lookup.get(img_key)
        if ann is None:
            continue

        if ann["label"] == "good":
            good_prompts.append(prompt_text)
        elif ann["label"] == "fair":
            fair_prompts.append(prompt_text)
        elif ann["label"] == "bad":
            bad_prompts.append(prompt_text)
            all_bad_reasons.extend(ann["bad_reasons"])

    # Build 5 unique prompts: good first, then fair, then bad
    seen = set()
    final_prompts = []
    for tier in (good_prompts, fair_prompts, bad_prompts):
        for p in tier:
            if p not in seen and len(final_prompts) < 5:
                final_prompts.append(p)
                seen.add(p)

    # Must have 5 unique total and at least MIN_GOOD_FAIR from good+fair
    if len(final_prompts) < 5:
        return None
    n_good_fair = sum(1 for p in final_prompts if p in set(good_prompts) | set(fair_prompts))
    if n_good_fair < MIN_GOOD_FAIR:
        return None

    prompts_block = "\n".join(
        f"<Prompt{i}>{p.strip()}</Prompt{i}>" for i, p in enumerate(final_prompts, 1)
    )
    cot_block = build_cot_block(anchor_row)
    assistant_content = f"{cot_block}\n{prompts_block}" if cot_block else prompts_block

    return {
        "id":                f"{url_hash}_original",
        "url_hash":          url_hash,
        "lp_url":            lp.get("LPURL", ""),
        "version":           "original",
        "good_prompt_count": len(good_prompts),
        "bad_prompt_count":  len(bad_prompts),
        "bad_reasons":       list(set(all_bad_reasons)),
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT_COT},
            {"role": "user",      "content": build_user_message(lp)},
            {"role": "assistant", "content": assistant_content},
        ],
    }


# ---------------------------------------------------------------------------
# Deduplication for existing samples
# ---------------------------------------------------------------------------

def extract_prompts_from_content(content: str) -> list[str]:
    prompts = []
    for i in range(1, 6):
        m = re.search(rf"<Prompt{i}>(.*?)</Prompt{i}>", content, re.DOTALL)
        if m:
            prompts.append(m.group(1).strip())
    return prompts


def is_clean(sample: dict) -> bool:
    """Return True if the sample has 5 unique prompts (no repeats)."""
    msgs = sample.get("messages", [])
    content = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")
    prompts = extract_prompts_from_content(content)
    return len(prompts) == 5 and len(set(prompts)) == 5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",          default="data")
    parser.add_argument("--eval_ratio",           type=float, default=0.1)
    parser.add_argument("--include_fair_as_bad",  action="store_true", default=False)
    parser.add_argument("--seed",                 type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load new raw data ────────────────────────────────────────────────────
    print("Loading new raw data ...")
    lp_by_url    = load_sd_file(RAW_DIR / "UHRS2K_SD.tsv")
    prompt_rows  = load_tsv(RAW_DIR / "UHRS2K_SD_Sample1000.GPT5PromptsCreator.0131-2-latest.tsv")
    label_rows   = load_tsv(RAW_DIR / "UHRS_Task_lp_labeling_0306_random1K_Quality.tsv")
    print(f"  SD LPs:       {len(lp_by_url)}")
    print(f"  Prompt rows:  {len(prompt_rows)}")
    print(f"  Label rows:   {len(label_rows)}")

    # ── Group prompts by UrlHash ─────────────────────────────────────────────
    prompts_by_hash = defaultdict(list)
    hash_to_url     = {}
    for row in prompt_rows:
        h = row["UrlHash"]
        prompts_by_hash[h].append(row)
        hash_to_url[h] = row["FinalDestinationURLUrl"].strip()

    # ── Aggregate labels ─────────────────────────────────────────────────────
    print("Aggregating labels ...")
    label_lookup = aggregate_labels(label_rows, args.include_fair_as_bad)
    print(f"  Label dist: {dict(Counter(v['label'] for v in label_lookup.values()))}")

    # ── Build new SFT samples ────────────────────────────────────────────────
    print("Building new SFT samples ...")
    new_samples     = []
    skipped_no_lp   = 0
    skipped_no_good = 0
    skipped_dup     = 0

    for url_hash, lp_rows in prompts_by_hash.items():
        lp_url = hash_to_url.get(url_hash, "")
        lp     = lp_by_url.get(lp_url)
        if lp is None:
            skipped_no_lp += 1
            continue

        sample = build_sft_sample(lp, url_hash, lp_rows, label_lookup)
        if sample is None:
            all_labels = [label_lookup.get(f"{url_hash}_{r['Tag']}", {}).get("label") for r in lp_rows]
            n_gf = sum(1 for l in all_labels if l in ("good", "fair"))
            if n_gf < MIN_GOOD_FAIR:
                skipped_no_good += 1
            else:
                skipped_dup += 1
            continue

        new_samples.append(sample)

    print(f"  New samples built:      {len(new_samples)}")
    print(f"  Skipped (no LP):               {skipped_no_lp}")
    print(f"  Skipped (<{MIN_GOOD_FAIR} good+fair):      {skipped_no_good}")
    print(f"  Skipped (can't fill 5 unique): {skipped_dup}")

    # ── Load existing data ───────────────────────────────────────────────────
    print("\nLoading existing data ...")
    train_path = out_dir / "sft_train_cot.jsonl"
    eval_path  = out_dir / "sft_eval_cot.jsonl"

    existing_train = load_jsonl(train_path) if train_path.exists() else []
    existing_eval  = load_jsonl(eval_path)  if eval_path.exists()  else []
    print(f"  Existing train: {len(existing_train)}, eval: {len(existing_eval)}")

    # ── Deduplicate existing samples (drop samples with repeated prompts) ────
    print("Deduplicating existing samples ...")
    clean_train = [s for s in existing_train if is_clean(s)]
    clean_eval  = [s for s in existing_eval  if is_clean(s)]
    print(f"  Train: {len(existing_train)} -> {len(clean_train)} "
          f"(dropped {len(existing_train)-len(clean_train)})")
    print(f"  Eval:  {len(existing_eval)} -> {len(clean_eval)} "
          f"(dropped {len(existing_eval)-len(clean_eval)})")

    # ── Split new samples into train/eval (LP-level, no leakage) ────────────
    existing_hashes = {s["url_hash"] for s in clean_train + clean_eval}
    new_unique_hashes = sorted({s["url_hash"] for s in new_samples
                                 if s["url_hash"] not in existing_hashes})
    random.shuffle(new_unique_hashes)
    n_eval    = max(1, int(len(new_unique_hashes) * args.eval_ratio))
    new_eval  = set(new_unique_hashes[:n_eval])
    new_train = set(new_unique_hashes[n_eval:])

    add_train = [s for s in new_samples if s["url_hash"] in new_train]
    add_eval  = [s for s in new_samples if s["url_hash"] in new_eval]
    print(f"\nNew samples split: train={len(add_train)}, eval={len(add_eval)}")

    # ── Merge ────────────────────────────────────────────────────────────────
    merged_train = clean_train + add_train
    merged_eval  = clean_eval  + add_eval
    print(f"Final: train={len(merged_train)}, eval={len(merged_eval)}")

    # ── Write ────────────────────────────────────────────────────────────────
    print("\nWriting output ...")
    write_jsonl(merged_train, train_path)
    write_jsonl(merged_eval,  eval_path)

    # ── Stats ────────────────────────────────────────────────────────────────
    stats = {
        "new_samples_built": len(new_samples),
        "new_skipped": {
            "no_lp": skipped_no_lp,
            "no_good": skipped_no_good,
            "lt5_unique": skipped_dup,
        },
        "existing_dedup": {
            "train_before": len(existing_train), "train_after": len(clean_train),
            "eval_before":  len(existing_eval),  "eval_after":  len(clean_eval),
        },
        "final": {
            "train": len(merged_train),
            "eval":  len(merged_eval),
        },
        "label_dist": dict(Counter(v["label"] for v in label_lookup.values())),
    }
    stats_path = out_dir / "dataset_stats_random1k.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Stats -> {stats_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
