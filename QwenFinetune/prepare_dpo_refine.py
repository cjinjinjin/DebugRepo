"""
Construct DPO training data from Internal100+Random100 orig vs. refine annotations.

Data sources:
  - orig prompts   : Internal100_Random100.GPT5PromptsCreator.0131-2.tsv
  - refine prompts : Internal100_Random100.GPT5PromptsCreator.0131-2_refineprompt.tsv
  - orig labels    : UHRS_Task_lp_quality_labeling_0204_LP200_0131-2_official_ZImage_images_cfg0_20260201-0426.tsv
  - refine labels  : UHRS_Task_lp_quality_labeling_0227-PromptRefiner_Internal100_LP200_0131-2_refinePrompt_official_ZImage_images_cfg0_20260227-0530.tsv

Strategy:
  Per LP (UrlHash), build three prompt pools from orig+refine prompt texts with labels:
    good_pool : prompts whose generated image was labelled good
    fair_pool : prompts whose generated image was labelled fair
    bad_pool  : prompts whose generated image was labelled bad
  (orig_text and refine_text are treated as two distinct entries; dedup by text)

  chosen  : fill 5 unique prompts: good > fair > at most 1 bad (filler only if needed
            AND bad_pool has >= 2 entries so rejected can still exceed chosen_bad)
  rejected: replace good slots with bad prompts until rejected_bad > chosen_bad

  Constraints:
    - >= 3 prompts in good_pool
    - chosen_bad <= 1
    - rejected_bad > chosen_bad  (strict signal)
    - all 5 prompts in chosen and rejected are unique
    - chosen != rejected

Output:
  data/dpo_refine_train_cot.jsonl
  data/dpo_refine_eval_cot.jsonl
  data/dataset_stats_dpo_refine.json

Usage:
  python prepare_dpo_refine.py
  python prepare_dpo_refine.py --eval_ratio 0.1 --seed 42
"""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RAW_DIR    = SCRIPT_DIR / "RawData"

SD_COLS = [
    "LPURL", "ImageURL", "label",
    "DocumentTitle", "VisualTitle", "Heading",
    "Title_CB", "VisualTitle_CB", "Heading_CB",
    "BestSnippet_CB", "MetaDescription_CB",
    "PrimaryContentNoTitleNoHeading",
]

MIN_GOOD = 3

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

def load_sd_file(path: Path, skip_col0: bool = False) -> dict:
    """Load a tab-separated LP file into a dict keyed by LPURL.

    Internal100_SD.tsv has an extra leading row-number column; set
    skip_col0=True to discard it before mapping to SD_COLS.
    """
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if skip_col0:
                parts = parts[1:]
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


def write_jsonl(samples: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(samples)} -> {path}")


# ---------------------------------------------------------------------------
# Label aggregation
# ---------------------------------------------------------------------------

def aggregate_labels(rows: list[dict], strip_prefix: str | None = None) -> dict:
    """
    Aggregate per-judge FinalDecision votes into a single label per prompt.
    lp_id format examples:
      orig:   'Internal10082_Prompt4.png'
      refine: '0227_Internal1002_Prompt3.jpg'
    """
    groups = defaultdict(list)
    for row in rows:
        lp_id = row.get("lp_id", "").strip()
        # strip image extension
        lp_id = re.sub(r"\.(png|jpg|jpeg)$", "", lp_id, flags=re.IGNORECASE)
        if strip_prefix:
            lp_id = re.sub(r"^" + strip_prefix, "", lp_id)
        groups[lp_id].append(row["FinalDecision"].strip().lower())

    result = {}
    for key, votes in groups.items():
        c = Counter(votes)
        g, b, f = c.get("good", 0), c.get("bad", 0), c.get("fair", 0)
        if   g >= 2: final = "good"
        elif b >= 2: final = "bad"
        elif f >= 2: final = "fair"
        elif g > b:  final = "good"
        elif b > g:  final = "bad"
        else:        final = "fair"
        result[key] = final
    return result


# ---------------------------------------------------------------------------
# CoT + user message builders (reused pattern from prepare_data.py)
# ---------------------------------------------------------------------------

def build_cot_block(row: dict) -> str:
    def clean(text):
        return text.strip().lstrip("-").strip()

    lines = []
    intent_raw = clean(row.get("ProductIntent", ""))
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

    cat_raw = clean(row.get("ProductCategory", ""))
    if cat_raw:
        cat = ""
        for part in cat_raw.splitlines():
            part = part.strip().lstrip("-").strip()
            if part.lower().startswith("category:"):
                cat = part.split(":", 1)[1].strip()
                break
        lines.append(f"Category: {cat or cat_raw}")

    visual_raw = clean(row.get("VisualContext", ""))
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

    val_raw = clean(row.get("ValueSignals", ""))
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


def build_response(cot_block: str, prompts: list[str]) -> str:
    prompts_block = "\n".join(
        f"<Prompt{i}>{p}</Prompt{i}>" for i, p in enumerate(prompts, 1)
    )
    return f"{cot_block}\n{prompts_block}" if cot_block else prompts_block


# ---------------------------------------------------------------------------
# DPO pair builder
# ---------------------------------------------------------------------------

def build_dpo_pair(
    url_hash: str,
    good_pool: list[str],
    fair_pool: list[str],
    bad_pool: list[str],
    lp: dict,
    anchor_row: dict,
) -> dict | None:
    """
    chosen  : good > fair > at most 1 bad (filler only when good+fair < 5
              AND bad_pool has >= 2 entries so rejected can exceed chosen_bad)
    rejected: replace good slots with bad prompts until rejected_bad > chosen_bad

    Returns None if constraints cannot be satisfied.
    """
    if len(good_pool) < MIN_GOOD or not bad_pool:
        return None

    bad_texts = set(bad_pool)

    # ── Build chosen ─────────────────────────────────────────────────────────
    seen = set()
    chosen = []
    bad_in_chosen = 0

    for p in good_pool + fair_pool:
        if p not in seen and len(chosen) < 5:
            chosen.append(p)
            seen.add(p)

    # Allow at most 1 bad filler, but only if bad_pool >= 2
    # (so there's still a spare bad for rejected to use)
    if len(chosen) < 5 and len(bad_pool) >= 2:
        for p in bad_pool:
            if len(chosen) >= 5:
                break
            if p not in seen and bad_in_chosen < 1:
                chosen.append(p)
                seen.add(p)
                bad_in_chosen += 1

    if len(chosen) < 5:
        return None

    chosen_bad_count = bad_in_chosen

    # ── Build rejected ────────────────────────────────────────────────────────
    # Start from chosen, replace good slots with bad prompts not already in chosen
    bad_spare = [p for p in bad_pool if p not in set(chosen)]
    if not bad_spare:
        return None

    rejected = chosen[:]
    good_positions = [i for i, p in enumerate(chosen) if p in set(good_pool)]
    rejected_bad_count = chosen_bad_count

    for pos in reversed(good_positions):
        if not bad_spare:
            break
        candidate = bad_spare.pop(0)
        rejected[pos] = candidate
        rejected_bad_count += 1
        if rejected_bad_count > chosen_bad_count:
            break

    # Final validation
    actual_rejected_bad = sum(1 for p in rejected if p in bad_texts)
    if actual_rejected_bad <= chosen_bad_count:
        return None
    if rejected == chosen or len(set(rejected)) < 5 or len(set(chosen)) < 5:
        return None

    cot_block    = build_cot_block(anchor_row)
    user_message = build_user_message(lp)

    user_turn = {"role": "user", "content": user_message}

    return {
        "id":                 f"{url_hash}_dpo_refine",
        "url_hash":           url_hash,
        "lp_url":             lp.get("LPURL", ""),
        "good_pool_size":     len(good_pool),
        "bad_pool_size":      len(bad_pool),
        "chosen_bad_count":   chosen_bad_count,
        "rejected_bad_count": actual_rejected_bad,
        "system":             SYSTEM_PROMPT_COT,
        "messages":           [user_turn,
                               {"role": "assistant",
                                "content": build_response(cot_block, chosen)}],
        "rejected_messages":  [user_turn,
                               {"role": "assistant",
                                "content": build_response(cot_block, rejected)}],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",  default="data")
    parser.add_argument("--eval_ratio",  type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    print("Loading data ...")
    lp_by_url = load_sd_file(RAW_DIR / "UHRS2K_SD.tsv")
    # Internal100_SD.tsv has an extra leading index column
    lp_by_url.update(load_sd_file(RAW_DIR / "Internal100_SD.tsv", skip_col0=True))
    orig_prompts = load_tsv(RAW_DIR / "Internal100_Random100.GPT5PromptsCreator.0131-2.tsv")
    refine_pr    = load_tsv(RAW_DIR / "Internal100_Random100.GPT5PromptsCreator.0131-2_refineprompt.tsv")
    orig_label_rows   = load_tsv(RAW_DIR / "UHRS_Task_lp_quality_labeling_0204_LP200_0131-2_official_ZImage_images_cfg0_20260201-0426.tsv")
    refine_label_rows = load_tsv(RAW_DIR / "UHRS_Task_lp_quality_labeling_0227-PromptRefiner_Internal100_LP200_0131-2_refinePrompt_official_ZImage_images_cfg0_20260227-0530.tsv")

    orig_lbl   = aggregate_labels(orig_label_rows)
    refine_lbl = aggregate_labels(refine_label_rows, strip_prefix=r"0227_")

    # key = UrlHash_Tag  (e.g. 'Internal1001_Prompt3')
    orig_by_key   = {r["UrlHash"] + "_" + r["Tag"]: r for r in orig_prompts}
    # key = UrlHash_Tag  (e.g. 'Internal10047_Prompt1')
    refine_by_key = {r["UrlHash"]: r for r in refine_pr}

    # URL lookup: UrlHash -> LP URL
    hash_to_url = {r["UrlHash"].split("_Prompt")[0]: r["FinalDestinationURLUrl"].strip()
                   for r in orig_prompts}

    # ── Build pools per LP ───────────────────────────────────────────────────
    print("Building DPO pairs ...")
    tags = ["Prompt1", "Prompt2", "Prompt3", "Prompt4", "Prompt5"]
    # Unique base hashes (without _PromptN suffix)
    base_hashes = sorted({r["UrlHash"] for r in orig_prompts})

    all_pairs = []
    skipped_no_lp = skipped_ineligible = 0

    for url_hash in base_hashes:
        lp_url = hash_to_url.get(url_hash, "")
        lp = lp_by_url.get(lp_url)
        if lp is None:
            skipped_no_lp += 1
            continue

        good_pool: list[str] = []
        fair_pool: list[str] = []
        bad_pool:  list[str] = []
        seen_texts: set[str] = set()
        anchor_row = None

        for tag in tags:
            key = f"{url_hash}_{tag}"
            orig_row = orig_by_key.get(key)
            if orig_row is None:
                continue
            if anchor_row is None:
                anchor_row = orig_row

            orig_text   = orig_row["Prompt"].strip()
            refine_row  = refine_by_key.get(key)
            refine_text = refine_row["RefinePrompt"].strip() if refine_row else None

            o_lbl = orig_lbl.get(key)
            r_lbl = refine_lbl.get(key)

            def add_to_pool(text, label):
                if text and text not in seen_texts and label in ("good", "fair", "bad"):
                    seen_texts.add(text)
                    if label == "good":
                        good_pool.append(text)
                    elif label == "fair":
                        fair_pool.append(text)
                    else:
                        bad_pool.append(text)

            add_to_pool(orig_text,   o_lbl)
            add_to_pool(refine_text, r_lbl)

        if anchor_row is None:
            skipped_ineligible += 1
            continue

        pair = build_dpo_pair(url_hash, good_pool, fair_pool, bad_pool, lp, anchor_row)
        if pair is None:
            skipped_ineligible += 1
            continue

        all_pairs.append(pair)

    print(f"  DPO pairs built:       {len(all_pairs)}")
    print(f"  Skipped (no LP):       {skipped_no_lp}")
    print(f"  Skipped (ineligible):  {skipped_ineligible}")

    # ── Train / eval split ───────────────────────────────────────────────────
    unique_hashes = sorted({p["url_hash"] for p in all_pairs})
    random.shuffle(unique_hashes)
    n_eval    = max(1, int(len(unique_hashes) * args.eval_ratio))
    eval_set  = set(unique_hashes[:n_eval])
    train_set = set(unique_hashes[n_eval:])

    train_pairs = [p for p in all_pairs if p["url_hash"] in train_set]
    eval_pairs  = [p for p in all_pairs if p["url_hash"] in eval_set]

    print(f"\nSplit: train={len(train_pairs)}, eval={len(eval_pairs)}")

    # ── Verify chosen_bad < rejected_bad for all pairs ───────────────────────
    violation = sum(1 for p in all_pairs
                    if p["chosen_bad_count"] >= p["rejected_bad_count"])
    print(f"Constraint violations (chosen_bad >= rejected_bad): {violation}")

    chosen_bad_dist   = Counter(p["chosen_bad_count"]   for p in all_pairs)
    rejected_bad_dist = Counter(p["rejected_bad_count"] for p in all_pairs)
    print(f"chosen_bad distribution:   {dict(sorted(chosen_bad_dist.items()))}")
    print(f"rejected_bad distribution: {dict(sorted(rejected_bad_dist.items()))}")

    # ── Write ─────────────────────────────────────────────────────────────────
    write_jsonl(train_pairs, out_dir / "dpo_refine_train_cot.jsonl")
    write_jsonl(eval_pairs,  out_dir / "dpo_refine_eval_cot.jsonl")

    stats = {
        "total_pairs": len(all_pairs),
        "train": len(train_pairs),
        "eval":  len(eval_pairs),
        "skipped": {"no_lp": skipped_no_lp, "ineligible": skipped_ineligible},
        "chosen_bad_dist":   dict(sorted(chosen_bad_dist.items())),
        "rejected_bad_dist": dict(sorted(rejected_bad_dist.items())),
    }
    stats_path = out_dir / "dataset_stats_dpo_refine.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Stats -> {stats_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
