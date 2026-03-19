"""
Construct DPO training data from Random1000 UHRS annotations.

Strategy:
  - chosen  : 5 prompts ordered good > fair > bad, all unique
              (good prompts first, then fill with fair, then bad if needed)
  - rejected: same 5 slots but replace good prompts with bad ones where available,
              keeping the rest unique
  - Require: >= 3 good prompts AND >= 1 distinct bad prompt (text != any good text)
  - <think> CoT block is identical in chosen and rejected (same LP)

Swift DPO format (sharegpt-style with chosen/rejected):
  {
    "system": "...",
    "conversation": [{"role": "user", "content": "..."}],
    "chosen":   [{"role": "assistant", "content": "..."}],
    "rejected": [{"role": "assistant", "content": "..."}]
  }

Output:
  data/dpo_train_cot.jsonl
  data/dpo_eval_cot.jsonl
  data/dataset_stats_dpo.json

Usage:
  python prepare_dpo_random1k.py
  python prepare_dpo_random1k.py --eval_ratio 0.1 --seed 42
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

MIN_GOOD = 3  # min unique good prompts required

# reuse system prompt from prepare_data.py
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
# Loaders (same as prepare_data_random1k.py)
# ---------------------------------------------------------------------------

def load_sd_file(path: Path) -> dict:
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


def write_jsonl(samples: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(samples)} -> {path}")


# ---------------------------------------------------------------------------
# Label aggregation
# ---------------------------------------------------------------------------

def aggregate_labels(label_rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in label_rows:
        lp_id = row.get("lp_id", "")
        m = re.match(r"(Random1000_\d+_Prompt\d+)_", lp_id)
        key = m.group(1) if m else lp_id
        groups[key].append(row["FinalDecision"].strip().lower())

    result = {}
    for key, votes in groups.items():
        c = Counter(votes)
        g, b, f = c.get("good", 0), c.get("bad", 0), c.get("fair", 0)
        if g >= 2:   final = "good"
        elif b >= 2: final = "bad"
        elif f >= 2: final = "fair"
        elif g > b:  final = "good"
        elif b > g:  final = "bad"
        else:        final = "fair"
        result[key] = final
    return result


# ---------------------------------------------------------------------------
# CoT + user message builders (same as prepare_data_random1k.py)
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
    lp: dict,
    url_hash: str,
    labeled: dict,   # {tag: (label, prompt_text)}
    anchor_row: dict,
) -> dict | None:
    """
    labeled: {tag: (label, prompt_text)} for Prompt1..5

    chosen  : good first, then fair, then at most 1 bad (filler only when
              good+fair < 5 AND bad_new >= 2, so rejected can still have more bad)
    rejected: replace good slots with bad prompts so that
              rejected_bad_count > chosen_bad_count  (strict signal)

    Eligibility:
      - >= MIN_GOOD unique good prompts
      - >= 1 bad prompt distinct from good (bad_new)
      - sufficient bad_new to guarantee rejected_bad > chosen_bad after build

    Returns None if not eligible.
    """
    tags = ["Prompt1", "Prompt2", "Prompt3", "Prompt4", "Prompt5"]

    goods = [(t, labeled[t][1]) for t in tags if labeled[t][0] == "good"]
    fairs = [(t, labeled[t][1]) for t in tags if labeled[t][0] == "fair"]
    bads  = [(t, labeled[t][1]) for t in tags if labeled[t][0] == "bad"]

    good_texts = set(p for _, p in goods)
    bad_new    = [(t, p) for t, p in bads if p not in good_texts]

    if len(goods) < MIN_GOOD or not bad_new:
        return None

    # ── Build chosen: good > fair > at most 1 bad ────────────────────────────
    # Only use bad as filler if we still need slots AND have >= 2 bad_new
    # (so rejected can replace a good slot with a fresh bad, giving rejected_bad > chosen_bad).
    seen = set()
    chosen_prompts = []
    bad_used_in_chosen = 0
    for _, p in goods + fairs:
        if p not in seen and len(chosen_prompts) < 5:
            chosen_prompts.append(p)
            seen.add(p)
    # Fill remaining slots with bad only if we still have a spare bad_new for rejected
    if len(chosen_prompts) < 5 and len(bad_new) >= 2:
        for _, p in bads:
            if len(chosen_prompts) >= 5:
                break
            if p not in seen and bad_used_in_chosen < 1:
                chosen_prompts.append(p)
                seen.add(p)
                bad_used_in_chosen += 1
    if len(chosen_prompts) < 5:
        return None

    chosen_bad_count = bad_used_in_chosen  # 0 or 1
    bad_texts_all    = set(p for _, p in bads)

    # ── Build rejected: replace good slots with bad_new ──────────────────────
    # Goal: rejected_bad_count = chosen_bad_count + N (N >= 1)
    # Each bad_new not already in rejected adds +1 bad to rejected.
    # bad_new already in rejected (filler slot) can be swapped to a good slot —
    # the good prompt moves to filler. That doesn't increase bad count,
    # but ensures bad sits in earlier (worse quality) positions.
    rejected_prompts   = chosen_prompts[:]
    good_positions     = [i for i, p in enumerate(chosen_prompts) if p in good_texts]
    rejected_bad_count = sum(1 for p in rejected_prompts if p in bad_texts_all)

    bad_idx = 0
    for pos in reversed(good_positions):
        if bad_idx >= len(bad_new):
            break
        _, candidate = bad_new[bad_idx]
        bad_idx += 1
        if candidate in set(rejected_prompts):
            # already a filler slot — swap to surface it at the good position
            old_pos = rejected_prompts.index(candidate)
            displaced_good = chosen_prompts[pos]
            rejected_prompts[pos]     = candidate
            rejected_prompts[old_pos] = displaced_good
            # bad count unchanged
        else:
            rejected_prompts[pos] = candidate
            rejected_bad_count += 1
        # Stop once we have strictly more bad than chosen
        if rejected_bad_count > chosen_bad_count:
            break

    # Final validation
    actual_rejected_bad = sum(1 for p in rejected_prompts if p in bad_texts_all)
    if actual_rejected_bad <= chosen_bad_count:
        return None
    if rejected_prompts == chosen_prompts or len(set(rejected_prompts)) < 5:
        return None

    cot_block    = build_cot_block(anchor_row)
    user_message = build_user_message(lp)

    return {
        "id":       f"{url_hash}_dpo",
        "url_hash": url_hash,
        "lp_url":   lp.get("LPURL", ""),
        "good_prompt_count":          len(goods),
        "bad_prompt_count":           len(bads),
        "chosen_bad_count":           chosen_bad_count,
        "rejected_bad_count":         actual_rejected_bad,
        "system":      SYSTEM_PROMPT_COT,
        "conversation": [{"role": "user", "content": user_message}],
        "chosen":   [{"role": "assistant",
                      "content": build_response(cot_block, chosen_prompts)}],
        "rejected": [{"role": "assistant",
                      "content": build_response(cot_block, rejected_prompts)}],
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
    lp_by_url   = load_sd_file(RAW_DIR / "UHRS2K_SD.tsv")
    prompt_rows = load_tsv(RAW_DIR / "UHRS2K_SD_Sample1000.GPT5PromptsCreator.0131-2-latest.tsv")
    label_rows  = load_tsv(RAW_DIR / "UHRS_Task_lp_labeling_0306_random1K_Quality.tsv")

    label_lookup = aggregate_labels(label_rows)

    by_hash = defaultdict(dict)
    hash_to_url = {}
    for r in prompt_rows:
        by_hash[r["UrlHash"]][r["Tag"]] = r
        hash_to_url[r["UrlHash"]] = r["FinalDestinationURLUrl"].strip()

    # ── Build pairs ───────────────────────────────────────────────────────────
    print("Building DPO pairs ...")
    all_pairs = []
    skipped_no_lp = skipped_ineligible = 0

    for url_hash, tag_rows in by_hash.items():
        lp_url = hash_to_url.get(url_hash, "")
        lp     = lp_by_url.get(lp_url)
        if lp is None:
            skipped_no_lp += 1
            continue

        tags = ["Prompt1", "Prompt2", "Prompt3", "Prompt4", "Prompt5"]
        labeled = {}
        anchor_row = None
        for tag in tags:
            row = tag_rows.get(tag)
            if row is None:
                continue
            key = f"{url_hash}_{tag}"
            lbl = label_lookup.get(key, "skip")
            labeled[tag] = (lbl, row["Prompt"].strip())
            if anchor_row is None:
                anchor_row = row

        if len(labeled) < 5 or anchor_row is None:
            skipped_ineligible += 1
            continue

        pair = build_dpo_pair(lp, url_hash, labeled, anchor_row)
        if pair is None:
            skipped_ineligible += 1
            continue

        all_pairs.append(pair)

    print(f"  DPO pairs built:       {len(all_pairs)}")
    print(f"  Skipped (no LP):       {skipped_no_lp}")
    print(f"  Skipped (ineligible):  {skipped_ineligible}")

    # ── Train / eval split at LP level ───────────────────────────────────────
    unique_hashes = sorted({p["url_hash"] for p in all_pairs})
    random.shuffle(unique_hashes)
    n_eval    = max(1, int(len(unique_hashes) * args.eval_ratio))
    eval_set  = set(unique_hashes[:n_eval])
    train_set = set(unique_hashes[n_eval:])

    train_pairs = [p for p in all_pairs if p["url_hash"] in train_set]
    eval_pairs  = [p for p in all_pairs if p["url_hash"] in eval_set]

    print(f"\nSplit: train={len(train_pairs)}, eval={len(eval_pairs)}")

    # ── Write ─────────────────────────────────────────────────────────────────
    write_jsonl(train_pairs, out_dir / "dpo_train_cot.jsonl")
    write_jsonl(eval_pairs,  out_dir / "dpo_eval_cot.jsonl")

    # ── Stats ─────────────────────────────────────────────────────────────────
    good_dist = Counter(p["good_prompt_count"] for p in all_pairs)
    stats = {
        "total_pairs": len(all_pairs),
        "train": len(train_pairs),
        "eval":  len(eval_pairs),
        "skipped": {"no_lp": skipped_no_lp, "ineligible": skipped_ineligible},
        "good_prompt_count_dist": dict(sorted(good_dist.items())),
    }
    stats_path = out_dir / "dataset_stats_dpo.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Stats -> {stats_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
