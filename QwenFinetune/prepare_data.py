"""
Data preparation for Qwen3.5-35B-A3B LoRA SFT finetuning.
TRUE ONE-STEP MODE: raw LP fields -> 5 good image prompts.
Supports optional Chain-of-Thought (CoT) format using GPT5 Step1 output.

This replaces the GPT5 two-step pipeline entirely.
At serving time, the model receives the same raw LP fields available online
(crawled from the landing page) and directly outputs 5 image prompts.

============================================================
Input files (all in RawData/ subdirectory):
============================================================

[Raw LP fields — available at serving time]
  1. Internal100_SD.tsv       (100 LPs, "Internal" set)
  2. UHRS2K_SD_Sample100.tsv  (100 LPs, "Random" set)

  No header row. Columns (13):
    RowId, LPURL, ImageURL, label,
    DocumentTitle, VisualTitle, Heading,
    Title_CB, VisualTitle_CB, Heading_CB,
    BestSnippet_CB, MetaDescription_CB, PrimaryContentNoTitleNoHeading

  Note: DocumentTitle is often empty in online serving; all other fields
  match exactly what the online crawler provides.

[Prompt text (GPT5 generated, used as supervision targets)]
  3. Internal100_Random100.GPT5PromptsCreator.0131-2.tsv
       Original GPT5 prompts
       Schema: Tag, UrlHash, RowId, FinalDestinationURLUrl, ImageURL,
               label (stale, ignored), ProductIntent, ProductCategory,
               VisualContext, AudienceAndContext, ValueSignals,
               ConfidenceLevel, Prompt
       Join key: FinalDestinationURLUrl == LPURL

  4. Internal100_Random100.GPT5PromptsCreator.0131-2_refineprompt.tsv
       GPT5 refine prompts
       Schema: UrlHash (= "{LPHash}_{PromptTag}"), Prompt, RefinePrompt

[Human annotation — 3 judges per image, majority vote]
  5. UHRS_Task_lp_quality_labeling_0204_LP200_0131-2_official_...tsv
       Labels for original GPT5 prompt images
       Image filename stem: {UrlHash}_{Tag}  e.g. Internal1001_Prompt3

  6. UHRS_Task_lp_quality_labeling_0227-PromptRefiner_...tsv
       Labels for refined prompt images
       Image filename stem: 0227_{UrlHash}_{Tag}  (strip "0227_" prefix)

============================================================
Label aggregation (majority vote across 3 judges):
  Good  : >= 2 votes Good
  Bad   : >= 2 votes Bad or Logo
  Fair  : ambiguous (excluded by default; --include_fair_as_bad to keep)
  Skip  : Imageloadfail or no annotation

============================================================
SFT training format:

  WITHOUT CoT (--no_cot):
    [system]    <role>
    [user]      Raw LP fields
    [assistant] <Prompt1>...</Prompt1> ... <Prompt5>...</Prompt5>

  WITH CoT (default, --cot):
    [system]    <role — instructs model to think first, then output prompts>
    [user]      Raw LP fields
    [assistant] <think>
                ProductType: ...
                SpecificProduct: ...
                Category: ...
                VisualAnchors: ...
                LifestyleVibe: ...
                CoreValueSignals: ...
                </think>
                <Prompt1>...</Prompt1> ... <Prompt5>...</Prompt5>

  CoT fields are taken from the GPT5 Step1 output stored in the prompt TSV.
  Both original and refine versions share the same Step1 output (same LP).

============================================================
Output:
  data/sft_train.jsonl        (or data/sft_train_cot.jsonl with CoT)
  data/sft_eval.jsonl         (or data/sft_eval_cot.jsonl with CoT)
  data/dataset_stats.json

Usage:
  python prepare_data.py                   # CoT mode (default)
  python prepare_data.py --no_cot          # plain mode, no CoT
  python prepare_data.py --include_fair_as_bad
  python prepare_data.py --eval_ratio 0.15 --seed 0
"""

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
RAW_DIR = SCRIPT_DIR / "RawData"

# Column names for SD files (no header row)
SD_COLS = [
    "RowId", "LPURL", "ImageURL", "label",
    "DocumentTitle", "VisualTitle", "Heading",
    "Title_CB", "VisualTitle_CB", "Heading_CB",
    "BestSnippet_CB", "MetaDescription_CB",
    "PrimaryContentNoTitleNoHeading",
]

# ---------------------------------------------------------------------------
# System prompts  (must match inference.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_PLAIN = (
    "You are an expert Ad Creative Director and Senior AI Image Prompt Engineer, "
    "specialized in high-performing Native Advertisement visuals.\n\n"
    "Given a landing page URL and its extracted content fields, your task is to "
    "generate five (5) high-quality English image generation prompts for Native Ads.\n\n"
    "Each prompt must:\n"
    "- Be <=150 words\n"
    "- Embed all safety, realism, quality, and exclusion constraints\n"
    "- Feel native and non-promotional\n"
    "- Show the product outcome or value naturally in context\n"
    "- Avoid stereotypes, text/logos in image, and stock-photo aesthetics\n"
    "- Ensure correct anatomy, natural hands, sharp focus, clean composition\n\n"
    "Output exactly 5 prompts in this format:\n"
    "<Prompt1>...</Prompt1>\n"
    "<Prompt2>...</Prompt2>\n"
    "<Prompt3>...</Prompt3>\n"
    "<Prompt4>...</Prompt4>\n"
    "<Prompt5>...</Prompt5>"
)

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
# File loaders
# ---------------------------------------------------------------------------

def load_sd_file(filename: str) -> list[dict]:
    """Load an SD TSV file (no header row) into list of dicts."""
    path = RAW_DIR / filename
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            while len(parts) < len(SD_COLS):
                parts.append("")
            row = dict(zip(SD_COLS, parts))
            rows.append(row)
    return rows


def load_tsv(filename: str) -> list[dict]:
    """Load a TSV file with header row."""
    path = RAW_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ---------------------------------------------------------------------------
# Label aggregation
# ---------------------------------------------------------------------------

def normalize_decision(label: str) -> str:
    label = label.strip().lower()
    if label == "good":
        return "good"
    if label in ("bad", "logo"):
        return "bad"
    if label == "fair":
        return "fair"
    return "skip"


def aggregate_labels(
    label_rows: list[dict],
    strip_prefix: str = "",
    include_fair_as_bad: bool = False,
) -> dict:
    """
    Group UHRS annotation rows by image key (URL stem), apply majority vote.
    Returns {img_key: {"label": ..., "votes": {...}, "bad_reasons": [...]}}
    """
    groups = defaultdict(list)
    for row in label_rows:
        stem = os.path.splitext(os.path.basename(row["ImgUrl"]))[0]
        if strip_prefix and stem.startswith(strip_prefix):
            stem = stem[len(strip_prefix):]
        groups[stem].append(row)

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
            g = votes.get("good", 0)
            b = votes.get("bad", 0)
            fair = votes.get("fair", 0)
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
# User message builder  (raw LP fields -> user turn text)
# ---------------------------------------------------------------------------

def build_user_message(lp: dict) -> str:
    """
    Build user turn from raw LP fields.
    Mirrors exactly what is available at online serving time.
    """
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


def format_prompts_output(prompts: list) -> str:
    return "\n".join(
        f"<Prompt{i}>{p.strip()}</Prompt{i}>" for i, p in enumerate(prompts, 1)
    )


def build_cot_block(anchor_row: dict) -> str:
    """
    Build the <think>...</think> CoT block from GPT5 Step1 fields.
    anchor_row is one row from the prompt TSV for this LP (any tag works,
    since Step1 fields are the same for all 5 prompts of the same LP).

    Extracts the 3 high-signal fields:
      ProductIntent  -> ProductType + SpecificProduct
      ProductCategory -> Category
      VisualContext  -> VisualAnchors + LifestyleVibe
      ValueSignals   -> CoreValueSignals  (optional, included when present)
    """
    def clean(text: str) -> str:
        """Strip leading bullet/dash, collapse whitespace."""
        text = text.strip()
        # Remove leading "- " patterns e.g. "- ProductType: Service"
        text = text.lstrip("-").strip()
        return text

    lines = []

    # --- ProductIntent: extract ProductType and SpecificProduct ---
    intent_raw = clean(anchor_row.get("ProductIntent", ""))
    if intent_raw:
        # Try to split "ProductType: X - SpecificProduct: Y"
        ptype, specific = "", ""
        for part in intent_raw.replace(" - ", "\n").splitlines():
            part = part.strip()
            if part.lower().startswith("producttype:"):
                ptype = part.split(":", 1)[1].strip()
            elif part.lower().startswith("specificproduct:"):
                specific = part.split(":", 1)[1].strip()
        if ptype:
            lines.append(f"ProductType: {ptype}")
        if specific:
            lines.append(f"SpecificProduct: {specific}")

    # --- ProductCategory: extract Category ---
    cat_raw = clean(anchor_row.get("ProductCategory", ""))
    if cat_raw:
        cat = ""
        for part in cat_raw.splitlines():
            part = part.strip().lstrip("-").strip()
            if part.lower().startswith("category:"):
                cat = part.split(":", 1)[1].strip()
                break
        if not cat:
            cat = cat_raw  # fallback: use raw value
        lines.append(f"Category: {cat}")

    # --- VisualContext: extract VisualAnchors and LifestyleVibe ---
    visual_raw = clean(anchor_row.get("VisualContext", ""))
    if visual_raw:
        anchors, vibe = "", ""
        for part in visual_raw.replace(" - ", "\n").splitlines():
            part = part.strip().lstrip("-").strip()
            if part.lower().startswith("visualanchors:"):
                anchors = part.split(":", 1)[1].strip()
            elif part.lower().startswith("lifestylevibe:"):
                vibe = part.split(":", 1)[1].strip()
        if anchors:
            lines.append(f"VisualAnchors: {anchors}")
        if vibe:
            lines.append(f"LifestyleVibe: {vibe}")

    # --- ValueSignals: CoreValueSignals ---
    val_raw = clean(anchor_row.get("ValueSignals", ""))
    if val_raw:
        signals = ""
        for part in val_raw.splitlines():
            part = part.strip().lstrip("-").strip()
            if part.lower().startswith("corevaluesignals:"):
                signals = part.split(":", 1)[1].strip()
                break
        if not signals:
            signals = val_raw
        lines.append(f"CoreValueSignals: {signals}")

    if not lines:
        return ""
    return "<think>\n" + "\n".join(lines) + "\n</think>"


# ---------------------------------------------------------------------------
# SFT sample builder
# ---------------------------------------------------------------------------

def build_sft_sample(
    lp: dict,
    url_hash: str,
    prompt_rows_for_lp: list,
    label_lookup: dict,
    version_tag: str,
    refine_map: dict,
    use_cot: bool = True,
):
    """
    Build one SFT training sample for a given LP + prompt version.

    Assistant turn format:
      use_cot=True:
        <think>
        ProductType: ...
        SpecificProduct: ...
        Category: ...
        VisualAnchors: ...
        LifestyleVibe: ...
        CoreValueSignals: ...
        </think>
        <Prompt1>...</Prompt1>
        ...
        <Prompt5>...</Prompt5>

      use_cot=False:
        <Prompt1>...</Prompt1>
        ...
        <Prompt5>...</Prompt5>

    Returns None if there are no good prompts.
    """
    tag_order = ["Prompt1", "Prompt2", "Prompt3", "Prompt4", "Prompt5"]
    row_by_tag = {r["Tag"]: r for r in prompt_rows_for_lp}

    good_prompts = []
    bad_prompts = []
    all_bad_reasons = []
    # anchor_row: use any row from this LP for Step1 fields (same for all 5 prompts)
    anchor_row = prompt_rows_for_lp[0] if prompt_rows_for_lp else {}

    for tag in tag_order:
        row = row_by_tag.get(tag)
        if row is None:
            continue
        img_key = f"{url_hash}_{tag}"

        if version_tag == "refine":
            prompt_text = refine_map.get(img_key, "").strip()
            if not prompt_text:
                continue
        else:
            prompt_text = row.get("Prompt", "").strip()

        ann = label_lookup.get(img_key)
        if ann is None:
            continue

        if ann["label"] == "good":
            good_prompts.append(prompt_text)
        elif ann["label"] == "bad":
            bad_prompts.append(prompt_text)
            all_bad_reasons.extend(ann["bad_reasons"])

    if not good_prompts:
        return None

    # Cycle good prompts to fill 5 slots
    padded = good_prompts[:]
    while len(padded) < 5:
        padded.append(good_prompts[len(padded) % len(good_prompts)])
    padded = padded[:5]

    # Build assistant content
    prompts_block = format_prompts_output(padded)
    if use_cot:
        cot_block = build_cot_block(anchor_row)
        assistant_content = f"{cot_block}\n{prompts_block}" if cot_block else prompts_block
        system_prompt = SYSTEM_PROMPT_COT
    else:
        assistant_content = prompts_block
        system_prompt = SYSTEM_PROMPT_PLAIN

    return {
        "id": f"{url_hash}_{version_tag}",
        "url_hash": url_hash,
        "lp_url": lp.get("LPURL", ""),
        "version": version_tag,
        "good_prompt_count": len(good_prompts),
        "bad_prompt_count": len(bad_prompts),
        "bad_reasons": list(set(all_bad_reasons)),
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": build_user_message(lp)},
            {"role": "assistant", "content": assistant_content},
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare one-step SFT data: raw LP fields -> good image prompts"
    )
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--include_fair_as_bad", action="store_true", default=False)
    parser.add_argument("--no_cot", action="store_true", default=False,
                        help="Disable CoT <think> block (default: CoT enabled)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_cot = not args.no_cot
    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {'CoT' if use_cot else 'Plain'} (use --no_cot to disable CoT)")

    # -----------------------------------------------------------------------
    # Load files
    # -----------------------------------------------------------------------
    print("Loading files ...")
    sd_internal     = load_sd_file("Internal100_SD.tsv")
    sd_uhrs         = load_sd_file("UHRS2K_SD_Sample100.tsv")
    prompt_rows     = load_tsv("Internal100_Random100.GPT5PromptsCreator.0131-2.tsv")
    refine_rows     = load_tsv("Internal100_Random100.GPT5PromptsCreator.0131-2_refineprompt.tsv")
    label_rows_0204 = load_tsv(
        "UHRS_Task_lp_quality_labeling_0204_LP200_0131-2_official_ZImage_images_cfg0_20260201-0426.tsv"
    )
    label_rows_0227 = load_tsv(
        "UHRS_Task_lp_quality_labeling_0227-PromptRefiner_Internal100_LP200_0131-2_refinePrompt_official_ZImage_images_cfg0_20260227-0530.tsv"
    )

    print(f"  Internal100_SD:  {len(sd_internal)} rows")
    print(f"  UHRS2K_SD:       {len(sd_uhrs)} rows")
    print(f"  Prompt rows:     {len(prompt_rows)}")
    print(f"  Refine rows:     {len(refine_rows)}")
    print(f"  Labels 0204:     {len(label_rows_0204)}")
    print(f"  Labels 0227:     {len(label_rows_0227)}")

    # -----------------------------------------------------------------------
    # LP lookup: LPURL -> raw SD fields dict
    # -----------------------------------------------------------------------
    lp_by_url = {}
    for row in sd_internal + sd_uhrs:
        url = row["LPURL"].strip()
        if url:
            lp_by_url[url] = row

    print(f"\nUnique LPs with raw fields: {len(lp_by_url)}")

    # -----------------------------------------------------------------------
    # Prompt lookup: UrlHash -> [prompt rows]  and  UrlHash -> LPURL
    # -----------------------------------------------------------------------
    prompts_by_hash = defaultdict(list)
    hash_to_url = {}
    for row in prompt_rows:
        h = row["UrlHash"]
        prompts_by_hash[h].append(row)
        hash_to_url[h] = row["FinalDestinationURLUrl"].strip()

    # Refine lookup: {UrlHash_Tag -> RefinePrompt}
    refine_map = {r["UrlHash"]: r["RefinePrompt"].strip() for r in refine_rows}

    # -----------------------------------------------------------------------
    # Aggregate UHRS labels
    # -----------------------------------------------------------------------
    print("Aggregating UHRS labels (majority vote) ...")
    labels_orig   = aggregate_labels(
        label_rows_0204, strip_prefix="",
        include_fair_as_bad=args.include_fair_as_bad
    )
    labels_refine = aggregate_labels(
        label_rows_0227, strip_prefix="0227_",
        include_fair_as_bad=args.include_fair_as_bad
    )

    print(f"  Original label dist: {dict(Counter(v['label'] for v in labels_orig.values()))}")
    print(f"  Refine   label dist: {dict(Counter(v['label'] for v in labels_refine.values()))}")

    # -----------------------------------------------------------------------
    # Build SFT samples
    # -----------------------------------------------------------------------
    print("\nBuilding SFT samples ...")
    all_samples = []
    skipped_no_lp = 0
    skipped_no_good = 0

    for url_hash, lp_rows in prompts_by_hash.items():
        lp_url = hash_to_url.get(url_hash, "")
        lp = lp_by_url.get(lp_url)
        if lp is None:
            skipped_no_lp += 1
            continue

        sample_orig = build_sft_sample(
            lp, url_hash, lp_rows,
            label_lookup=labels_orig,
            version_tag="original",
            refine_map=refine_map,
            use_cot=use_cot,
        )
        if sample_orig:
            all_samples.append(sample_orig)
        else:
            skipped_no_good += 1

        sample_refine = build_sft_sample(
            lp, url_hash, lp_rows,
            label_lookup=labels_refine,
            version_tag="refine",
            refine_map=refine_map,
            use_cot=use_cot,
        )
        if sample_refine:
            all_samples.append(sample_refine)
        else:
            skipped_no_good += 1

    print(f"  Total SFT samples:      {len(all_samples)}")
    print(f"  Skipped (no LP fields): {skipped_no_lp}")
    print(f"  Skipped (no good):      {skipped_no_good}")
    print(f"  Good-prompt dist:       {dict(sorted(Counter(s['good_prompt_count'] for s in all_samples).items()))}")

    # -----------------------------------------------------------------------
    # Train / eval split at LP level (no data leakage)
    # -----------------------------------------------------------------------
    unique_hashes = sorted({s["url_hash"] for s in all_samples})
    random.shuffle(unique_hashes)
    n_eval = max(1, int(len(unique_hashes) * args.eval_ratio))
    eval_set  = set(unique_hashes[:n_eval])
    train_set = set(unique_hashes[n_eval:])

    train_samples = [s for s in all_samples if s["url_hash"] in train_set]
    eval_samples  = [s for s in all_samples if s["url_hash"] in eval_set]

    print(f"\nSplit (eval_ratio={args.eval_ratio}):")
    print(f"  Train: {len(train_set)} LPs, {len(train_samples)} samples")
    print(f"  Eval:  {len(eval_set)} LPs, {len(eval_samples)} samples")

    # -----------------------------------------------------------------------
    # Write JSONL
    # -----------------------------------------------------------------------
    suffix = "_cot" if use_cot else ""

    def write_jsonl(samples, path):
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(samples)} -> {path}")

    write_jsonl(train_samples, out_dir / f"sft_train{suffix}.jsonl")
    write_jsonl(eval_samples,  out_dir / f"sft_eval{suffix}.jsonl")

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------
    def vstats(samples):
        by_v = defaultdict(list)
        for s in samples:
            by_v[s["version"]].append(s["good_prompt_count"])
        return {
            v: {"count": len(c), "avg_good": round(sum(c)/len(c), 2),
                "dist": dict(Counter(c))}
            for v, c in by_v.items()
        }

    stats = {
        "mode": f"one-step ({'CoT' if use_cot else 'plain'})",
        "include_fair_as_bad": args.include_fair_as_bad,
        "label_dist": {
            "original": dict(Counter(v["label"] for v in labels_orig.values())),
            "refine":   dict(Counter(v["label"] for v in labels_refine.values())),
        },
        "train": vstats(train_samples),
        "eval":  vstats(eval_samples),
        "skipped": {"no_lp_fields": skipped_no_lp, "no_good_prompts": skipped_no_good},
    }
    stats_path = out_dir / f"dataset_stats{suffix}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Stats -> {stats_path}")

    # -----------------------------------------------------------------------
    # Sanity check: show one sample
    # -----------------------------------------------------------------------
    if train_samples:
        s = train_samples[0]
        def safe(text, n=120):
            return text[:n].encode("ascii", errors="replace").decode("ascii")
        print(f"\n--- Sample: {s['id']} (version={s['version']}, good={s['good_prompt_count']}) ---")
        print(f"  [user snippet]")
        for part in s["messages"][1]["content"].split("\n\n")[:4]:
            if part.strip():
                print(f"    {safe(part)}")
        print(f"  [assistant]")
        # Show <think> block if present
        asst = s["messages"][2]["content"]
        if "<think>" in asst:
            think_end = asst.find("</think>") + len("</think>")
            print(f"    {safe(asst[:think_end], 400)}")
            print(f"    {safe(asst[think_end:think_end+180])}...")
        else:
            print(f"    {safe(asst, 200)}...")

    print("\nDone.")


if __name__ == "__main__":
    main()
