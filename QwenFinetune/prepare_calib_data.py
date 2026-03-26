"""
Prepare a mixed calibration dataset for AWQ/GPTQ quantization.

Mixes:
  - Private data : infer_input.jsonl built from UHRS2K_SD_Random200_0324.tsv
                   (user-only turns, no assistant — matches real serving distribution)
  - Public data  : wikitext-2-raw-v1 (plain text wrapped as user/assistant turns)

Output format per line (swift --calib_dataset compatible):
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Usage:
  # Basic (equal split, total 128)
  python prepare_calib_data.py

  # Custom counts
  python prepare_calib_data.py --n_private 64 --n_public 64 --output_jsonl data/calib_data.jsonl

  # Use a pre-built infer_input.jsonl directly
  python prepare_calib_data.py --private_jsonl data/infer_input_custom.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent

# System prompt used during DPO training — must match for calibration to be meaningful
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
    "<think>\nProductType: ...\nSpecificProduct: ...\nCategory: ...\n"
    "VisualAnchors: ...\nLifestyleVibe: ...\nCoreValueSignals: ...\n</think>\n"
    "<Prompt1>...</Prompt1>\n<Prompt2>...</Prompt2>\n<Prompt3>...</Prompt3>\n"
    "<Prompt4>...</Prompt4>\n<Prompt5>...</Prompt5>"
)

WIKITEXT_SYSTEM = (
    "You are a helpful assistant. Answer the user's question clearly and concisely."
)


# ---------------------------------------------------------------------------
# Private data: read from dpo_refine_eval_cot.jsonl (has real assistant turns)
# or build on-the-fly from the raw TSV + eval JSONL for assistant turns
# ---------------------------------------------------------------------------

# Dummy assistant response used when no real GT is available
_DUMMY_ASSISTANT = (
    "<think>\nProductType: Physical Product\nSpecificProduct: consumer product\n"
    "Category: retail\nVisualAnchors: product, person, environment\n"
    "LifestyleVibe: everyday lifestyle\nCoreValueSignals: reliable, simple\n</think>\n"
    "<Prompt1>A person using the product in a natural setting. Correct anatomy, "
    "natural hands, sharp focus, clean composition, no logos, no watermark.</Prompt1>\n"
    "<Prompt2>Close-up detail of the product on a clean surface. Sharp focus, "
    "realistic textures, no text, no logos, no watermark.</Prompt2>\n"
    "<Prompt3>Lifestyle scene showing product outcome. Natural lighting, candid, "
    "subject not looking at camera, no logos, no watermark.</Prompt3>\n"
    "<Prompt4>Outdoor context with product in use. Natural environment, correct anatomy, "
    "clean composition, no logos, no watermark.</Prompt4>\n"
    "<Prompt5>Indoor everyday scene featuring the product. Warm lighting, realistic, "
    "no extra text, no logos, no watermark.</Prompt5>"
)


def load_private_records(jsonl_path: Path, n: int, seed: int) -> list[dict]:
    """
    Load records from an infer_input.jsonl (user-turn-only format).
    Adds a dummy assistant turn so GPTQ calibration has complete sequences.
    """
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            messages = r.get("messages", [])
            system   = r.get("system", SYSTEM_PROMPT_COT)
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            if not user_msg:
                continue
            records.append({
                "messages": [
                    {"role": "system",    "content": system},
                    {"role": "user",      "content": user_msg},
                    {"role": "assistant", "content": _DUMMY_ASSISTANT},
                ]
            })

    random.seed(seed)
    random.shuffle(records)
    sampled = records[:n]
    print(f"[private]  loaded {len(records)} records, sampled {len(sampled)}")
    return sampled


def load_private_from_gt(gt_jsonl: Path, n: int, seed: int) -> list[dict]:
    """
    Load from dpo_refine_eval_cot.jsonl which has real assistant turns.
    Preferred over TSV when available — higher calibration quality.
    """
    records = []
    with open(gt_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            messages = r.get("messages", [])
            system   = r.get("system", SYSTEM_PROMPT_COT)
            user_msg  = next((m["content"] for m in messages if m["role"] == "user"), "")
            asst_msg  = next((m["content"] for m in messages if m["role"] == "assistant"), "")
            if not user_msg or not asst_msg:
                continue
            records.append({
                "messages": [
                    {"role": "system",    "content": system},
                    {"role": "user",      "content": user_msg},
                    {"role": "assistant", "content": asst_msg},
                ]
            })

    random.seed(seed)
    random.shuffle(records)
    sampled = records[:n]
    print(f"[private-gt] loaded {len(records)} GT records, sampled {len(sampled)}")
    return sampled


def build_private_from_tsv(tsv_path: Path, n: int, seed: int) -> list[dict]:
    """
    Build private records from the raw TSV.
    Uses dummy assistant turn for GPTQ calibration completeness.
    For higher quality calibration, prefer load_private_from_gt() with the eval JSONL.
    """
    import csv
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

    def build_user_message(row):
        parts = [
            "Generate 5 image generation prompts for a Native Ad based on the "
            "following landing page information:\n"
        ]
        for key, label in field_labels:
            val = (row.get(key) or "").strip()
            if val:
                parts.append(f"[{label}]\n{val}")
        return "\n\n".join(parts)

    records = []
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            user_msg = build_user_message(row)
            records.append({
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT_COT},
                    {"role": "user",      "content": user_msg},
                    {"role": "assistant", "content": _DUMMY_ASSISTANT},
                ]
            })

    random.seed(seed)
    random.shuffle(records)
    sampled = records[:n]
    print(f"[private]  built {len(records)} records from TSV, sampled {len(sampled)}")
    return sampled


# ---------------------------------------------------------------------------
# Public data: wikitext-2-raw-v1 via HuggingFace datasets
# ---------------------------------------------------------------------------

def load_public_records(n: int, seed: int) -> list[dict]:
    """
    Sample n records from wikitext-2-raw-v1 (test split).
    Each article is wrapped as a user question + assistant answer turn
    so the model sees realistic assistant-style text during calibration.

    Offline / mirror support:
      - Set HF_DATASETS_OFFLINE=1 to use local cache only
      - Set HF_ENDPOINT=https://hf-mirror.com to use a mirror
      - Set WIKITEXT_CACHE_DIR=/path/to/local/wikitext to load from a
        pre-downloaded directory (plain .txt files, one passage per line)
    """
    import os

    # ── Fallback: local plain-text cache ─────────────────────────────────────
    local_cache = os.environ.get("WIKITEXT_CACHE_DIR", "")
    if local_cache and Path(local_cache).exists():
        print(f"[public]   loading wikitext from local cache: {local_cache}")
        passages = []
        for txt_file in sorted(Path(local_cache).glob("*.txt")):
            for line in txt_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if len(line.split()) >= 50:
                    passages.append(line)
        random.seed(seed)
        random.shuffle(passages)
        records = []
        for passage in passages[:n]:
            records.append({
                "messages": [
                    {"role": "system",    "content": WIKITEXT_SYSTEM},
                    {"role": "user",      "content": f"Summarize the following passage:\n\n{passage}"},
                    {"role": "assistant", "content": passage[:500]},
                ]
            })
        print(f"[public]   sampled {len(records)} passages from local cache")
        return records

    # ── HuggingFace (online or offline cache) ─────────────────────────────────
    try:
        from datasets import load_dataset
    except ImportError:
        print("[WARN] 'datasets' package not installed. Skipping public data.")
        print("       Install with: pip install datasets")
        return []

    # Allow mirror via env var (e.g. HF_ENDPOINT=https://hf-mirror.com)
    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if hf_endpoint:
        print(f"[public]   using HF mirror: {hf_endpoint}")

    print("[public]   loading wikitext-2-raw-v1 ...")
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as e:
        print(f"[WARN] Failed to load wikitext from HuggingFace: {e}")
        print("       Tips:")
        print("         1. Set HF_ENDPOINT=https://hf-mirror.com to use a mirror")
        print("         2. Pre-download and set WIKITEXT_CACHE_DIR=/path/to/local/wikitext")
        print("         3. Set HF_DATASETS_OFFLINE=1 if dataset is already cached")
        return []

    # Filter out empty / header lines, then chunk into ~300-word passages
    passages = []
    buf = []
    for item in ds:
        text = item["text"].strip()
        if not text or text.startswith(" ="):
            if buf:
                passages.append(" ".join(buf))
                buf = []
        else:
            buf.append(text)
            if len(" ".join(buf).split()) >= 300:
                passages.append(" ".join(buf))
                buf = []
    if buf:
        passages.append(" ".join(buf))

    random.seed(seed)
    random.shuffle(passages)

    records = []
    for passage in passages[:n]:
        # Wrap as a reading-comprehension style turn so the model generates text
        records.append({
            "messages": [
                {"role": "system",    "content": WIKITEXT_SYSTEM},
                {"role": "user",      "content": f"Summarize the following passage:\n\n{passage}"},
                {"role": "assistant", "content": passage[:500]},  # use passage itself as pseudo-answer
            ]
        })

    print(f"[public]   sampled {len(records)} wikitext passages")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--private_jsonl",
        default="",
        help="Pre-built infer_input.jsonl (from prepare_infer_input.py). "
             "If not given, --private_tsv is used instead.",
    )
    parser.add_argument(
        "--private_tsv",
        default=str(SCRIPT_DIR / "RawData" / "UHRS2K_SD_Random200_0324.tsv"),
        help="Raw TSV file to build private records from (used when --private_jsonl not given)",
    )
    parser.add_argument(
        "--gt_jsonl",
        default=str(SCRIPT_DIR / "data" / "dpo_refine_eval_cot.jsonl"),
        help="GT eval JSONL with real assistant turns (highest quality calibration data). "
             "Used in addition to TSV private data when available.",
    )
    parser.add_argument("--n_private",    type=int, default=64)
    parser.add_argument("--n_public",     type=int, default=64)
    parser.add_argument(
        "--output_jsonl",
        default=str(SCRIPT_DIR / "data" / "calib_data.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Private records ───────────────────────────────────────────────────────
    # Priority: GT eval JSONL (real assistant) > infer_input JSONL > raw TSV
    gt_path = Path(args.gt_jsonl)
    if gt_path.exists():
        private = load_private_from_gt(gt_path, args.n_private, args.seed)
    elif args.private_jsonl and Path(args.private_jsonl).exists():
        private = load_private_records(Path(args.private_jsonl), args.n_private, args.seed)
    else:
        tsv_path = Path(args.private_tsv)
        if not tsv_path.exists():
            print(f"[ERROR] TSV not found: {tsv_path}")
            sys.exit(1)
        private = build_private_from_tsv(tsv_path, args.n_private, args.seed)

    # ── Public records ───────────────────────────────────────────────────────
    public = load_public_records(args.n_public, args.seed)

    # ── Mix & shuffle ────────────────────────────────────────────────────────
    all_records = private + public
    random.seed(args.seed)
    random.shuffle(all_records)

    # ── Write ────────────────────────────────────────────────────────────────
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[done] {len(private)} private + {len(public)} public = {len(all_records)} total")
    print(f"       -> {out_path}")


if __name__ == "__main__":
    main()
