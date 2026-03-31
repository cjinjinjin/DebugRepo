"""
Prepare GRPO training data (prompt-only, no ground truth).

Sources:
  1. SFT train JSONL  -- strip assistant turn, keep system+user
     SFT user message contains many fields; we rebuild a simplified user message
     with only [Landing Page URL] and [Page Content] (PrimaryContentNoTitleNoHeading).

  2. New LP context CSV (UHRS2K_SD_Random300_0331_LPContext.csv)
     Columns: URL, MainBlock
     Filter out crawl failures by MainBlock length < MIN_CONTENT_LEN chars.

Output: data/grpo_train.jsonl
  {
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user",   "content": "..."}
    ]
  }

Usage:
  python prepare_grpo_data.py
  python prepare_grpo_data.py --sft_train data/sft_train_cot.jsonl \
      --lp_csv RawData/UHRS2K_SD_Random300_0331_LPContext.csv \
      --output data/grpo_train.jsonl \
      --min_content_len 200
"""

import argparse
import csv
import json
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# System prompt  (identical to SFT/DPO training)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_COT = (
    "You are an expert Ad Creative Director and Senior AI Image Prompt Engineer, "
    "specialized in high-performing Native Advertisement visuals.\n\n"
    "Given a landing page URL and its page content, your task is to "
    "generate five (5) high-quality English image generation prompts for Native Ads.\n\n"
    "First, reason about the product inside <think>...</think> tags (keep it concise, under 200 words). "
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
    "ProductType: [Physical Product / Digital Product / Service]\n"
    "SpecificProduct: [concise noun phrase]\n"
    "Category: [broad category]\n"
    "VisualAnchors: [2-3 physical elements]\n"
    "LifestyleVibe: [emotional tone]\n"
    "CoreValueSignals: [up to 3 values]\n"
    "</think>\n"
    "<Prompt1>[your first image generation prompt here]</Prompt1>\n"
    "<Prompt2>[your second image generation prompt here]</Prompt2>\n"
    "<Prompt3>[your third image generation prompt here]</Prompt3>\n"
    "<Prompt4>[your fourth image generation prompt here]</Prompt4>\n"
    "<Prompt5>[your fifth image generation prompt here]</Prompt5>"
)


# ---------------------------------------------------------------------------
# User message builder  (serving-aligned: only LPURL + LP context)
# ---------------------------------------------------------------------------

def build_user_message(lp_url: str, lp_context: str) -> str:
    parts = [
        "Generate 5 image generation prompts for a Native Ad based on the "
        "following landing page information:\n"
    ]
    if lp_url.strip():
        parts.append(f"[Landing Page URL]\n{lp_url.strip()}")
    if lp_context.strip():
        parts.append(f"[Page Content]\n{lp_context.strip()}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Source 1: extract from SFT JSONL
# ---------------------------------------------------------------------------

_LP_URL_RE    = re.compile(r"\[Landing Page URL\]\n(.+?)(?=\n\n|\Z)", re.S)
_LP_CONTEXT_RE = re.compile(r"\[Page Content\]\n(.+?)(?=\n\n\[|\Z)", re.S)


def _extract_field(user_content: str, pattern: re.Pattern) -> str:
    m = pattern.search(user_content)
    return m.group(1).strip() if m else ""


def from_sft_jsonl(sft_path: Path) -> list[dict]:
    """
    Read SFT JSONL, extract LPURL and PrimaryContentNoTitleNoHeading from the
    user turn, rebuild a simplified user message with only those two fields.
    """
    samples = []
    with open(sft_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            msgs = s.get("messages", [])
            if len(msgs) < 2:
                continue

            user_content = msgs[1]["content"]
            lp_url     = _extract_field(user_content, _LP_URL_RE)
            lp_context = _extract_field(user_content, _LP_CONTEXT_RE)

            samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_COT},
                    {"role": "user",   "content": build_user_message(lp_url, lp_context)},
                ]
            })
    return samples


# ---------------------------------------------------------------------------
# Source 2: new LP context CSV
# ---------------------------------------------------------------------------

def from_lp_csv(csv_path: Path, min_content_len: int) -> tuple[list[dict], int]:
    """
    Read CSV with columns URL, MainBlock.
    Filter rows where len(MainBlock) < min_content_len (crawl failures).
    Returns (samples, n_filtered).
    """
    samples = []
    n_filtered = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url     = (row.get("URL") or "").strip()
            context = (row.get("MainBlock") or "").strip()
            if len(context) < min_content_len:
                n_filtered += 1
                continue
            samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_COT},
                    {"role": "user",   "content": build_user_message(url, context)},
                ]
            })
    return samples, n_filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare GRPO prompt-only data from SFT JSONL + new LP CSV"
    )
    parser.add_argument(
        "--sft_train", default="data/sft_train_cot.jsonl",
        help="SFT training JSONL (system+user+assistant); assistant turn is dropped"
    )
    parser.add_argument(
        "--lp_csv", default="RawData/UHRS2K_SD_Random300_0331_LPContext.csv",
        help="New LP context CSV with columns: URL, MainBlock"
    )
    parser.add_argument(
        "--output", default="data/grpo_train.jsonl",
        help="Output JSONL path"
    )
    parser.add_argument(
        "--min_content_len", type=int, default=200,
        help="Minimum MainBlock length to keep a CSV row (filters crawl failures)"
    )
    args = parser.parse_args()

    sft_path = SCRIPT_DIR / args.sft_train
    csv_path = SCRIPT_DIR / args.lp_csv
    out_path = SCRIPT_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Source 1: SFT ---
    sft_samples = from_sft_jsonl(sft_path)
    print(f"SFT input samples:          {len(sft_samples)}")

    # --- Source 2: new LP CSV ---
    csv_samples, n_filtered = from_lp_csv(csv_path, args.min_content_len)
    print(f"New LP CSV total rows:       {len(csv_samples) + n_filtered}")
    print(f"  Filtered (len < {args.min_content_len}):     {n_filtered}")
    print(f"  Kept:                       {len(csv_samples)}")

    # --- Merge ---
    all_samples = sft_samples + csv_samples
    print(f"\nTotal GRPO samples:         {len(all_samples)}")
    print(f"  from SFT:                  {len(sft_samples)}")
    print(f"  from new LP CSV:           {len(csv_samples)}")

    # --- Write ---
    with open(out_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nWrote -> {out_path}")

    # --- Sanity check ---
    sample = all_samples[0]
    user_msg = sample["messages"][1]["content"]
    print(f"\n--- Sample 0 (SFT source) ---")
    print(user_msg[:300] + ("..." if len(user_msg) > 300 else ""))

    if csv_samples:
        sample2 = csv_samples[0]
        user_msg2 = sample2["messages"][1]["content"]
        print(f"\n--- Sample from CSV ---")
        print(user_msg2[:300] + ("..." if len(user_msg2) > 300 else ""))


if __name__ == "__main__":
    main()
