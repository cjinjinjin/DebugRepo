"""
Preprocess a raw SD TSV file (with header row) into JSONL format for swift infer.

Input TSV columns (with header):
  UrlHash, RowId, LPURL, ImageURL, label,
  DocumentTitle, VisualTitle, Heading,
  Title_CB, VisualTitle_CB, Heading_CB,
  BestSnippet_CB, MetaDescription_CB, PrimaryContentNoTitleNoHeading

Output JSONL (one record per row):
  {
    "id": "<UrlHash>",
    "lp_url": "<LPURL>",
    "system": "<system prompt>",
    "messages": [
      {"role": "user", "content": "<LP fields formatted as user turn>"}
    ]
  }

Usage:
  python prepare_infer_input.py \
      --input_tsv  RawData/UHRS2K_SD_Random200_0324.tsv \
      --output_jsonl data/infer_input.jsonl
"""

import argparse
import csv
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# System prompt  (CoT mode — matches the DPO fine-tuned model's training format)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_COT = (
    "You are an expert Ad Creative Director and Senior AI Image Prompt Engineer, "
    "specialized in high-performing Native Advertisement visuals.\n\n"
    "Given a landing page URL and its extracted content fields, your task is to "
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
# User message builder  (mirrors prepare_data.py / inference.py)
# ---------------------------------------------------------------------------

def build_user_message(row: dict) -> str:
    field_labels = [
        ("LPURL",                          "Landing Page URL",  None),
        ("DocumentTitle",                  "Document Title",    200),
        ("VisualTitle",                    "Visual Title",      200),
        ("Heading",                        "Heading",           400),
        ("Title_CB",                       "Title (CB)",        200),
        ("VisualTitle_CB",                 "Visual Title (CB)", 200),
        ("Heading_CB",                     "Heading (CB)",      200),
        ("BestSnippet_CB",                 "Best Snippet (CB)", 400),
        ("MetaDescription_CB",             "Meta Description",  300),
        ("PrimaryContentNoTitleNoHeading", "Page Content",      800),
    ]
    parts = [
        "Generate 5 image generation prompts for a Native Ad based on the "
        "following landing page information:\n"
    ]
    for key, label, max_chars in field_labels:
        val = (row.get(key) or "").strip()
        if val:
            if max_chars and len(val) > max_chars:
                val = val[:max_chars] + "..."
            parts.append(f"[{label}]\n{val}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tsv",     required=True, help="Input TSV file with header row")
    parser.add_argument("--output_jsonl",  required=True, help="Output JSONL for swift infer")
    args = parser.parse_args()

    input_path  = Path(args.input_tsv)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            url_hash = row.get("UrlHash", row.get("RowId", "")).strip()
            records.append({
                "id":      url_hash,
                "lp_url":  row.get("LPURL", "").strip(),
                "system":  SYSTEM_PROMPT_COT,
                "messages": [
                    {"role": "user", "content": build_user_message(row)}
                ],
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote {len(records)} records -> {output_path}")


if __name__ == "__main__":
    main()
