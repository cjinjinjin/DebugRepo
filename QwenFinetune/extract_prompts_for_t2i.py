"""
Extract individual prompts from swift infer output AND ground-truth,
align with gt metadata, and write a unified TSV (.txt) for t2i model consumption.

Input:
  --infer_file  : swift infer output JSONL  (has "response" field)
  --gt_file     : sft_eval_cot.jsonl        (has "id", "url_hash", "lp_url", "messages" fields)
  --output_file : output TSV (.txt) for t2i model

Output format: tab-separated, first line is header
  id\turl_hash\tlp_url\tprompt_index\tprompt_id\tsource\tprompt

Usage:
  python extract_prompts_for_t2i.py \\
      --infer_file  /path/to/checkpoint-30/eval_results/eval_swift_output.jsonl \\
      --gt_file     ./data/sft_eval_cot.jsonl \\
      --output_file /path/to/checkpoint-30/eval_results/prompts_for_t2i.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

COLUMNS = ["id", "url_hash", "lp_url", "prompt_index", "prompt_id", "source", "prompt"]


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_prompts(response: str) -> dict[int, str]:
    """Extract Prompt1..Prompt5 from model response string."""
    prompts = {}
    for i in range(1, 6):
        m = re.search(rf"<Prompt{i}>(.*?)</Prompt{i}>", response, re.DOTALL)
        if m:
            # collapse newlines/tabs inside prompt text to keep TSV valid
            prompts[i] = re.sub(r"[\t\n\r]+", " ", m.group(1).strip())
    return prompts


def write_prompts(out_f, rec_id, url_hash, lp_url, prompts, source):
    """Write one TSV row per prompt. Returns number of rows written."""
    count = 0
    for prompt_idx, prompt_text in sorted(prompts.items()):
        row = [
            rec_id,
            url_hash,
            lp_url,
            str(prompt_idx),
            f"{rec_id}_p{prompt_idx}_{source}",
            source,
            prompt_text,
        ]
        out_f.write("\t".join(row) + "\n")
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_file",  required=True, help="swift infer output JSONL")
    parser.add_argument("--gt_file",     required=True, help="sft_eval_cot.jsonl with id/url_hash")
    parser.add_argument("--output_file", required=True, help="output TSV (.txt) for t2i model")
    args = parser.parse_args()

    infer_records = load_jsonl(args.infer_file)
    gt_records    = load_jsonl(args.gt_file)

    gt_by_id = {r["id"]: r for r in gt_records if "id" in r}

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_model = 0
    total_gt    = 0
    missing_gt  = 0

    with open(output_path, "w", encoding="utf-8") as out_f:

        # header
        out_f.write("\t".join(COLUMNS) + "\n")

        # ── Model outputs ────────────────────────────────────────────────────
        for idx, record in enumerate(infer_records):
            response = record.get("response", "")
            prompts  = extract_prompts(response)

            rec_id = record.get("id", "")
            if rec_id and rec_id in gt_by_id:
                gt = gt_by_id[rec_id]
            elif idx < len(gt_records):
                gt = gt_records[idx]
                rec_id = gt.get("id", f"sample_{idx}")
            else:
                gt = {}
                rec_id = rec_id or f"sample_{idx}"
                missing_gt += 1

            url_hash = gt.get("url_hash", "")
            lp_url   = gt.get("lp_url", "")

            if not prompts:
                print(f"[WARN] No model prompts for record {idx} (id={rec_id})", file=sys.stderr)
                continue

            total_model += write_prompts(out_f, rec_id, url_hash, lp_url, prompts, source="model")

        # ── Ground-truth outputs ─────────────────────────────────────────────
        for gt in gt_records:
            rec_id   = gt.get("id", "")
            url_hash = gt.get("url_hash", "")
            lp_url   = gt.get("lp_url", "")

            messages = gt.get("messages", [])
            gt_response = ""
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    gt_response = msg.get("content", "")
                    break

            prompts = extract_prompts(gt_response)
            if not prompts:
                print(f"[WARN] No GT prompts for id={rec_id}", file=sys.stderr)
                continue

            total_gt += write_prompts(out_f, rec_id, url_hash, lp_url, prompts, source="gt")

    print(f"Done. model={total_model} prompts, gt={total_gt} prompts → {output_path}")
    if missing_gt:
        print(f"[WARN] {missing_gt} model records had no matching gt entry (metadata left empty)")


if __name__ == "__main__":
    main()
