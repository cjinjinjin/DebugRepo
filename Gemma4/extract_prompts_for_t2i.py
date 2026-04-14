"""
Extract individual prompts from Two-Step vLLM output and write a TSV (.txt)
for T2I model consumption.

Input:
  --infer_file  : Two-Step vLLM output JSONL (has "id", "generated_prompts", "raw_output")
  --input_file  : original inference input JSONL (has "id", "lp_url") for metadata
  --output_file : output TSV (.txt) for T2I model

Output format: tab-separated, first line is header
  id\turl_hash\tlp_url\tprompt_index\tprompt_id\tsource\tprompt

Usage:
  python Gemma4/extract_prompts_for_t2i.py \
      --infer_file  /path/to/gemma4_random200_two_step_vllm_eval.jsonl \
      --input_file  Gemma4/data/random200_infer_input.jsonl \
      --output_file /path/to/prompts_for_t2i.txt
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


def extract_prompts_from_raw(response: str) -> dict[int, str]:
    """Fallback: extract Prompt1..Prompt5 from raw_output string via regex."""
    prompts = {}
    for i in range(1, 6):
        m = re.search(rf"<Prompt{i}>(.*?)</Prompt{i}>", response, re.DOTALL)
        if m:
            prompts[i] = re.sub(r"[\t\n\r]+", " ", m.group(1).strip())
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_file", required=True, help="Two-Step vLLM output JSONL")
    parser.add_argument("--input_file", required=True, help="Original input JSONL (for lp_url metadata)")
    parser.add_argument("--output_file", required=True, help="Output TSV (.txt) for T2I model")
    args = parser.parse_args()

    infer_records = load_jsonl(args.infer_file)
    input_records = load_jsonl(args.input_file)

    # Build lookup: id -> metadata from input JSONL
    input_by_id = {r["id"]: r for r in input_records if "id" in r}

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_prompts = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("\t".join(COLUMNS) + "\n")

        for idx, record in enumerate(infer_records):
            rec_id = record.get("id", f"sample_{idx}")

            # Get metadata from input file
            input_rec = input_by_id.get(rec_id, {})
            url_hash = rec_id  # id is UrlHash from prepare_infer_input.py
            lp_url = input_rec.get("lp_url", "")

            # Prefer generated_prompts (already parsed list), fallback to raw_output regex
            gen_prompts = record.get("generated_prompts", [])
            if gen_prompts and len(gen_prompts) > 0:
                prompts = {i + 1: re.sub(r"[\t\n\r]+", " ", p.strip())
                           for i, p in enumerate(gen_prompts) if p and p.strip()}
            else:
                raw = record.get("raw_output", "")
                prompts = extract_prompts_from_raw(raw)

            if not prompts:
                print(f"[WARN] No prompts for record {idx} (id={rec_id})", file=sys.stderr)
                continue

            seen_texts = set()
            for prompt_idx, prompt_text in sorted(prompts.items()):
                if prompt_text in seen_texts:
                    print(f"[WARN] Duplicate prompt skipped: id={rec_id} prompt_index={prompt_idx}", file=sys.stderr)
                    continue
                seen_texts.add(prompt_text)
                row = [
                    rec_id,
                    url_hash,
                    lp_url,
                    str(prompt_idx),
                    f"{rec_id}_p{prompt_idx}_model",
                    "model",
                    prompt_text,
                ]
                out_f.write("\t".join(row) + "\n")
                total_prompts += 1

    print(f"Done. {total_prompts} prompts from {len(infer_records)} records -> {output_path}")


if __name__ == "__main__":
    main()
