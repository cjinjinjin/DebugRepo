"""
Convert Gemma 4 inference JSONL output to TSV format for Z-Image T2I pipeline.

Input JSONL format (per line):
  {"id": "Random200_0324_1", "generated_prompts": ["...", "...", ...], "raw_output": "...", "format_compliant": true}

Output TSV format (for Z-Image_ForQwen3.py):
  prompt_id\tprompt
  Random200_0324_1_p1\tA cinematic photograph ...
  Random200_0324_1_p2\tA wide-angle shot ...
  ...

Usage:
  python Gemma4/convert_jsonl_to_t2i_prompts.py \
      --input_file /path/to/gemma4_random200_eval.jsonl \
      --output_file /path/to/prompts_for_t2i.txt \
      [--compliant_only]
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Convert Gemma 4 JSONL to Z-Image TSV")
    parser.add_argument("--input_file", required=True, help="Input JSONL from Gemma 4 inference")
    parser.add_argument("--output_file", required=True, help="Output TSV for Z-Image")
    parser.add_argument("--compliant_only", action="store_true",
                        help="Only include format-compliant samples")
    args = parser.parse_args()

    total = 0
    skipped = 0
    prompt_count = 0

    with open(args.input_file, "r", encoding="utf-8") as fin, \
         open(args.output_file, "w", encoding="utf-8") as fout:
        # Write TSV header
        fout.write("prompt_id\tprompt\n")

        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1

            if args.compliant_only and not record.get("format_compliant", False):
                skipped += 1
                continue

            sample_id = record.get("id", f"sample_{total}")
            prompts = record.get("generated_prompts", [])

            for i, prompt_text in enumerate(prompts, 1):
                prompt_text = prompt_text.strip().replace("\n", " ").replace("\t", " ")
                if not prompt_text:
                    continue
                prompt_id = f"{sample_id}_p{i}"
                fout.write(f"{prompt_id}\t{prompt_text}\n")
                prompt_count += 1

    print(f"Done: {total} samples, {skipped} skipped, {prompt_count} prompts written -> {args.output_file}")


if __name__ == "__main__":
    main()
