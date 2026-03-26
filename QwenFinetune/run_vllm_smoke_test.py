"""
Smoke test for the quantized model using vLLM directly (no swift dependency).

Usage:
  python run_vllm_smoke_test.py \
      --model        /path/to/merged_model_awq_int4 \
      --input_jsonl  data/calib_smoke_10.jsonl \
      --output_jsonl /path/to/smoke_output.jsonl \
      --tp           4
"""

import argparse
import json
from pathlib import Path


def build_prompt(rec: dict, tokenizer) -> str:
    """Apply chat template to a messages record."""
    messages = []
    system = rec.get("system", "")
    if system:
        messages.append({"role": "system", "content": system})
    for msg in rec.get("messages", []):
        messages.append(msg)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        required=True)
    parser.add_argument("--input_jsonl",  required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--tp",           type=int, default=4,
                        help="tensor_parallel_size")
    parser.add_argument("--max_tokens",   type=int, default=2048)
    args = parser.parse_args()

    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit("[ERROR] vllm or transformers not installed.")

    print(f"[INFO] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"[INFO] Loading quantized model (tp={args.tp}): {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        quantization="awq",
        trust_remote_code=True,
        max_model_len=8192,
    )

    records = []
    with open(args.input_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"[INFO] Building prompts for {len(records)} records ...")
    prompts = [build_prompt(r, tokenizer) for r in records]

    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    print("[INFO] Running inference ...")
    outputs = llm.generate(prompts, sampling)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec, out in zip(records, outputs):
            response = out.outputs[0].text
            f.write(json.dumps({
                "id":       rec.get("id", ""),
                "lp_url":   rec.get("lp_url", ""),
                "response": response,
            }, ensure_ascii=False) + "\n")
            # Print first 200 chars of each response for quick sanity check
            print(f"  [{rec.get('id','')}] {response[:200]!r}")

    print(f"\n[INFO] Smoke test done. Output: {out_path}")


if __name__ == "__main__":
    main()
