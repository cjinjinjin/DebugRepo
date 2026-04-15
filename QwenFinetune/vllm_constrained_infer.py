"""
vLLM offline batch inference with optional regex-constrained decoding.

Uses vLLM's tensor-parallel engine for fast multi-GPU inference, with
optional guided_regex to enforce output format compliance.

Usage:
  # Constrained + no_think (fastest)
  python vllm_constrained_infer.py \
      --model_path /path/to/merged_model \
      --input_file data/dpo_combined_eval_cot.jsonl \
      --output_file results/constrained_no_think.jsonl \
      --constrained --no_think

  # Constrained with COT
  python vllm_constrained_infer.py \
      --model_path /path/to/merged_model \
      --input_file data/dpo_combined_eval_cot.jsonl \
      --output_file results/constrained_cot.jsonl \
      --constrained

  # Unconstrained (baseline)
  python vllm_constrained_infer.py \
      --model_path /path/to/merged_model \
      --input_file data/dpo_combined_eval_cot.jsonl \
      --output_file results/unconstrained.jsonl
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Regex patterns for constrained decoding
# ---------------------------------------------------------------------------

PATTERN_COT = (
    r"<think>[^<]+</think>\s*"
    r"<Prompt1>[^<]+</Prompt1>\s*"
    r"<Prompt2>[^<]+</Prompt2>\s*"
    r"<Prompt3>[^<]+</Prompt3>\s*"
    r"<Prompt4>[^<]+</Prompt4>\s*"
    r"<Prompt5>[^<]+</Prompt5>"
)

PATTERN_NO_THINK = (
    r"<Prompt1>[^<]+</Prompt1>\s*"
    r"<Prompt2>[^<]+</Prompt2>\s*"
    r"<Prompt3>[^<]+</Prompt3>\s*"
    r"<Prompt4>[^<]+</Prompt4>\s*"
    r"<Prompt5>[^<]+</Prompt5>"
)


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} results -> {path}")


def build_messages(record: dict) -> list[dict]:
    """Build chat messages from a record (compatible with eval JSONL format)."""
    system = record.get("system", "")
    messages = record.get("messages", [])

    # Only include user message (drop assistant response if present)
    chat = []
    if system:
        chat.append({"role": "system", "content": system})
    for msg in messages:
        if msg["role"] == "user":
            chat.append(msg)
            break
    return chat


def make_guided_params(pattern: str):
    """Create guided decoding params, compatible with both old and new vLLM API."""
    # Try new API first (vLLM >= 0.8)
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return {"structured_outputs": StructuredOutputsParams(regex=pattern)}
    except ImportError:
        pass

    # Fall back to old API (vLLM < 0.8)
    try:
        from vllm.sampling_params import GuidedDecodingParams
        return {"guided_decoding": GuidedDecodingParams(regex=pattern)}
    except ImportError:
        pass

    print("ERROR: Cannot find vLLM guided decoding params class. "
          "Please upgrade vLLM.", file=sys.stderr)
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        description="vLLM batch inference with optional constrained decoding")
    p.add_argument("--model_path", required=True,
                   help="Path to merged model directory")
    p.add_argument("--input_file", required=True,
                   help="Input JSONL (same format as swift eval data)")
    p.add_argument("--output_file", required=True,
                   help="Output JSONL (compatible with evaluate.py)")
    p.add_argument("--constrained", action="store_true", default=False,
                   help="Enable regex-constrained decoding")
    p.add_argument("--no_think", action="store_true", default=False,
                   help="Skip <think> block (faster, fewer tokens)")
    p.add_argument("--tp_size", type=int, default=8,
                   help="Tensor parallel size (default: 8)")
    p.add_argument("--max_tokens", type=int, default=0,
                   help="Max new tokens (default: 1024 for no_think, 2048 for COT)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--max_model_len", type=int, default=8192,
                   help="Max model sequence length (default: 8192)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.max_tokens == 0:
        args.max_tokens = 1024 if args.no_think else 2048

    # Load input data
    records = load_jsonl(args.input_file)
    print(f"Loaded {len(records)} records from {args.input_file}")

    # Build messages for each record
    all_messages = [build_messages(r) for r in records]

    # Initialize vLLM
    from vllm import LLM, SamplingParams

    print(f"Loading model from {args.model_path} (tp={args.tp_size}) ...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )

    # Build sampling params
    sp_kwargs = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    if args.constrained:
        pattern = PATTERN_NO_THINK if args.no_think else PATTERN_COT
        mode = "no_think" if args.no_think else "COT"
        print(f"Constrained decoding enabled ({mode} mode)")
        sp_kwargs.update(make_guided_params(pattern))

    sampling_params = SamplingParams(**sp_kwargs)

    # Run batch inference
    enable_thinking = not args.no_think
    print(f"Running inference (enable_thinking={enable_thinking}, "
          f"max_tokens={args.max_tokens}) ...")
    t0 = time.time()

    outputs = llm.chat(
        messages=all_messages,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )

    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s ({elapsed/len(records):.1f}s/sample)")

    # Format compliance check
    if args.no_think:
        format_regex = re.compile(
            r"<Prompt1>[\s\S]+?</Prompt1>\s*"
            r"<Prompt2>[\s\S]+?</Prompt2>\s*"
            r"<Prompt3>[\s\S]+?</Prompt3>\s*"
            r"<Prompt4>[\s\S]+?</Prompt4>\s*"
            r"<Prompt5>[\s\S]+?</Prompt5>"
        )
    else:
        format_regex = re.compile(
            r"<think>[\s\S]+?</think>\s*"
            r"<Prompt1>[\s\S]+?</Prompt1>\s*"
            r"<Prompt2>[\s\S]+?</Prompt2>\s*"
            r"<Prompt3>[\s\S]+?</Prompt3>\s*"
            r"<Prompt4>[\s\S]+?</Prompt4>\s*"
            r"<Prompt5>[\s\S]+?</Prompt5>"
        )

    n_compliant = 0
    results = []
    for record, output in zip(records, outputs):
        response = output.outputs[0].text
        compliant = bool(format_regex.search(response))
        if compliant:
            n_compliant += 1

        # Output format compatible with evaluate.py (uses "response" field)
        results.append({
            "id": record.get("id", ""),
            "url_hash": record.get("url_hash", ""),
            "lp_url": record.get("lp_url", ""),
            "system": record.get("system", ""),
            "messages": record.get("messages", []),
            "response": response,
            "format_compliant": compliant,
        })

    write_jsonl(results, args.output_file)

    total = len(results)
    print(f"\nFormat compliance: {n_compliant}/{total} ({100*n_compliant/total:.1f}%)")
    print(f"Time: {elapsed:.1f}s total, {elapsed/total:.1f}s/sample")


if __name__ == "__main__":
    main()
