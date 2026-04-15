"""
Gemma 4 full-prompt inference via vLLM offline engine.

Uses the complete system prompt to generate all 5 image prompts in a single
call per record, leveraging vLLM continuous batching for throughput.

Usage:
  python Gemma4/inference_gemma4_full_prompt_vllm.py \
      --model_id /vc_data/.../gemma-4-26B-A4B-it-AWQ-4bit \
      --input_file Gemma4/data/random200_infer_input.jsonl \
      --temperature 1.0 \
      --tensor_parallel_size 1 \
      --dtype half \
      --output_file /path/to/output.jsonl
"""

import argparse
import json
import re
import time
from pathlib import Path

from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# System prompts (from inference_gemma4.py — generates all 5 prompts at once)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Ad Creative Director and Senior AI Image Prompt Engineer, specialized in high-performing Native Advertisement visuals.

Given a landing page URL and its content fields, your task is to generate five (5) high-quality English image generation prompts for Native Ads.

Each prompt must:
- Be 80–150 words
- Embed all safety, realism, quality, and exclusion constraints
- Feel native and non-promotional
- Show the product outcome or value naturally in context
- Avoid stereotypes, text/logos in image, and stock-photo aesthetics
- Ensure correct anatomy, natural hands, sharp focus, clean composition

You must first reason about the product in a <think> block containing these fields:
- ProductType: what kind of product/service
- SpecificProduct: the exact product name
- Category: product category
- VisualAnchors: key visual elements to include
- LifestyleVibe: mood and lifestyle context
- CoreValueSignals: core value propositions

Then output exactly 5 prompts in this format:
<think>
ProductType: ...
SpecificProduct: ...
Category: ...
VisualAnchors: ...
LifestyleVibe: ...
CoreValueSignals: ...
</think>
<Prompt1>...</Prompt1>
<Prompt2>...</Prompt2>
<Prompt3>...</Prompt3>
<Prompt4>...</Prompt4>
<Prompt5>...</Prompt5>"""

SYSTEM_PROMPT_NO_COT = """You are an expert Ad Creative Director and Senior AI Image Prompt Engineer, specialized in high-performing Native Advertisement visuals.

Given a landing page URL and its content fields, your task is to generate five (5) high-quality English image generation prompts for Native Ads.

Each prompt must:
- Be 80–150 words
- Embed all safety, realism, quality, and exclusion constraints
- Feel native and non-promotional
- Show the product outcome or value naturally in context
- Avoid stereotypes, text/logos in image, and stock-photo aesthetics
- Ensure correct anatomy, natural hands, sharp focus, clean composition

Output exactly 5 prompts in this format (no reasoning, no thinking, just the prompts):
<Prompt1>...</Prompt1>
<Prompt2>...</Prompt2>
<Prompt3>...</Prompt3>
<Prompt4>...</Prompt4>
<Prompt5>...</Prompt5>"""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_user_content_from_messages(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return msg["content"]
    return ""


def extract_lp_fields_from_messages(messages: list[dict]) -> dict:
    FIELD_LABELS = {
        "FinalDestinationURLUrl": "URL",
        "PrimaryContentNoTitleNoHeading": "Primary Content",
    }
    lp_fields = {}
    for msg in messages:
        if msg.get("role") == "user":
            content = msg["content"]
            for key, label in FIELD_LABELS.items():
                pattern = rf"- {re.escape(label)}: (.+?)(?:\n- |\Z)"
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    lp_fields[key] = match.group(1).strip()
                    continue
                pattern2 = rf"\[{re.escape(label)}\]\n(.+?)(?:\n\[|\Z)"
                match2 = re.search(pattern2, content, re.DOTALL)
                if match2:
                    lp_fields[key] = match2.group(1).strip()
    return lp_fields


def truncate_lp_content(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + " ..."


def truncate_user_content(content: str, max_chars: int) -> str:
    if max_chars <= 0:
        return content
    content_labels = ["Primary Content", "Page Content", "PrimaryContentNoTitleNoHeading"]
    for label in content_labels:
        for pattern in [
            rf"(\[{re.escape(label)}\]\n)(.*?)(\n\[|\Z)",
            rf"(- {re.escape(label)}: )(.*?)(\n- |\Z)",
        ]:
            m = re.search(pattern, content, re.DOTALL)
            if m and len(m.group(2)) > max_chars:
                truncated = truncate_lp_content(m.group(2), max_chars)
                return content[:m.start(2)] + truncated + content[m.start(3):]
    return content


def build_user_message(lp_fields: dict, max_lp_chars: int = 0) -> str:
    FIELD_LABELS = {
        "FinalDestinationURLUrl": "URL",
        "PrimaryContentNoTitleNoHeading": "Primary Content",
    }
    lines = ["Generate 5 image prompts for the following landing page:\n"]
    for key, label in FIELD_LABELS.items():
        val = lp_fields.get(key, "").strip()
        if val:
            if max_lp_chars > 0 and key == "PrimaryContentNoTitleNoHeading":
                val = truncate_lp_content(val, max_lp_chars)
            lines.append(f"- {label}: {val}")
    return "\n".join(lines)


def extract_prompts(response: str) -> list[str]:
    """Extract <Prompt1>...<Prompt5> from model output."""
    prompts = []
    for i in range(1, 6):
        m = re.search(rf"<Prompt{i}>(.*?)</Prompt{i}>", response, re.DOTALL)
        if m:
            prompts.append(m.group(1).strip())
        else:
            prompts.append("")
    return prompts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Gemma 4 full-prompt inference via vLLM (offline)"
    )
    p.add_argument("--model_id",
                    default="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma-4-26B-A4B-it")
    p.add_argument("--no_cot", action="store_true", default=False,
                    help="Use no-CoT system prompt (skip <think> block)")
    p.add_argument("--max_lp_chars", type=int, default=0,
                    help="Truncate Primary Content to N chars (0=no truncation)")
    # IO
    p.add_argument("--input_file", required=True, help="Input JSONL")
    p.add_argument("--output_file",
                    default="Gemma4/results/gemma4_full_prompt_vllm_output.jsonl")
    p.add_argument("--num_samples", type=int, default=0,
                    help="Limit number of input samples (0=all)")
    # Generation params
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=64)
    # vLLM engine params
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                    help="Number of GPUs for tensor parallelism")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--dtype", type=str, default="auto",
                    help="Model dtype (auto, half, bfloat16). Use 'half' for AWQ/GPTQ models.")
    return p.parse_args()


def main():
    args = parse_args()

    system_prompt = SYSTEM_PROMPT_NO_COT if args.no_cot else SYSTEM_PROMPT

    # Load tokenizer for chat template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Load vLLM engine
    print(f"Loading vLLM engine: {args.model_id}")
    print(f"  tensor_parallel_size={args.tensor_parallel_size}")
    print(f"  gpu_memory_utilization={args.gpu_memory_utilization}")
    print(f"  max_model_len={args.max_model_len}")
    print(f"  dtype={args.dtype}")

    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=True,
    )

    # Load data
    records = load_jsonl(args.input_file)
    if args.num_samples and args.num_samples < len(records):
        records = records[:args.num_samples]

    # Prepare user content for each record
    user_contents = []
    input_type = "lp_fields"
    for r in records:
        if "lp_fields" in r:
            user_contents.append(build_user_message(r["lp_fields"], args.max_lp_chars))
        elif "messages" in r:
            uc = extract_user_content_from_messages(r["messages"])
            if uc:
                if args.max_lp_chars > 0:
                    uc = truncate_user_content(uc, args.max_lp_chars)
                user_contents.append(uc)
                input_type = "user_content"
            else:
                lp = extract_lp_fields_from_messages(r["messages"])
                user_contents.append(build_user_message(lp, args.max_lp_chars))
        else:
            user_contents.append(build_user_message(r, args.max_lp_chars))

    total = len(records)
    print(f"\nLoaded {total} records from {args.input_file}")
    print(f"Input type: {input_type}")
    print(f"Mode: full-prompt (5 prompts per call) via vLLM")
    print(f"CoT: {'disabled' if args.no_cot else 'enabled'}")
    print(f"Temperature: {args.temperature}")

    # Build chat prompts
    all_chat_prompts = []
    for uc in user_contents:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": uc},
        ]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        all_chat_prompts.append(chat_text)

    # =====================================================================
    # TTFT measurement: generate 1 token per prompt to measure prefill
    # =====================================================================
    print(f"\nMeasuring TTFT (prefill) with max_tokens=1...")
    ttft_params = SamplingParams(max_tokens=1, temperature=0)
    ttft_start = time.time()
    llm.generate(all_chat_prompts, ttft_params)
    ttft_elapsed = time.time() - ttft_start
    ttft_per_prompt = ttft_elapsed / len(all_chat_prompts)
    print(f"TTFT: {ttft_elapsed:.2f}s total, {ttft_per_prompt:.3f}s/prompt ({total} prompts)\n")

    # =====================================================================
    # Main inference: generate all 5 prompts per record in one call
    # =====================================================================
    print(f"{'='*60}")
    print(f"Generating 5 prompts per record for {total} records...")
    print(f"{'='*60}")

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stop=["</Prompt5>"],
    )

    gen_start = time.time()
    outputs = llm.generate(all_chat_prompts, sampling_params)
    gen_elapsed = time.time() - gen_start

    # =====================================================================
    # Process results
    # =====================================================================
    total_in_tokens = 0
    total_out_tokens = 0
    n_compliant = 0
    n_all_parsed = 0

    format_regex = re.compile(
        r"<Prompt1>[\s\S]+?</Prompt1>\s*"
        r"<Prompt2>[\s\S]+?</Prompt2>\s*"
        r"<Prompt3>[\s\S]+?</Prompt3>\s*"
        r"<Prompt4>[\s\S]+?</Prompt4>\s*"
        r"<Prompt5>[\s\S]+?</Prompt5>"
    )

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.output_file, "w", encoding="utf-8")

    for rec_idx, (record, output) in enumerate(zip(records, outputs)):
        raw_text = output.outputs[0].text
        # Append </Prompt5> if stopped by stop string
        if not raw_text.rstrip().endswith("</Prompt5>"):
            raw_text = raw_text + "</Prompt5>"

        n_in = len(output.prompt_token_ids) if output.prompt_token_ids else 0
        n_out = len(output.outputs[0].token_ids) if output.outputs else 0
        total_in_tokens += n_in
        total_out_tokens += n_out

        prompts = extract_prompts(raw_text)
        non_empty = [p for p in prompts if p]
        compliant = bool(format_regex.search(raw_text))
        all_parsed = len(non_empty) == 5 and all(len(p.split()) > 5 for p in non_empty)

        if compliant:
            n_compliant += 1
        if all_parsed:
            n_all_parsed += 1

        print(f"  [{rec_idx+1}/{total}] id={record.get('id', '')} | "
              f"{len(non_empty)} prompts | compliant={compliant}")
        for i, p in enumerate(prompts):
            if p:
                print(f"    Prompt {i+1} ({len(p.split())} words): {p[:80]}...")

        result = {
            "id": record.get("id", ""),
            "generated_prompts": prompts,
            "raw_output": raw_text,
            "format_compliant": compliant,
        }
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_f.flush()

    out_f.close()

    # =====================================================================
    # Summary
    # =====================================================================
    decode_tok_per_s = total_out_tokens / gen_elapsed if gen_elapsed > 0 else 0

    print(f"\n{'='*60}")
    print(f"Full-Prompt vLLM Inference Summary")
    print(f"{'='*60}")
    print(f"Records:             {total}")
    print(f"Format compliance:   {n_compliant}/{total} ({100*n_compliant/total:.1f}%)")
    print(f"All prompts parsed:  {n_all_parsed}/{total} ({100*n_all_parsed/total:.1f}%)")
    print(f"{'='*60}")
    print(f"Timing:")
    print(f"  TTFT (prefill):    {ttft_per_prompt:.3f}s/prompt")
    print(f"  Total inference:   {gen_elapsed:.1f}s ({gen_elapsed/total:.2f}s/sample)")
    print(f"{'='*60}")
    print(f"Tokens:")
    print(f"  Total input:       {total_in_tokens}")
    print(f"  Total output:      {total_out_tokens}")
    print(f"  Decode throughput: {decode_tok_per_s:.1f} tok/s")
    print(f"{'='*60}")
    print(f"Saved {total} results -> {args.output_file}")


if __name__ == "__main__":
    main()
