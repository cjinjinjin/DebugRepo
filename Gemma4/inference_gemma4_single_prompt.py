"""
Gemma 4 single-prompt inference: generate ONE prompt per call, N parallel samples.

Tests diversity and LP-relevance by generating N independent single-prompt
samples in one batched model.generate() call (batch input expansion).
The N prompts are combined into <Prompt1>...<PromptN> format, compatible with
evaluate.py.

Usage:
  CUDA_VISIBLE_DEVICES=0 python Gemma4/inference_gemma4_single_prompt.py \
      --model_id /vc_data/.../gemma-4-26B-A4B-it \
      --input_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
      --num_samples 2 \
      --no_think \
      --temperature 1.2 \
      --output_file Gemma4/results/gemma4_single_prompt_test.jsonl
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch

from inference_gemma4 import (
    Gemma4PromptGenerator,
    build_user_message,
    extract_lp_fields_from_messages,
    extract_user_content_from_messages,
    load_jsonl,
    parse_gemma_response,
    truncate_user_content,
)

# ---------------------------------------------------------------------------
# System prompt: generate exactly ONE prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_SINGLE = """You are an expert Ad Creative Director and Senior AI Image Prompt Engineer, specialized in high-performing Native Advertisement visuals.

Given a landing page URL and its content fields, generate ONE high-quality English image generation prompt for a Native Ad.

The prompt must:
- Be 80–150 words
- Embed all safety, realism, quality, and exclusion constraints
- Feel native and non-promotional
- Show the product outcome or value naturally in context
- Avoid stereotypes, text/logos in image, and stock-photo aesthetics
- Ensure correct anatomy, natural hands, sharp focus, clean composition

Output exactly one prompt (no reasoning, no thinking):
<Prompt>...</Prompt>"""

SYSTEM_PROMPT_SINGLE_COT = """You are an expert Ad Creative Director and Senior AI Image Prompt Engineer, specialized in high-performing Native Advertisement visuals.

Given a landing page URL and its content fields, generate ONE high-quality English image generation prompt for a Native Ad.

The prompt must:
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

Then output exactly one prompt:
<think>
ProductType: ...
SpecificProduct: ...
Category: ...
VisualAnchors: ...
LifestyleVibe: ...
CoreValueSignals: ...
</think>
<Prompt>...</Prompt>"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_single_prompt(text: str) -> str:
    """Extract <Prompt>...</Prompt> from model output."""
    m = re.search(r"<Prompt>(.*?)</Prompt>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


# ---------------------------------------------------------------------------
# Batch-parallel single-prompt generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_n_samples(
    gen: Gemma4PromptGenerator,
    user_content: str = None,
    lp_fields: dict = None,
    num_samples: int = 5,
    max_new_tokens: int = 512,
    temperature: float = 1.2,
    top_p: float = 0.95,
    top_k: int = 64,
    do_sample: bool = True,
) -> list[str]:
    """Generate N independent single-prompt samples via true batch decoding.

    Replicates the same input N times into a batch so all N sequences are
    decoded in parallel in a single model.generate() call.
    Returns a list of N raw decoded strings.
    """
    if user_content:
        input_text = gen.build_input_from_content(user_content)
    else:
        input_text = gen.build_input(lp_fields)

    # Tokenize once, then replicate to batch of N
    single = gen.processor(
        text=input_text,
        return_tensors="pt",
    )
    input_len = single["input_ids"].shape[-1]

    # Expand to (num_samples, seq_len)
    batch_input_ids = single["input_ids"].expand(num_samples, -1).contiguous().to(gen.model.device)
    batch_attention = single["attention_mask"].expand(num_samples, -1).contiguous().to(gen.model.device)

    outputs = gen.model.generate(
        input_ids=batch_input_ids,
        attention_mask=batch_attention,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
    )

    # outputs shape: (num_samples, seq_len)
    raw_texts = []
    for i in range(outputs.shape[0]):
        decoded = gen.processor.decode(
            outputs[i][input_len:], skip_special_tokens=False
        )
        # Strip EOS / pad tokens
        for eos in ["<eos>", "<end_of_turn>", "</s>", "<pad>"]:
            decoded = decoded.replace(eos, "")
        raw_texts.append(decoded.strip())

    return raw_texts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Gemma 4 single-prompt inference: 1 prompt x N parallel samples"
    )
    p.add_argument("--model_id",
                    default="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma-4-26B-A4B-it")
    p.add_argument("--processor_id", default="")
    p.add_argument("--adapter_path", default="")
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    p.add_argument("--use_gptq", action="store_true", default=False)
    p.add_argument("--no_think", action="store_true", default=False,
                    help="Disable thinking mode")
    p.add_argument("--no_cot", action="store_true", default=False,
                    help="Use no-CoT system prompt (skip <think> block)")
    p.add_argument("--max_lp_chars", type=int, default=0,
                    help="Truncate Primary Content to N chars (0=no truncation)")
    p.add_argument("--attn_impl", type=str, default="")
    # IO
    p.add_argument("--input_file", required=True, help="Input JSONL")
    p.add_argument("--output_file", default="Gemma4/results/gemma4_single_prompt_output.jsonl")
    p.add_argument("--num_samples", type=int, default=0,
                    help="Limit number of input samples (0=all)")
    # Single-prompt specific
    p.add_argument("--num_calls", type=int, default=5,
                    help="Number of parallel samples per input (default: 5)")
    # Generation params
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.2)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--do_sample", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    gen = Gemma4PromptGenerator(
        model_id=args.model_id,
        processor_id=args.processor_id or None,
        adapter_path=args.adapter_path or None,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        use_gptq=args.use_gptq,
        no_cot=args.no_cot,
        max_lp_chars=args.max_lp_chars,
        enable_thinking=not args.no_think,
        attn_impl=args.attn_impl,
    )

    # Override system prompt to single-prompt version
    if args.no_cot:
        gen.system_prompt = SYSTEM_PROMPT_SINGLE
    else:
        gen.system_prompt = SYSTEM_PROMPT_SINGLE_COT

    records = load_jsonl(args.input_file)
    if args.num_samples and args.num_samples < len(records):
        records = records[:args.num_samples]

    # Prepare inputs
    batch_inputs = []
    input_type = "lp_fields"
    for r in records:
        if "lp_fields" in r:
            batch_inputs.append(r["lp_fields"])
        elif "messages" in r:
            user_content = extract_user_content_from_messages(r["messages"])
            if user_content:
                batch_inputs.append(user_content)
                input_type = "user_content"
            else:
                batch_inputs.append(extract_lp_fields_from_messages(r["messages"]))
        else:
            batch_inputs.append(r)

    num_calls = args.num_calls
    total = len(records)
    print(f"Loaded {total} records from {args.input_file}")
    print(f"Input type: {input_type}")
    print(f"Mode: single-prompt x {num_calls} parallel samples (batch decoding)")
    print(f"Temperature: {args.temperature}")

    # Format compliance regex (combined output should have Prompt1..PromptN)
    format_regex = re.compile(
        r"<Prompt1>[\s\S]+?</Prompt1>\s*"
        r"<Prompt2>[\s\S]+?</Prompt2>\s*"
        r"<Prompt3>[\s\S]+?</Prompt3>\s*"
        r"<Prompt4>[\s\S]+?</Prompt4>\s*"
        r"<Prompt5>[\s\S]+?</Prompt5>"
    )

    n_compliant = 0
    n_all_parsed = 0
    start_time = time.time()

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.output_file, "w", encoding="utf-8")

    for idx, (record, inp) in enumerate(zip(records, batch_inputs)):
        print(f"\n{'='*60}")
        print(f"[Sample {idx+1}/{total}] id={record.get('id', '')}")

        t0 = time.time()
        if input_type == "user_content":
            raw_texts = generate_n_samples(
                gen, user_content=inp, num_samples=num_calls,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p, top_k=args.top_k,
                do_sample=args.do_sample,
            )
        else:
            raw_texts = generate_n_samples(
                gen, lp_fields=inp, num_samples=num_calls,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p, top_k=args.top_k,
                do_sample=args.do_sample,
            )
        elapsed_sample = time.time() - t0

        prompts = []
        for i, raw in enumerate(raw_texts):
            prompt = parse_single_prompt(raw)
            word_count = len(prompt.split())
            prompts.append(prompt)
            print(f"  [{i+1}] {word_count} words | {prompt[:80]}...")

        print(f"  Time: {elapsed_sample:.1f}s for {num_calls} samples")

        # Combine into <Prompt1>...<PromptN> format
        formatted_parts = []
        for i, p in enumerate(prompts, 1):
            formatted_parts.append(f"<Prompt{i}>{p}</Prompt{i}>")
        combined_raw = "\n\n".join(formatted_parts)

        compliant = bool(format_regex.search(combined_raw)) if num_calls >= 5 else len(prompts) == num_calls
        all_parsed = all(p and len(p.split()) > 5 for p in prompts)

        if compliant:
            n_compliant += 1
        if all_parsed:
            n_all_parsed += 1

        result = {
            "id": record.get("id", ""),
            "generated_prompts": prompts,
            "raw_output": combined_raw,
            "individual_raw_outputs": raw_texts,
            "format_compliant": compliant,
        }
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_f.flush()

        done = idx + 1
        print(f"  => {len(prompts)} prompts | compliant: {n_compliant}/{done} | all_parsed: {n_all_parsed}/{done}")

    out_f.close()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Total inference time: {elapsed:.1f}s ({elapsed/total:.1f}s/sample)")
    print(f"Saved {total} results -> {args.output_file}")
    print(f"\nFormat compliance: {n_compliant}/{total} ({100*n_compliant/total:.1f}%)")
    print(f"All prompts parsed: {n_all_parsed}/{total} ({100*n_all_parsed/total:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
