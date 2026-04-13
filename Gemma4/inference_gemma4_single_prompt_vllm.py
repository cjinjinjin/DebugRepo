"""
Gemma 4 single-prompt inference via vLLM offline engine.

Generates ONE prompt per call, N parallel samples per input, using vLLM's
continuous batching for true parallelism. Combines results into
<Prompt1>...<PromptN> format compatible with evaluate.py.

Usage:
  python Gemma4/inference_gemma4_single_prompt_vllm.py \
      --model_id /vc_data/.../gemma-4-26B-A4B-it \
      --input_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
      --num_samples 2 \
      --no_cot \
      --temperature 1.2 \
      --num_calls 5 \
      --output_file Gemma4/results/gemma4_single_prompt_vllm_test.jsonl
"""

import argparse
import json
import re
import time
from pathlib import Path

from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# System prompts: generate exactly ONE prompt
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

IMPORTANT: Output ONLY the prompt inside tags. No explanations, no options, no reasoning, no commentary. Just:
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
    lines = ["Generate 1 image prompt for the following landing page:\n"]
    for key, label in FIELD_LABELS.items():
        val = lp_fields.get(key, "").strip()
        if val:
            if max_lp_chars > 0 and key == "PrimaryContentNoTitleNoHeading":
                val = truncate_lp_content(val, max_lp_chars)
            lines.append(f"- {label}: {val}")
    return "\n".join(lines)


def parse_single_prompt(text: str) -> str:
    """Extract <Prompt>...</Prompt> from model output."""
    m = re.search(r"<Prompt>(.*?)</Prompt>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Gemma 4 single-prompt inference via vLLM (offline)"
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
                    default="Gemma4/results/gemma4_single_prompt_vllm_output.jsonl")
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
    # vLLM engine params
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                    help="Number of GPUs for tensor parallelism")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=8192)
    return p.parse_args()


def main():
    args = parse_args()

    system_prompt = SYSTEM_PROMPT_SINGLE if args.no_cot else SYSTEM_PROMPT_SINGLE_COT

    # Load tokenizer for chat template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Load vLLM engine
    print(f"Loading vLLM engine: {args.model_id}")
    print(f"  tensor_parallel_size={args.tensor_parallel_size}")
    print(f"  gpu_memory_utilization={args.gpu_memory_utilization}")
    print(f"  max_model_len={args.max_model_len}")

    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
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
                # Replace "Generate 5" with "Generate 1" to match single-prompt system prompt
                uc = re.sub(r"Generate \d+ image", "Generate 1 image", uc, count=1)
                if args.max_lp_chars > 0:
                    uc = truncate_user_content(uc, args.max_lp_chars)
                user_contents.append(uc)
                input_type = "user_content"
            else:
                lp = extract_lp_fields_from_messages(r["messages"])
                user_contents.append(build_user_message(lp, args.max_lp_chars))
        else:
            user_contents.append(build_user_message(r, args.max_lp_chars))

    num_calls = args.num_calls
    total = len(records)
    print(f"\nLoaded {total} records from {args.input_file}")
    print(f"Input type: {input_type}")
    print(f"Mode: single-prompt x {num_calls} parallel samples (vLLM)")
    print(f"Temperature: {args.temperature}")

    # Build all prompts: N copies per input for parallel sampling
    all_prompts = []
    prompt_to_record = []  # maps each prompt index -> (record_idx, call_idx)
    for rec_idx, uc in enumerate(user_contents):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": uc},
        ]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        for call_idx in range(num_calls):
            all_prompts.append(chat_text)
            prompt_to_record.append((rec_idx, call_idx))

    print(f"Total vLLM requests: {len(all_prompts)} ({total} records x {num_calls} calls)")
    print(f"Generating...\n")

    # Single vLLM generate call for all prompts
    start_time = time.time()
    outputs = llm.generate(all_prompts, sampling_params)
    elapsed = time.time() - start_time

    # Extract per-request latency metrics from vLLM outputs
    all_latency_metrics = []
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        m = {}
        if metrics:
            # V1 engine: RequestStateStats has first_token_ts, last_token_ts, scheduled_ts
            if hasattr(metrics, "first_token_ts") and hasattr(metrics, "last_token_ts"):
                prefill = (metrics.first_token_ts - metrics.scheduled_ts) if metrics.scheduled_ts else None
                decode = (metrics.last_token_ts - metrics.first_token_ts) if metrics.first_token_ts else None
                ttft = getattr(metrics, "first_token_latency", None)
                m = {"ttft": ttft, "prefill_s": prefill, "decode_s": decode}
            # V0 engine: RequestMetrics has first_token_time, finished_time, arrival_time
            elif hasattr(metrics, "first_token_time") and hasattr(metrics, "finished_time"):
                arrival = getattr(metrics, "arrival_time", None)
                first_tok = metrics.first_token_time
                finished = metrics.finished_time
                ttft = (first_tok - arrival) if (first_tok and arrival) else None
                decode = (finished - first_tok) if (finished and first_tok) else None
                m = {"ttft": ttft, "prefill_s": ttft, "decode_s": decode}
        num_output_tokens = len(output.outputs[0].token_ids) if output.outputs else 0
        m["output_tokens"] = num_output_tokens
        if m.get("decode_s") and num_output_tokens > 1:
            m["decode_tok_per_s"] = (num_output_tokens - 1) / m["decode_s"]
        all_latency_metrics.append(m)

    # Group results by record
    grouped = {}  # rec_idx -> list of (call_idx, raw_text, latency_metrics)
    for i, (output, (rec_idx, call_idx)) in enumerate(zip(outputs, prompt_to_record)):
        raw_text = output.outputs[0].text
        if rec_idx not in grouped:
            grouped[rec_idx] = []
        grouped[rec_idx].append((call_idx, raw_text, all_latency_metrics[i]))

    # Sort each group by call_idx
    for rec_idx in grouped:
        grouped[rec_idx].sort(key=lambda x: x[0])

    # Format compliance regex
    format_regex = re.compile(
        r"<Prompt1>[\s\S]+?</Prompt1>\s*"
        r"<Prompt2>[\s\S]+?</Prompt2>\s*"
        r"<Prompt3>[\s\S]+?</Prompt3>\s*"
        r"<Prompt4>[\s\S]+?</Prompt4>\s*"
        r"<Prompt5>[\s\S]+?</Prompt5>"
    )

    n_compliant = 0
    n_all_parsed = 0

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.output_file, "w", encoding="utf-8")

    for rec_idx, record in enumerate(records):
        print(f"{'='*60}")
        print(f"[Sample {rec_idx+1}/{total}] id={record.get('id', '')}")

        raw_texts = [text for _, text, _ in grouped.get(rec_idx, [])]
        per_sample_metrics = [m for _, _, m in grouped.get(rec_idx, [])]
        prompts = []
        for i, raw in enumerate(raw_texts):
            prompt = parse_single_prompt(raw)
            word_count = len(prompt.split())
            prompts.append(prompt)
            print(f"  [{i+1}] {word_count} words | {prompt[:80]}...")

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
            "latency_metrics": per_sample_metrics,
        }
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_f.flush()

        print(f"  => {len(prompts)} prompts | compliant: {n_compliant}/{rec_idx+1} | all_parsed: {n_all_parsed}/{rec_idx+1}")

    out_f.close()

    print(f"\n{'='*60}")
    print(f"Total inference time: {elapsed:.1f}s ({elapsed/total:.1f}s/sample)")
    print(f"Saved {total} results -> {args.output_file}")
    print(f"\nFormat compliance: {n_compliant}/{total} ({100*n_compliant/total:.1f}%)")
    print(f"All prompts parsed: {n_all_parsed}/{total} ({100*n_all_parsed/total:.1f}%)")

    # Latency summary
    ttfts = [m["ttft"] for m in all_latency_metrics if m.get("ttft") is not None]
    prefills = [m["prefill_s"] for m in all_latency_metrics if m.get("prefill_s") is not None]
    decodes = [m["decode_s"] for m in all_latency_metrics if m.get("decode_s") is not None]
    out_toks = [m["output_tokens"] for m in all_latency_metrics if m.get("output_tokens")]
    decode_speeds = [m["decode_tok_per_s"] for m in all_latency_metrics if m.get("decode_tok_per_s") is not None]

    if ttfts or prefills or decodes:
        print(f"\n{'='*60}")
        print(f"Latency Summary ({len(all_latency_metrics)} requests)")
        print(f"{'='*60}")

        def _stats(vals, unit="s"):
            if not vals:
                return "N/A"
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            avg = sum(vals_sorted) / n
            med = vals_sorted[n // 2]
            p95 = vals_sorted[int(n * 0.95)] if n >= 2 else vals_sorted[-1]
            return f"avg={avg:.3f}{unit}  med={med:.3f}{unit}  p95={p95:.3f}{unit}  min={vals_sorted[0]:.3f}{unit}  max={vals_sorted[-1]:.3f}{unit}"

        if ttfts:
            print(f"TTFT (prefill):    {_stats(ttfts)}")
        if prefills and prefills != ttfts:
            print(f"Prefill time:      {_stats(prefills)}")
        if decodes:
            print(f"Decode time:       {_stats(decodes)}")
        if out_toks:
            avg_tok = sum(out_toks) / len(out_toks)
            print(f"Output tokens:     avg={avg_tok:.0f}  total={sum(out_toks)}")
        if decode_speeds:
            print(f"Decode speed:      {_stats(decode_speeds, unit=' tok/s')}")
    else:
        print(f"\n(No per-request latency metrics available from this vLLM version)")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
