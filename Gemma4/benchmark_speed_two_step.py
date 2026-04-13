"""
Speed benchmark for Gemma 4 two-step inference (scene planning + prompt expansion).

Measures per-sample, per-step:
  - Prefill time (TTFT for Step 1; forward pass for Step 2 batch)
  - Decode time and tok/s
  - Total wall-clock time

Step 1: single-sequence scene planning (uses TextIteratorStreamer for TTFT)
Step 2: batch=5 scene expansion (uses separate forward pass for prefill timing)

Usage:
  CUDA_VISIBLE_DEVICES=0 python Gemma4/benchmark_speed_two_step.py \
      --model_id ./gemma-4-26B-A4B-it \
      --input_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
      --num_samples 20
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference_gemma4 import (
    Gemma4PromptGenerator,
    build_user_message,
    extract_lp_fields_from_messages,
    extract_user_content_from_messages,
    load_jsonl,
    truncate_user_content,
)
from inference_gemma4_two_step import (
    SYSTEM_PROMPT_STEP1_SCENES,
    SYSTEM_PROMPT_STEP2_EXPAND,
    parse_scenes,
    parse_single_prompt,
)


# ---------------------------------------------------------------------------
# TTFT measurement for single-sequence generation (Step 1)
# ---------------------------------------------------------------------------

def generate_with_ttft(model, inputs, tokenizer, max_new_tokens, gen_kwargs):
    """Run model.generate() with TTFT measurement using TextIteratorStreamer.

    Returns (outputs, t_start, t_first_token, t_end).
    """
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    full_kwargs = {
        **{k: v for k, v in inputs.items()},
        "max_new_tokens": max_new_tokens,
        **gen_kwargs,
        "streamer": streamer,
    }

    outputs_container = {}

    def generate_fn():
        with torch.inference_mode():
            outputs_container["outputs"] = model.generate(**full_kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    thread = threading.Thread(target=generate_fn)
    thread.start()

    t_first_token = None
    for _ in streamer:
        if t_first_token is None:
            t_first_token = time.perf_counter()

    thread.join()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_end = time.perf_counter()

    if t_first_token is None:
        t_first_token = t_end

    return outputs_container.get("outputs"), t_start, t_first_token, t_end


# ---------------------------------------------------------------------------
# Prefill timing for batch generation (Step 2)
# ---------------------------------------------------------------------------

def measure_batch_prefill(model, batch_inputs):
    """Run a single forward pass on batch to measure prefill time.

    Returns prefill_time in seconds.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    with torch.inference_mode():
        model(**batch_inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    return t_end - t_start


# ---------------------------------------------------------------------------
# Main benchmark function
# ---------------------------------------------------------------------------

def benchmark_two_step(
    gen: Gemma4PromptGenerator,
    user_content: str = None,
    lp_fields: dict = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
    do_sample: bool = True,
) -> dict:
    """Run two-step generation with prefill/decode timing for each step."""
    orig_system_prompt = gen.system_prompt
    gen_kwargs = dict(temperature=temperature, top_p=top_p, top_k=top_k, do_sample=do_sample)

    if hasattr(gen.processor, "tokenizer"):
        tokenizer = gen.processor.tokenizer
    else:
        tokenizer = gen.processor

    # ---------------------------------------------------------------
    # Step 1: Generate 5 scene concepts (single sequence, with TTFT)
    # ---------------------------------------------------------------
    gen.system_prompt = SYSTEM_PROMPT_STEP1_SCENES

    if user_content:
        step1_content = user_content
        if gen.max_lp_chars > 0:
            step1_content = truncate_user_content(step1_content, gen.max_lp_chars)
        step1_input_text = gen.build_input_from_content(step1_content)
    elif lp_fields:
        step1_content = build_user_message(lp_fields, gen.max_lp_chars)
        step1_input_text = gen.build_input_from_content(step1_content)
    else:
        step1_input_text = gen.build_input_from_content("")

    step1_inputs = tokenizer(step1_input_text, return_tensors="pt").to(gen.model.device)
    step1_input_len = step1_inputs["input_ids"].shape[-1]

    step1_outputs, t1_start, t1_first_token, t1_end = generate_with_ttft(
        gen.model, step1_inputs, tokenizer, max_new_tokens=128, gen_kwargs=gen_kwargs,
    )

    step1_new_tokens = step1_outputs[0].shape[-1] - step1_input_len if step1_outputs is not None else 0
    step1_prefill = t1_first_token - t1_start
    step1_decode = t1_end - t1_first_token
    step1_total = t1_end - t1_start
    step1_decode_tok_s = step1_new_tokens / step1_decode if step1_decode > 0 else 0

    # Parse scenes
    step1_response = tokenizer.decode(step1_outputs[0][step1_input_len:], skip_special_tokens=False)
    if hasattr(gen.processor, "parse_response"):
        parsed = gen.processor.parse_response(step1_response)
    else:
        parsed = gen._parse_response_fallback(step1_response)
    step1_text = parsed.get("content", "") if isinstance(parsed, dict) else str(parsed)
    scenes = parse_scenes(step1_text)

    if len(scenes) < 5:
        gen.system_prompt = orig_system_prompt
        return {
            "step1_input_tokens": step1_input_len,
            "step1_output_tokens": int(step1_new_tokens),
            "step1_prefill_s": round(step1_prefill, 3),
            "step1_decode_s": round(step1_decode, 3),
            "step1_decode_tok_s": round(step1_decode_tok_s, 1),
            "step1_total_s": round(step1_total, 3),
            "step2_input_tokens": 0,
            "step2_output_tokens": 0,
            "step2_prefill_s": 0,
            "step2_decode_s": 0,
            "step2_decode_tok_s": 0,
            "step2_total_s": 0,
            "total_time_s": round(step1_total, 3),
            "total_output_tokens": int(step1_new_tokens),
            "scenes_parsed": len(scenes),
            "error": f"Only {len(scenes)} scenes parsed",
        }

    # ---------------------------------------------------------------
    # Step 2: Batch-expand 5 scenes (batch=5, prefill via forward)
    # ---------------------------------------------------------------
    gen.system_prompt = SYSTEM_PROMPT_STEP2_EXPAND

    if user_content:
        base_content = user_content
        if gen.max_lp_chars > 0:
            base_content = truncate_user_content(base_content, gen.max_lp_chars)
    elif lp_fields:
        base_content = build_user_message(lp_fields, gen.max_lp_chars)
    else:
        base_content = ""

    input_texts = []
    for scene in scenes:
        scene_content = (
            f"{base_content}\n\n"
            f"Expand this scene concept into a detailed prompt:\n"
            f"<Scene>{scene}</Scene>"
        )
        input_texts.append(gen.build_input_from_content(scene_content))

    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    batch_inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
    ).to(gen.model.device)
    input_lens = batch_inputs["attention_mask"].sum(dim=1).tolist()
    step2_total_input_tokens = sum(input_lens)

    # Measure prefill: single forward pass on the batch
    step2_prefill = measure_batch_prefill(gen.model, batch_inputs)

    # Full generate (includes prefill + decode)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_step2_start = time.perf_counter()
    with torch.inference_mode():
        step2_outputs = gen.model.generate(
            **batch_inputs,
            max_new_tokens=256,
            **gen_kwargs,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_step2_end = time.perf_counter()

    tokenizer.padding_side = orig_padding_side

    step2_total = t_step2_end - t_step2_start
    step2_decode = step2_total - step2_prefill  # approximate: subtract measured prefill time

    # Count step 2 output tokens
    step2_total_output_tokens = 0
    for i in range(len(scenes)):
        pad_len = batch_inputs["input_ids"].shape[1] - input_lens[i]
        gen_start = pad_len + int(input_lens[i])
        gen_tokens = step2_outputs[i][gen_start:]
        step2_total_output_tokens += len(gen_tokens)

    step2_decode_tok_s = step2_total_output_tokens / step2_decode if step2_decode > 0 else 0

    gen.system_prompt = orig_system_prompt

    total_time = step1_total + step2_total
    total_output_tokens = int(step1_new_tokens) + step2_total_output_tokens

    return {
        "step1_input_tokens": step1_input_len,
        "step1_output_tokens": int(step1_new_tokens),
        "step1_prefill_s": round(step1_prefill, 3),
        "step1_decode_s": round(step1_decode, 3),
        "step1_decode_tok_s": round(step1_decode_tok_s, 1),
        "step1_total_s": round(step1_total, 3),
        "step2_input_tokens": step2_total_input_tokens,
        "step2_output_tokens": step2_total_output_tokens,
        "step2_prefill_s": round(step2_prefill, 3),
        "step2_decode_s": round(step2_decode, 3),
        "step2_decode_tok_s": round(step2_decode_tok_s, 1),
        "step2_total_s": round(step2_total, 3),
        "total_time_s": round(total_time, 3),
        "total_output_tokens": total_output_tokens,
        "scenes_parsed": len(scenes),
    }


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(timings):
    """Print summary statistics with prefill/decode breakdown."""
    if not timings:
        print("No benchmark samples.")
        return

    def stats(vals):
        return {
            "avg": np.mean(vals),
            "median": np.median(vals),
            "p95": np.percentile(vals, 95),
            "min": np.min(vals),
            "max": np.max(vals),
        }

    s1_prefill = stats([t["step1_prefill_s"] for t in timings])
    s1_decode = stats([t["step1_decode_s"] for t in timings])
    s1_decode_tok_s = stats([t["step1_decode_tok_s"] for t in timings])
    s1_total = stats([t["step1_total_s"] for t in timings])
    s1_out_tok = stats([t["step1_output_tokens"] for t in timings])

    s2_prefill = stats([t["step2_prefill_s"] for t in timings])
    s2_decode = stats([t["step2_decode_s"] for t in timings])
    s2_decode_tok_s = stats([t["step2_decode_tok_s"] for t in timings])
    s2_total = stats([t["step2_total_s"] for t in timings])
    s2_out_tok = stats([t["step2_output_tokens"] for t in timings])

    total_time = stats([t["total_time_s"] for t in timings])
    total_out_tok = stats([t["total_output_tokens"] for t in timings])

    print(f"\n{'='*70}")
    print("Two-Step Speed Benchmark Summary (Prefill + Decode)")
    print(f"{'='*70}")
    print(f"Samples:              {len(timings)}")

    print(f"\n{'-'*70}")
    print(f"Step 1 — Scene Planning (batch=1, max_new_tokens=128)")
    print(f"{'-'*70}")
    print(f"  Prefill (TTFT):")
    print(f"    Avg:              {s1_prefill['avg']:.3f}s")
    print(f"    Median:           {s1_prefill['median']:.3f}s")
    print(f"  Decode:")
    print(f"    Avg time:         {s1_decode['avg']:.2f}s")
    print(f"    Median time:      {s1_decode['median']:.2f}s")
    print(f"    Avg tok/s:        {s1_decode_tok_s['avg']:.1f}")
    print(f"    Median tok/s:     {s1_decode_tok_s['median']:.1f}")
    print(f"  Total:")
    print(f"    Avg time:         {s1_total['avg']:.2f}s")
    print(f"    Avg output tokens:{s1_out_tok['avg']:.0f}")

    print(f"\n{'-'*70}")
    print(f"Step 2 — Batch Expand (batch=5, max_new_tokens=256)")
    print(f"{'-'*70}")
    print(f"  Prefill (forward pass):")
    print(f"    Avg:              {s2_prefill['avg']:.3f}s")
    print(f"    Median:           {s2_prefill['median']:.3f}s")
    print(f"  Decode:")
    print(f"    Avg time:         {s2_decode['avg']:.1f}s")
    print(f"    Median time:      {s2_decode['median']:.1f}s")
    print(f"    Avg tok/s:        {s2_decode_tok_s['avg']:.1f} (aggregate across 5 seqs)")
    print(f"    Median tok/s:     {s2_decode_tok_s['median']:.1f}")
    print(f"  Total:")
    print(f"    Avg time:         {s2_total['avg']:.1f}s")
    print(f"    Avg output tokens:{s2_out_tok['avg']:.0f} (sum of 5 seqs)")

    print(f"\n{'-'*70}")
    print(f"Overall (Step 1 + Step 2)")
    print(f"{'-'*70}")
    print(f"  Avg time:           {total_time['avg']:.1f}s")
    print(f"  Median time:        {total_time['median']:.1f}s")
    print(f"  P95 time:           {total_time['p95']:.1f}s")
    print(f"  Min time:           {total_time['min']:.1f}s")
    print(f"  Max time:           {total_time['max']:.1f}s")
    print(f"  Avg output tokens:  {total_out_tok['avg']:.0f}")
    print(f"{'='*70}")

    # Time breakdown
    avg_s1_prefill = np.mean([t["step1_prefill_s"] for t in timings])
    avg_s1_decode = np.mean([t["step1_decode_s"] for t in timings])
    avg_s2_prefill = np.mean([t["step2_prefill_s"] for t in timings])
    avg_s2_decode = np.mean([t["step2_decode_s"] for t in timings])
    avg_total = np.mean([t["total_time_s"] for t in timings])

    if avg_total > 0:
        print(f"\nTime Breakdown (avg per sample):")
        print(f"  Step1 prefill:  {avg_s1_prefill:.2f}s  ({100*avg_s1_prefill/avg_total:.0f}%)")
        print(f"  Step1 decode:   {avg_s1_decode:.2f}s  ({100*avg_s1_decode/avg_total:.0f}%)")
        print(f"  Step2 prefill:  {avg_s2_prefill:.2f}s  ({100*avg_s2_prefill/avg_total:.0f}%)")
        print(f"  Step2 decode:   {avg_s2_decode:.2f}s  ({100*avg_s2_decode/avg_total:.0f}%)")
        print(f"  ─────────────────────────────")
        print(f"  Total:          {avg_total:.1f}s")

    # Error rate
    errors = [t for t in timings if "error" in t]
    if errors:
        print(f"\nErrors: {len(errors)}/{len(timings)}")
        for e in errors[:3]:
            print(f"  {e['error']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Speed benchmark for Gemma 4 two-step inference (prefill + decode)"
    )
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--processor_id", default="")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--use_gptq", action="store_true", default=False)
    parser.add_argument("--no_think", action="store_true", default=False,
                        help="Disable thinking mode")
    parser.add_argument("--max_lp_chars", type=int, default=0,
                        help="Truncate Primary Content to N chars (0=no truncation)")
    parser.add_argument("--attn_impl", type=str, default="")
    parser.add_argument("--input_file", required=True, help="Input JSONL")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup samples (excluded from stats)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--output_file", default="",
                        help="Save detailed results JSON")
    args = parser.parse_args()

    # --- Load model via Gemma4PromptGenerator ---
    gen = Gemma4PromptGenerator(
        model_id=args.model_id,
        processor_id=args.processor_id or None,
        adapter_path=args.adapter_path or None,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        use_gptq=args.use_gptq,
        no_cot=True,
        max_lp_chars=args.max_lp_chars,
        enable_thinking=not args.no_think,
        attn_impl=args.attn_impl,
    )

    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": args.do_sample,
    }

    # --- Load data ---
    total_needed = args.num_samples + args.warmup
    records = load_jsonl(args.input_file)
    if len(records) > total_needed:
        records = records[:total_needed]

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

    print(f"Loaded {len(records)} records ({args.warmup} warmup + {args.num_samples} benchmark)")
    print(f"Input type: {input_type}")
    print(f"Mode: two-step (scene planning + batch expansion)")
    print()

    # --- Run benchmark ---
    timings = []

    for idx, inp in enumerate(batch_inputs):
        is_warmup = idx < args.warmup
        label = "WARMUP" if is_warmup else f"Sample {idx - args.warmup + 1}/{args.num_samples}"

        if input_type == "user_content":
            result = benchmark_two_step(gen, user_content=inp, **gen_kwargs)
        else:
            result = benchmark_two_step(gen, lp_fields=inp, **gen_kwargs)

        error_tag = " [ERR]" if "error" in result else ""
        print(
            f"  [{label}] "
            f"S1: prefill={result['step1_prefill_s']:.2f}s decode={result['step1_decode_s']:.2f}s ({result['step1_output_tokens']}tok) | "
            f"S2: prefill={result['step2_prefill_s']:.2f}s decode={result['step2_decode_s']:.1f}s ({result['step2_output_tokens']}tok) | "
            f"Total: {result['total_time_s']:.1f}s{error_tag}"
        )

        if not is_warmup:
            result["sample_idx"] = idx - args.warmup
            timings.append(result)

    print_summary(timings)

    # --- Save results ---
    if args.output_file and timings:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump({
                "config": {
                    "num_samples": len(timings),
                    "warmup": args.warmup,
                    "model_id": args.model_id,
                    "max_lp_chars": args.max_lp_chars,
                },
                "summary": {
                    "avg_step1_prefill_s": round(float(np.mean([t["step1_prefill_s"] for t in timings])), 3),
                    "avg_step1_decode_s": round(float(np.mean([t["step1_decode_s"] for t in timings])), 2),
                    "avg_step1_decode_tok_s": round(float(np.mean([t["step1_decode_tok_s"] for t in timings])), 1),
                    "avg_step2_prefill_s": round(float(np.mean([t["step2_prefill_s"] for t in timings])), 3),
                    "avg_step2_decode_s": round(float(np.mean([t["step2_decode_s"] for t in timings])), 1),
                    "avg_step2_decode_tok_s": round(float(np.mean([t["step2_decode_tok_s"] for t in timings])), 1),
                    "avg_total_time_s": round(float(np.mean([t["total_time_s"] for t in timings])), 1),
                    "avg_total_output_tokens": round(float(np.mean([t["total_output_tokens"] for t in timings]))),
                },
                "results": timings,
            }, f, indent=2)
        print(f"\nDetailed results saved to {args.output_file}")


if __name__ == "__main__":
    main()
