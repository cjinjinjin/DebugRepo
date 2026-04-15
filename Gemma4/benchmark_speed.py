"""
Speed benchmark for Gemma 4 26B-A4B-it inference.

Measures per-sample:
  - TTFT (Time to First Token) — prefill latency
  - Decode time and tok/s
  - Total wall-clock time

Supports:
  - LP content length sweeps (--lp_char_lengths 400,1000,2000,5000)
  - Attention implementation selection (--attn_impl sdpa/flash_attention_2/eager)
  - torch.compile (--torch_compile)
  - Output saving for quality inspection (--save_outputs)

Usage:
  CUDA_VISIBLE_DEVICES=0 python Gemma4/benchmark_speed.py \
      --model_id ./gemma-4-26B-A4B-it \
      --input_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
      --num_samples 20 \
      --lp_char_lengths 400,1000,2000,5000
"""

import argparse
import json
import time
import threading
import torch
import numpy as np
from pathlib import Path


def load_jsonl(path, max_records=None):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                if max_records and len(records) >= max_records:
                    break
    return records


def extract_user_content(record):
    messages = record.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return msg["content"]
    return ""


def load_processor(proc_id):
    """Load processor with fallback to AutoTokenizer.

    Always patches tokenizer_config.json into a temp dir so that
    AutoTokenizer sees a clean local path (avoids HF repo-ID validation
    errors with absolute paths in transformers 5.x).
    """
    try:
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(proc_id)
    except Exception:
        from transformers import AutoTokenizer
        import shutil, tempfile
        tmpdir = tempfile.mkdtemp(prefix="tok_patch_")
        # Copy all tokenizer-related files to temp dir
        for fp in Path(proc_id).iterdir():
            if fp.is_file() and ("token" in fp.name.lower() or fp.suffix in (".json", ".model")):
                shutil.copy2(fp, tmpdir)
        # Patch extra_special_tokens if needed
        tok_cfg = Path(tmpdir) / "tokenizer_config.json"
        if tok_cfg.exists():
            with open(tok_cfg, "r") as f:
                cfg = json.load(f)
            if isinstance(cfg.get("extra_special_tokens"), list):
                cfg["extra_special_tokens"] = {}
                with open(tok_cfg, "w") as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)
        processor = AutoTokenizer.from_pretrained(tmpdir)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return processor


def load_model(args):
    """Load model with optional GPTQ, attention impl, and torch.compile."""
    if args.use_gptq:
        from gptqmodel import GPTQModel
        model = GPTQModel.load(args.model_id, device_map="auto")
    else:
        from transformers import AutoModelForCausalLM
        kwargs = {
            "device_map": "auto",
            "dtype": torch.bfloat16,
        }
        if args.attn_impl:
            kwargs["attn_implementation"] = args.attn_impl
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)

    if args.torch_compile:
        print("Compiling model with torch.compile(mode='reduce-overhead') ...")
        model = torch.compile(model, mode="reduce-overhead")

    model.eval()
    return model


def run_generate_with_ttft(model, inputs, processor, max_new_tokens):
    """Run model.generate() with TTFT measurement using TextIteratorStreamer."""
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(processor, skip_special_tokens=True)

    gen_kwargs = {
        **{k: v for k, v in inputs.items()},
        "max_new_tokens": max_new_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "do_sample": True,
        "streamer": streamer,
    }

    outputs_container = {}

    def generate_fn():
        with torch.inference_mode():
            outputs_container["outputs"] = model.generate(**gen_kwargs)

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


def run_benchmark_group(model, processor, user_contents, system_prompt,
                        enable_thinking, args, lp_chars_label=""):
    """Run benchmark on a list of user_contents, return list of timing dicts."""
    timings = []

    for i, content in enumerate(user_contents):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        input_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = processor(text=input_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        is_warmup = i < args.warmup
        label = "WARMUP" if is_warmup else f"Sample {i - args.warmup + 1}/{args.num_samples}"

        outputs, t_start, t_first_token, t_end = run_generate_with_ttft(
            model, inputs, processor, args.max_new_tokens,
        )

        new_tokens = outputs[0].shape[-1] - input_len if outputs is not None else 0
        ttft = t_first_token - t_start
        decode_time = t_end - t_first_token
        total_time = t_end - t_start
        decode_tok_s = new_tokens / decode_time if decode_time > 0 else 0

        # Decode output text for saving
        output_text = ""
        if outputs is not None and args.save_outputs:
            output_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

        lp_suffix = f" | LP: {lp_chars_label}" if lp_chars_label else ""
        print(f"  [{label}] {new_tokens} tokens in {total_time:.1f}s | "
              f"TTFT: {ttft:.2f}s | Decode: {decode_time:.1f}s ({decode_tok_s:.1f} tok/s) | "
              f"Input: {input_len} tokens{lp_suffix}")

        if not is_warmup:
            entry = {
                "sample_idx": i - args.warmup,
                "input_tokens": input_len,
                "output_tokens": int(new_tokens),
                "ttft_s": round(ttft, 3),
                "decode_s": round(decode_time, 3),
                "decode_tok_s": round(decode_tok_s, 1),
                "total_s": round(total_time, 3),
            }
            if lp_chars_label:
                entry["lp_chars"] = lp_chars_label
            if output_text:
                entry["output_text"] = output_text
            timings.append(entry)

    return timings


def print_summary(timings, model_type, cot_label, lp_chars_label=""):
    """Print summary statistics for a group of timings."""
    if not timings:
        print("No benchmark samples.")
        return

    ttft_list = [t["ttft_s"] for t in timings]
    decode_list = [t["decode_s"] for t in timings]
    decode_tok_s_list = [t["decode_tok_s"] for t in timings]
    total_list = [t["total_s"] for t in timings]
    output_tokens_list = [t["output_tokens"] for t in timings]
    input_tokens_list = [t["input_tokens"] for t in timings]

    title = f"Speed Benchmark Summary — {model_type} ({cot_label})"
    if lp_chars_label:
        title += f" [LP: {lp_chars_label} chars]"

    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    print(f"Samples:           {len(timings)}")
    print(f"Avg input tokens:  {np.mean(input_tokens_list):.0f}")
    print(f"Avg output tokens: {np.mean(output_tokens_list):.0f}")
    print(f"{'-'*70}")
    print(f"TTFT (prefill):")
    print(f"  Avg:             {np.mean(ttft_list):.2f}s")
    print(f"  Median:          {np.median(ttft_list):.2f}s")
    print(f"  P95:             {np.percentile(ttft_list, 95):.2f}s")
    print(f"  Min:             {np.min(ttft_list):.2f}s")
    print(f"  Max:             {np.max(ttft_list):.2f}s")
    print(f"{'-'*70}")
    print(f"Decode:")
    print(f"  Avg tok/s:       {np.mean(decode_tok_s_list):.1f}")
    print(f"  Median tok/s:    {np.median(decode_tok_s_list):.1f}")
    print(f"  Avg time:        {np.mean(decode_list):.1f}s")
    print(f"  Median time:     {np.median(decode_list):.1f}s")
    print(f"{'-'*70}")
    print(f"Total:")
    print(f"  Avg time:        {np.mean(total_list):.1f}s")
    print(f"  Median time:     {np.median(total_list):.1f}s")
    print(f"{'='*70}")


def print_group_comparison(all_group_timings):
    """Print comparison table across LP char length groups."""
    print(f"\n{'='*90}")
    print("LP Length Comparison")
    print(f"{'='*90}")
    header = f"{'LP Chars':>10} | {'Avg Input Tok':>13} | {'Avg TTFT':>10} | {'Avg Decode':>10} | {'Avg Total':>10} | {'Avg tok/s':>10}"
    print(header)
    print(f"{'-'*90}")

    for lp_chars, timings in all_group_timings:
        if not timings:
            continue
        avg_input = np.mean([t["input_tokens"] for t in timings])
        avg_ttft = np.mean([t["ttft_s"] for t in timings])
        avg_decode = np.mean([t["decode_s"] for t in timings])
        avg_total = np.mean([t["total_s"] for t in timings])
        avg_tok_s = np.mean([t["decode_tok_s"] for t in timings])
        print(f"{lp_chars:>10} | {avg_input:>13.0f} | {avg_ttft:>9.2f}s | {avg_decode:>9.1f}s | {avg_total:>9.1f}s | {avg_tok_s:>10.1f}")

    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 speed benchmark with TTFT measurement")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--processor_id", default="")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--use_gptq", action="store_true", default=False)
    parser.add_argument("--no_think", action="store_true", default=False)
    parser.add_argument("--cot", action="store_true", default=False,
                        help="Use CoT system prompt (default: no-CoT)")
    # Keep --no_cot for backward compatibility, but default is now no-CoT
    parser.add_argument("--no_cot", action="store_true", default=False,
                        help="(deprecated, no-CoT is now default)")
    parser.add_argument("--max_lp_chars", type=int, default=0,
                        help="Truncate Primary Content to this many chars (0=no truncation)")
    parser.add_argument("--lp_char_lengths", type=str, default="",
                        help="Comma-separated LP char lengths for sweep, e.g. 400,1000,2000,5000")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup samples (excluded from stats)")
    parser.add_argument("--attn_impl", type=str, default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"],
                        help="Attention implementation (default: sdpa)")
    parser.add_argument("--torch_compile", action="store_true", default=False,
                        help="Apply torch.compile to model")
    parser.add_argument("--save_outputs", type=str, default="",
                        help="Save per-sample outputs to this JSONL file")
    args = parser.parse_args()

    proc_id = args.processor_id or args.model_id

    # --- Load processor ---
    print(f"Loading processor from {proc_id} ...")
    processor = load_processor(proc_id)

    # --- Load model ---
    print(f"Loading model from {args.model_id} ...")
    if not args.use_gptq:
        print(f"Attention implementation: {args.attn_impl}")
    if args.torch_compile:
        print("torch.compile: enabled")
    model = load_model(args)
    print("Model ready.")

    # --- Load data ---
    total_needed = args.num_samples + args.warmup
    records = load_jsonl(args.input_file, max_records=total_needed)
    raw_user_contents = [extract_user_content(r) for r in records]
    print(f"Loaded {len(raw_user_contents)} samples ({args.warmup} warmup + {args.num_samples} benchmark)")

    # --- System prompt ---
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from inference_gemma4 import SYSTEM_PROMPT, SYSTEM_PROMPT_NO_COT, truncate_user_content
    except ImportError:
        SYSTEM_PROMPT = ""
        SYSTEM_PROMPT_NO_COT = ""
        truncate_user_content = None

    # Default is no-CoT; --cot enables CoT
    use_cot = args.cot
    system_prompt = SYSTEM_PROMPT if use_cot else SYSTEM_PROMPT_NO_COT
    cot_label = "with-CoT" if use_cot else "no-CoT"
    enable_thinking = not args.no_think

    model_type = "GPTQ-4bit" if args.use_gptq else "BF16"

    print(f"Mode: {cot_label}, thinking: {'on' if enable_thinking else 'off'}")

    # --- Determine LP char lengths to test ---
    lp_char_lengths = []
    if args.lp_char_lengths:
        lp_char_lengths = [int(x.strip()) for x in args.lp_char_lengths.split(",")]
    elif args.max_lp_chars > 0:
        lp_char_lengths = [args.max_lp_chars]

    all_outputs = []

    if lp_char_lengths:
        # --- Sweep mode: test each LP char length ---
        all_group_timings = []
        for lp_chars in lp_char_lengths:
            print(f"\n--- LP char length: {lp_chars} ---")
            if truncate_user_content:
                user_contents = [truncate_user_content(c, lp_chars) for c in raw_user_contents]
            else:
                user_contents = raw_user_contents
            timings = run_benchmark_group(
                model, processor, user_contents, system_prompt,
                enable_thinking, args, lp_chars_label=str(lp_chars),
            )
            all_group_timings.append((str(lp_chars), timings))
            print_summary(timings, model_type, cot_label, lp_chars_label=str(lp_chars))
            all_outputs.extend(timings)

        # Also run without truncation
        print(f"\n--- LP char length: unlimited ---")
        timings = run_benchmark_group(
            model, processor, raw_user_contents, system_prompt,
            enable_thinking, args, lp_chars_label="unlimited",
        )
        all_group_timings.append(("unlimited", timings))
        print_summary(timings, model_type, cot_label, lp_chars_label="unlimited")
        all_outputs.extend(timings)

        print_group_comparison(all_group_timings)
    else:
        # --- Single run mode ---
        if args.max_lp_chars > 0 and truncate_user_content:
            user_contents = [truncate_user_content(c, args.max_lp_chars) for c in raw_user_contents]
            print(f"Truncated LP content to max {args.max_lp_chars} chars")
        else:
            user_contents = raw_user_contents

        timings = run_benchmark_group(
            model, processor, user_contents, system_prompt,
            enable_thinking, args,
        )
        print_summary(timings, model_type, cot_label)
        all_outputs.extend(timings)

    # --- Save outputs ---
    if args.save_outputs and all_outputs:
        out_path = Path(args.save_outputs)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for entry in all_outputs:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\nOutputs saved to {args.save_outputs}")


if __name__ == "__main__":
    main()
