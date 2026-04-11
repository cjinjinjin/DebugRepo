"""
Speed benchmark: BF16 vs GPTQ 4-bit inference on Gemma 4 26B-A4B-it.

Runs N samples on a single GPU with both model variants and reports:
  - Time to first token (TTFT)
  - Tokens per second (generation throughput)
  - Total wall-clock time per sample

Usage:
  # BF16
  CUDA_VISIBLE_DEVICES=0 python Gemma4/benchmark_speed.py \
      --model_id ./gemma-4-26B-A4B-it \
      --input_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
      --num_samples 20

  # GPTQ 4-bit
  CUDA_VISIBLE_DEVICES=0 python Gemma4/benchmark_speed.py \
      --model_id ./gemma-4-26B-A4B-it-GPTQ-Int4 \
      --processor_id ./gemma-4-26B-A4B-it \
      --input_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
      --num_samples 20 \
      --use_gptq
"""

import argparse
import json
import time
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


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 speed benchmark")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--processor_id", default="")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--use_gptq", action="store_true", default=False)
    parser.add_argument("--no_think", action="store_true", default=False)
    parser.add_argument("--no_cot", action="store_true", default=False,
                        help="Use no-CoT system prompt (skip <think> block)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup samples (excluded from stats)")
    args = parser.parse_args()

    proc_id = args.processor_id or args.model_id

    # --- Load processor ---
    print(f"Loading processor from {proc_id} ...")
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(proc_id)
    except Exception:
        from transformers import AutoTokenizer
        import shutil, tempfile
        tok_cfg = Path(proc_id) / "tokenizer_config.json"
        load_path = proc_id
        tmpdir = None
        if tok_cfg.exists():
            with open(tok_cfg, "r") as f:
                cfg = json.load(f)
            if isinstance(cfg.get("extra_special_tokens"), list):
                tmpdir = tempfile.mkdtemp(prefix="tok_patch_")
                for fp in Path(proc_id).iterdir():
                    if fp.is_file() and ("token" in fp.name.lower() or fp.suffix in (".json", ".model")):
                        shutil.copy2(fp, tmpdir)
                cfg["extra_special_tokens"] = {}
                with open(Path(tmpdir) / "tokenizer_config.json", "w") as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)
                load_path = tmpdir
        processor = AutoTokenizer.from_pretrained(load_path)
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # --- Load model ---
    print(f"Loading model from {args.model_id} ...")
    if args.use_gptq:
        from gptqmodel import GPTQModel
        model = GPTQModel.load(args.model_id, device_map="auto")
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, device_map="auto", dtype=torch.bfloat16,
        )
    model.eval()
    print("Model ready.")

    # --- Load data ---
    total_needed = args.num_samples + args.warmup
    records = load_jsonl(args.input_file, max_records=total_needed)
    user_contents = [extract_user_content(r) for r in records]
    print(f"Loaded {len(user_contents)} samples ({args.warmup} warmup + {args.num_samples} benchmark)")

    # --- System prompt ---
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from inference_gemma4 import SYSTEM_PROMPT, SYSTEM_PROMPT_NO_COT
    except ImportError:
        SYSTEM_PROMPT = ""
        SYSTEM_PROMPT_NO_COT = ""

    system_prompt = SYSTEM_PROMPT_NO_COT if args.no_cot else SYSTEM_PROMPT
    enable_thinking = not args.no_think

    # --- Benchmark ---
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

        # Sync before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                do_sample=True,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        new_tokens = outputs[0].shape[-1] - input_len
        elapsed = t_end - t_start
        tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0

        print(f"  [{label}] {new_tokens} tokens in {elapsed:.2f}s = {tok_per_sec:.1f} tok/s (input: {input_len} tokens)")

        if not is_warmup:
            timings.append({
                "sample_idx": i - args.warmup,
                "input_tokens": input_len,
                "output_tokens": int(new_tokens),
                "elapsed_s": round(elapsed, 3),
                "tokens_per_sec": round(tok_per_sec, 1),
            })

    # --- Summary ---
    if not timings:
        print("No benchmark samples run.")
        return

    tok_per_sec_list = [t["tokens_per_sec"] for t in timings]
    elapsed_list = [t["elapsed_s"] for t in timings]
    output_tokens_list = [t["output_tokens"] for t in timings]

    model_type = "GPTQ-4bit" if args.use_gptq else "BF16"
    cot_label = "no-CoT" if args.no_cot else "with-CoT"
    print(f"\n{'='*60}")
    print(f"Speed Benchmark Summary — {model_type} ({cot_label})")
    print(f"{'='*60}")
    print(f"Samples:           {len(timings)}")
    print(f"Avg output tokens: {np.mean(output_tokens_list):.0f}")
    print(f"Avg time/sample:   {np.mean(elapsed_list):.2f}s")
    print(f"Median time:       {np.median(elapsed_list):.2f}s")
    print(f"Avg tok/s:         {np.mean(tok_per_sec_list):.1f}")
    print(f"Median tok/s:      {np.median(tok_per_sec_list):.1f}")
    print(f"Min tok/s:         {np.min(tok_per_sec_list):.1f}")
    print(f"Max tok/s:         {np.max(tok_per_sec_list):.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
