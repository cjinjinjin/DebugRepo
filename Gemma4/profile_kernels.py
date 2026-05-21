"""
Kernel-level CUDA profiling for Gemma4 26B-A4B-it.

Uses torch.profiler to capture GPU kernel execution and produces a breakdown
similar to ZImage's profiling results (GEMM %, attention %, overhead %, etc.).

Outputs:
  - Console summary: top kernels, category breakdown (GEMM / Attention / etc.)
  - Chrome trace: profile_gemma4.json (viewable at chrome://tracing)

Usage:
  CUDA_VISIBLE_DEVICES=0 python Gemma4/profile_kernels.py \
      --model_id ./gemma-4-26B-A4B-it \
      --input_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
      --num_tokens 128 \
      --output profile_gemma4.json

Notes:
  - This profiles HF Transformers inference (not vLLM), since torch.profiler
    needs direct model access.
  - Run with --warmup_steps 2 to let CUDA caches settle before profiling.
  - The profile captures both prefill and decode phases.
"""

import argparse
import json
import re
import sys
import torch
from pathlib import Path
from collections import defaultdict

# Reuse model/processor loading from benchmark_speed.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_speed import load_processor, load_jsonl, extract_user_content


def _patch_compressed_tensors_group_size():
    """Monkey-patch compressed_tensors to fix group_size=0 validation error.

    The compressed-tensors library constructs QuantizationArgs with group_size=0
    for ignore-listed layers, which fails its own pydantic validator. We wrap
    __init__ to silently fix group_size=0 → 128 before pydantic validates.
    """
    try:
        from compressed_tensors.quantization.quant_args import QuantizationArgs
        _orig_init = QuantizationArgs.__init__

        def _patched_init(self, **data):
            if data.get("group_size") == 0:
                data["group_size"] = 128
            _orig_init(self, **data)

        QuantizationArgs.__init__ = _patched_init
        print("[INFO] Patched QuantizationArgs.__init__ for group_size=0 bug")
    except (ImportError, Exception) as e:
        print(f"[WARN] Could not patch compressed_tensors: {e}")


def load_model_for_profile(args):
    """Load model for profiling. Supports BF16 and AWQ (compressed-tensors) models."""
    from transformers import AutoModelForCausalLM, AutoConfig

    # Workaround: transformers 5.x bug where config.quantization_config is None
    # for compressed-tensors format (AWQ). Manually load and inject it.
    config = AutoConfig.from_pretrained(args.model_id)
    if getattr(config, "quantization_config", None) is None:
        config_path = Path(args.model_id) / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if "quantization_config" in raw:
                config.quantization_config = raw["quantization_config"]
                print(f"[INFO] Manually injected quantization_config "
                      f"(quant_method: {raw['quantization_config'].get('quant_method', 'unknown')})")

    # Patch compressed_tensors group_size=0 bug before loading
    _patch_compressed_tensors_group_size()

    kwargs = {
        "config": config,
        "device_map": "auto",
        "dtype": torch.bfloat16,
    }
    if args.attn_impl:
        kwargs["attn_implementation"] = args.attn_impl
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)
    model.eval()
    return model


def categorize_kernel(name: str) -> str:
    """Classify a CUDA kernel name into a high-level category."""
    n = name.lower()

    # GEMM / matrix multiply
    if any(k in n for k in ["gemm", "aten::mm", "aten::addmm", "aten::bmm",
                             "cutlass", "cublas", "sm80_xmma", "sm90_xmma",
                             "ampere_", "volta_", "turing_"]):
        return "GEMM (matmul)"

    # Attention
    if any(k in n for k in ["sdpa", "flash", "attention", "mem_efficient",
                             "fmha", "scaled_dot"]):
        return "Attention (SDPA)"

    # Softmax
    if "softmax" in n:
        return "Softmax"

    # LayerNorm / RMSNorm
    if any(k in n for k in ["layernorm", "rmsnorm", "layer_norm", "rms_norm"]):
        return "LayerNorm/RMSNorm"

    # Elementwise (activation, add, mul, etc.)
    if any(k in n for k in ["elementwise", "gelu", "silu", "relu", "sigmoid",
                             "aten::add", "aten::mul", "aten::div",
                             "vectorized", "unrolled"]):
        return "Elementwise"

    # Copy / dtype cast
    if any(k in n for k in ["copy_", "cast", "aten::to", "aten::_to_copy",
                             "convert"]):
        return "dtype cast / copy"

    # Embedding / gather / scatter
    if any(k in n for k in ["embedding", "gather", "scatter", "index"]):
        return "Embedding/Gather"

    # RoPE / rotary
    if any(k in n for k in ["rope", "rotary"]):
        return "RoPE"

    # MoE routing
    if any(k in n for k in ["topk", "moe", "router", "expert"]):
        return "MoE routing"

    # Command buffer / sync
    if any(k in n for k in ["command buffer", "cudalaunchkernel"]):
        return "Command Buffer / Sync"

    return "Other"


def run_profile(model, processor, user_content, system_prompt, args):
    """Run a single inference under torch.profiler and return the profiler."""
    try:
        from inference_gemma4 import SYSTEM_PROMPT_NO_COT
        if not system_prompt:
            system_prompt = SYSTEM_PROMPT_NO_COT
    except ImportError:
        pass

    messages = [
        {"role": "system", "content": system_prompt or "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]

    input_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    print(f"Input length: {input_len} tokens, generating up to {args.num_tokens} tokens")

    # Warmup runs (no profiling)
    for i in range(args.warmup_steps):
        print(f"  Warmup {i+1}/{args.warmup_steps}...")
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=16, do_sample=False)
        torch.cuda.synchronize()

    # Profiled run
    print("  Profiling...")
    trace_path = args.output or "profile_gemma4.json"

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=args.num_tokens, do_sample=False)
        torch.cuda.synchronize()

    new_tokens = outputs[0].shape[-1] - input_len
    print(f"  Generated {new_tokens} tokens")

    # Export chrome trace
    prof.export_chrome_trace(trace_path)
    print(f"  Chrome trace saved to: {trace_path}")

    return prof, new_tokens


def analyze_profile(prof):
    """Analyze profiler results and print kernel-level breakdown."""
    # Get CUDA kernel events
    events = prof.key_averages()

    # Collect CUDA time per kernel
    kernel_times = []
    for evt in events:
        cuda_time = evt.cuda_time_total  # microseconds
        if cuda_time > 0:
            kernel_times.append({
                "name": evt.key,
                "cuda_us": cuda_time,
                "count": evt.count,
                "category": categorize_kernel(evt.key),
            })

    if not kernel_times:
        print("\nNo CUDA kernels found. Are you running on GPU?")
        return

    total_cuda_us = sum(k["cuda_us"] for k in kernel_times)

    # --- Top 20 kernels ---
    kernel_times.sort(key=lambda x: x["cuda_us"], reverse=True)
    print(f"\n{'='*90}")
    print(f"Top 20 CUDA Kernels by Time")
    print(f"{'='*90}")
    print(f"{'Rank':>4} | {'CUDA Time':>12} | {'%':>6} | {'Count':>6} | {'Category':<22} | Name")
    print(f"{'-'*90}")
    for i, k in enumerate(kernel_times[:20]):
        pct = k["cuda_us"] / total_cuda_us * 100
        time_ms = k["cuda_us"] / 1000
        print(f"{i+1:>4} | {time_ms:>9.1f} ms | {pct:>5.1f}% | {k['count']:>6} | {k['category']:<22} | {k['name'][:60]}")

    # --- Category breakdown ---
    cat_times = defaultdict(float)
    cat_counts = defaultdict(int)
    for k in kernel_times:
        cat_times[k["category"]] += k["cuda_us"]
        cat_counts[k["category"]] += k["count"]

    cat_sorted = sorted(cat_times.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*70}")
    print(f"CUDA Time Breakdown by Category")
    print(f"{'='*70}")
    print(f"{'Category':<25} | {'CUDA Time':>12} | {'%':>6} | {'Kernel Count':>12}")
    print(f"{'-'*70}")
    for cat, us in cat_sorted:
        pct = us / total_cuda_us * 100
        time_ms = us / 1000
        print(f"{cat:<25} | {time_ms:>9.1f} ms | {pct:>5.1f}% | {cat_counts[cat]:>12}")
    print(f"{'-'*70}")
    print(f"{'TOTAL':<25} | {total_cuda_us/1000:>9.1f} ms | 100.0%")
    print(f"{'='*70}")

    # --- CPU vs CUDA time comparison ---
    cpu_events = [e for e in events if e.cpu_time_total > 0]
    total_cpu_us = sum(e.cpu_time_total for e in cpu_events)
    print(f"\nTotal CPU time: {total_cpu_us/1000:.1f} ms")
    print(f"Total CUDA time: {total_cuda_us/1000:.1f} ms")
    if total_cpu_us > 0:
        ratio = total_cuda_us / total_cpu_us
        print(f"CUDA/CPU ratio: {ratio:.2f}x")
        if ratio < 0.5:
            print("⚠️  CPU overhead dominant — torch.compile may help significantly")
        else:
            print("✅  GPU utilization looks reasonable")


def main():
    parser = argparse.ArgumentParser(description="Gemma4 kernel-level CUDA profiling")
    parser.add_argument("--model_id", required=True, help="Path to model")
    parser.add_argument("--processor_id", default="", help="Path to processor (default: same as model_id)")
    parser.add_argument("--input_file", required=True, help="JSONL input file")
    parser.add_argument("--sample_idx", type=int, default=0, help="Which sample to profile (0-indexed)")
    parser.add_argument("--num_tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--warmup_steps", type=int, default=2, help="Warmup iterations before profiling")
    parser.add_argument("--attn_impl", type=str, default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--output", type=str, default="profile_gemma4.json",
                        help="Chrome trace output path")
    args = parser.parse_args()

    proc_id = args.processor_id or args.model_id

    print(f"Loading processor from {proc_id} ...")
    processor = load_processor(proc_id)

    print(f"Loading model from {args.model_id} ...")
    model = load_model_for_profile(args)
    print("Model ready.")

    records = load_jsonl(args.input_file, max_records=args.sample_idx + 1)
    if args.sample_idx >= len(records):
        print(f"ERROR: sample_idx {args.sample_idx} out of range (only {len(records)} records)")
        sys.exit(1)
    user_content = extract_user_content(records[args.sample_idx])

    # Try to get system prompt
    system_prompt = ""
    try:
        from inference_gemma4 import SYSTEM_PROMPT_NO_COT
        system_prompt = SYSTEM_PROMPT_NO_COT
    except ImportError:
        pass

    prof, new_tokens = run_profile(model, processor, user_content, system_prompt, args)
    analyze_profile(prof)

    print(f"\n📊 Done! Open {args.output} in chrome://tracing for detailed visualization.")


if __name__ == "__main__":
    main()
