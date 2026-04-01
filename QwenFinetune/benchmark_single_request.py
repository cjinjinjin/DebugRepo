"""
Single-request inference latency benchmark for Qwen model on one GPU.

Measures:
  - Total generation time
  - Tokens/second throughput
  - GPU memory usage

NOTE: vLLM 0.8.5 does NOT support GPTQ for Qwen3-MoE (load_weights bug).
Use the merged (bfloat16) model. GPTQ support requires vLLM >= 0.9.x.

Usage:
  python benchmark_single_request.py --model /path/to/merged_model
  python benchmark_single_request.py --model /path/to/model --n_runs 5 --max_tokens 512
  python benchmark_single_request.py --model /path/to/model --prompt_file data/sft_eval_cot.jsonl --sample_idx 0
"""

import argparse
import json
import time
from pathlib import Path


SAMPLE_PROMPT = """Generate 5 image prompts for the following landing page:

- URL: https://example.com/skincare-serum
- Document Title: Revitalizing Face Serum | Glow & Hydrate
- Heading: Transform Your Skin in 30 Days
- Best Snippet (CB): Our clinically tested serum with hyaluronic acid and vitamin C visibly reduces fine lines and boosts radiance in just 4 weeks.
- Primary Content: Formulated with 10% pure Vitamin C, 2% retinol, and deep-hydration hyaluronic acid complex. Suitable for all skin types. Dermatologist tested and approved."""

SYSTEM_PROMPT = """You are an expert Ad Creative Director and Senior AI Image Prompt Engineer, specialized in high-performing Native Advertisement visuals.

Given a landing page URL and its content fields, your task is to generate five (5) high-quality English image generation prompts for Native Ads.

Each prompt must:
- Be ≤150 words
- Embed all safety, realism, quality, and exclusion constraints
- Feel native and non-promotional
- Show the product outcome or value naturally in context
- Avoid stereotypes, text/logos in image, and stock-photo aesthetics
- Ensure correct anatomy, natural hands, sharp focus, clean composition

Output exactly 5 prompts in this format:
<Prompt1>...</Prompt1>
<Prompt2>...</Prompt2>
<Prompt3>...</Prompt3>
<Prompt4>...</Prompt4>
<Prompt5>...</Prompt5>"""


def build_prompt_from_jsonl(rec: dict, tokenizer) -> str:
    messages = []
    system = rec.get("system", "")
    if system:
        messages.append({"role": "system", "content": system})
    for msg in rec.get("messages", []):
        if msg.get("role") != "assistant":
            messages.append(msg)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_default_prompt(tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": SAMPLE_PROMPT},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def get_gpu_memory_gb() -> dict:
    try:
        import torch
        mem = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved  = torch.cuda.memory_reserved(i)  / 1024**3
            total     = torch.cuda.get_device_properties(i).total_memory / 1024**3
            mem[i] = {"allocated_gb": allocated, "reserved_gb": reserved, "total_gb": total}
        return mem
    except Exception:
        return {}


def run_benchmark(llm, prompt: str, tokenizer, sampling_params, n_runs: int, warmup: int):
    from vllm import SamplingParams  # noqa: F401

    prompt_tokens = len(tokenizer.encode(prompt))
    print(f"\n[INFO] Prompt token count : {prompt_tokens}")
    print(f"[INFO] Warmup runs        : {warmup}")
    print(f"[INFO] Benchmark runs     : {n_runs}")
    print(f"[INFO] Max new tokens     : {sampling_params.max_tokens}")

    # ── Warmup ────────────────────────────────────────────────────────────────
    if warmup > 0:
        print("\n[INFO] Warming up ...")
        for _ in range(warmup):
            llm.generate([prompt], sampling_params)
        print("[INFO] Warmup done.")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    latencies   = []
    output_lens = []

    print("\n[INFO] Running benchmark ...")
    for i in range(n_runs):
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        t1 = time.perf_counter()

        elapsed   = t1 - t0
        out_text  = outputs[0].outputs[0].text
        out_tokens = len(tokenizer.encode(out_text))

        latencies.append(elapsed)
        output_lens.append(out_tokens)

        tps = out_tokens / elapsed if elapsed > 0 else 0
        print(f"  Run {i+1:2d}: {elapsed:.3f}s  |  {out_tokens} tokens  |  {tps:.1f} tok/s")

    # ── Summary ───────────────────────────────────────────────────────────────
    avg_lat  = sum(latencies) / n_runs
    min_lat  = min(latencies)
    max_lat  = max(latencies)
    avg_toks = sum(output_lens) / n_runs
    avg_tps  = avg_toks / avg_lat if avg_lat > 0 else 0

    print("\n" + "=" * 55)
    print("  BENCHMARK RESULTS (single GPU, single request)")
    print("=" * 55)
    print(f"  Runs               : {n_runs}")
    print(f"  Prompt tokens      : {prompt_tokens}")
    print(f"  Avg output tokens  : {avg_toks:.1f}")
    print(f"  Avg latency        : {avg_lat * 1000:.1f} ms")
    print(f"  Min latency        : {min_lat * 1000:.1f} ms")
    print(f"  Max latency        : {max_lat * 1000:.1f} ms")
    print(f"  Avg throughput     : {avg_tps:.1f} tok/s")
    print("=" * 55)

    mem = get_gpu_memory_gb()
    if mem:
        print("\n  GPU Memory (GPU 0):")
        g = mem[0]
        print(f"    Allocated : {g['allocated_gb']:.2f} GB")
        print(f"    Reserved  : {g['reserved_gb']:.2f} GB")
        print(f"    Total     : {g['total_gb']:.2f} GB")
        print("=" * 55)

    return {
        "prompt_tokens": prompt_tokens,
        "avg_output_tokens": avg_toks,
        "avg_latency_ms": avg_lat * 1000,
        "min_latency_ms": min_lat * 1000,
        "max_latency_ms": max_lat * 1000,
        "avg_throughput_tok_per_s": avg_tps,
        "latencies_ms": [l * 1000 for l in latencies],
        "output_lens": output_lens,
        "gpu_memory": mem,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        required=True,
                        help="Path to model (merged bfloat16 or quantized)")
    parser.add_argument("--n_runs",       type=int, default=3,
                        help="Number of benchmark runs (default: 3)")
    parser.add_argument("--warmup",       type=int, default=1,
                        help="Number of warmup runs (default: 1)")
    parser.add_argument("--max_tokens",   type=int, default=1024,
                        help="Max new tokens to generate (default: 1024)")
    parser.add_argument("--max_model_len", type=int, default=8192,
                        help="Max model context length (default: 8192)")
    parser.add_argument("--prompt_file",  default="",
                        help="Optional: JSONL file to load a real prompt from")
    parser.add_argument("--sample_idx",  type=int, default=0,
                        help="Index of the sample in --prompt_file (default: 0)")
    parser.add_argument("--enable_reasoning", action="store_true", default=False,
                        help="Enable reasoning mode (Qwen3 thinking)")
    parser.add_argument("--output_json", default="",
                        help="Optional: save results to this JSON file")
    parser.add_argument("--quantization", default="none",
                        choices=["gptq", "gptq_marlin", "awq", "none"],
                        help="Quantization type (default: none). NOTE: GPTQ requires vLLM >= 0.9.x for Qwen3-MoE")
    args = parser.parse_args()

    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit("[ERROR] vllm or transformers not installed.")

    print("=" * 55)
    print("  Single-Request Inference Latency Benchmark")
    print("=" * 55)
    print(f"  Model      : {args.model}")
    print(f"  GPU        : single (CUDA_VISIBLE_DEVICES=0)")
    print(f"  Quant      : {args.quantization}")
    print(f"  Reasoning  : {args.enable_reasoning}")
    print("=" * 55)

    print("\n[INFO] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("[INFO] Loading model (tp=1) ...")
    quant = None if args.quantization == "none" else args.quantization
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        **({"quantization": quant} if quant else {}),
        dtype="float16" if quant in ("gptq", "gptq_marlin") else "bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        enable_reasoning=args.enable_reasoning,
        reasoning_parser="deepseek_r1" if args.enable_reasoning else None,
    )
    print("[INFO] Model loaded.")

    # ── Build prompt ──────────────────────────────────────────────────────────
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if args.sample_idx >= len(lines):
            raise SystemExit(f"[ERROR] sample_idx {args.sample_idx} out of range (file has {len(lines)} lines)")
        rec = json.loads(lines[args.sample_idx])
        prompt = build_prompt_from_jsonl(rec, tokenizer)
        print(f"[INFO] Using prompt from {args.prompt_file} (index {args.sample_idx})")
    else:
        prompt = build_default_prompt(tokenizer)
        print("[INFO] Using built-in sample prompt")

    sampling = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        max_tokens=args.max_tokens,
    )

    results = run_benchmark(llm, prompt, tokenizer, sampling, args.n_runs, args.warmup)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Results saved to {out_path}")


if __name__ == "__main__":
    main()
