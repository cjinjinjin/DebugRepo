"""
vLLM serving throughput & latency benchmark for Gemma 4.

Sends N requests concurrently to a running vLLM server and measures:
  - Per-request latency (time to full response)
  - Throughput (requests/second)
  - Output token counts and tok/s

Usage:
  python Gemma4/benchmark_vllm.py \
      --input_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
      --num_samples 50 \
      --concurrency 8 \
      --base_url http://localhost:8000 \
      --no_cot
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp


# System prompts (same as inference_gemma4.py)
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
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            return msg["content"]
    return ""


async def send_request(session, base_url, model_name, system_prompt,
                       user_content, max_tokens, sample_idx):
    """Send a single chat completion request and return timing info."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
    }

    t_start = time.perf_counter()
    try:
        async with session.post(url, json=payload) as resp:
            result = await resp.json()
        t_end = time.perf_counter()

        if "error" in result:
            return {
                "sample_idx": sample_idx,
                "error": result["error"],
                "elapsed_s": round(t_end - t_start, 3),
            }

        usage = result.get("usage", {})
        output_tokens = usage.get("completion_tokens", 0)
        elapsed = t_end - t_start
        tok_per_sec = output_tokens / elapsed if elapsed > 0 else 0

        return {
            "sample_idx": sample_idx,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": output_tokens,
            "elapsed_s": round(elapsed, 3),
            "tokens_per_sec": round(tok_per_sec, 1),
        }
    except Exception as e:
        t_end = time.perf_counter()
        return {
            "sample_idx": sample_idx,
            "error": str(e),
            "elapsed_s": round(t_end - t_start, 3),
        }


async def run_benchmark(args):
    """Run concurrent benchmark."""
    records = load_jsonl(args.input_file, max_records=args.num_samples)
    user_contents = [extract_user_content(r) for r in records]

    system_prompt = SYSTEM_PROMPT_NO_COT if args.no_cot else SYSTEM_PROMPT
    cot_label = "no-CoT" if args.no_cot else "with-CoT"

    print(f"Samples: {len(user_contents)}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Mode: {cot_label}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Server: {args.base_url}")
    print()

    # Discover model name
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{args.base_url}/v1/models") as resp:
            models_resp = await resp.json()
        model_name = models_resp["data"][0]["id"]
        print(f"Model: {model_name}")

    # Run with concurrency limit
    semaphore = asyncio.Semaphore(args.concurrency)
    results = []

    async def bounded_request(session, idx, content):
        async with semaphore:
            result = await send_request(
                session, args.base_url, model_name, system_prompt,
                content, args.max_tokens, idx,
            )
            status = "OK" if "error" not in result else "ERR"
            tokens = result.get("output_tokens", "?")
            elapsed = result["elapsed_s"]
            print(f"  [{idx+1}/{len(user_contents)}] {status} "
                  f"{tokens} tokens in {elapsed:.1f}s")
            return result

    t_total_start = time.perf_counter()

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as session:
        tasks = [
            bounded_request(session, i, content)
            for i, content in enumerate(user_contents)
        ]
        results = await asyncio.gather(*tasks)

    t_total_end = time.perf_counter()
    total_wall = t_total_end - t_total_start

    # Summary
    ok_results = [r for r in results if "error" not in r]
    err_results = [r for r in results if "error" in r]

    if not ok_results:
        print("\nAll requests failed!")
        for r in err_results[:5]:
            print(f"  Error: {r['error']}")
        return

    latencies = [r["elapsed_s"] for r in ok_results]
    output_tokens = [r["output_tokens"] for r in ok_results]
    tok_per_sec = [r["tokens_per_sec"] for r in ok_results]

    import numpy as np
    throughput = len(ok_results) / total_wall

    print(f"\n{'='*60}")
    print(f"vLLM Benchmark Summary — BF16 ({cot_label})")
    print(f"{'='*60}")
    print(f"Successful:        {len(ok_results)}/{len(results)}")
    print(f"Concurrency:       {args.concurrency}")
    print(f"Total wall time:   {total_wall:.1f}s")
    print(f"Throughput:        {throughput:.2f} req/s")
    print(f"{'='*60}")
    print(f"Latency (per req):")
    print(f"  Avg:             {np.mean(latencies):.1f}s")
    print(f"  Median:          {np.median(latencies):.1f}s")
    print(f"  P95:             {np.percentile(latencies, 95):.1f}s")
    print(f"  Min:             {np.min(latencies):.1f}s")
    print(f"  Max:             {np.max(latencies):.1f}s")
    print(f"{'='*60}")
    print(f"Output tokens:")
    print(f"  Avg:             {np.mean(output_tokens):.0f}")
    print(f"  Median:          {np.median(output_tokens):.0f}")
    print(f"{'='*60}")
    print(f"Per-request tok/s:")
    print(f"  Avg:             {np.mean(tok_per_sec):.1f}")
    print(f"  Median:          {np.median(tok_per_sec):.1f}")
    print(f"{'='*60}")

    if err_results:
        print(f"\nErrors ({len(err_results)}):")
        for r in err_results[:5]:
            print(f"  Sample {r['sample_idx']}: {r['error']}")

    # Save detailed results
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump({
                "config": {
                    "concurrency": args.concurrency,
                    "num_samples": len(results),
                    "max_tokens": args.max_tokens,
                    "mode": cot_label,
                    "model": model_name,
                },
                "summary": {
                    "throughput_rps": round(throughput, 2),
                    "avg_latency_s": round(float(np.mean(latencies)), 1),
                    "median_latency_s": round(float(np.median(latencies)), 1),
                    "p95_latency_s": round(float(np.percentile(latencies, 95)), 1),
                    "avg_output_tokens": round(float(np.mean(output_tokens))),
                    "avg_tok_per_sec": round(float(np.mean(tok_per_sec)), 1),
                    "total_wall_s": round(total_wall, 1),
                    "success": len(ok_results),
                    "errors": len(err_results),
                },
                "results": results,
            }, f, indent=2)
        print(f"\nDetailed results saved to {args.output_file}")


def main():
    parser = argparse.ArgumentParser(description="vLLM serving benchmark")
    parser.add_argument("--input_file", required=True, help="Input JSONL file")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Max concurrent requests")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--base_url", default="http://localhost:8000")
    parser.add_argument("--no_cot", action="store_true", default=False,
                        help="Use no-CoT system prompt")
    parser.add_argument("--output_file", default="",
                        help="Save detailed results JSON")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
