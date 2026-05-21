#!/usr/bin/env python3
"""
Benchmark: Baseline vs Inductor+FBC vs TRT
==========================================
Measures end-to-end inference latency for ZImage pipeline.
"""

import sys
import time
import gc
import torch
import numpy as np

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
PROMPT = "A red apple on a white table"
HEIGHT, WIDTH = 768, 1344
STEPS = 9
WARMUP = 2
RUNS = 5


def timed_run(pipe, label, warmup=WARMUP, runs=RUNS):
    """Run warmup + timed iterations, return per-iteration times."""
    print(f"\n  [{label}] Warming up ({warmup} iters)...")
    for i in range(warmup):
        gen = torch.Generator(DEVICE).manual_seed(42)
        _ = pipe(prompt=PROMPT, height=HEIGHT, width=WIDTH,
                 guidance_scale=0, num_inference_steps=STEPS, generator=gen)
        if i == 0:
            print(f"    warmup 1 done (includes compilation)")

    torch.cuda.synchronize()
    times = []
    print(f"  [{label}] Timing ({runs} iters)...")
    for i in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen = torch.Generator(DEVICE).manual_seed(42)
        _ = pipe(prompt=PROMPT, height=HEIGHT, width=WIDTH,
                 guidance_scale=0, num_inference_steps=STEPS, generator=gen)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"    run {i+1}: {times[-1]:.3f}s")

    avg = np.mean(times)
    std = np.std(times)
    print(f"  [{label}] avg={avg:.3f}s ± {std:.3f}s  (min={min(times):.3f}s)")
    return avg


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def bench_baseline():
    """Baseline: no compile, no optimizations."""
    print("=" * 60)
    print("Benchmark 1: Baseline (eager, no compile)")
    print("=" * 60)
    from diffusers import ZImagePipeline
    pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    pipe.to(DEVICE)
    avg = timed_run(pipe, "baseline")
    del pipe
    cleanup()
    return avg


def bench_inductor_fbc():
    """Inductor + first-batch-cache (current best)."""
    print("\n" + "=" * 60)
    print("Benchmark 2: Inductor + FBC (current best)")
    print("=" * 60)
    from diffusers import ZImagePipeline
    pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    pipe.to(DEVICE)

    # Apply FBC
    try:
        from diffusers.hooks import apply_first_batch_cache
        apply_first_batch_cache(pipe.transformer, threshold=0.3)
        print("  FBC applied (threshold=0.3)")
    except ImportError:
        try:
            from diffusers import apply_first_batch_cache
            apply_first_batch_cache(pipe.transformer, threshold=0.3)
            print("  FBC applied via diffusers top-level (threshold=0.3)")
        except ImportError:
            import diffusers
            print(f"  ⚠️ FBC not available (diffusers {diffusers.__version__})")
            print(f"    Available in diffusers.hooks: {[x for x in dir(diffusers.hooks) if 'cache' in x.lower() or 'batch' in x.lower()]}" if hasattr(diffusers, 'hooks') else "    diffusers.hooks module not found")

    pipe.transformer = torch.compile(pipe.transformer, backend="inductor")
    avg = timed_run(pipe, "inductor+FBC")
    del pipe
    cleanup()
    return avg


def bench_trt():
    """TRT-compatible transformer with torch.compile(backend='torch_tensorrt')."""
    print("\n" + "=" * 60)
    print("Benchmark 3: TRT (torch_tensorrt)")
    print("=" * 60)

    try:
        import torch_tensorrt
        print(f"  torch_tensorrt: {torch_tensorrt.__version__}")
    except ImportError:
        print("  ❌ torch_tensorrt not installed, skipping")
        return None

    from diffusers import ZImagePipeline
    pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    pipe.to(DEVICE)

    # Replace with TRT-compatible transformer
    orig_state = pipe.transformer.state_dict()
    orig_config = pipe.transformer.config

    sys.path.insert(0, "/tmp")
    from transformer_z_image_trt import ZImageTransformer2DModel as TRTModel
    trt_transformer = TRTModel(**orig_config)
    trt_transformer.load_state_dict(orig_state, strict=True)
    trt_transformer.to(device=DEVICE, dtype=DTYPE)
    trt_transformer.eval()

    pipe.transformer = torch.compile(trt_transformer, backend="torch_tensorrt")
    avg = timed_run(pipe, "TRT", warmup=3, runs=RUNS)  # extra warmup for TRT
    del pipe
    cleanup()
    return avg


def bench_inductor_trt_transformer():
    """TRT-compatible transformer with inductor (not TRT backend) + FBC."""
    print("\n" + "=" * 60)
    print("Benchmark 4: TRT-compatible transformer + Inductor + FBC")
    print("=" * 60)
    from diffusers import ZImagePipeline
    pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    pipe.to(DEVICE)

    orig_state = pipe.transformer.state_dict()
    orig_config = pipe.transformer.config

    sys.path.insert(0, "/tmp")
    from transformer_z_image_trt import ZImageTransformer2DModel as TRTModel
    trt_transformer = TRTModel(**orig_config)
    trt_transformer.load_state_dict(orig_state, strict=True)
    trt_transformer.to(device=DEVICE, dtype=DTYPE)
    trt_transformer.eval()

    pipe.transformer = trt_transformer

    try:
        from diffusers.hooks import apply_first_batch_cache
        apply_first_batch_cache(pipe.transformer, threshold=0.3)
        print("  FBC applied (threshold=0.3)")
    except ImportError:
        try:
            from diffusers import apply_first_batch_cache
            apply_first_batch_cache(pipe.transformer, threshold=0.3)
            print("  FBC applied via diffusers top-level (threshold=0.3)")
        except ImportError:
            print("  FBC not available")

    pipe.transformer = torch.compile(pipe.transformer, backend="inductor")
    avg = timed_run(pipe, "TRT-compat+inductor+FBC")
    del pipe
    cleanup()
    return avg


if __name__ == "__main__":
    results = {}

    results["baseline"] = bench_baseline()
    results["inductor+FBC"] = bench_inductor_fbc()
    results["TRT-compat+inductor+FBC"] = bench_inductor_trt_transformer()
    results["TRT"] = bench_trt()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    baseline = results["baseline"]
    for name, avg in results.items():
        if avg is None:
            print(f"  {name:30s}  SKIPPED")
        else:
            speedup = baseline / avg if avg > 0 else 0
            print(f"  {name:30s}  {avg:.3f}s  ({speedup:.2f}x)")
    print("=" * 60)
