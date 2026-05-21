#!/usr/bin/env python3
"""
TRT 优化 benchmark v2: 修正版
==============================
发现：
  - use_explicit_typing=False 会导致 attention 编译失败（TRT 要求 strongly typed network）
  - use_explicit_typing=True 时，TRT 已经在用原始 bf16 精度，enabled_precisions={f32} 只是 fallback
  - 因此关键不是精度配置，而是 optimization_level / workspace / decompose_attention

本脚本测试：
  1. TRT + optimization_level=5 + 大 workspace（强类型模式）
  2. TRT + decompose_attention=True（把 SDPA 拆成小算子让 TRT 融合）
  3. Inductor max-autotune
  4. Inductor reduce-overhead (CUDA graphs)
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
    print(f"\n  [{label}] Warming up ({warmup} iters)...")
    for i in range(warmup):
        gen = torch.Generator(DEVICE).manual_seed(42)
        _ = pipe(prompt=PROMPT, height=HEIGHT, width=WIDTH,
                 guidance_scale=0, num_inference_steps=STEPS, generator=gen)
        if i == 0:
            print(f"    warmup 1 done")

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


def load_trt_transformer():
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

    return pipe, trt_transformer


def test_trt_opt5_workspace():
    """TRT 强类型 + opt5 + 大 workspace"""
    import torch_tensorrt
    print("=" * 70)
    print("Test 1: TRT + opt_level=5 + workspace=2GB (强类型 bf16)")
    print("=" * 70)

    pipe, trt_transformer = load_trt_transformer()

    compiled = torch.compile(
        trt_transformer,
        backend="torch_tensorrt",
        options={
            "use_explicit_typing": True,
            "enabled_precisions": {torch.float32},  # fallback only, bf16 保留
            "workspace_size": 2 << 30,  # 2GB
            "min_block_size": 1,
            "optimization_level": 5,
            "use_python_runtime": False,
            "use_fast_partitioner": True,
        }
    )
    pipe.transformer = compiled
    avg = timed_run(pipe, "TRT+opt5+ws2G", warmup=3, runs=RUNS)
    del pipe
    cleanup()
    return avg


def test_trt_decompose_attention():
    """TRT + decompose_attention=True: 把 SDPA 拆成 matmul+softmax+matmul"""
    import torch_tensorrt
    print("\n" + "=" * 70)
    print("Test 2: TRT + decompose_attention=True")
    print("=" * 70)

    pipe, trt_transformer = load_trt_transformer()

    compiled = torch.compile(
        trt_transformer,
        backend="torch_tensorrt",
        options={
            "use_explicit_typing": True,
            "enabled_precisions": {torch.float32},
            "workspace_size": 2 << 30,
            "min_block_size": 1,
            "optimization_level": 5,
            "use_python_runtime": False,
            "decompose_attention": True,  # 拆分 SDPA
        }
    )
    pipe.transformer = compiled
    avg = timed_run(pipe, "TRT+decompose_attn", warmup=3, runs=RUNS)
    del pipe
    cleanup()
    return avg


def test_trt_experimental_decomp():
    """TRT + enable_experimental_decompositions=True"""
    import torch_tensorrt
    print("\n" + "=" * 70)
    print("Test 3: TRT + experimental decompositions")
    print("=" * 70)

    pipe, trt_transformer = load_trt_transformer()

    compiled = torch.compile(
        trt_transformer,
        backend="torch_tensorrt",
        options={
            "use_explicit_typing": True,
            "enabled_precisions": {torch.float32},
            "workspace_size": 2 << 30,
            "min_block_size": 1,
            "optimization_level": 5,
            "enable_experimental_decompositions": True,
        }
    )
    pipe.transformer = compiled
    avg = timed_run(pipe, "TRT+exp_decomp", warmup=3, runs=RUNS)
    del pipe
    cleanup()
    return avg


def test_inductor_max_autotune():
    """Inductor max-autotune"""
    print("\n" + "=" * 70)
    print("Test 4: Inductor max-autotune")
    print("=" * 70)

    pipe, trt_transformer = load_trt_transformer()
    compiled = torch.compile(trt_transformer, backend="inductor", mode="max-autotune")
    pipe.transformer = compiled
    avg = timed_run(pipe, "inductor+max-autotune", warmup=3, runs=RUNS)
    del pipe
    cleanup()
    return avg


def test_inductor_reduce_overhead():
    """Inductor reduce-overhead (CUDA graphs)"""
    print("\n" + "=" * 70)
    print("Test 5: Inductor reduce-overhead (CUDA graphs)")
    print("=" * 70)

    pipe, trt_transformer = load_trt_transformer()
    compiled = torch.compile(trt_transformer, backend="inductor", mode="reduce-overhead")
    pipe.transformer = compiled
    avg = timed_run(pipe, "inductor+reduce-overhead", warmup=3, runs=RUNS)
    del pipe
    cleanup()
    return avg


if __name__ == "__main__":
    results = {}

    tests = [
        ("TRT+opt5+ws2G", test_trt_opt5_workspace),
        ("TRT+decompose_attn", test_trt_decompose_attention),
        ("TRT+exp_decomp", test_trt_experimental_decomp),
        ("inductor+max-autotune", test_inductor_max_autotune),
        ("inductor+reduce-overhead", test_inductor_reduce_overhead),
    ]

    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  ❌ {name} 失败: {type(e).__name__}: {str(e)[:200]}")
            results[name] = None
            cleanup()

    BASELINE = 6.442
    INDUCTOR_DEFAULT = 5.224

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"参考: baseline(eager)=6.442s, inductor(default)=5.224s (1.23x)")
    print("=" * 70)
    for name, avg in results.items():
        if avg is None:
            print(f"  {name:35s}  FAILED")
        else:
            speedup = BASELINE / avg
            print(f"  {name:35s}  {avg:.3f}s  ({speedup:.2f}x vs baseline)")
    print("=" * 70)
