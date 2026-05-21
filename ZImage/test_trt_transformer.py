#!/usr/bin/env python3
"""
测试 TRT-compatible ZImage transformer
=====================================
Step 1: 用修改后的 transformer 替换原始 transformer，验证 eager 推理正确
Step 2: 测试 torch.compile(backend='torch_tensorrt')
"""

import sys
import time
import torch
import numpy as np

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"


def step1_verify_eager():
    """加载原始模型，替换 transformer 模块，验证输出一致"""
    print("=" * 60)
    print("Step 1: Verify eager inference with TRT-compatible transformer")
    print("=" * 60)

    from diffusers import ZImagePipeline

    # 加载原始 pipeline
    pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    pipe.to(DEVICE)

    # 原始推理
    print("\n  [1a] Running original pipeline...")
    gen = torch.Generator(DEVICE).manual_seed(42)
    orig_out = pipe(
        prompt="A red apple on a white table",
        height=768, width=1344,
        guidance_scale=0, num_inference_steps=4,
        generator=gen,
    )
    orig_images = orig_out.images

    # 替换 transformer
    print("\n  [1b] Replacing transformer with TRT-compatible version...")

    # 保存原始 transformer 的 state_dict
    orig_state = pipe.transformer.state_dict()
    orig_config = pipe.transformer.config

    # 导入修改后的 transformer
    sys.path.insert(0, "/tmp")
    from transformer_z_image_trt import ZImageTransformer2DModel as TRTModel

    # 创建新 transformer 并加载权重
    trt_transformer = TRTModel(**orig_config)
    trt_transformer.load_state_dict(orig_state, strict=True)
    trt_transformer.to(device=DEVICE, dtype=DTYPE)
    trt_transformer.eval()

    print(f"    Loaded state_dict: {len(orig_state)} keys, strict=True OK")

    # 替换
    pipe.transformer = trt_transformer

    # TRT-compatible 推理
    print("\n  [1c] Running TRT-compatible pipeline...")
    gen = torch.Generator(DEVICE).manual_seed(42)
    trt_out = pipe(
        prompt="A red apple on a white table",
        height=768, width=1344,
        guidance_scale=0, num_inference_steps=4,
        generator=gen,
    )
    trt_images = trt_out.images

    # 比较
    orig_np = np.array(orig_images[0])
    trt_np = np.array(trt_images[0])

    diff = np.abs(orig_np.astype(float) - trt_np.astype(float))
    max_diff = diff.max()
    mean_diff = diff.mean()
    psnr = 10 * np.log10(255**2 / (mean_diff**2 + 1e-10)) if mean_diff > 0 else float('inf')

    print(f"\n  [1d] Comparison:")
    print(f"    Max pixel diff:  {max_diff}")
    print(f"    Mean pixel diff: {mean_diff:.4f}")
    print(f"    PSNR:           {psnr:.1f} dB")

    if max_diff <= 2:
        print("    ✅ PASS: Output matches (max diff ≤ 2)")
    elif max_diff <= 10:
        print("    ⚠️  WARN: Small diff (likely bf16 RoPE rounding)")
    else:
        print("    ❌ FAIL: Large diff — check Real RoPE implementation")

    return pipe


def step2_test_torch_compile_inductor(pipe):
    """先用 inductor 测试 torch.compile 是否工作"""
    print("\n" + "=" * 60)
    print("Step 2: torch.compile(backend='inductor')")
    print("=" * 60)

    pipe.transformer = torch.compile(pipe.transformer, backend="inductor")

    print("  Running compiled inference...")
    try:
        gen = torch.Generator(DEVICE).manual_seed(42)
        t0 = time.time()
        _ = pipe(
            prompt="A red apple on a white table",
            height=768, width=1344,
            guidance_scale=0, num_inference_steps=2,
            generator=gen,
        )
        t1 = time.time()
        print(f"    ✅ Inductor compile OK ({t1-t0:.1f}s including compilation)")
    except Exception as e:
        print(f"    ❌ Inductor failed: {e}")
        return False
    return True


def step3_test_torch_compile_trt(pipe):
    """测试 torch_tensorrt backend"""
    print("\n" + "=" * 60)
    print("Step 3: torch.compile(backend='torch_tensorrt')")
    print("=" * 60)

    try:
        import torch_tensorrt
        print(f"  torch_tensorrt: {torch_tensorrt.__version__}")
    except ImportError:
        print("  ❌ torch_tensorrt not installed")
        return

    # 重新加载 TRT transformer（compile 不能复用）
    from diffusers import ZImagePipeline
    orig_pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    orig_pipe.to(DEVICE)
    orig_state = orig_pipe.transformer.state_dict()
    orig_config = orig_pipe.transformer.config

    from transformer_z_image_trt import ZImageTransformer2DModel as TRTModel
    trt_transformer = TRTModel(**orig_config)
    trt_transformer.load_state_dict(orig_state, strict=True)
    trt_transformer.to(device=DEVICE, dtype=DTYPE)
    trt_transformer.eval()

    compiled = torch.compile(trt_transformer, backend="torch_tensorrt")
    orig_pipe.transformer = compiled

    print("  Running TRT-compiled inference...")
    try:
        gen = torch.Generator(DEVICE).manual_seed(42)
        t0 = time.time()
        _ = orig_pipe(
            prompt="A red apple on a white table",
            height=768, width=1344,
            guidance_scale=0, num_inference_steps=1,
            generator=gen,
        )
        t1 = time.time()
        print(f"    ✅ TRT compile SUCCESS! ({t1-t0:.1f}s including compilation)")
    except Exception as e:
        import traceback
        print(f"    ❌ TRT compile failed: {type(e).__name__}: {str(e)[:500]}")
        tb = traceback.format_exc()
        with open("/tmp/trt_fix_traceback.txt", "w") as f:
            f.write(tb)
        print(f"    Full traceback: /tmp/trt_fix_traceback.txt")

        # 分析剩余障碍
        error_str = str(e) + tb
        if "complex" in error_str.lower():
            print("    → Still has complex64 issue")
        if "scatter" in error_str.lower():
            print("    → Still has scatter issue")
        if "GetItemSource" in error_str:
            print("    → Dynamo can't trace nested list inputs (try fullgraph=False)")


if __name__ == "__main__":
    pipe = step1_verify_eager()
    # step2 会改 pipe.transformer，所以用一个 flag
    inductor_ok = step2_test_torch_compile_inductor(pipe)
    # 释放 GPU 内存，避免 step3 OOM
    del pipe
    import gc; gc.collect()
    torch.cuda.empty_cache()
    step3_test_torch_compile_trt(None)
    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
