#!/usr/bin/env python3
"""
ZImage Cache Optimization 全量对比测试
修复 pipeline patch + 测试 Baseline / FBC / MagCache / TaylorSeer

运行命令:
    python /tmp/all_in_one.py

注意: 此脚本会修改 pipeline_z_image.py，请在容器环境中运行
"""

import time, gc, os, sys, re, io, base64, json, torch
import numpy as np
from pathlib import Path

MODEL_DIR = "Tongyi-MAI/Z-Image-Turbo"
PIPELINE_PATH = "/usr/local/lib/python3.12/dist-packages/diffusers/pipelines/z_image/pipeline_z_image.py"
PROMPT = "A beautiful sunset over the ocean, golden light reflecting on calm waters, dramatic clouds"
WIDTH, HEIGHT = 1344, 768
STEPS = 9
SEED = 42
WARMUP = 1
REPEAT = 3

results = {}

# ============================================================
# Step 1: Fix pipeline_z_image.py
# ============================================================
def fix_pipeline():
    print("=" * 60)
    print("Step 1: Fixing pipeline_z_image.py")
    print("=" * 60)

    with open(PIPELINE_PATH, "r") as f:
        content = f.read()
    lines = content.split("\n")

    # Find the transformer call area (look for "model_out_list = self.transformer(")
    # and remove any broken cache_context or _set_context patches
    new_lines = []
    skip_next = 0
    i = 0
    while i < len(lines):
        line = lines[i]

        if skip_next > 0:
            skip_next -= 1
            i += 1
            continue

        # Remove broken "with self.transformer.cache_context" lines
        if "cache_context" in line and "self.transformer" in line:
            print(f"  Removing broken cache_context at line {i+1}: {line.strip()}")
            # This is a `with` statement - the next lines are indented under it
            # We need to de-indent them
            i += 1
            # Collect the indented block and de-indent by one level
            while i < len(lines):
                next_line = lines[i]
                # If it's more indented than the with statement, de-indent
                stripped = next_line.lstrip()
                if stripped and not next_line.startswith(line[:len(line) - len(line.lstrip())] + "    "):
                    break
                # Remove one level of indentation (4 spaces)
                if next_line.startswith("    " * 7):  # 28 spaces -> 24
                    new_lines.append(next_line[4:])
                elif len(next_line) > 4 and next_line[:4] == "    ":
                    new_lines.append(next_line[4:])
                else:
                    new_lines.append(next_line)
                i += 1
            continue

        # Remove any previous _set_context patches
        if "_set_context" in line and ("HookRegistry" in line or "_reg" in line or "_HR" in line):
            print(f"  Removing old _set_context patch at line {i+1}: {line.strip()}")
            i += 1
            continue

        # Remove import lines for HookRegistry that we previously added
        if "from diffusers.hooks import HookRegistry as _HR" in line:
            print(f"  Removing old import at line {i+1}")
            i += 1
            continue

        if "_reg = _HR.check_if_exists_or_initialize" in line:
            print(f"  Removing old reg init at line {i+1}")
            i += 1
            continue

        new_lines.append(line)
        i += 1

    content = "\n".join(new_lines)

    # Now find the transformer call and wrap it with _set_context
    # Pattern: "model_out_list = self.transformer("
    # We need to add _set_context before and after
    pattern = r"(\s+)(model_out_list = self\.transformer\()"
    match = re.search(pattern, content)
    if match:
        indent = match.group(1)
        transformer_call = match.group(2)
        # Find the closing )[0] line
        start_pos = match.start()
        # Build replacement with context setup
        replacement = (
            f"{indent}from diffusers.hooks import HookRegistry as _HR\n"
            f"{indent}_reg = _HR.check_if_exists_or_initialize(self.transformer)\n"
            f"{indent}_reg._set_context(\"default\")\n"
            f"{indent}{transformer_call}"
        )
        content = content[:match.start()] + replacement + content[match.end():]

        # Find the )[0] line after the transformer call and add _set_context(None) after it
        # Search for the line ending with )[0]
        rest = content[match.start():]
        close_match = re.search(r"\)\[0\]\n", rest)
        if close_match:
            insert_pos = match.start() + close_match.end()
            content = content[:insert_pos] + f"{indent}_reg._set_context(None)\n" + content[insert_pos:]
            print("  Inserted _set_context(\"default\") before transformer call")
            print("  Inserted _set_context(None) after transformer call")
        else:
            print("  WARNING: Could not find )[0] closing line")
    else:
        print("  WARNING: Could not find transformer call pattern")
        # Check if it's already patched correctly
        if '_set_context("default")' in content:
            print("  Pipeline appears already correctly patched")
        else:
            print("  ERROR: Cannot patch pipeline!")
            sys.exit(1)

    with open(PIPELINE_PATH, "w") as f:
        f.write(content)

    # Verify
    with open(PIPELINE_PATH, "r") as f:
        verify = f.read()

    if "cache_context" in verify:
        print("  ERROR: cache_context still present after fix!")
        sys.exit(1)

    if '_set_context("default")' not in verify:
        print("  ERROR: _set_context not found after fix!")
        sys.exit(1)

    print("  Pipeline fixed successfully!")
    print()


# ============================================================
# Step 2: Load pipeline helper
# ============================================================
def load_pipe():
    """Load a fresh ZImagePipeline."""
    from diffusers import ZImagePipeline
    pipe = ZImagePipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    return pipe


def free_pipe(pipe):
    """Properly free pipeline GPU memory."""
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)


def generate(pipe, seed=SEED):
    """Run single generation, return (image, time_seconds)."""
    gen = torch.Generator("cuda").manual_seed(seed)
    t0 = time.time()
    result = pipe(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        guidance_scale=0,
        num_inference_steps=STEPS,
        generator=gen,
    )
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return result.images[0], elapsed


def benchmark(pipe, label):
    """Warmup + repeated runs, return avg time."""
    print(f"\n--- {label} ---")
    # Warmup
    for i in range(WARMUP):
        img, t = generate(pipe)
        print(f"  Warmup {i+1}: {t:.3f}s")

    # Timed runs
    times = []
    for i in range(REPEAT):
        img, t = generate(pipe, seed=SEED + i)
        times.append(t)
        print(f"  Run {i+1}: {t:.3f}s")

    avg = np.mean(times)
    print(f"  Average: {avg:.3f}s")

    # Save last image
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    print(f"  Image size: {len(img_b64)} chars (base64)")

    return avg, img_b64


# ============================================================
# Step 3: Run tests
# ============================================================
def test_baseline():
    print("\n" + "=" * 60)
    print("Test: Baseline (no cache)")
    print("=" * 60)
    pipe = load_pipe()
    avg, img = benchmark(pipe, "Baseline")
    results["baseline"] = {"avg_time": avg, "image": img[:100]}
    free_pipe(pipe)
    return avg


def test_fbc():
    print("\n" + "=" * 60)
    print("Test: First Block Cache (FBC)")
    print("=" * 60)
    pipe = load_pipe()
    from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig
    config = FirstBlockCacheConfig(threshold=0.05)
    apply_first_block_cache(pipe.transformer, config)
    print(f"  FBC applied with threshold=0.05")

    avg, img = benchmark(pipe, "FBC")
    results["fbc"] = {"avg_time": avg, "image": img[:100]}
    free_pipe(pipe)
    return avg


def test_taylorseer():
    print("\n" + "=" * 60)
    print("Test: TaylorSeer Cache")
    print("=" * 60)
    pipe = load_pipe()
    from diffusers.hooks import apply_taylorseer_cache, TaylorSeerCacheConfig

    # With disable_cache_before_step=1 and disable_cache_after_step=STEPS-1
    # to force first/last steps without cache (colleague's suggestion)
    config = TaylorSeerCacheConfig(
        cache_interval=2,
        max_order=1,
        disable_cache_before_step=1,
        disable_cache_after_step=STEPS - 1,
    )
    apply_taylorseer_cache(pipe.transformer, config)
    print(f"  TaylorSeer applied: interval=2, max_order=1, disable_before=1, disable_after={STEPS-1}")

    avg, img = benchmark(pipe, "TaylorSeer")
    results["taylorseer"] = {"avg_time": avg, "image": img[:100]}
    free_pipe(pipe)
    return avg


def test_taylorseer_aggressive():
    print("\n" + "=" * 60)
    print("Test: TaylorSeer Cache (aggressive, interval=3)")
    print("=" * 60)
    pipe = load_pipe()
    from diffusers.hooks import apply_taylorseer_cache, TaylorSeerCacheConfig

    config = TaylorSeerCacheConfig(
        cache_interval=3,
        max_order=1,
        disable_cache_before_step=1,
        disable_cache_after_step=STEPS - 1,
    )
    apply_taylorseer_cache(pipe.transformer, config)
    print(f"  TaylorSeer applied: interval=3, max_order=1, disable_before=1, disable_after={STEPS-1}")

    avg, img = benchmark(pipe, "TaylorSeer-aggressive")
    results["taylorseer_aggressive"] = {"avg_time": avg, "image": img[:100]}
    free_pipe(pipe)
    return avg


def test_magcache():
    print("\n" + "=" * 60)
    print("Test: MagCache (calibrate + infer)")
    print("=" * 60)

    # Phase 1: Calibration
    print("\n  Phase 1: Calibration run...")
    pipe = load_pipe()
    from diffusers.hooks import apply_mag_cache, MagCacheConfig

    cal_config = MagCacheConfig(
        num_inference_steps=STEPS,
        calibrate=True,
    )
    apply_mag_cache(pipe.transformer, cal_config)

    # Run calibration
    gen = torch.Generator("cuda").manual_seed(SEED)
    pipe(
        prompt=PROMPT, height=HEIGHT, width=WIDTH,
        guidance_scale=0, num_inference_steps=STEPS, generator=gen,
    )

    # Extract mag_ratios
    mag_ratios = cal_config.mag_ratios
    print(f"  Calibration done. mag_ratios type: {type(mag_ratios)}")
    if mag_ratios is not None:
        if isinstance(mag_ratios, dict):
            print(f"  mag_ratios keys: {list(mag_ratios.keys())[:5]}...")
        elif isinstance(mag_ratios, list):
            print(f"  mag_ratios length: {len(mag_ratios)}")
    free_pipe(pipe)

    if mag_ratios is None:
        print("  ERROR: Calibration failed, mag_ratios is None")
        results["magcache"] = {"avg_time": -1, "error": "calibration failed"}
        return -1

    # Phase 2: Inference with calibrated ratios
    print("\n  Phase 2: Inference with calibrated ratios...")
    pipe = load_pipe()
    infer_config = MagCacheConfig(
        num_inference_steps=STEPS,
        mag_ratios=mag_ratios,
    )
    apply_mag_cache(pipe.transformer, infer_config)

    avg, img = benchmark(pipe, "MagCache")
    results["magcache"] = {"avg_time": avg, "image": img[:100]}
    free_pipe(pipe)
    return avg


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("ZImage Cache Optimization - Full Comparison")
    print(f"Model: {MODEL_DIR}")
    print(f"Prompt: {PROMPT[:50]}...")
    print(f"Resolution: {WIDTH}x{HEIGHT}, Steps: {STEPS}, Seed: {SEED}")
    print(f"Warmup: {WARMUP}, Repeat: {REPEAT}")
    print()

    # Step 1: Fix pipeline
    fix_pipeline()

    # Step 2: Run tests
    baseline = test_baseline()

    fbc = test_fbc()

    taylor = test_taylorseer()

    taylor_agg = test_taylorseer_aggressive()

    magcache = test_magcache()

    # Step 3: Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<25} {'Avg Time (s)':<15} {'Speedup':<10}")
    print("-" * 50)

    for name, key in [
        ("Baseline", "baseline"),
        ("FBC", "fbc"),
        ("TaylorSeer (i=2)", "taylorseer"),
        ("TaylorSeer (i=3)", "taylorseer_aggressive"),
        ("MagCache", "magcache"),
    ]:
        if key in results and results[key]["avg_time"] > 0:
            t = results[key]["avg_time"]
            speedup = baseline / t if t > 0 else 0
            print(f"{name:<25} {t:<15.3f} {speedup:<10.2f}x")
        elif key in results:
            print(f"{name:<25} {'FAILED':<15} {'N/A':<10}")
        else:
            print(f"{name:<25} {'SKIPPED':<15} {'N/A':<10}")

    print()
    print("Done!")
