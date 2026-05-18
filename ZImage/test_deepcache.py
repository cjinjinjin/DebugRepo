"""
Test DeepCache acceleration on ZImage pipeline.
Run inside Docker container with ZImage model mounted at /Model.

Usage: python /test_deepcache.py
"""
import time
import torch
from diffusers import ZImagePipeline

MODEL_PATH = "/Model"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
PROMPT = "A beautiful sunset over the ocean with vibrant colors"
WIDTH, HEIGHT = 1344, 768
STEPS = 9
SEED = 42
WARMUP = 1
RUNS = 3


def run_inference(pipe, label, seed=SEED):
    """Run inference and return latency in ms."""
    latencies = []
    for i in range(WARMUP + RUNS):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        t0 = time.time()
        images = pipe(
            prompt=PROMPT, height=HEIGHT, width=WIDTH,
            guidance_scale=0, num_inference_steps=STEPS,
            num_images_per_prompt=1, generator=gen,
        ).images
        latency = int((time.time() - t0) * 1000)
        if i < WARMUP:
            print(f"  [{label}] warmup: {latency}ms")
        else:
            print(f"  [{label}] run {i-WARMUP+1}: {latency}ms")
            latencies.append(latency)
    avg = sum(latencies) // len(latencies)
    print(f"  [{label}] average: {avg}ms")
    return avg, images[0]


def main():
    print("=" * 60)
    print("ZImage DeepCache Benchmark")
    print("=" * 60)

    # Load pipeline
    print(f"\nLoading ZImagePipeline from {MODEL_PATH}...")
    pipe = ZImagePipeline.from_pretrained(MODEL_PATH, torch_dtype=DTYPE).to(DEVICE)
    print("Pipeline loaded.\n")

    # --- Baseline (no optimization) ---
    print("--- Baseline (no optimization) ---")
    baseline_avg, baseline_img = run_inference(pipe, "baseline")
    baseline_img.save("/tmp/baseline.png")
    print()

    # --- DeepCache ---
    print("--- DeepCache ---")
    DeepCacheSDHelper = None
    try:
        from diffusers.utils import DeepCacheSDHelper
        print("DeepCacheSDHelper imported from diffusers")
    except ImportError:
        print("Not in diffusers, trying DeepCache package...")
        try:
            from DeepCache import DeepCacheSDHelper
            print("DeepCacheSDHelper imported from DeepCache package")
        except ImportError:
            print("DeepCache not available. Try: pip install DeepCache")

    if DeepCacheSDHelper is not None:
        for cache_interval in [2, 3]:
            print(f"\n  cache_interval={cache_interval}:")
            helper = DeepCacheSDHelper(pipe=pipe)
            helper.set_params(cache_interval=cache_interval, cache_branch_id=0)
            helper.enable()

            try:
                avg, img = run_inference(pipe, f"deepcache_ci{cache_interval}")
                img.save(f"/tmp/deepcache_ci{cache_interval}.png")
                speedup = (baseline_avg - avg) / baseline_avg * 100
                print(f"  speedup vs baseline: {speedup:.1f}%")
            except Exception as e:
                print(f"  ERROR with cache_interval={cache_interval}: {e}")

            helper.disable()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Results saved to /tmp/baseline.png, /tmp/deepcache_ci*.png")
    print("Compare images visually to check quality.")
    print("=" * 60)


if __name__ == "__main__":
    main()
