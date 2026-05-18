"""
Round 2 Optimization Benchmark for ZImage
Tests: TF32, max-autotune, step reduction, prompt cache
"""
import sys, time, torch, gc

print("=" * 60)
print("ZImage Round 2 Optimization Benchmark")
print("=" * 60)

# === Test 9: TF32 Status ===
print("\n[Test 9] TF32 Status Check")
print(f"  matmul.allow_tf32 = {torch.backends.cuda.matmul.allow_tf32}")
print(f"  cudnn.allow_tf32  = {torch.backends.cudnn.allow_tf32}")
# Enable TF32 (should be default on Ampere+, but let's be explicit)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print(f"  [After enable] matmul.allow_tf32 = {torch.backends.cuda.matmul.allow_tf32}")
print(f"  [After enable] cudnn.allow_tf32  = {torch.backends.cudnn.allow_tf32}")

# === Load Pipeline ===
print("\nLoading ZImage pipeline...")
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "/home/lixinqian/jinjin/Z-Image",
    torch_dtype=torch.bfloat16,
    local_files_only=True
)
pipe = pipe.to("cuda")

PROMPT = "a futuristic cityscape at sunset, highly detailed, 8k"
NEG_PROMPT = "blurry, low quality"
GEN_KWARGS = dict(
    height=768, width=1344,
    num_inference_steps=9,
    guidance_scale=4.0,
    generator=torch.Generator("cuda").manual_seed(42),
)

def benchmark(label, pipe_obj, num_warmup=3, num_runs=3, **extra_kwargs):
    kwargs = {**GEN_KWARGS, **extra_kwargs}
    # Reset seed each time for consistency
    for i in range(num_warmup):
        kwargs["generator"] = torch.Generator("cuda").manual_seed(42)
        _ = pipe_obj(PROMPT, negative_prompt=NEG_PROMPT, **kwargs)

    times = []
    for i in range(num_runs):
        kwargs["generator"] = torch.Generator("cuda").manual_seed(42)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = pipe_obj(PROMPT, negative_prompt=NEG_PROMPT, **kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000
        times.append(ms)
        print(f"  {label} run {i+1}: {ms:.0f}ms")

    avg = sum(times) / len(times)
    print(f"  {label} avg: {avg:.0f}ms")
    return avg

# ============================================================
# Test A: Baseline (no compile)
# ============================================================
print("\n" + "=" * 60)
print("[A] Baseline (no compile, TF32 enabled)")
print("=" * 60)
baseline = benchmark("Baseline", pipe)

# ============================================================
# Test B: torch.compile mode="reduce-overhead" (current prod)
# ============================================================
print("\n" + "=" * 60)
print("[B] torch.compile(mode='reduce-overhead') - Current Production")
print("=" * 60)
pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
reduce_overhead = benchmark("ReduceOverhead", pipe)

# Reset for next test
del pipe
gc.collect()
torch.cuda.empty_cache()

pipe = DiffusionPipeline.from_pretrained(
    "/home/lixinqian/jinjin/Z-Image",
    torch_dtype=torch.bfloat16,
    local_files_only=True
)
pipe = pipe.to("cuda")

# ============================================================
# Test C: torch.compile mode="max-autotune" (Test 8 + 10)
# ============================================================
print("\n" + "=" * 60)
print("[C] torch.compile(mode='max-autotune') - Test 8+10")
print("    (includes CUDA Graph + autotuning of kernel configs)")
print("=" * 60)
pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune")
max_autotune = benchmark("MaxAutotune", pipe)

# ============================================================
# Test D: Step Reduction (Test 14)
# Using current best compile mode from above
# ============================================================
print("\n" + "=" * 60)
print("[D] Step Reduction Tests (Test 14)")
print("    Using max-autotune compile from Test C")
print("=" * 60)

# Already compiled from Test C, reuse
for steps in [7, 5, 3]:
    print(f"\n--- {steps} steps ---")
    step_result = benchmark(
        f"Steps={steps}",
        pipe,
        num_warmup=2,
        num_runs=3,
        num_inference_steps=steps
    )

# ============================================================
# Test E: Prompt Embedding Cache (Test 13)
# ============================================================
print("\n" + "=" * 60)
print("[E] Prompt Embedding Cache (Test 13)")
print("    Cache text encoder output, skip re-encoding on each call")
print("=" * 60)

# Check if pipe has encode_prompt or text_encoder
if hasattr(pipe, 'encode_prompt'):
    print("  pipe.encode_prompt exists, testing cached embeddings...")

    # Pre-compute prompt embeddings
    try:
        # Try to get cached embeddings
        prompt_embeds_result = pipe.encode_prompt(
            PROMPT,
            negative_prompt=NEG_PROMPT,
            do_classifier_free_guidance=True,
        )

        if isinstance(prompt_embeds_result, tuple):
            if len(prompt_embeds_result) == 2:
                prompt_embeds, negative_embeds = prompt_embeds_result
                cached_kwargs = dict(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                )
            elif len(prompt_embeds_result) == 3:
                prompt_embeds, negative_embeds, pooled = prompt_embeds_result
                cached_kwargs = dict(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    pooled_prompt_embeds=pooled,
                )
            else:
                print(f"  encode_prompt returned {len(prompt_embeds_result)} items, skipping cache test")
                cached_kwargs = None
        else:
            print(f"  encode_prompt returned unexpected type, skipping")
            cached_kwargs = None

        if cached_kwargs:
            cache_result = benchmark(
                "PromptCached",
                pipe,
                num_warmup=2,
                num_runs=3,
                **cached_kwargs,
                # Don't pass prompt/neg_prompt when using embeds
            )
    except Exception as e:
        print(f"  Prompt cache test failed: {e}")
elif hasattr(pipe, 'text_encoder'):
    print("  No encode_prompt method, but text_encoder exists")
    print("  Skipping - would need custom caching logic")
else:
    print("  No text_encoder found, skipping prompt cache test")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  [A] Baseline (no compile):        {baseline:.0f}ms")
print(f"  [B] reduce-overhead (current):    {reduce_overhead:.0f}ms  ({(1-reduce_overhead/baseline)*100:+.1f}%)")
print(f"  [C] max-autotune:                 {max_autotune:.0f}ms  ({(1-max_autotune/baseline)*100:+.1f}%)")
print(f"  [C vs B] max-autotune vs reduce:  {(1-max_autotune/reduce_overhead)*100:+.1f}%")
print("=" * 60)
