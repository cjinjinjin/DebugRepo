# Inference Acceleration in Practice: ZImage × Gemma4

> The complete decision path from 12+ optimization techniques to production deployment

---

## Part 1: General Inference Acceleration Framework

All acceleration techniques fall into **3 broad categories**:

### Category 1: Make Each Computation Faster

> Core idea: Execute the same computation more efficiently

| Technique | Principle | Applicable Scenarios | Prerequisites |
|-----------|-----------|---------------------|---------------|
| **torch.compile** | Python → Triton/C++ compilation, operator fusion, eliminate Python overhead | Nearly all PyTorch models | PyTorch 2.0+; no unsupported ops or dynamic control flow |
| **TensorRT** | NVIDIA inference engine: layer fusion, precision calibration, kernel auto-tuning | CNNs, standard Transformers | All ops must be TRT-supported; no complex64 or certain scatter ops |
| **ONNX Runtime** | Cross-platform inference engine, graph optimization + multi-backend | Models exportable as static graphs | Successful torch.export; no nested dynamic inputs; encoder must be supported by optimum |
| **FlashAttention** | Tiled attention computation, reduced HBM access | Standard attention with head_dim ≤ 256 | head_dim limit (FA v2 ≤ 256); PyTorch SDPA auto-selects optimal kernel |
| **Quantization (INT8/FP8/4-bit)** | Lower weight/activation precision, reduce memory bandwidth and compute | LLMs (decode-bound, memory bandwidth bottleneck) | Requires hardware kernel support (FP8→H100, INT8→optimized kernels); precision-sensitive models may suffer quality collapse |

#### What is torch.compile?

```
Eager Mode (no compile) — 5 ops = 5 kernels, each dispatched separately by CPU:

  CPU:  prep→dispatch   prep→dispatch   prep→dispatch   prep→dispatch   prep→dispatch
          │                │                │                │                │
  GPU:    ▼                ▼                ▼                ▼                ▼
       ┌────────┐ idle ┌────────┐ idle ┌────────┐ idle ┌────────┐ idle ┌────────┐
       │ Linear │~~~~~~│  Add   │~~~~~~│  GELU  │~~~~~~│ Linear │~~~~~~│  Add   │
       └────────┘      └────────┘      └────────┘      └────────┘      └────────┘
                  ↑              ↑              ↑              ↑
             GPU waits      GPU waits      GPU waits      GPU waits
             for CPU to     for CPU to     for CPU to     for CPU to
             dispatch next  dispatch next  dispatch next  dispatch next

  Problem: GPU finishes a kernel in ~μs, but waits ~10s μs for CPU to prepare
           and dispatch the next one → "Command Buffer Full" in profiling (36.8%)

torch.compile (default) — fuse 5 ops into 2 kernels, CPU dispatches 2 instead of 5:

  CPU:  prep→dispatch              prep→dispatch
          │                            │
  GPU:    ▼                            ▼
       ┌──────────────────────────┐ ┌──────────────────────────┐
       │  Linear + Add + GELU     │ │  Linear + Add            │
       │  (fused into 1 kernel)   │ │  (fused into 1 kernel)   │
       └──────────────────────────┘ └──────────────────────────┘

  Benefit: fewer kernels → fewer CPU dispatches → less GPU idle time
  Remaining cost: CPU still dispatches each fused kernel one by one

torch.compile (reduce-overhead) — above fusion + CUDA Graphs:

  First run:  record entire GPU execution sequence
  Every subsequent run:
  CPU:  replay! (single command)
          │
  GPU:    ▼
       ┌──────────────────────────────────────────────────────────┐
       │  All fused kernels replayed back-to-back, zero idle gaps │
       │  (CPU sends 1 command instead of N)                      │
       └──────────────────────────────────────────────────────────┘

  Benefit: eliminates ALL CPU→GPU dispatch overhead
  Constraint: execution path must be fixed — conflicts with FBC (dynamic skip)
```

| Compile Mode | What It Does | Speedup | Tradeoff |
|-------------|-------------|---------|----------|
| `eager` (no compile) | Python interpreter executes each op one by one, dispatching a separate CUDA kernel for each; GPU idles between dispatches | 1x (baseline) | ✅ Maximum compatibility, easiest to debug |
| `default` | Operator fusion + eliminate Python overhead | ~1.25x | Compatible with dynamic hooks (FBC) |
| `reduce-overhead` | Above + CUDA Graphs (replay recorded GPU commands) | ~1.36x | CUDA Graphs conflicts with stateful hooks — cannot combine with FBC |
| `max-autotune` | Above + auto-select fastest kernel per op | ~1.30x | Longer compilation time; marginal gain over reduce-overhead |

#### What is TensorRT?

TensorRT does the same optimizations as torch.compile, but goes deeper:

```
torch.compile:                          TensorRT:

① Fusion (Triton)                       ① Fusion (TRT engine)
  Linear+Add+GELU → 1 kernel             Linear+Add+GELU → 1 kernel
  ✅ same idea                            ✅ same idea, supports more
                                             fusion patterns

② Kernel params                         ② Kernel Auto-Tuning
  default: heuristic                      benchmark all candidates
  max-autotune: benchmark                 on your GPU, pick fastest
  ✅ same idea                            ✅ same idea (always does it)

  (nothing)                             ③ Precision Calibration
                                          auto-select per-layer precision:
                                          FP32 / FP16 / INT8
                                          sensitive layers keep FP32
                                          others drop to INT8
                                          → speed up + preserve quality

③ Dispatch                              ④ Dispatch
  default: per-kernel dispatch            compiled TRT engine runs
  reduce-overhead: CUDA Graphs            directly on GPU, no Python
  ✅ similar                              ✅ similar to CUDA Graphs
```

| Step | Eager Mode (PyTorch default) | torch.compile | TensorRT |
|------|------------------------------|--------------|----------|
| ① Operator Fusion | ❌ Each op runs as a separate kernel | ✅ Triton: Linear+Add+GELU → 1 kernel | ✅ TRT engine: same, but supports more fusion patterns |
| ② Kernel Tuning | ❌ Uses generic pre-built kernels | default: heuristic; max-autotune: benchmark | Always benchmark all candidates on your GPU |
| ③ Precision Calibration | ❌ Manual only (user sets dtype) | ❌ Not available | ✅ Auto per-layer FP32/FP16/INT8 selection |
| ④ Dispatch | Python interpreter dispatches each op one by one | default: per-kernel; reduce-overhead: CUDA Graphs | Native engine replay (similar to CUDA Graphs) |
| **Compatibility** | ✅ Everything works | High — most PyTorch ops; graceful fallback | Low — ALL ops must be supported; one fails = all fails |
| **Typical speedup** | 1x (baseline) | 1.25–1.36x | 2–5x (when it works) |

### Category 2: Reduce the Number of Computations

> Core idea: Skip unnecessary computation steps or tokens

| Technique | Principle | Applicable Scenarios | Prerequisites |
|-----------|-----------|---------------------|---------------|
| **First Block Cache (FBC)** | Skip subsequent blocks when intermediate features are similar across adjacent denoising steps | Diffusion models | diffusers 0.38+; model must support `_set_context` interface |
| **TeaCache** | Use timestep embedding deltas to predict whether the entire transformer can be skipped | Diffusion models | Requires model-specific polynomial coefficient calibration |
| **DeepCache** | Cache high-level UNet features, compute only low-level features each step | UNet-based Diffusion models | **UNet architecture only** — incompatible with Transformer-based models |
| **Reduce Inference Steps** | 9→7→5 steps, linear latency reduction | Diffusion models | Image quality floor must be validated |
| **No-CoT / No-Think** | Reduce LLM output token count | LLMs (decode-bound) | Only effective in decode-bound scenarios; no effect when prefill-bound |
| **Input Truncation** | Limit input length to reduce prefill and KV cache usage | LLMs | Only effective when prefill-bound or KV cache constrained |
| **Two-Step Generation** | Split task into planning + execution, use stop_strings to truncate | LLM structured output | Task must be decomposable; batch decode more efficient than sequential |

### Category 3: Increase Parallel Throughput

> Core idea: Process more requests simultaneously

| Technique | Principle | Applicable Scenarios | Prerequisites |
|-----------|-----------|---------------------|---------------|
| **Continuous Batching** | Dynamic batching — completed requests release immediately, new requests join instantly | LLM serving | Requires vLLM / TGI or similar serving framework |
| **Tensor Parallel (TP)** | Split model across multiple GPUs, reduce per-GPU computation | Large models (> single GPU VRAM) | More TP is not always better — communication overhead can exceed compute gains |
| **Concurrency Scaling** | Increase simultaneous request count | High-throughput scenarios | Trades single-request latency for throughput |
| **Quantize → Multi-Replica** | Quantization reduces VRAM → deploy more replicas per GPU | Throughput-first scenarios | Quality loss from quantization must be acceptable |

---

## Part 2: ZImage Case Study — The Optimization Challenge of Diffusion Transformers

### 2.1 Model Overview

- **ZImage**: Diffusion Transformer-based text-to-image model (ZImagePipeline)
- Architecture: 30 Transformer blocks (not UNet), Qwen3 text encoder, VAE decoder
- Deployment: DLIS (A100/A6000)
- Inference: 9-step denoising, each step runs through all 30 Transformer layers

### 2.2 Profiling: Where Is the Bottleneck?

| Component | Share | Notes |
|-----------|-------|-------|
| **Transformer** | **95.5%** | Absolute dominant — optimization must focus here |
| VAE | 2.9% | Not worth optimizing |
| Text Encoder | 1.6% | Not worth optimizing |

**CUDA Time Breakdown:**

| Category | Share | Implication |
|----------|-------|-------------|
| **GEMM (aten::mm)** | **40.3%** | 493 matmul calls (Q/K/V/Out + FFN) — target for quantization |
| **Command Buffer Full** | **36.8%** | CPU can't dispatch kernels fast enough → core value of torch.compile |
| SDPA (attention) | 7.7% | Already using mem_efficient kernel; limited room for FlashAttention |
| dtype cast (copy_) | 8.3% | bf16↔fp32 conversions, hard to eliminate |

**Key finding:** All 30 Transformer blocks are perfectly uniform (~19ms/step) — no single hotspot block.

### 2.3 Technique Selection Results

All techniques mapped to the 3 acceleration categories:

| Category | Technique | Result | Status | Key Detail |
|----------|-----------|--------|--------|------------|
| **① Make Each Computation Faster** | torch.compile (reduce-overhead) | **1.36x** | ✅ | CUDA Graphs eliminates Command Buffer Full (36.8%) |
| | torch.compile (default) | **1.25x** | ✅ | Operator fusion only, no CUDA Graphs |
| | TensorRT | ❌ blocked | ❌ | `complex64` RoPE not supported → conversion fails at step 1 |
| | ONNX Runtime | ❌ blocked | ❌ | `complex64` + nested inputs + Qwen3 encoder — multiple blockers |
| | FlashAttention | No room | ❌ | SDPA already auto-selects flash_fwd_kernel; only 7.7% of CUDA time |
| | INT8 W8A8 (torchao) | **2.1x slower** | ❌ | A100 lacks optimized INT8 GEMM kernel (need H100 SM90a) |
| | INT8 (bitsandbytes) | PSNR **9.6dB** | ❌ | Quality collapse — 9-step denoise accumulates quantization error |
| | torch.compile + INT8 | **155x slower** | ❌ | AffineQuantizedTensor subclass breaks Triton fusion |
| | FP8 | ❌ blocked | ❌ | A100/A6000 lack FP8 hardware (need H100/Ada SM90+) |
| **② Reduce Computation Count** | FBC (t=0.3) | **1.20x** | ✅ | Skip blocks when adjacent step features are similar |
| | MagCache | **1.21x** | ✅ | Hardcoded calibrated skip ratios |
| | Reduce steps 9→7 | **1.28x** | ✅ | Linear relationship, simplest acceleration |
| | DeepCache | ❌ blocked | ❌ | Requires UNet; ZImage is Transformer — inapplicable |
| | TeaCache | ❌ unreliable | ❌ | Initial 1.24x results were fake (mosaic bug); real quality too unstable |
| **③ Increase Parallel Throughput** | — | — | — | Single-request scenario, not applicable for ZImage |
| **① + ②** | **torch.compile + FBC** | **1.55x** | ⭐ | Must use mode='default' (CUDA Graphs conflicts with FBC) |

> **Pattern:** Category ① compute acceleration engines (TRT/ONNX/FA) all blocked by non-standard ops; quantization all failed due to hardware + quality; only torch.compile works. Category ② cache methods partially work, but tightly coupled to architecture.

#### 🔬 Deep Dive: Why INT8 Quantization Fails Across the Board on Diffusion Models

**Three paths attempted, three paths failed:**

1. **torchao W8A8**: A100 lacks SM90a INT8 GEMM optimized kernels → falls back to slow path → 2.1x slower
2. **bitsandbytes INT8**: Marginal 1.09x speedup, but each GEMM becomes 5-6 kernels (quantize→INT8 GEMM→dequantize), and **PSNR drops from 40+dB to 9.6dB** — in diffusion models, small per-step errors accumulate and amplify
3. **torch.compile + INT8**: AffineQuantizedTensor is a PyTorch subclass that completely breaks Triton operator fusion → 155x slower

**Root cause:** Diffusion models' 9-step iterative denoising **accumulates quantization error**, unlike LLM autoregressive decode where each step is independent. Combined with A100's lack of optimized INT8 kernels, quantization is currently not viable for diffusion models.

### 2.4 Final Deployment Configuration

```
torch.compile(mode='default') + FBC(threshold=0.3) = 1.55x speedup
```

- Latency: 4666ms → ~3010ms (A100)
- Quality: PSNR 30.6dB (acceptable)
- No model modifications required — pure runtime optimization
- Alternative: torch.compile(reduce-overhead) alone = 1.36x, PSNR 37.3dB (near-lossless, quality-first option)

---

## Part 3: Gemma4 Case Study — LLM Serving Architecture Optimization

### 3.1 Model Overview

- **Gemma4 26B-A4B-it**: Google MoE LLM, 26B total params / 3.8B active params
- Use case: Ad Landing Page → ad copy generation
- Deployment: DLIS (A100/A6000), vLLM serving
- Evaluation: Zero-shot format compliance 95.9% (vs Qwen3 47.9%)

### 3.2 Profiling: Where Is the Bottleneck?

Measured with `benchmark_speed.py` — Two-Step generation on 4 samples:

| Phase | Time | Share |
|-------|------|-------|
| Step 1 prefill (TTFT) | 0.02s | ~0% |
| Step 1 decode | 7.70s | 19% |
| Step 2 prefill | 1.54s | 4% |
| Step 2 decode | 30.30s | **77%** |
| **Total** | **39.6s** | |

**Key observations:**

| Observation | Evidence | Implication |
|-------------|----------|-------------|
| **Decode-bound** | Decode = 77% of total time, TTFT = 0.02s | Optimize decode, not prefill |
| **Latency ∝ output tokens** | 1024 tok / 632 tok ≈ 1.62x → 82.65s / 52.0s ≈ 1.59x | Fewer tokens = linearly faster |
| **MoE routing ≈ free** | Dense 4.5B: 12.9 tok/s ≈ MoE 3.8B: 12.1 tok/s | MoE gives 7x param at ~same speed |

**Conclusion:** Since it's decode-bound, **reducing output token count = linear latency reduction**.

### 3.3 Technique Selection Results

All techniques mapped to the 3 acceleration categories:

| Category | Technique | Result | Status | Key Detail |
|----------|-----------|--------|--------|------------|
| **① Make Each Computation Faster** | CUDA Graphs (ENFORCE_EAGER=false) | **~12x** latency | ✅ | 17s→1.4s; vLLM captures and replays kernel sequence, eliminates CPU-GPU sync overhead |
| | AWQ 4-bit (vLLM) | **1.27x** throughput | ✅ | VRAM 52GB→19GB → larger KV cache → higher batch capacity; single-GPU AWQ matches dual-GPU BF16 |
| | TP=2 (vs TP=4) | **~5x faster** | ⚪ | 3.8B active params fits 2 GPUs; TP=4 AllReduce overhead dominates. Superseded by AWQ single-GPU |
| | GPTQ 4-bit | -14% tok/s | ⚪ | Pure dequant overhead; no fused kernel like AWQ |
| | FP8 KV Cache | ❌ not supported | ❌ | A100 lacks FP8 hardware (need H100/Ada SM90+) |
| | Prefix Caching | Marginal | ⚪ | Reuse KV cache for shared prompt prefix; limited gain with diverse inputs |
| | TP=1 vs TP=2 | TP=1 sufficient | ⚪ | AWQ 4-bit fits single GPU (19GB); TP=2 adds communication overhead for no gain |
| **② Reduce Computation Count** | No-CoT | **-18%** time | ✅ | Fewer output tokens → linearly less decode time |
| | Two-Step + stop_strings | **-46%** per-record | ✅ | Output tokens 1041→311 (-70%); stop at `</Prompt>` |
| | LP truncation (2000 chars) | <5% effect | ⚪ | Decode-bound → input length barely matters |
| | E2B small model (2B) | 20% format compliance | ❌ | Fast but unusable — model capability too low |
| **③ Increase Parallel Throughput** | vLLM continuous batching | **28x** (1 GPU) | ✅ | 30.5→869.5 tok/s; dynamic batch fill/drain |
| | vLLM (2×A100, TP=2) | **35x** | ✅ | 1,075 tok/s |
| | Concurrency C=8→32 | **2x** throughput | ✅ | Latency +39% (42s→58s) — classic tradeoff |
| **① + ② + ③** | **CUDA Graphs + AWQ + vLLM + Two-Step + C=32** | **59x** | ⭐ | 67 min → 68.8s for 190 records |

> **Pattern:** Unlike ZImage, **all 3 categories contribute** — and Category ③ (serving architecture) is the biggest lever (28-59x). LLM optimization ecosystem is mature.

### 3.4 Final Deployment Configuration

```
CUDA Graphs + AWQ 4-bit + vLLM + Two-Step + C=32
```

- Per-request latency: ~0.36s (A100) / ~1.7s (A6000)
- Batch of 190 records: 68.8s (vs HF Transformers 67 min = **59x speedup**)
- Format compliance: 98.9%
- Single-GPU VRAM: ~19GB (deployable on A6000 48GB)

---

## Part 4: Comparative Summary

### Technique × Model Type Compatibility Matrix

| Optimization Technique | ZImage (Diffusion Transformer) | Gemma4 (MoE LLM) | Reason for Difference |
|----------------------|-------------------------------|-------------------|----------------------|
| **torch.compile** | ✅ 1.30x (core optimization) | ⚪ Not needed (built into vLLM) | Diffusion has severe Command Buffer Full issue |
| **TensorRT** | ❌ complex64 + scatter unsupported | ⚪ Not needed (built into vLLM) | Non-standard ops block conversion |
| **FlashAttention** | ❌ head_dim=512 > 256 limit | ✅ Auto-enabled by vLLM | Architecture params don't meet prerequisites |
| **INT8/FP8 Quantization** | ❌ Quality collapse + no optimized kernels | — Not tested | Diffusion iterative error accumulation |
| **4-bit Quantization (AWQ/GPTQ)** | — Not tested | ✅ AWQ 1.27x throughput gain | LLM decode-bound: bandwidth gain > dequant overhead |
| **FBC** | ✅ 1.20x | ❌ Not applicable | Only for diffusion multi-step denoising |
| **DeepCache** | ❌ Requires UNet architecture | ❌ Not applicable | Architecture constraint |
| **Continuous Batching** | ⚪ Single-request scenario | ✅ **28–59x** (biggest lever) | Mature LLM serving ecosystem |
| **Tensor Parallel** | ⚪ Single GPU sufficient | ✅ TP=2 optimal (TP=4 is slower) | Balance between communication overhead and model size |
| **Reduce Output Volume** | ✅ Fewer steps (linear) | ✅ No-CoT -18%, Two-Step -44% | Greatest gains for decode-bound models |

### Key Takeaways

#### 1. Model Architecture Determines the Optimization Path

- **Diffusion Transformer**: Optimization ecosystem immature (TRT/FA/ONNX/quantization all failed) — only torch.compile + cache skipping are viable
- **MoE LLM**: Optimization ecosystem mature — serving architecture (vLLM + batching + TP) is the biggest lever

#### 2. Profile First, Don't Guess

- ZImage profiling revealed Command Buffer Full at 36.8% → directly pointed to torch.compile
- Gemma4 profiling revealed decode at 77% + TTFT 0.02s → directly pointed to reduce output tokens + batching
- **Optimizing without profiling = shooting arrows in the dark**

#### 3. "Textbook Optimizations" Don't Always Work

- INT8 quantization: "Should be faster in theory" → actually 2.1x slower (no optimized kernels) or quality collapse (9.6dB)
- TP=4: "More GPUs should be faster" → actually 5x slower than TP=2 (communication overhead)
- Channels-Last: "NHWC should be faster" → actually 4% slower (conversion overhead)
- **Always measure, always look at the data**

#### 4. Combined Optimizations ≠ Simple Addition

- torch.compile(reduce-overhead) + FBC → conflict (CUDA Graphs vs dynamic hooks)
- torch.compile(default) + FBC → **1.55x** (must choose the right compile mode)
- torch.compile + INT8 → **155x slower** (subclass breaks fusion)
- **Understand each optimization's implementation mechanism before combining**

#### 5. Final Acceleration Results

| Model | Before | After | Total Speedup |
|-------|--------|-------|---------------|
| **ZImage** | 4666ms / request | ~3010ms / request | **1.55x** |
| **Gemma4** | 67 min / 190 records (HF) | 68.8s / 190 records (vLLM) | **59x** |
