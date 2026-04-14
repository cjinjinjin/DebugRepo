# Native Ad Image Prompt Generation: Weekly Update
## 2026/04/06 — 2026/04/14

---

# Slide 1: Background & Goal

**Task**: Auto-generate 5 high-quality image prompts per Native Ad

**Input**: Landing Page URL + content fields → **Output**: 5 `<Prompt1>`~`<Prompt5>`

**Metrics**: Format compliance / UHRS human Good Rate / Inference speed

**Current baseline**: GPT5 zero-shot (UHRS Good Rate 65.4%)

---

# Slide 2: Qwen3 Training Exploration (4/6 - 4/11) — Dead End

## DPO: Likelihood Displacement

| Checkpoint | Fully Compliant |
|------------|----------------|
| SFT baseline | ~30% |
| **DPO ckpt-1 (best)** | **47.9%** |
| DPO ckpt-4 | 25.3% (below baseline) |

- Only 1-step DPO helped; more steps → chosen prob drops → quality degrades

## GRPO: Reward Hacking → v2 Fix

- v1: empty shell tags scored +1.45 → model learned 18.7-word shells
- v2: empty shells score -0.40, step 1 min_length=737 tokens ✅
- But ~6h/step, 34 steps ≈ 8 days

**Conclusion**: Qwen3 best = 47.9% compliance, far from production (>95%)

---

# Slide 3: Gemma4 Zero-shot Breakthrough

## No fine-tuning needed, works out of the box

| Model | Format Compliance | Usability |
|-------|-------------------|-----------|
| Qwen3 SFT baseline | ~30% | Unusable |
| Qwen3 DPO best | 47.9% | Unusable |
| **Gemma4 26B-A4B-it** | **98.5-100%** | **Production-ready** |
| Gemma4 E4B-it (4.5B) | 100% | Weak quality |
| Gemma4 E2B-it (2B) | 20% | Unusable |

---

# Slide 4: UHRS Human Evaluation — Gemma4 vs GPT5

**Random 200 LPs, 3 judges/image, ~1000 images**

| Metric | Gemma4 | GPT5 | Delta |
|--------|--------|------|-------|
| **Good Rate** | **75.4%** | 65.4% | **+10pp** |
| Bad Rate | 17.3% | 28.1% | -10.8pp |

| LP-level threshold | Gemma4 | GPT5 | Delta |
|--------------------|--------|------|-------|
| >= 3/5 Good | 87.9% | 75.5% | +12.4pp |
| >= 4/5 Good | 63.6% | 43.0% | **+20.6pp** |

**Gemma4 zero-shot surpasses GPT5 baseline without any fine-tuning**

---

# Slide 5: Inference Speed Optimization

| Config | 190 samples | Per sample | Speedup |
|--------|-------------|------------|---------|
| Transformers 1×A100 | ~67min | 52s | 1x |
| vLLM Two-Step 2×A100 | 68.8s | 0.36s | **~144x** |
| vLLM Two-Step 1×A100 | 84.9s | 0.45s | ~116x |
| **vLLM Two-Step AWQ 1×A100** | **70.0s** | **0.37s** | **~140x** |

**Key optimizations**:
- vLLM continuous batching: ~100x throughput gain
- Two-Step + stop_strings: halved output tokens, 39.6s → 21.2s
- AWQ 4-bit: model weights 52GB → 19GB, 1×GPU ≈ 2×GPU perf

**Recommended**: AWQ 4-bit + 1×A100 → 70s/190 samples, 1105 tok/s

---

# Slide 6: Conclusions & Next Steps

## Key Conclusions

1. **Gemma4 26B-A4B-it zero-shot is the best approach**
   - No fine-tuning, compliance 98.5%+, UHRS +10pp vs GPT5
2. **AWQ 4-bit + vLLM is the recommended deployment config**
   - 1×A100 0.37s/sample, quality intact
3. **Qwen3 training paused**
   - DPO/GRPO both have fundamental issues, best = 47.9%

## Next Steps

- vLLM serving deployment + concurrency stress test
- Single-Prompt vs Two-Step UHRS human comparison
- System prompt optimization (reduce forbidden words)
