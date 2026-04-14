# Native Ad Image Prompt Generation: Weekly Update
## 2026/04/06 — 2026/04/14

---

# Slide 1: Background & Goal

**Task**: Auto-generate 5 high-quality image prompts per Native Ad

**Input**: Landing Page URL + content fields → **Output**: 5 `<Prompt1>`~`<Prompt5>`

**Metrics**: Format compliance / UHRS human Good Rate / Inference speed

**Current baseline**: GPT5 zero-shot (UHRS Good Rate 65.4%)

---

# Slide 2: Qwen3 Training — All Stages (4/6 - 4/11) — Dead End

## Format Compliance Across 4 Stages

| Stage | Fully Compliant | Avg Words | Core Issue |
|-------|----------------|-----------|------------|
| Zero-shot (base) | 14.8% | 10.2 | Think block repetition collapse |
| SFT ckpt-50 | ~30% | — | Learned format but low quality |
| **DPO ckpt-1 (best)** | **47.9%** | 68.2 | Likelihood displacement |
| GRPO v1 ckpt-6 | 22.6% | 18.7 | Reward hacking (empty shells) |
| GRPO v2 step-1 | — | 737 tok | Too slow (~6h/step) |

## Key Findings Per Stage

- **Zero-shot**: Repetition loops in think blocks, only 39.8% samples had think block, format unusable
- **SFT**: Suppressed repetition, taught output format, but compliance only ~30%
- **DPO**: Only 1-step helped (47.9%); more steps → chosen prob drops → degrades to 25.3%
- **GRPO v1**: Empty shell tags scored +1.45 → model learned 18.7-word shells (reward hacking)
- **GRPO v2**: Fixed empty shells, but ~6h/step, 34 steps ≈ 8 days — impractical

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

**Image-level Good Rate (max-vote):**

| Metric | Gemma4 | GPT5 | Delta |
|--------|--------|------|-------|
| **Good Rate** | **75.4%** | 65.4% | **+10pp** |
| Bad Rate | 17.3% | 28.1% | -10.8pp |

**LP-level cumulative distribution:**

| LP-level threshold | Gemma4 | GPT5 | Delta |
|--------------------|--------|------|-------|
| >= 3/5 Good | 87.9% | 75.5% | +12.4pp |
| >= 4/5 Good | 63.6% | 43.0% | **+20.6pp** |

**Gemma4 zero-shot surpasses GPT5 baseline without any fine-tuning**

---

# Slide 5: Two-Step vs Single-Prompt Inference

## Approach Comparison

| Dimension | Single-Prompt | Two-Step |
|-----------|--------------|---------|
| Pipeline | Generate 5 prompts at once | Step 1: plan 5 scenes → Step 2: expand each |
| Prompt length target | 80-150 words | 30-50 words |
| Diversity source | Temperature sampling | 5 scenes with forced different angles |
| Tag misalignment risk | Yes (at high temp) | None (generated individually) |

## vLLM Inference Performance (190 samples, 2×A100)

| Metric | Single-Prompt | Two-Step | Delta |
|--------|--------------|---------|-------|
| Total time | 122.7s | **68.8s** | **-44%** |
| Total output tokens | 159,474 | 74,003 | -54% |
| Forbidden words | 4.0/5 | **0.1/5** | **Major improvement** |
| Format compliance | 100% | 100% | — |

**Two-Step advantages**: Shorter prompts naturally reduce forbidden words, scene planning ensures diversity, 54% fewer tokens → faster inference

---

# Slide 6: Inference Speed Optimization

| Config | 190 samples | Per sample | Speedup |
|--------|-------------|------------|---------|
| Transformers 1×A100 | ~67min | 52s | 1x |
| vLLM Single-Prompt 2×A100 | 122.7s | 0.6s | ~87x |
| vLLM Two-Step 2×A100 | 68.8s | 0.36s | **~144x** |
| vLLM Two-Step 1×A100 | 84.9s | 0.45s | ~116x |
| **vLLM Two-Step AWQ 1×A100** | **70.0s** | **0.37s** | **~140x** |

**Key optimizations**:
- vLLM continuous batching: ~100x throughput gain
- Two-Step + stop_strings: halved output tokens, 39.6s → 21.2s
- AWQ 4-bit: model weights 52GB → 19GB, 1×GPU ≈ 2×GPU perf

**Recommended**: AWQ 4-bit + 1×A100 → 70s/190 samples, 1105 tok/s

---

# Slide 7: Conclusions & Next Steps

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
