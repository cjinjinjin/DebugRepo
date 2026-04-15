# GRPO / DPO Training for Qwen3-30B-A3B: Progress & Challenges
## Group Meeting — April 6, 2026

---

## Slide 1: Project Overview

**Goal**: Fine-tune Qwen3-30B-A3B (MoE) to generate high-quality, format-compliant image prompts

**Training Pipeline**:
- **Base**: Qwen3-30B-A3B (MoE, BF16 ~60GB)
- **SFT** (completed): LoRA rank-64, checkpoint-50 merged
- **GRPO** (in progress): Reinforcement learning from format+quality reward
- **DPO** (evaluated): Format-preference training as alternative path
- **Constrained Decoding** (validated): Regex-based hard constraint at inference time

**Infrastructure**: 8x A100-SXM4-80GB (Azure ML Singularity)

---

## Slide 2: Key Challenges — 18 Failed Configurations Before Success

| # | Approach | Failure | Category |
|---|----------|---------|----------|
| 1-2 | ZeRO-3 + gather_for_generation | Deadlock: rank0 allgather 30B params, NCCL hang | Multi-GPU |
| 3 | vLLM colocate | NCCL OOM, torch+NCCL version incompatibility | vLLM |
| 4-5 | vLLM server + split GPUs | NCCL invalid usage (v1 EngineCore conflict) | vLLM |
| 6-8 | ZeRO-2/3 + QLoRA 4bit | OOM / BNB 4bit can't allgather to BF16 | Quantization |
| 9 | ZeRO-3 + BF16 (top_k=-1) | transformers rejects top_k=-1 | Config bug |
| **10** | **ZeRO-3 + BF16 + comp512** | **clipped_ratio=1.0: 100% outputs truncated, reward=0.07 (no learning)** | **Truncation** |
| 15 | ZeRO-2 + QLoRA, no vLLM | NCCL ALLREDUCE 600s timeout (async generation) | Multi-GPU |
| 16 | ZeRO-2 + QLoRA | `VLLM_MODE` env leak -> colocate mode silently activated | Config bug |
| 17 | vLLM server TP=2 (v0.8.5) | **FusedMoE `_load_w2` IndexError** — MoE weight shard OOB | **vLLM bug** |
| 18 | vLLM server TP=2 (v0.10.2) | **Same bug** — only fixed in vLLM 0.19.0 (PR #37010) | **vLLM bug** |

**Two hardest problems**:
- **vLLM FusedMoE TP bug**: Blocks Plan A (10-50x faster generation). Traced across 7 vLLM versions — only 0.19.0 has the fix
- **Completion truncation**: Model needs ~900 tokens but was given 512. All outputs cut -> noisy reward -> zero learning. Step 14 gradient explosion (80x) confirmed instability

---

## Slide 3: Truncation Fix — comp512 vs comp1024 vs comp2048

| Metric | comp512 | comp1024 | comp2048 |
|--------|---------|----------|----------|
| **Reward** | 0.072 | **0.447** | **0.484** |
| **clipped_ratio** | 1.0 (all cut) | 0.16 | **0.0** |
| **mean_length** | 512 (= max) | 900 | 900 |
| **frac_reward_zero_std** | 0.06-0.13 | 0.0-0.09 | **0.0** |
| **step_time** | ~6,100s | ~10,600s | ~11,000s |
| **memory (GiB)** | ~70 | ~70 | ~73 |

- Model needs ~900 tokens; comp512 was a dead end from the start
- comp2048 eliminates all truncation with only 4% speed cost over comp1024
- SFT baseline reward = 0.211 — both comp1024 and comp2048 exceed it from step 1

---

## Slide 4: Current GRPO Training Results

### comp1024 (10/34 steps, ~59h elapsed)
- Reward: **stable plateau at ~0.45** (2x SFT baseline 0.211)
- clipped_ratio: 0.11-0.22 (some truncation remains)
- KL ~0.007, grad_norm stable, no anomalies
- Speed: ~10,600s/step (~2.9h), ETA ~8 days total

### comp2048 (6/34 steps, ~40h elapsed) — Best Config
- Reward: **0.45-0.51** (highest across all experiments)
- clipped_ratio: **0.0** (zero truncation)
- frac_reward_zero_std: **0.0** (100% effective gradients)
- Speed: ~11,000s/step (only 4% slower than comp1024!)
- Memory: 73.15 GiB (7 GiB headroom, no OOM risk)

**Key insight**: comp2048 dominates comp1024 on all metrics with negligible cost increase

---

## Slide 5: Reward Plateau — Open Question

**Observation**: Reward plateaus at ~0.45-0.51 across 6-10 steps, no clear upward trend

**Possible explanations**:
1. **KL ~0.007 = very slow policy update**: Model barely deviates from reference policy
2. **Learning rate too low**: Current 5e-6 may be insufficient for RL signal
3. **NUM_GENERATIONS=2**: Limited reward variance within groups (GRPO needs contrast)
4. **Normal for large MoE models**: Known to learn slowly with GRPO
5. **Need more steps**: 6-10 steps may be too few — need 20-30+ to see trends

**Next**: Continue training, consider LR/generation tuning if plateau persists at step 20+

---

## Slide 6: DPO Alternative — Negative Result

**Motivation**: GRPO blocked by infra issues -> try DPO format-preference training

**Data**: 1774 train pairs (95.8% format, 4.2% quality), 12 corruption strategies

**Training**: Converged in ~5 steps (loss 0.19->0.0, accuracy 100%)

**Result: DPO FAILED to improve format compliance**

| Metric | DPO checkpoint-10 | SFT Baseline |
|--------|-------------------|--------------|
| Format compliance (5 tags) | **31.6%** | ~30% |

**Why it failed**:
1. Task too easy — corruption too extreme, model trivially discriminates
2. Likelihood displacement — chosen logprobs dropped (-1114 -> -1277)
3. Train-inference gap — DPO optimizes relative probs, not generation quality

---

## Slide 7: Constrained Decoding — What It Is

**Problem**: SFT model only ~30% format compliant. DPO failed to improve it. Can we enforce format at inference time?

**Constrained Decoding** = modifying the token sampling process so the model can ONLY generate outputs matching a predefined structure.

**How it works**:
1. Define a **regex pattern** describing valid output format (e.g., `<think>...</think><Prompt1>...</Prompt1>...`)
2. Compile the regex into a **Finite State Machine (FSM)** over the model's vocabulary
3. At each decoding step, the FSM determines which tokens are **legal next tokens** given the current state
4. **Mask out all illegal tokens** (set logits to -inf) before sampling
5. Result: the model generates freely within structural constraints — **100% format compliance by construction**

```
Normal decoding:     P(token) = softmax(logits)
Constrained:         P(token) = softmax(logits * mask_FSM)   # illegal tokens get -inf
```

**Key trade-off**: Guarantees structure, but constrains the model's generation space — content quality depends on how restrictive the pattern is.

---

## Slide 8: Constrained Decoding — Our Implementation

**Tool**: outlines 0.1.x (regex -> FSM -> logit masking)

**Pattern** (simplified to avoid FSM state explosion):
```
<think>[^<]+</think>\s*
<Prompt1>[^<]+</Prompt1>\s* ... <Prompt5>[^<]+</Prompt5>
```

**Challenges overcome**:
- Naive regex `[\s\S]{10,3000}` -> FSM state explosion (O(3000 x 150K vocab)), stuck overnight
- `interegular` doesn't support lookahead `(?!...)` -> simplified to `[^<]+`
- outlines API version mismatch (0.1.x vs 0.2.x) -> used legacy API

**Result**: 
- FSM compiles in seconds (was infinite before)
- Single-sample test: **100% format compliant, good content quality**
- Batch evaluation on 190 samples in progress

---

## Slide 9: Environment Journey — 3 Full Rebuilds

| Generation | Key Stack | Trigger |
|------------|-----------|---------|
| Gen 1 | swift 4.0.3, torch 2.6.0, vllm 0.11.0 | trl import crash |
| Gen 2 | swift 4.1.0.dev0, torch 2.8.0, vllm 0.10.2 | vLLM FusedMoE bug |
| **Gen 3** | swift 4.1.0.dev0, **torch 2.10.0**, **vllm 0.19.0** | FusedMoE fix only in 0.19+ |

**Key lessons**:
- vLLM + MoE + TP>1 is bleeding-edge — most versions are broken
- Environment variable leakage (`VLLM_MODE`) caused silent config override
- `NCCL_BLOCKING_WAIT` and `ASYNC_ERROR_HANDLING` are mutually exclusive
- System CUDA toolkit vs PyTorch CUDA version mismatch blocks DeepSpeed JIT

---

## Slide 10: Summary & Next Steps

### What Works
- **GRPO comp2048**: Best config — zero truncation, reward 2x SFT baseline, stable training
- **Constrained decoding**: Hard guarantee on format compliance at inference time

### What Doesn't Work
- **DPO format-preference**: No improvement over SFT baseline (too-easy negatives)
- **comp512**: Useless — 100% truncation kills all learning signal

### Next Steps
1. **Continue comp2048 GRPO** to 34 steps, watch for reward uptrend at step 15-20+
2. **Validate vLLM 0.19.0** on new machine -> unlock Plan A (10-50x faster generation)
3. **Batch evaluate constrained decoding** on full eval set (190 samples)
4. If reward plateau persists: tune LR, increase NUM_GENERATIONS, or try longer training

### Timeline
| Task | ETA |
|------|-----|
| comp2048 completes 34 steps | ~April 12 |
| comp1024 completes 34 steps | ~April 12 |
| Plan A (vLLM server mode) validation | Pending new machine access |
