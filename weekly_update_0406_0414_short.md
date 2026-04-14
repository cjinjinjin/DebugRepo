# Native Ad Image Prompt 生成：周进展
## 2026/04/06 — 2026/04/14

---

# Slide 1: 背景与目标

**任务**：为 Native Ad 自动生成 5 个高质量 image prompt

**输入**：Landing Page URL + 内容字段 → **输出**：5 个 `<Prompt1>`~`<Prompt5>`

**评估维度**：格式合规率 / UHRS 人工 Good Rate / 推理速度

**当前 baseline**：GPT5 zero-shot（UHRS Good Rate 65.4%）

---

# Slide 2: Qwen3 训练全阶段总结（4/6 - 4/11）— 碰壁

## 四阶段 Format Compliance 对比

| 阶段 | Fully Compliant | Avg Words | 核心问题 |
|------|----------------|-----------|----------|
| Zero-shot（base） | 14.8% | 10.2 | think block 重复坍缩，token 浪费 |
| SFT ckpt-50 | ~30% | — | 学会格式但质量不足 |
| **DPO ckpt-1（最佳）** | **47.9%** | 68.2 | Likelihood displacement |
| GRPO v1 ckpt-6 | 22.6% | 18.7 | Reward hacking（空壳 tag）|
| GRPO v2 step-1 | — | 737 tok | 训练太慢（~6h/step）|

## 各阶段关键发现

- **Zero-shot**：think block 中产生重复循环，仅 39.8% 样本有 think block，格式几乎不可用
- **SFT**：抑制重复、教会输出格式，但 compliance 仅 ~30%
- **DPO**：仅 1 步有效（47.9%），更多步数 → chosen 概率被压低 → 退化至 25.3%
- **GRPO v1**：空壳 tag 得 +1.45 分 → 模型学会写 18.7 词空壳（reward hacking）
- **GRPO v2**：修复空壳问题，但单步 ~6h，34 步需 ~8 天，不实际

**结论**：Qwen3 最佳 47.9% compliance，距生产要求（>95%）差距大

---

# Slide 3: Gemma4 Zero-shot 突破

## 无需微调，开箱即用

| 模型 | Format Compliance | 可用性 |
|------|-------------------|--------|
| Qwen3 SFT baseline | ~30% | 不可用 |
| Qwen3 DPO 最佳 | 47.9% | 不可用 |
| **Gemma4 26B-A4B-it** | **98.5-100%** | **生产可用** |
| Gemma4 E4B-it (4.5B) | 100% | 质量偏弱 |
| Gemma4 E2B-it (2B) | 20% | 不可用 |

---

# Slide 4: UHRS 人工评测 — Gemma4 vs GPT5

**Random 200 LPs, 3 judges/image, ~1000 images**

| 指标 | Gemma4 | GPT5 | 差异 |
|------|--------|------|------|
| **Good Rate** | **75.4%** | 65.4% | **+10pp** |
| Bad Rate | 17.3% | 28.1% | -10.8pp |

| LP 级别阈值 | Gemma4 | GPT5 | 差异 |
|-------------|--------|------|------|
| >= 3/5 Good | 87.9% | 75.5% | +12.4pp |
| >= 4/5 Good | 63.6% | 43.0% | **+20.6pp** |

**Gemma4 zero-shot 无需微调即超越 GPT5 baseline**

---

# Slide 5: 推理速度优化

| 方案 | 190 条总耗时 | 每条 | 加速比 |
|------|-------------|------|--------|
| Transformers 单卡 | ~67min | 52s | 1x |
| vLLM Two-Step 双卡 | 68.8s | 0.36s | **~144x** |
| vLLM Two-Step 单卡 | 84.9s | 0.45s | ~116x |
| **vLLM Two-Step AWQ 单卡** | **70.0s** | **0.37s** | **~140x** |

**关键优化**：
- vLLM continuous batching：批量推理 ~100x 加速
- Two-Step + stop_strings：总 token 减半，39.6s → 21.2s
- AWQ 4-bit：模型权重 52GB → 19GB，单卡 ≈ 双卡性能

**推荐配置**：AWQ 4-bit + 单卡 A100 → 70s/190 条，1105 tok/s

---

# Slide 6: 结论与下一步

## 关键结论

1. **Gemma4 26B-A4B-it zero-shot 是最优方案**
   - 无需微调，compliance 98.5%+，UHRS +10pp vs GPT5
2. **AWQ 4-bit + vLLM 是推荐部署配置**
   - 单卡 A100 0.37s/sample，质量无损
3. **Qwen3 训练路线暂停**
   - DPO/GRPO 均有根本性问题，最佳仅 47.9%

## 下一步

- vLLM serving 上线 + 并发压测
- Single-Prompt vs Two-Step UHRS 人工对比
- System prompt 优化（降低 forbidden words）
