# Native Ad Image Prompt 生成：周进展汇报
## 2026/04/06 — 2026/04/14

---

# Slide 1: Agenda

1. **背景回顾**：任务目标与技术路线
2. **Qwen3 训练探索**：DPO / GRPO 实验与结论
3. **Gemma4 Zero-shot 突破**：模型选择与 UHRS 人工评测
4. **推理速度优化**：Transformers → vLLM → Two-Step → AWQ 量化
5. **关键结论与下一步**

---

# Slide 2: 背景回顾

**任务**：为 Native Ad 自动生成 5 个高质量 image generation prompt

**输入**：Landing Page URL + 内容字段（标题、描述、正文等）

**输出**：5 个 80-150 word 的英文图片生成 prompt（`<Prompt1>`~`<Prompt5>` 格式）

**要求**：
- 格式合规（5 个标签完整、互不重复、≤150 words）
- 内容质量（原生感、无 stock 感、无 logo/文字、解剖正确）
- 多样性（5 个 prompt 视角不同）
- 推理速度（满足线上吞吐需求）

**当前 baseline**：GPT5 zero-shot（UHRS 人工评测 Good Rate 65.4%）

---

# Slide 3: 技术路线概览

```
Qwen3-30B-A3B                          Gemma4-26B-A4B-it
┌──────────────────┐                  ┌────────────────────────┐
│  SFT → DPO → GRPO  │                  │  Zero-shot（无微调）     │
│  format compliance  │                  │  format compliance      │
│  ~30% → 47.9% → ?  │                  │  98.5% ~ 100%           │
│  训练困难：likelihood │                  │  UHRS Good: 75.4%       │
│  displacement,      │                  │  (+10pp vs GPT5)        │
│  reward hacking     │                  └────────────────────────┘
└──────────────────┘                           ↓
        ↓                              ┌────────────────────────┐
   训练方向暂停                         │  推理速度优化            │
   模型能力瓶颈明确                     │  Transformers → vLLM    │
                                       │  → Two-Step → AWQ       │
                                       │  单卡 70s / 190条       │
                                       └────────────────────────┘
```

---

# Slide 4: Qwen3 DPO 实验（4/6 - 4/8）

## DPO 核心实验结果

| Checkpoint | Fully Compliant | Avg Word Count | 说明 |
|------------|----------------|----------------|------|
| SFT baseline | ~30% | — | 微调起点 |
| **DPO ckpt-1** | **47.9%** | 68.2 | 最佳（仅 1 步训练）|
| DPO ckpt-2 | 46.8% | 62.2 | 开始退化 |
| DPO ckpt-3 | 39.5% | 55.5 | 持续下降 |
| DPO ckpt-4 | 25.3% | 47.5 | 低于 SFT baseline |
| DPO ckpt-5 | 41.6% | 53.6 | 略有回弹 |

## 关键发现：Likelihood Displacement

- ckpt-1 是唯一全面优于 SFT 的 checkpoint
- 步数越多 → chosen 概率被压低 → 生成质量退化
- **根因**：负样本太简单（极端 format corruption），模型轻松区分但没学到细粒度约束
- **结论**：DPO 对此任务提升有限（47.9% fully compliant），极易过拟合

---

# Slide 5: Qwen3 GRPO 实验（4/8 - 4/11）

## Reward v1 → v2 演进

**v1 问题：Reward Hacking**
- GRPO ckpt-6 fully compliant 仅 22.6%（低于 SFT ~30%）
- 平均 prompt 仅 18.7 词 — 模型学会写空壳标签拿格式分
- 5 个 3 词空壳 tag 在 v1 可得 +1.45 分

**v2 修复：**

| 场景 | v1 分数 | v2 分数 |
|------|---------|---------|
| 5 个空壳 tag | **+1.45** | **-0.40** |
| 5 个 80 词质量 prompt | +1.34 | **+1.61** |

- 新增 `min_length` 惩罚：<20 词 → -0.2/prompt
- 新增 `descriptiveness` 奖励：内容词比例 → +0.3 max
- 新增 `CoT fields` 奖励：6 个推理字段 → +0.4 max
- 格式分权重减半

**v2 Step 1 结果**：reward=0.731，min_length=737 tokens，空壳策略被成功堵死

## Base Model 评估（4/11）

| 指标 | Base Model | SFT |
|------|-----------|-----|
| Fully compliant | 14.8% | ~30% |
| Avg word count | 10.2 | — |

- Base model 重复循环（repetition collapse），think block 耗尽全部 token
- SFT 核心价值：抑制重复 + 学习格式

## 结论

- Qwen3 训练路线受限：DPO likelihood displacement + GRPO 训练速度慢（单步 ~6h）
- 最佳结果 47.9% fully compliant，距离生产要求（>95%）差距大

---

# Slide 6: Gemma4 Zero-shot 突破（4/10 - 4/13）

## 模型信息

| 维度 | Qwen3-30B-A3B | Gemma4-26B-A4B-it |
|------|--------------|-------------------|
| 架构 | MoE, 3B 激活/30B 总 | MoE, 3.8B 激活/25.2B 总 |
| 上下文 | 128K | **256K** |
| Zero-shot compliance | ~14.8% (base) / ~47.9% (最佳微调) | **98.5% ~ 100%** |

## 核心优势

- **格式合规率开箱即用 98.5%+**：无需任何微调
- **多模型对比**：

| 模型 | Format Compliance | 备注 |
|------|-------------------|------|
| Gemma4 26B-A4B-it | **98.5-100%** | 生产可用 |
| Gemma4 E4B-it (4.5B) | 100% | 质量偏弱（forbidden words 2.1/5）|
| Gemma4 E2B-it (2B) | 20% | 不可用 |

- **格式遵循能力是硬门槛**：E2B 内容质量指标反而更好，但格式不合规导致完全不可用

---

# Slide 7: UHRS 人工评测 — Gemma4 vs GPT5

## Image Level Good Rate（Random 200 LPs, 3 judges/image）

| 指标 | Gemma4 Zero-shot | GPT5 | 差异 |
|------|-----------------|------|------|
| **Good** | **75.4%** | 65.4% | **+10.0pp** |
| Fair | 7.3% | 6.5% | +0.8pp |
| **Bad** | **17.3%** | 28.1% | **-10.8pp** |

## LP Level 质量分布

| 阈值 | Gemma4 | GPT5 | 差异 |
|------|--------|------|------|
| >= 3/5 Good | 87.9% | 75.5% | **+12.4pp** |
| >= 4/5 Good | 63.6% | 43.0% | **+20.6pp** |
| >= 5/5 Good | 28.8% | 13.5% | **+15.3pp** |

## 关键结论

- **Gemma4 zero-shot 无需微调即超越 GPT5 baseline +10pp**
- LP 级别质量分布明显右移：高质量 LP（≥4/5 Good）占比 63.6% vs 43.0%
- Bad Rate 从 28.1% 降至 17.3%

---

# Slide 8: 推理速度优化路线

## 优化方向演进

```
Transformers 单卡          vLLM Serving          vLLM Offline
  52s/sample        →      42s/sample      →     0.36s/sample
  12.1 tok/s                15.2 tok/s            1299 tok/s
```

## vLLM Serving Benchmark 关键发现（4/11）

| 配置 | Throughput | Avg Latency | 效果 |
|------|-----------|-------------|------|
| TP=4, CoT, C=8 | 0.04 req/s | ~200s | 基线（通信开销大）|
| TP=2, No-CoT, C=8 | 0.16 req/s | 42.0s | **4x 提升** |
| TP=2, No-CoT, C=32 | **0.32 req/s** | 58.2s | **8x 提升** |

- **TP=4→TP=2 是最大优化**：通信开销减半
- 截断 LP 对平均延迟无影响（瓶颈在 decode 而非 prefill）

---

# Slide 9: Two-Step 生成方案 + stop_strings 优化

## Two-Step 设计

```
Step 1: Scene Planning          Step 2: Batch Expand
生成 5 个场景概念               将每个场景扩展为完整 prompt
(5-10 words each)               (30-50 words each, batch=5)
~10s                             ~10s (with stop_strings)
```

5 个强制不同视角：close-up / lifestyle / environmental / outcome / mood

## stop_strings 优化效果（Transformers, 4/13）

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| **Avg total time** | 39.6s | **21.2s** | **-46%** |
| Step 2 output tokens | 1041 | 311 | -70% |
| Step 2 decode | 30.3s | 10.2s | -66% |

- 模型在 `</Prompt>` 后继续生成无用内容 → 添加 `stop_strings` 后立即停止
- 时间分布均衡：Step 1 decode 44% + Step 2 decode 48%

---

# Slide 10: vLLM Offline 全量结果对比（190 条, 4/14）

## 推理方案对比

| 方案 | 总耗时 | 每条 | Throughput | GPU |
|------|--------|------|------------|-----|
| Transformers Two-Step | ~67min | 21.2s | 30.5 tok/s | 1×A100 |
| **vLLM Single-Prompt** | **122.7s** | 0.6s | 1,299 tok/s | 2×A100 |
| **vLLM Two-Step** | **68.8s** | 0.36s | 1,075 tok/s | 2×A100 |
| **vLLM Two-Step 单卡** | **84.9s** | 0.45s | 869 tok/s | 1×A100 |
| **vLLM Two-Step AWQ** | **70.0s** | 0.37s | 1,105 tok/s | 1×A100 |

## AWQ 4-bit 量化（4/14）

| 指标 | BF16 单卡 | AWQ 单卡 | BF16 双卡 |
|------|-----------|----------|-----------|
| 总耗时 | 84.9s | **70.0s** | 68.8s |
| Throughput | 869 tok/s | **1,105 tok/s** | 1,075 tok/s |
| 模型权重 | ~52GB | **~19GB** | ~26GB/卡 |
| Compliance | 99.5% | 98.9% | 100% |

- **AWQ 单卡 = BF16 双卡性能**：模型权重省 63%，更多显存给 KV cache
- 质量几乎无损（98.9% compliance, forbidden words 0.1/5）

---

# Slide 11: Benchmark 脚本升级（4/13）

## TTFT + LP 长度 Sweep 结果

| LP Chars | Avg Input Tok | Avg TTFT | Avg Total | Avg tok/s |
|----------|--------------|----------|-----------|-----------|
| 400 | 533 | 0.02s | 54.8s | 11.8 |
| 1000 | 636 | 0.02s | 53.9s | 11.7 |
| 2000 | 779 | 0.02s | 54.1s | 11.7 |
| 5000 | 1032 | 0.02s | 51.7s | 11.9 |
| unlimited | 1095 | 0.02s | 52.2s | 12.2 |

**关键发现**：LP 长度对速度几乎无影响（瓶颈完全在 decode）

---

# Slide 12: 质量评估对比总结

## Single-Prompt vs Two-Step（vLLM, 190 条）

| 指标 | Single-Prompt | Two-Step |
|------|---------------|----------|
| Format compliance | 100% | 99.5% |
| Avg word count | 126.5 | **42.4** |
| Forbidden words | 4.0/5 | **0.1/5** |
| Quality hints | 4.2/5 | 1.8/5 |
| 速度 | 122.7s | **68.8s** |

**Two-Step 优势**：
- Forbidden words 大幅改善（4.0 → 0.1/5）
- 速度快 44%（输出 token 减少 54%）
- 多样性好（5 个强制不同视角）

**Single-Prompt 优势**：
- Prompt 更长更详细（126.5 vs 42.4 words）
- Quality hints 更多

---

# Slide 13: 关键结论

## 1. 模型选择
- **Gemma4 26B-A4B-it zero-shot 是最佳方案**
- 无需微调，format compliance 98.5%+，UHRS Good Rate 75.4%（+10pp vs GPT5）
- Qwen3 训练路线受限（DPO/GRPO 均有根本性问题），最佳仅 47.9%

## 2. 推理优化
- **vLLM offline + Two-Step + AWQ 4-bit 是推荐配置**
- 单卡 A100 70s 跑完 190 条（0.37s/sample），throughput 1,105 tok/s
- 相比 Transformers 单卡加速 **~57x**

## 3. 生成策略
- Two-Step（场景规划+扩展）vs Single-Prompt 各有优势，按业务需求选择
- Two-Step 适合：短 prompt（30-50w）、低 forbidden words、强多样性
- Single-Prompt 适合：长 prompt（80-150w）、高细节度

---

# Slide 14: 下一步计划

1. **生产部署评估**
   - vLLM serving 模式 + AWQ 量化上线测试
   - 并发压测（C=32/64），验证线上吞吐达标

2. **Prompt 质量优化**
   - Single-Prompt 的 forbidden words 偏高（4.0/5），优化 system prompt
   - 探索 Two-Step + 更长 prompt 目标（50-80 words）

3. **UHRS 扩展评测**
   - Two-Step vs Single-Prompt 的 UHRS 人工对比
   - AWQ 量化 vs BF16 的质量差异评估

4. **Qwen3 GRPO v2 监控**
   - 继续跟踪 reward v2 训练（当前 step 1 reward=0.731）
   - 若训练收敛，评估是否能接近 Gemma4 水平

---

# Appendix: 实验时间线

| 日期 | 工作内容 | 关键结果 |
|------|---------|---------|
| 4/6 | DPO ckpt-10 评估 | 31.6% ≈ SFT baseline，DPO 失败 |
| 4/6 | DPO 早期 checkpoint 策略 | ckpt-1 最佳 47.9% |
| 4/8 | DPO v12 全 5 ckpts 对比 | 确认 likelihood displacement |
| 4/8 | GRPO ckpt-6 190 条评估 | 22.6%，reward hacking 确认 |
| 4/8 | Reward v2 设计实现 | 空壳 +1.45→-0.40 |
| 4/8 | vLLM 0.19.0 成功 | FusedMoE bug 已修复 |
| 4/10 | GRPO v2 step 1 | reward=0.731，空壳策略被堵 |
| 4/10 | Gemma4 环境搭建 | zero-shot 测试开始 |
| 4/11 | vLLM Serving benchmarks | TP=2 最优, C=32 0.32 req/s |
| 4/11 | Qwen3 Base model 评估 | 14.8% baseline |
| 4/13 | TTFT + LP sweep | LP 长度对速度无影响 |
| 4/13 | Random200 No-CoT+Trunc | 98.5% compliance |
| 4/13 | Two-Step + stop_strings | 39.6s → 21.2s (-46%) |
| 4/13 | E4B/E2B 对比 | E4B 100% 可用, E2B 20% 不可用 |
| 4/13 | **UHRS 人工评测** | **Gemma4 75.4% vs GPT5 65.4%** |
| 4/14 | Single-Prompt vLLM | 122.7s/190条, 1299 tok/s |
| 4/14 | Two-Step vLLM | 68.8s/190条, 1075 tok/s |
| 4/14 | AWQ 4-bit 量化 | 70.0s 单卡, 1105 tok/s |
