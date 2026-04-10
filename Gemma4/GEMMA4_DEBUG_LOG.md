# Gemma 4 训练调试日志

## 背景

从 Qwen3-30B-A3B (MoE) 迁移到 Gemma 4，原因：
- Qwen3 MoE 训练遇到大量基础设施问题（ZeRO-3 死锁、vLLM FusedMoE TP bug、NCCL 超时等）
- GRPO 存在 reward hacking（空壳短 prompt 拿高分），DPO 存在 likelihood displacement
- Qwen3 最好结果：DPO v12 ckpt-1 = 47.9% fully compliant（190 条 eval）
- 训练迭代极慢（~3h/step），调试周期长

## 模型选择

### 候选模型对比（Gemma 4，2026-04-09 发布）

| 维度 | Qwen3-30B-A3B | Gemma 4 26B-A4B-it | Gemma 4 31B-it |
|------|--------------|---------------------|----------------|
| 架构 | MoE, 激活 3B | **MoE, 激活 3.8B** | Dense, 30.7B |
| 总参数 | 30B | 25.2B | 30.7B |
| Expert 配置 | — | 8/128 active + 1 shared | — |
| 显存 BF16 | ~60GB | **~50GB** | ~61GB |
| 上下文 | 32K | **256K** | **256K** |
| MMLU Pro | — | 82.6% | 85.2% |
| MMMU Pro (Vision) | — | 73.8% | 76.9% |
| MATH-Vision | — | 82.4% | 85.6% |
| AIME 2026 | — | 88.3% | 89.2% |
| 推理速度 | ~3B 激活 | **~4B 激活** | 30.7B 全激活 |
| Thinking 模式 | 有 | **有（原生）** | **有（原生）** |

### 选择：Gemma 4 26B-A4B-it

理由：
1. 与 Qwen3-30B-A3B 同为 MoE 架构，推理速度相当
2. Benchmark 全面碾压 Gemma 3 27B（MMMU Pro 73.8% vs 49.7%，MATH-Vision 82.4% vs 46.0%）
3. 原生支持 thinking 模式（`<|think|>`），与现有 CoT 流程匹配
4. 256K 上下文，处理长 Landing Page 无压力
5. Google 新架构，预期训练工具链兼容性更好

---

## 任务定义

**输入**：Landing Page 内容字段（URL, Title, Heading, Content 等）
**输出**：5 个高质量 Native Ad 图像生成 prompt（`<Prompt1>`~`<Prompt5>`），每个 ≤150 words
**可选**：Chain-of-Thought 推理（`<think>` block，含 6 字段分析）

### 评估指标
- Format compliance（5 tags 全部存在 + 唯一 + 闭合）
- Think block 完整性（6 字段：ProductType, SpecificProduct, Category, VisualAnchors, LifestyleVibe, CoreValueSignals）
- 平均 prompt 字数（目标 40-150 words）
- 关键词覆盖率
- 禁用词检查

### Baseline 参照
| 模型 | Fully Compliant | Avg Word Count |
|------|----------------|----------------|
| Qwen3 SFT baseline | ~30% | — |
| Qwen3 DPO v12 ckpt-1（最佳） | 47.9% | 68.2 |
| Qwen3 GRPO comp2048 ckpt-6 | 22.6% | 18.7 |

---

## 阶段一：Zero-shot 测试（2026-04-10）

### 环境配置

**硬件**：8× A100-SXM4-80GB，CUDA Driver 12.8，nvcc 11.8

**环境搭建记录**：
1. `bash Gemma4/setup_env.sh` 创建 `gemma4` conda 环境
2. `huggingface-cli download` 下载模型到本地 `./gemma-4-26B-A4B-it`
3. 尝试 `cp -r` 到 vc_data 共享存储 → 跨 mount 拷贝极慢，放弃，改用本地路径
4. `rm -rf` 清理 vc_data 上的残留也很慢（mount 路径 I/O 瓶颈）

**踩坑记录**：

| # | 问题 | 原因 | 解决方案 |
|---|------|------|----------|
| 1 | `AutoProcessor.from_pretrained()` 报 `Unrecognized processing class` | `setup_env.sh` 安装的 `transformers==4.57.6` 不支持 Gemma 4 的 `Gemma4Processor` class | `pip install -U transformers` 升级到最新版 |
| 2 | 升级 transformers 后报 `PyTorch and torchvision compiled with different CUDA major versions` (torch cu130 vs torchvision cu126) | `pip install -U torch` 默认装了 cu130 版本，与 torchvision cu126 不匹配 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --force-reinstall` |
| 3 | vc_data 共享存储 `mv` 报 `preserving permissions: Operation not permitted` | mount 路径不支持 permission 保留 | 改用 `cp -r`，但速度太慢；最终决定直接用本地路径 |

**最终依赖版本**（待确认 torch 重装后）：
```
torch: 需 cu126 版本
transformers: 最新版（需支持 Gemma4Processor）
accelerate: >=1.0.0
peft: >=0.13.0
```

### 配置
- 模型路径：`./gemma-4-26B-A4B-it`（本地）
- 方式：直接用现有 system prompt + eval 数据，不做任何训练
- 推理脚本：`inference_gemma4.py`（使用 `AutoProcessor`，对齐官方 HF card）
- 评估脚本：复用 `QwenFinetune/evaluate.py`
- 评估数据：`QwenFinetune/data/sft_eval_cot.jsonl`

### 运行命令

```bash
# Zero-shot 推理（batch 模式）
python Gemma4/inference_gemma4.py \
    --model_id google/gemma-4-26B-A4B-it \
    --input_file QwenFinetune/data/sft_eval_cot.jsonl \
    --output_file Gemma4/results/gemma4_zeroshot_eval.jsonl \
    --max_new_tokens 2048 \
    --batch_size 1

# 评估
python QwenFinetune/evaluate.py \
    --generated_file Gemma4/results/gemma4_zeroshot_eval.jsonl \
    --report_file Gemma4/results/gemma4_zeroshot_report.json
```

### Zero-shot 结果

⬜ 待运行

---

## 训练调研（2026-04-10）

### 关键发现

#### 1. QLoRA 不适用于 MoE 模型
BitsAndBytes 4-bit 量化与 MoE 路由不兼容（Unsloth 文档明确指出 "MoE QLoRA not recommended"）。
**必须用 bf16 LoRA，不能用 QLoRA。**

#### 2. 推荐 LoRA 超参

| 参数 | 推荐值 | 来源 |
|------|--------|------|
| LoRA rank | 8（纯文本）/ 32（多模态） | Unsloth |
| LoRA alpha | = rank | Unsloth |
| dropout | 0 | Unsloth |
| 学习率 | 1e-4 ~ 2e-4（LoRA）/ 2e-5（full SFT） | TRL + Unsloth |
| optimizer | adamw_8bit | Unsloth |
| target_modules | all-linear（`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`） | TRL |
| gradient accumulation | 4 | Unsloth |
| warmup | 5 steps | Unsloth |
| weight decay | 0.001（LoRA）/ 0.01（full） | Unsloth |
| max grad norm | 1.0（文本）/ 0.3（多模态） | Unsloth |
| 精度 | **bf16**（MoE 不能用 4-bit） | All |
| gradient checkpointing | 必须开 | All |

#### 3. DeepSpeed 配置
- **用 ZeRO Stage 2**，不能用 Stage 3（与 expert parallelism 不兼容）
- 与 Qwen3 MoE 的 ZeRO-3 死锁问题一致
- 建议配置：
```json
{
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "reduce_scatter": true,
    "allgather_bucket_size": 50000000,
    "reduce_bucket_size": 50000000,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "cpu_offload": true
  }
}
```

#### 4. MoE Router Loss
如需 load-balancing auxiliary loss 参与训练，设置：
```python
model.config.output_router_logits = True
```
（TRL SFTTrainer 文档建议对 MoE 模型开启）

#### 5. ms-swift 当前状态（v4.0.4+）
- Gemma 4 LoRA SFT 已支持（PR #8508，2026-04-03）
- ⚠️ **Thinking 模式不支持**（issue #9065，open）
- ⚠️ 全量微调保存模型会 crash（issue #9056，open）
- LoRA SFT 文本/视觉可用

#### 6. 其他注意事项
- **冻结 Vision/Audio Tower**：Google/HF 推荐 SFT 时冻结（我们是纯文本任务，不影响）
- **显存需求**：26B-A4B bf16 LoRA 训练约需 >40GB VRAM，8×A100-80GB 足够
- **Chat template**：Gemma 4 用 `<|turn>user\n` / `<|turn>model\n` 分隔符
- **Thinking 格式**：system prompt 含 `<|think|>`，输出 `<|channel>thought\n...<channel|>`
- **`use_cache` bug**（Unsloth 报告）：gradient checkpointing 强制 `use_cache=False` 时，KV-shared layers 可能丢失共享状态，Unsloth 已修复
- **response-only training**：用 `assistant_only_loss=True`（TRL），只在 assistant 回复上计算 loss

### 参考链接
- [HF Blog: Gemma 4](https://huggingface.co/blog/gemma4)
- [Unsloth Gemma 4 训练指南](https://unsloth.ai/docs/models/gemma-4/train)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [ms-swift Gemma 4 PR #8508](https://github.com/modelscope/ms-swift/pull/8508)
- [ms-swift Gemma 4 Issues](https://github.com/modelscope/ms-swift/issues?q=gemma+4)
- [DeepSpeed MoE Tutorial](https://www.deepspeed.ai/tutorials/mixture-of-experts/)

---

## 阶段二：SFT 微调（如 zero-shot 效果不理想）

### 计划
- 使用 `QwenFinetune/data/sft_train_cot.jsonl`（833 条）
- **bf16 LoRA**（不用 QLoRA），rank 8，alpha 8，target all-linear
- DeepSpeed ZeRO Stage 2
- 学习率 1e-4，gradient accumulation 4
- 框架：TRL SFTTrainer（ms-swift thinking 模式暂不支持）
- 预期训练速度远快于 Qwen3（dense 部分更小，MoE 路由清晰）

⬜ 待实施

---

## 阶段三：Preference 优化（DPO/GRPO）

### 计划
- 基于 SFT checkpoint
- 使用 reward v2（已修复 reward hacking 问题）
- 或直接 DPO（save_steps=1，用极早期 checkpoint 避免 likelihood displacement）

⬜ 待实施

---

## 待办
1. ⬜ 运行 zero-shot 推理（debug 空输出问题已修复 parse_response key 名）
2. ⬜ 评估 zero-shot 结果，对比 Qwen3 baseline
3. ⬜ 如果 < 50% compliant，开始 SFT
4. ⬜ SFT 后评估，决定是否需要 DPO/GRPO
