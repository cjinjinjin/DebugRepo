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
- 评估数据：`QwenFinetune/data/dpo_combined_eval_cot.jsonl`（190 条，由 `combine_dpo_data.py` 合并 format + quality eval）

### 运行命令

```bash
# 生成 190 条合并 eval 数据（如不存在）
cd QwenFinetune && python combine_dpo_data.py && cd ..

# Zero-shot 推理（8 GPU 数据并行，推荐 --no_think 模式）
python Gemma4/inference_gemma4_multi_gpu.py \
    --model_id ./gemma-4-26B-A4B-it \
    --input_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
    --output_file Gemma4/results/gemma4_zeroshot_eval.jsonl \
    --num_gpus 8 \
    --max_new_tokens 2048 \
    --no_think

# 评估
python QwenFinetune/evaluate.py \
    --generated_file Gemma4/results/gemma4_zeroshot_eval.jsonl \
    --report_file Gemma4/results/gemma4_zeroshot_report.json
```

### Zero-shot 结果

#### 单个 Case 测试（2026-04-10）

**测试输入**：Premium Wireless Headphones 虚构 Landing Page

```
URL: https://www.example.com/premium-wireless-headphones
Title: Premium Wireless Headphones - Crystal Clear Sound
Heading: Experience Music Like Never Before
Content: Our premium wireless headphones deliver studio-quality sound with active noise cancellation...
```

**推理统计**：
- 生成 token 数：1951（偏多，正式内容约需 ~800，thinking 冗余严重）
- `max_new_tokens=2048`

**格式合规性** — ✅ 通过：
| 检查项 | 结果 |
|--------|------|
| 5 个 `<PromptN>` 标签全部存在且闭合 | ✅ |
| 5 个 prompt 内容唯一 | ✅ |
| 所有 prompt ≤ 150 words | ✅（各 ~90-110 words） |

**CoT Think Block** — ✅ 通过（6 字段全部存在）：
- ProductType: Consumer Electronics / Audio Gear
- SpecificProduct: Premium Wireless Over-Ear Headphones
- Category: Audio & Lifestyle
- VisualAnchors: ✅
- LifestyleVibe: ✅
- CoreValueSignals: ✅（但未严格遵循枚举约束）

**内容质量**：
- 场景多样性好：通勤者(ANC)、远程办公(舒适)、音乐沉浸(音质)、户外(无线自由)、静物(工艺)
- 摄影语言专业：含具体镜头参数（35mm, f/2.8, Sony A7R IV）、灯光、景深描述
- Native 感强：所有场景都是生活方式导向，无硬推销
- 安全约束嵌入完整：包含 "no logos", "no text", "no watermark", "correct anatomy", "natural hands"

**发现的问题**：
1. **Thinking 冗余** — 输出了两层 `<think>` block：第一层是极长的链式推理（含完整 5 条 prompt 草稿 + 自我纠正），第二层才是结构化 6 字段摘要。Token 浪费约 60%
2. **CoreValueSignals 枚举约束被忽略** — 应从 `[professional, premium, affordable, efficient, reliable, simple]` 中选择，但模型自由发挥为 "ANC, battery, comfort, studio-quality sound"
3. **Prompt 字数偏长** — 平均 ~95-105 words，高于 Qwen3 DPO 的 68.2 但仍在 150 限制内

**初步判断**：
- 单个 case 格式完全合规，内容质量高
- 如果 190 条 eval 能保持类似表现，zero-shot 大概率超越 Qwen3 DPO v12 的 47.9% baseline
- 需完成全量 190 条评估才能定论

⬜ 全量 196 条评估待运行

#### 全量推理 Bug：0% Format Compliance（2026-04-10）

**现象**：196 条全量推理，format compliance = 0/196 (0%)，与单 case 测试 100% compliance 形成巨大反差。

**根因分析**：

`inference_gemma4.py` 的 `extract_lp_fields_from_messages()` 函数使用 `FIELD_LABELS` 的短标签构造正则来提取 user message 中的字段：
```python
FIELD_LABELS = {
    "FinalDestinationURLUrl": "URL",           # → 正则搜索 [URL]
    "PrimaryContentNoTitleNoHeading": "Primary Content",  # → 正则搜索 [Primary Content]
}
```

但 SFT/DPO eval 数据中的 user message 使用的是**全名 bracket label**：
```
[Landing Page URL]
https://www.example.com/...

[Primary Content]
Product description...
```

- `\[URL\]` **匹配不到** `[Landing Page URL]` → URL 字段丢失
- `\[Primary Content\]` 恰好能匹配（标签名一致） → 但仅靠 content 不够
- 最终发给模型的 user message 几乎为空 → 模型无法生成有效输出

**单 case 测试为何通过**：单 case 模式直接用 `--url` 和 `--content` CLI 参数构造 `lp_fields` dict，绕过了 `extract_lp_fields_from_messages()`，所以不受影响。

**修复方案**：

新增 `extract_user_content_from_messages()` 函数，直接提取 user message 的 content 透传给模型，而非拆字段再重建。SFT/DPO 数据中的 user message 本身就是格式良好的完整 prompt：

```python
def extract_user_content_from_messages(messages):
    for msg in messages:
        if msg.get("role") == "user":
            return msg["content"]
    return ""
```

批量推理逻辑改为：有 `messages` 的数据 → 直接透传 user content → 不再走 field extraction + rebuild 路径。

**修改文件**：`Gemma4/inference_gemma4.py`
- 新增 `extract_user_content_from_messages()` 函数
- `generate()` 方法新增 `user_content` 参数，支持直接传入用户消息内容
- `generate_batch()` 支持 `input_type="user_content"` 模式
- `main()` 中 batch 逻辑优先使用 user content 透传

⬜ 需重新运行全量 196 条推理验证修复效果

#### 全量 196 条评估结果（2026-04-10）

修复 field extraction bug 后，重新跑 196 条全量推理（8 GPU 并行，no-think 模式）。

**第一轮（system prompt: ≤150 words）**：

| 指标 | Gemma 4 Zero-shot | Qwen3 DPO v12 (Baseline) |
|------|-------------------|--------------------------|
| **Fully compliant** | **93.4%** | 47.9% |
| Avg word count/prompt | 69.1 | 68.2 |
| Prompts within 150 words | 4.8 / 5 | — |
| `<think>` block present | 100.0% | — |
| All 6 CoT fields present | 100.0% | — |

问题：avg 69.1 words 远低于 SFT 训练数据的 111.3 words（80-120 区间占 76%）。

**第二轮（system prompt: 80–150 words）**：

| 指标 | Gemma 4 v2 | Gemma 4 v1 | Qwen3 DPO v12 |
|------|-----------|-----------|----------------|
| **Fully compliant** | **95.9%** | 93.4% | 47.9% |
| All 5 tags present | 95.9% | 93.4% | — |
| All 5 prompts unique | 95.9% | 93.4% | — |
| Prompts within 150 words | 4.9 / 5 | 4.8 / 5 | — |
| **Avg word count/prompt** | **89.9** | 69.1 | 68.2 |
| `<think>` block present | 100.0% | 100.0% | — |
| All 6 CoT fields present | 100.0% | 100.0% | — |
| Quality hints per sample | 2.9 / 5 | 2.5 / 5 | — |
| Forbidden words per sample | 1.6 / 5 | 1.0 / 5 | — |
| LP keyword coverage | 0.0% | 0.0% | — |

**关键发现**：
1. **Format compliance 95.9%** — 加下限后反而提升 2.5pp，**Qwen3 DPO 最佳的两倍**
2. **Avg word count 89.9** — 从 69→90，更接近训练数据分布（111），但仍有提升空间
3. **CoT 100% 完整** — 所有样本都有完整的 `<think>` block 和 6 字段分析
4. **Quality hints 2.9/5** — 略有提升（更长的 prompt 能容纳更多质量约束描述）
5. **Forbidden words 1.6/5** — 略有上升，prompt 更长导致更多触发（可通过 SFT 改善）
6. **Keyword coverage 0.0%** — 评估脚本的 LP keyword 提取逻辑问题，需检查

**结论**：
- Gemma 4 26B-A4B-it zero-shot **95.9% fully compliant**，远超 Qwen3 DPO 的 47.9%
- 无需任何微调，直接可用于生产评估
- 下一步：inference Random200 进行额外验证

#### No-think 模式测试（2026-04-10）

同一 case，加 `--no_think` 关闭 Gemma 原生 thinking 模式。

**对比**：
| 维度 | Think 模式 | No-think 模式 |
|------|-----------|--------------|
| 生成 token 数 | 1951 | **603**（节省 69%） |
| 格式合规 | ✅ | ✅ |
| 5 tags 完整 | ✅ | ✅ |
| Think block | ✅（冗余两层） | ✅（模型自发输出单层） |
| Prompt 平均字数 | ~95-110 | ~75-90 |

**关键发现**：
1. **Token 效率提升 ~3x** — 原生 thinking 产生冗余链式推理（草稿 + 自我纠正），no-think 模式完全消除
2. **模型仍自发输出 `<think>` block** — system prompt 指令驱动，无需原生 thinking
3. **质量未降** — 场景多样性、摄影语言、Native 感均保持高质量
4. **字数更紧凑** — 更接近 Qwen3 DPO baseline 的 68.2

**结论**：全量评估使用 `--no_think` 模式。

#### 多 GPU 数据并行方案

新建 `Gemma4/inference_gemma4_multi_gpu.py`：
- Gemma 4 26B ~50GB bf16，单张 A100-80GB 可装下
- 8 GPU 各加载独立模型副本，数据 round-robin 分片
- 用 subprocess 启 8 个 `inference_gemma4.py` 进程
- 自动合并结果，保持原始顺序
- 预期 190 条 + no_think ≈ **4-8 分钟**完成

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
1. ✅ 运行 zero-shot 全量 196 条推理（8 GPU 并行 + no_think 模式）
2. ✅ 评估 zero-shot 结果，对比 Qwen3 baseline → **93.4% vs 47.9%，大幅超越**
3. ~~⬜ 如果 < 50% compliant，开始 SFT~~ → 不需要，zero-shot 已达 93.4%
4. ⬜ 考虑 SFT 微调进一步提升（forbidden words、150-word 限制）
5. ✅ inference Random200（`RawData/UHRS2K_SD_Random200_0324.tsv`）→ **99.0% compliant, avg 94.1 words**

---

## Random200 Inference 准备（2026-04-10）

### 脚本
- 新建 `Gemma4/eval_gemma4_random200.sh`：TSV → JSONL → 8-GPU 并行推理 → evaluate
- 数据预处理复用 `QwenFinetune/prepare_infer_input.py`（将 TSV 转为 JSONL）
- 推理复用 `inference_gemma4_multi_gpu.py`（8 GPU 数据并行）

### 数据流
1. `QwenFinetune/RawData/UHRS2K_SD_Random200_0324.tsv`（200 条原始数据）
2. → `Gemma4/data/random200_infer_input.jsonl`（JSONL 中间格式，本地）
3. → `/vc_data/.../Gemma4_results/gemma4_random200_eval.jsonl`（推理结果，vc_data）
4. → `/vc_data/.../Gemma4_results/gemma4_random200_report.json`（评估报告，vc_data）

### 注意事项
- `prepare_infer_input.py` 使用全部 10 个 LP 字段（vs eval 数据只有 2 个生产字段）
- `inference_gemma4.py` 的 `extract_user_content_from_messages()` 直通 user message，不受字段数量影响
- 使用 no-think 模式（与 zeroshot eval 一致）

### 用法
```bash
bash Gemma4/eval_gemma4_random200.sh
```

### Random200 结果（2026-04-10）

| 指标 | DPO 196条 (v2) | Random200 |
|------|---------------|-----------|
| Fully Compliant | 95.9% | **99.0%** |
| All 5 tags present | 95.9% | 99.0% |
| All 5 prompts unique | 95.9% | 99.0% |
| Avg word count | 89.9 | **94.1** |
| Prompts within 150 words | 5.0/5 | 5.0/5 |
| `<think>` block present | — | 100.0% |
| All 6 CoT fields | — | 100.0% |
| Quality hints | — | 3.0/5 |
| Forbidden words | — | 1.8/5 |
| LP keyword coverage | — | 0.0% |

**分析**：
- Random200 使用 10 个 LP 字段输入（vs DPO eval 仅 2 个生产字段），信息更丰富
- Format compliance 从 95.9% 提升到 99.0%，说明更丰富的输入有助于模型生成
- 平均字数 94.1，符合 80-150 words 要求
- CoT 完整性 100%
- Forbidden words 1.8/5 需关注（后续 SFT 可改善）

---

## AWQ 4-bit 量化验证（2026-04-10）

### 背景
Online serving 需要低延迟，AWQ 4-bit 量化可减少显存（50GB → ~13GB）并提升推理速度。
使用 `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`（HuggingFace, 209k downloads, 28 likes）。

### 对比目标
| 指标 | BF16 (baseline) | AWQ 4-bit (待验证) |
|------|-----------------|-------------------|
| Format Compliance | 95.9% | ? |
| Avg Word Count | 89.9 | ? |
| 显存/卡 | ~52GB | ~13GB |

### 脚本
- 下载：`Gemma4/download_model_awq.sh`
- 评估：`Gemma4/eval_gemma4_awq_zeroshot.sh`（196 条 DPO eval, 8 GPU, no-think）
- 输出：`/vc_data/.../Gemma4_results/gemma4_awq_zeroshot_eval.jsonl`
- 报告：`/vc_data/.../Gemma4_results/gemma4_awq_zeroshot_report.json`

### 技术要点
- AWQ 预量化模型使用 `compressed-tensors` 格式（`quant_method: compressed-tensors`），不是标准 AWQ
- `AutoModelForCausalLM.from_pretrained()` 自动检测并加载，需要安装 `compressed-tensors` 包
- Processor 和 tokenizer 文件与原始 BF16 模型完全一致，可直接从 AWQ 模型路径加载
- 新增 `--processor_id` 参数支持从不同路径加载 processor（备用方案）

### 环境兼容性问题（2026-04-10）

AWQ 模型加载遇到严重的包版本兼容性问题：

| # | 现象 | 原因 | 解决方案 |
|---|------|------|----------|
| 1 | `pip install compressed-tensors` 后 `AutoProcessor` 报 `Unrecognized processing class` | `compressed-tensors` 依赖把 `transformers` 降级到 `4.57.6`，该版本没有 `Gemma4Processor` class | — |
| 2 | `AutoTokenizer` 也报 `AttributeError: 'list' object has no attribute 'keys'` | `transformers 4.57.6` 的 tokenizer 对 Gemma 4 的 `extra_special_tokens` 格式不兼容 | — |
| 3 | `pip install -U transformers` 升级到 `5.5.3` 后 processor OK，但模型加载报 `AttributeError: 'NoneType' object has no attribute 'get'` | `transformers 5.5.3` 的 `auto.py` 第 270 行 `config.quantization_config` 返回 `None`，与 `compressed-tensors` 量化格式反序列化不兼容 | — |
| 4 | 用 `compressed-tensors 0.14.0.1`（vllm 要求的版本）模型能加载但大量 UNEXPECTED/MISSING 权重 | 旧版 `compressed-tensors` 与 `transformers 5.5.3` 的权重序列化格式不匹配 | — |
| 5 | `compressed-tensors 0.15.0.1`（最新版）+ `transformers 5.5.3`，同样有 UNEXPECTED/MISSING 权重 | `transformers 5.x` 对 `compressed-tensors` 格式支持不完善 | ⬜ 调查中 |

**核心矛盾**：
- `Gemma4Processor` 需要 `transformers >= 5.x`
- `compressed-tensors` 量化模型加载需要 `transformers` 与 `compressed-tensors` 版本精确匹配
- 当前没有找到一个同时满足两个条件的版本组合

**当前环境**：
```
transformers: 5.5.3
compressed-tensors: 0.15.0.1（--no-deps 安装，避免降级 transformers）
```

**已尝试的所有方案和结果**：

| # | 方案 | 结果 |
|---|------|------|
| 1 | `transformers 4.57.6` + `compressed-tensors 0.15.0.1` | 模型权重加载正常（无警告），但 `AutoModelForCausalLM` 不认识 `gemma4` 架构（`KeyError: 'gemma4'`） |
| 2 | `transformers 5.5.3` + `compressed-tensors 0.15.0.1` | 认识 `gemma4` 架构，但 `config.quantization_config` 返回 `None`（`auto.py:270` bug） |
| 3 | 方案 2 + 手动注入 `quantization_config` | 注入成功，但 `compressed-tensors` 解压 `ignore` 列表中的层时 `group_size=0` 触发 pydantic 校验错误 |
| 4 | `transformers 5.5.3` + `compressed-tensors 0.14.1a20260326`（模型量化时用的版本） | 同方案 3，同样的 `group_size=0` 错误 |
| 5 | `transformers 5.5.3` + `compressed-tensors 0.14.0.1` | 模型能加载但大量 UNEXPECTED/MISSING 权重 |

**根本原因分析**：
- `compressed-tensors` 在 `transformers 5.x` 下解压权重时，会对 `ignore` 列表中的层（不该量化的层）也执行解压，这些层没有 `group_size`（默认为 0），触发 `QuantizationArgs` 的 pydantic 校验
- `transformers 4.57.6` 走了不同的加载路径，不会触发这个问题，但该版本没有 `gemma4` 模型架构注册
- 这是 `compressed-tensors` 与 `transformers 5.x` 交互的 bug

### vLLM 方案调研（2026-04-10）

**GitHub Issues 发现**：
- [vllm#39133](https://github.com/vllm-project/vllm/issues/39133)：有人成功用 `cyankiwi/gemma-4-31B-it-AWQ-4bit`（同系列）在 vLLM 上 serve，用 2×RTX 3090 TP=2
- [vllm#39204](https://github.com/vllm-project/vllm/issues/39204)：`vllm 0.19.0` 要求 `transformers<5`，但 `gemma4` 需要 `>=5`，存在同样的版本冲突
- [vllm#39392](https://github.com/vllm-project/vllm/issues/39392)：Gemma4 tool-call-parser 在并发请求下会产生 `<pad>` token
- HuggingFace 模型卡上用户用 `vllm/vllm-openai:gemma4-cu130` Docker 镜像成功 serve

**vLLM 可行方案**：
1. 使用专门的 Docker 镜像 `vllm/vllm-openai:gemma4-cu130`（已有人验证可用）
2. vLLM serve 配置参考（来自 HuggingFace 讨论）：
   ```yaml
   model: cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit
   container_image: vllm/vllm-openai:gemma4-cu130
   serve_args:
     trust-remote-code: true
     dtype: auto
     tensor-parallel-size: 1  # AWQ 4-bit ~13GB，单卡即可
     gpu-memory-utilization: 0.9
     max-model-len: 120000
     reasoning-parser: gemma4
   ```
3. Eval 脚本通过 OpenAI-compatible API 调用 vLLM 服务
4. vLLM 原生支持 `compressed-tensors` AWQ 格式，不需要绕版本兼容性问题

**vLLM 已知问题**：
- tool calling 在并发下可能不稳定（[vllm#39392](https://github.com/vllm-project/vllm/issues/39392)）
- 需要专门的 Docker 镜像（`gemma4-cu130`），普通 `pip install vllm` 不够

**下一步计划**：
- 线上环境有 Docker 后，用 `vllm/vllm-openai:gemma4-cu130` 启动 serving
- 写基于 OpenAI API 的 eval 脚本调用 vLLM 服务
- 对比 AWQ 4-bit vs BF16 的 compliance 和 word count

### 用法
```bash
# 1. 下载 AWQ 模型
HF_TOKEN=hf_xxx bash Gemma4/download_model_awq.sh

# 2. 跑 eval
bash Gemma4/eval_gemma4_awq_zeroshot.sh
```

### 结果
⬜ AWQ 方案已放弃（见上方根本原因分析），改用 GPTQ

---

## GPTQ 4-bit 量化验证（2026-04-10）

### 背景
AWQ `compressed-tensors` 格式与 `transformers 5.x` 存在不可解决的兼容性问题（见上），改用 GPTQ 格式。
使用 `raydelossantos/gemma-4-26B-A4B-it-GPTQ-Int4`（HuggingFace）。

### 对比目标
| 指标 | BF16 (baseline) | GPTQ 4-bit (待验证) |
|------|-----------------|---------------------|
| Format Compliance | 95.9% | ? |
| Avg Word Count | 89.9 | ? |
| 显存/卡 | ~52GB | ~13GB |

### 专用环境：`gemma4-quant`

AWQ/GPTQ 量化推理的依赖与原始 `gemma4` 环境冲突（`transformers` 版本、`gptqmodel` vs `auto-gptq`），
**必须用独立 conda 环境**，避免污染原始推理/训练环境。

```bash
conda create -n gemma4-quant python=3.10 -y
conda activate gemma4-quant
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install "transformers>=5.5" accelerate
pip install gptqmodel optimum==1.24.0
```

**最终依赖版本**：
```
Python:          3.10
torch:           2.11.0+cu126
transformers:    5.5.3
gptqmodel:       6.0.3
optimum:         1.24.0
accelerate:      最新
```

### 踩坑记录

| # | 问题 | 原因 | 解决方案 |
|---|------|------|----------|
| 1 | `huggingface-hub>=0.34.0,<1.0 is required for gptqmodel` | `pip install gptqmodel` 默认拉了 `huggingface-hub 1.10.1` | `pip install "huggingface-hub>=0.34.0,<1.0"` |
| 2 | `AutoTokenizer` 报 `'list' object has no attribute 'keys'` on `extra_special_tokens` | Gemma 4 的 `tokenizer_config.json` 中 `extra_special_tokens` 是 list 格式，`transformers 5.x` 期望 dict | 在 `inference_gemma4.py` 中加 fallback：复制 tokenizer 到临时目录，把 list 转为 dict `{}` |
| 3 | `KeyError: 'gemma4'` （transformers 4.57.6） | `transformers 4.57.6` 没有注册 `gemma4` 模型架构 | 必须用 `transformers >= 5.x` |
| 4 | `NameError: 'QuantizeConfig' not defined` （optimum 2.1.0） | `optimum 2.x` 对 GPTQ 的 API 不兼容 | `pip install optimum==1.24.0` |
| 5 | `ImportError: cannot import name 'no_init_weights'` （auto-gptq 0.7.1） | `auto-gptq` 与 `transformers 5.x` 不兼容（`no_init_weights` 在 5.x 中被移除） | 卸载 `auto-gptq`，改用 `gptqmodel` |
| 6 | `ValueError: Block pattern could not be match` （optimum） | `optimum` 不识别 Gemma 4 MoE 架构的 block 结构 | 不走 `AutoModelForCausalLM.from_pretrained()`，直接用 `GPTQModel.load()` |
| 7 | `ImportError: requires gptqmodel` | 缺少 `gptqmodel` 包 | `pip install gptqmodel` |

### 关键技术发现

1. **GPTQ 模型加载必须用 `GPTQModel.load()`**，不能用 `AutoModelForCausalLM.from_pretrained()`
   - `transformers` + `optimum` 走的 `auto.py` 路径不认识 Gemma 4 MoE 的 block 结构
   - `gptqmodel` 库有专门的 `Gemma4ForConditionalGenerationGPTQ` 实现

2. **`auto-gptq` 已废弃，用 `gptqmodel` 替代**
   - `auto-gptq` 最新版（0.7.1）与 `transformers 5.x` 不兼容
   - `gptqmodel` 是 `auto-gptq` 的继任者，API 类似但维护更积极
   - `GPTQModel.load()` 对应原来的 `AutoGPTQForCausalLM.from_quantized()`

3. **`optimum` 版本必须是 `1.24.0`**
   - `optimum 2.x` 的 `QuantizeConfig` API 变更导致 NameError
   - `1.24.0` 与 `gptqmodel 6.0.3` 配合正常

4. **模型加载验证成功**：
   ```python
   from gptqmodel import GPTQModel
   model = GPTQModel.load('./gemma-4-26B-A4B-it-GPTQ-Int4', device_map='auto')
   # → <class 'gptqmodel.models.definitions.gemma4.Gemma4ForConditionalGenerationGPTQ'>
   ```

### 代码改动

在现有 inference pipeline 中新增 `--use_gptq` 参数：

- **`Gemma4/inference_gemma4.py`**：
  - `Gemma4PromptGenerator.__init__()` 新增 `use_gptq: bool` 参数
  - GPTQ 分支：`from gptqmodel import GPTQModel; self.model = GPTQModel.load(model_id, device_map=device)`
  - 非 GPTQ 分支：保持原有 `AutoModelForCausalLM.from_pretrained()` 路径
  - argparse 新增 `--use_gptq` flag

- **`Gemma4/inference_gemma4_multi_gpu.py`**：
  - argparse 新增 `--use_gptq`，透传给子进程

- **`Gemma4/eval_gemma4_gptq_zeroshot.sh`**：
  - 调用时加 `--use_gptq` flag
  - `--processor_id` 指向 BF16 原始模型的 processor（GPTQ 模型目录可能缺少 processor 文件）

### 脚本
- 下载：`Gemma4/download_model_gptq.sh`
- 评估：`Gemma4/eval_gemma4_gptq_zeroshot.sh`（196 条 DPO eval, 8 GPU, no-think, `--use_gptq`）
- 输出：`/vc_data/.../Gemma4_results/gemma4_gptq_zeroshot_eval.jsonl`
- 报告：`/vc_data/.../Gemma4_results/gemma4_gptq_zeroshot_report.json`

### 注意事项
- multi-GPU launcher 给每个子进程设 `CUDA_VISIBLE_DEVICES=N`（单卡），`GPTQModel.load(device_map='auto')` 会自动映射到该可见卡
- GPTQ 4-bit ~13GB/卡，A100-80GB 绑定没有问题
- 如遇问题可先用 `--num_gpus 1` 单卡测试

### 用法
```bash
conda activate gemma4-quant
bash Gemma4/eval_gemma4_gptq_zeroshot.sh
```

### 结果（2026-04-11）

| 指标 | BF16 (baseline) | GPTQ 4-bit | 差异 |
|------|-----------------|------------|------|
| **Fully Compliant** | **95.9%** | **92.9%** | **-3.0pp** |
| All 5 tags present | 95.9% | 92.9% | -3.0pp |
| All 5 prompts unique | 95.9% | 92.9% | -3.0pp |
| Avg Word Count | 89.9 | 90.9 | +1.0 |
| Prompts ≤150 words | 4.9/5 | 4.8/5 | -0.1 |
| `<think>` block present | 100% | 100% | 0 |
| All 6 CoT fields | 100% | 100% | 0 |
| Quality hints | 2.9/5 | 2.8/5 | -0.1 |
| Forbidden words | 1.6/5 | 1.7/5 | +0.1 |
| LP keyword coverage | 0.0% | 0.0% | 0 |

**结论**：
- GPTQ 4-bit 质量损失很小（-3pp），92.9% 仍远超 Qwen3 DPO 的 47.9%
- 字数、CoT、质量约束基本持平
- **可接受用于线上 serving** — 显存从 ~52GB 降到 ~13GB，质量损失在噪点范围内

### 速度对比（2026-04-11）

单卡（A100-80GB）、20 条样本、no-think 模式、batch_size=1：

| 指标 | BF16 | GPTQ 4-bit | 差异 |
|------|------|------------|------|
| **Avg tok/s** | **11.9** | **10.2** | **-14%** |
| Median tok/s | 12.1 | 10.3 | -15% |
| Avg time/sample | 63.1s | 73.2s | +16% |
| Avg output tokens | 753 | 750 | ~0 |
| 显存占用 | ~52GB | ~13GB | **-75%** |

**分析**：
- GPTQ 4-bit 单卡推理比 BF16 慢 ~15%，反量化计算开销 > 减少显存带宽的收益
- MoE 模型本身只激活 ~4B 参数，BF16 生成速度已经不慢（12 tok/s）
- GPTQ 的优势不在单条推理速度，而在**显存节省 75%**：
  - 同一张 A100-80GB 可跑 4-6 个 GPTQ 副本（vs BF16 只能 1 个）
  - 可部署在 RTX 3090/4090（24GB）等消费级卡上
  - vLLM continuous batching + GPTQ = 吞吐量远高于 BF16 单条推理

#### BF16 No-CoT 速度测试（2026-04-11）

单卡 A100-80GB、20 条样本、no-think + no-CoT（system prompt 不要求 `<think>` block）、batch_size=1：

| 指标 | BF16 CoT | BF16 No-CoT | 差异 |
|------|----------|------------|------|
| **Avg tok/s** | 11.9 | **12.1** | +2% |
| Median tok/s | 12.1 | 12.2 | +1% |
| Min tok/s | — | 10.7 | — |
| Max tok/s | — | 12.5 | — |
| Avg time/sample | 63.1s | **52.0s** | **-18%** |
| Median time/sample | — | 50.8s | — |
| Avg output tokens | 753 | **632** | **-16%** |

**分析**：
- 去掉 CoT `<think>` block 后，平均输出 token 从 753 → 632（减少 16%），推理时间从 63s → 52s（缩短 18%）
- tok/s 基本持平（12.1 vs 11.9），说明生成速度不受 CoT 影响，时间节省完全来自更少的 output tokens
- input token 变化范围大（307 ~ 3137），但对生成速度影响很小（MoE 激活参数少，prefill 快）
- No-CoT 模式更适合纯生产场景（不需要 CoT 推理过程），每条可节省 ~11 秒

**Benchmark 脚本**：`Gemma4/benchmark_speed.py`

#### BF16 No-CoT + 截断 2000 + 固定 1024 output tokens（2026-04-11）

单卡 A100-80GB、20 条样本、no-think + no-CoT、`--max_lp_chars 2000`、`--max_new_tokens 1024`、batch_size=1：

```bash
CUDA_VISIBLE_DEVICES=0 python Gemma4/benchmark_speed.py \
    --model_id /vc_data/.../gemma-4-26B-A4B-it \
    --input_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
    --num_samples 20 \
    --no_cot \
    --max_lp_chars 2000 \
    --max_new_tokens 1024
```

| 指标 | 值 |
|------|-----|
| Avg output tokens | 1024（固定） |
| Avg time/sample | 82.65s |
| Median time/sample | 82.58s |
| **Avg tok/s** | **12.4** |
| Median tok/s | 12.4 |
| Min tok/s | 12.2 |
| Max tok/s | 12.5 |
| Input token 范围 | 305 ~ 1264 |

**分析**：
- 固定 1024 output tokens，每条延迟非常稳定（82.0-84.3s），方差极小
- tok/s 稳定在 12.2-12.5，**与输入长度完全无关**（input 305 vs 1264 tokens，tok/s 差异 <2%）
- 进一步验证：**瓶颈完全在 decode，prefill 对延迟几乎无贡献**
- 截断到 2000 chars 后 input token 范围 305-1264，均为短输入，prefill 可忽略
- 对比自然输出（avg 632 tokens, 52.0s）：1024 tokens / 632 tokens ≈ 1.62x，82.65s / 52.0s ≈ 1.59x，**延迟与输出长度成线性关系**

### 经验教训

1. **先搜后试**：遇到第三方包兼容性问题时，应先搜索 GitHub Issues 了解已知问题和可行方案，再动手尝试。AWQ 问题本可通过搜索 `compressed-tensors` + `transformers 5.x` 的 issues 更早发现死路。

2. **独立环境隔离**：量化推理的依赖栈（`gptqmodel`/`optimum`/`compressed-tensors`）与训练环境（`trl`/`peft`/`deepspeed`）有大量冲突。**任何实验性依赖都应在独立 conda 环境中安装**，避免破坏已有工作环境。

3. **AWQ `compressed-tensors` 格式是死路**：`transformers 4.x` 能加载权重但不认识 `gemma4` 架构；`transformers 5.x` 认识架构但 `compressed-tensors` 解压 `ignore` 层时 `group_size=0` 报错。两个版本都走不通。这是上游 bug。

4. **GPTQ 是可行的后备方案**：`gptqmodel` 库有专门的 Gemma 4 MoE 支持（`Gemma4ForConditionalGenerationGPTQ`），绕过了 `transformers` + `optimum` 的架构识别问题。关键是用 `GPTQModel.load()` 直接加载，不走 `AutoModel` 路径。

5. **vLLM 是线上最优解**：对于生产部署，`vllm/vllm-openai:gemma4-cu130` Docker 镜像内部处理了所有版本兼容性问题，是最稳定的方案。本地用 GPTQ + `gptqmodel` 做离线验证。

---

## LPContext 长度分析与截断优化

### 背景

LPContext（Landing Page Content）是模型输入中最长的部分。过长的 LPContext 增加推理时间但增益有限——模型只需要核心产品信息即可生成高质量 prompt。

### 数据集 LPContext 长度分布

统计对象：user message 中所有内容的总字符数。

| 数据集 | 样本数 | 平均 | 中位数 | P90 | P95 | 最大 |
|--------|--------|------|--------|------|------|--------|
| sft_eval_cot | 87 | 4,689 | 4,055 | 8,618 | 10,988 | 17,187 |
| sft_train_cot | 833 | 5,186 | 3,779 | — | — | 343,978 |
| grpo_train | 1,100 | 3,867 | 2,890 | — | — | 35,687 |

**按字符数分桶（sft_eval_cot 87 条为例）：**

| 截断阈值 | 覆盖比例 | 说明 |
|----------|---------|------|
| ≤ 500 | 1.1% | 几乎所有样本都超过 |
| ≤ 1,000 | 8.0% | — |
| ≤ 2,000 | 16.1% | — |
| ≤ 3,000 | 32.2% | 约 1/3 不需要截断 |
| ≤ 5,000 | 63.2% | — |
| ≤ 10,000 | 92.0% | 极少数超长尾 |

### 截断方案

新增 `--max_lp_chars` 参数（默认 0 = 不截断，推荐 2000）：

```bash
python Gemma4/inference_gemma4.py --max_lp_chars 2000 ...
```

**实现逻辑：**
- `build_user_message()` 模式：截断 `PrimaryContentNoTitleNoHeading` 字段
- `user_content` 直通模式：自动识别 `[Page Content]`、`[Primary Content]`、`- Primary Content:` 等格式并截断
- 截断在最后一个完整单词处断开，追加 ` ...`

**预期效果（截断到 2000 字符）：**
- 约 84% 样本的内容会被截断（sft_eval_cot 中仅 16.1% ≤ 2000）
- 输入 token 数大幅减少 → 推理更快
- 核心产品信息通常在前 1000-2000 字符内，对 prompt 质量影响极小

### 待实验

- [ ] 对比 `--max_lp_chars 2000` vs 不截断的 format compliance 和 prompt 质量
- [ ] 测试截断后的推理速度提升幅度

---

## vLLM Serving 部署与 Benchmark（2026-04-11）

### 背景

单条推理延迟 ~60s（BF16 CoT）/ ~52s（BF16 No-CoT），无法满足线上 8 req/s 吞吐量目标。
同事使用 Qwen-VL-30B-A3B + vLLM 实现了 2-3s/request（短输出场景），说明 vLLM continuous batching 是关键。

### 环境搭建

**硬件**：8× A100-SXM4-80GB（新机器 node-0）

**Conda 环境**：
```bash
# 重要：切换 conda 环境前必须先重启 shell
conda init bash && exec bash
conda create -n gemma4-vllm python=3.10 -y
conda activate gemma4-vllm
pip install vllm
pip install "transformers>=5.5.0"
```

**踩坑**：
- 直接 `conda activate` 可能不生效，`pip` 仍指向旧环境（如 `ptca`），导致安装到错误位置并报 `Permission denied`
- 解决方案：`conda init bash && exec bash` 重启 shell 后再 activate
- 确认方式：`which pip` 应指向 `/home/aiscuser/.conda/envs/gemma4-vllm/bin/pip`

**最终依赖版本**：
```
Python: 3.10
vllm: 0.19.x
transformers: 5.5.x
```

### vLLM 服务启动

```bash
# BF16 模型，4 卡 tensor parallel
vllm serve /vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma-4-26B-A4B-it \
    --tensor-parallel-size 4 \
    --max-model-len 96000 \
    --port 8000
```

**启动日志关键信息**：
- 模型加载 + CUDA graph capture 约需 110s
- KV cache: 982,992 tokens（56.25 GiB per worker）
- 最大并发（96K tokens/request）: 52.93x
- 每卡显存占用: ~76GB（模型 + KV cache + CUDA graphs）
- GPU 4-7 空闲（仅 GPU 0-3 用于 TP=4）

### 初步 Benchmark 结果（TP=4, with-CoT, 8 并发）

**配置**：
- 模型：BF16 Gemma 4 26B-A4B-it
- Tensor Parallel: 4
- Concurrency: 8
- Mode: with-CoT（system prompt 含 `<think>` block）
- Max tokens: 2048
- 数据：`dpo_combined_eval_cot.jsonl`（190 条，用户消息平均 5643 chars）

**前 15 条结果**：

| 样本 | Output Tokens | 延迟 (s) |
|------|--------------|----------|
| 1-8 (首批) | 715-816 | 186-220 |
| 9-15 (第二批) | 737-823 | 151-174 |

**分析**：
- 单条延迟 ~150-220s，**远高于 transformers 单卡的 ~60s**
- 原因：TP=4 的跨卡通信开销 + 输入过长（平均 5643 chars ≈ 2000+ tokens，prefill 慢）
- vLLM TP 模式为降低单卡显存而设计，不是为减少单条延迟
- 8 并发总吞吐量约 8 / 200s ≈ 0.04 req/s，离 8 req/s 目标差距巨大

### 问题分析与优化方向

**当前瓶颈**：
1. **TP=4 通信开销**：BF16 模型 ~50GB，TP=4 每卡只存 ~12GB 模型权重，但每次 decode step 需要 4 卡 all-reduce，通信延迟大
2. **输入过长**：用户消息平均 5643 chars（含完整网页内容），prefill 阶段占用大量时间
3. **输出过长**：CoT 模式平均 750 tokens output，生成时间长

**优化方向**：

| 方向 | 预期效果 | 风险 |
|------|---------|------|
| **减少 TP 数量**（TP=2 或 TP=1） | 减少通信开销，单条延迟可能降低 | TP=1 需 80GB 放 50GB 模型 + KV cache，可能 OOM |
| **限制 max-model-len**（`--max-model-len 4096`） | 减少 KV cache 占用，留更多空间给 batch | 截断过长输入 |
| **截断输入内容**（`--max_lp_chars 2000`） | 减少 prefill 时间 | 需验证质量影响 |
| **No-CoT 模式** | 输出 tokens 从 ~750 降到 ~400-500 | 需验证 compliance |
| **量化模型**（FP8/GPTQ） | 减少显存占用，可用 TP=1 | vLLM GPTQ 支持实验性 |
| **多副本部署**（非 TP） | 每卡独立副本，无通信开销 | 需量化到单卡装得下 |

### Benchmark 2: TP=2, No-CoT, 8 并发（2026-04-11）

**配置**：
- Tensor Parallel: 2（GPU 0-1）
- `--max-model-len 8192`
- Concurrency: 8
- Mode: no-CoT
- Max tokens: 1024
- 数据：20 条

**结果**：

| 指标 | 值 |
|------|-----|
| Successful | 20/20 |
| Total wall time | 126.3s |
| **Throughput** | **0.16 req/s** |
| Avg latency | 42.0s |
| Median latency | 41.2s |
| P95 latency | 49.8s |
| Min / Max latency | 34.6s / 51.6s |
| Avg output tokens | 631 |
| **Avg tok/s** | **15.2** |
| Median tok/s | 15.2 |

**横向对比**：

| 指标 | Transformers 单卡 (No-CoT) | vLLM TP=4 CoT | vLLM TP=2 No-CoT |
|------|---------------------------|---------------|-------------------|
| Avg 延迟 | 52.0s | ~170-220s | **42.0s** |
| Avg tok/s | 12.1 | ~4 | **15.2** |
| 吞吐量 | ~0.02 req/s | ~0.04 req/s | **0.16 req/s** |
| Avg output tokens | 632 | ~750 | 631 |

**分析**：
- TP=2 比 TP=4 延迟大幅降低（42s vs 170-220s），通信开销减半
- tok/s 提升 26%（15.2 vs 12.1），vLLM 的 CUDA graph + continuous batching 发挥作用
- 8 并发吞吐量是 transformers 单条的 8 倍
- **瓶颈仍是输出长度**：631 tokens / 15.2 tok/s ≈ 42s/request，要 8 req/s 需要延迟 < 1s

### 待实验

- [ ] 提高并发数（32/64），测吞吐量是否线性增长
- [ ] 截断输入（减少 prefill 时间）
- [ ] FP8 量化模型 vLLM 部署
- [ ] 多 GPTQ 副本部署（单卡 13GB，可放 4-6 副本/机器）

### Benchmark 脚本

`Gemma4/benchmark_vllm.py` — 基于 asyncio + aiohttp 的并发 benchmark 工具：
- 支持 `--concurrency N` 设置并发数
- 支持 `--no_cot` 切换 system prompt
- 支持 `--max_lp_chars N` 截断输入内容
- 自动发现 vLLM 模型名称
- 输出：吞吐量（req/s）、延迟分布（avg/median/p95）、每条 tok/s
- 可导出详细 JSON 结果

### Benchmark 3: TP=2, No-CoT, C=32（2026-04-11）

**配置**：
- Tensor Parallel: 2（GPU 0-1）
- `--max-model-len 8192`
- Concurrency: **32**
- Mode: no-CoT
- Max tokens: 1024
- 数据：50 条

**结果**：

| 指标 | 值 |
|------|-----|
| Successful | 50/50 |
| Total wall time | 156.5s |
| **Throughput** | **0.32 req/s** |
| Avg latency | 58.2s |
| Median latency | 53.2s |
| P95 latency | 91.2s |
| Avg output tokens | 636 |
| **Avg tok/s** | **10.9** |
| Median tok/s | 11.3 |

**与 C=8 对比**：

| 指标 | C=8 (20条) | C=32 (50条) | 变化 |
|------|-----------|-------------|------|
| **Throughput** | 0.16 req/s | **0.32 req/s** | **+100%** |
| Avg latency | 42.0s | 58.2s | +39% |
| Avg tok/s | 15.2 | 10.9 | -28% |
| Avg output tokens | 631 | 636 | ~0 |

**分析**：
- 并发从 8→32 后，吞吐量翻倍（0.16→0.32 req/s），vLLM continuous batching 有效利用了 GPU 计算
- 代价是单条延迟上升 39%（42→58s），每条 tok/s 下降 28%（15.2→10.9）
- P95 延迟达到 91s，长尾效应明显（batch 中后排请求等待时间长）
- 输出 token 量不变（~636），说明并发不影响生成内容

### Benchmark 4: TP=2, No-CoT, C=8, 截断 2000 字符（2026-04-11）

**配置**：
- Tensor Parallel: 2（GPU 0-1）
- `--max-model-len 8192`
- Concurrency: 8
- Mode: no-CoT
- Max tokens: 1024
- **`--max_lp_chars 2000`**（截断 LP 内容到 2000 字符）
- 数据：20 条

**结果**：

| 指标 | 值 |
|------|-----|
| Successful | 20/20 |
| Total wall time | 132.3s |
| **Throughput** | **0.15 req/s** |
| Avg latency | 44.5s |
| Median latency | 43.2s |
| P95 latency | 53.1s |
| Avg output tokens | 647 |
| **Avg tok/s** | **14.6** |
| Median tok/s | 14.8 |

**与不截断（Benchmark 2）对比**：

| 指标 | 不截断 | 截断 2000 | 变化 |
|------|--------|-----------|------|
| Avg latency | 42.0s | 44.5s | +6% |
| Avg tok/s | 15.2 | 14.6 | -4% |
| Throughput | 0.16 req/s | 0.15 req/s | -6% |
| Avg output tokens | 631 | 647 | +3% |

**分析**：
- 对当前 20 条样本（平均输入 ~5600 chars），截断到 2000 后**平均延迟无改善**
- 原因：这批数据的主要瓶颈在 **decode（逐 token 生成 ~640 tokens ≈ 42s）**，prefill 占比很小
- 但**截断对超长输入仍有价值**：P95/P99 的超长 LP（10K-300K+ chars）prefill 时间显著，截断可降低这些 case 的延迟和显存峰值
- 截断后模型输出略长（647 vs 631），可能是信息减少后模型更易发散
- **建议线上保留截断**：对尾部超长 case 有防御作用，对普通 case 无负面影响

### 吞吐量优化总结

| 方案 | Throughput | Avg Latency | Avg tok/s | 效果 |
|------|-----------|-------------|-----------|------|
| TP=4, CoT, C=8 | ~0.04 req/s | ~200s | ~4 | 基线 |
| TP=2, No-CoT, C=8 | 0.16 req/s | 42.0s | 15.2 | **4x** |
| TP=2, No-CoT, C=32 | **0.32 req/s** | 58.2s | 10.9 | **8x** |
| TP=2, No-CoT, C=8, 截断 2000 | 0.15 req/s | 44.5s | 14.6 | 平均无效，超长 case 有防御作用 |

**关键结论**：
1. **TP=4→TP=2 是最大优化**：通信开销减半，延迟从 200s 降到 42s
2. **提高并发有效但延迟增加**：C=32 吞吐翻倍到 0.32 req/s，但单条延迟增加 39%
3. **截断输入对平均 case 无效，但对超长输入有防御价值**：主要瓶颈是 decode，但超长 LP（10K+ chars）的 prefill 时间和显存峰值不可忽视，建议线上保留截断
4. **BF16 单实例极限约 0.3 req/s**：要达到 8 req/s 需要 ~25 个并行实例
5. **要大幅降低延迟，只有两条路**：减少输出 tokens 或 提高 decode 速度（量化/更小模型/硬件升级）

---

## Benchmark 脚本升级：TTFT + LP 长度 Sweep + 加速选项（2026-04-13）

### 背景

之前的 `benchmark_speed.py` 无法区分 prefill 和 decode 耗时，也无法验证不同 LP 长度对延迟的影响。需要：
1. 测量 TTFT（Time to First Token）分离 prefill vs decode
2. 按 LP content 长度（400/1000/2000/5000 chars）分组测试
3. 探索加速选项（SDPA、Flash Attention 2、torch.compile）

### 改动

#### `Gemma4/benchmark_speed.py`（完全重写）

核心改动：

1. **TTFT 计量**：使用 `TextIteratorStreamer` + threading，在单独线程中 `model.generate()`，主线程监听 streamer 拿到 first token 的时间
   - `t_start` → `t_first_token`：TTFT（prefill 时间）
   - `t_first_token` → `t_end`：decode 时间
   - 每条输出 `ttft_ms`、`decode_s`、`decode_tok_s`、`total_s`

2. **LP 长度 Sweep**：新增 `--lp_char_lengths` 参数（逗号分隔），对每个长度值截断 LP content 并分别跑全部样本，输出对比表

3. **加速选项**：
   - `--attn_impl`：选择 attention 实现（`sdpa`/`flash_attention_2`/`eager`，默认 `sdpa`）
   - `--torch_compile`：启用 `torch.compile(model, mode="reduce-overhead")`

4. **默认 No-CoT**：`--no_cot` 改为默认 True，新增 `--cot` 可显式启用 CoT

5. **输出保存**：`--save_outputs` 将每条输入输出保存到 JSONL，用于后续检查输出质量

#### `Gemma4/inference_gemma4.py`（小改动）

新增 `attn_impl` 参数传入模型加载：
```python
if attn_impl:
    load_kwargs["attn_implementation"] = attn_impl
```

### 运行环境

```
conda 环境: gemma4
transformers: 5.5.3（TextIteratorStreamer 可用）
torch: 2.11.0+cu126（torch.compile 可用）
flash-attn: 未安装（CUDA 版本不匹配：系统 11.8 vs PyTorch 12.6，安装报错）
```

### 运行命令

```bash
conda activate gemma4

CUDA_VISIBLE_DEVICES=0 python Gemma4/benchmark_speed.py \
    --model_id /vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma-4-26B-A4B-it/ \
    --input_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
    --num_samples 20 \
    --lp_char_lengths 400,1000,2000,5000 \
    --save_outputs /vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/Gemma4_results/gemma4_bf16_nocot_lpsweep_ttft_benchmark.jsonl
```

### 预期输出

每条样本：
```
[Sample 1/20] 634 tokens in 52.3s | TTFT: 1.2s | Decode: 51.1s (12.4 tok/s) | Input: 1024 tokens | LP: 2000 chars
```

LP 长度对比表：
```
LP Chars | Avg Input Tokens | Avg TTFT | Avg Decode | Avg Total | Avg tok/s
---------|------------------|----------|------------|-----------|----------
  400    |      312         |   0.8s   |   51.0s    |   51.8s   |   12.5
 1000    |      587         |   1.0s   |   51.2s    |   52.2s   |   12.4
 2000    |      923         |   1.3s   |   51.3s    |   52.6s   |   12.3
 5000    |     1854         |   2.1s   |   51.5s    |   53.6s   |   12.2
```

### 结果（2026-04-13）

| LP Chars | Avg Input Tok | Avg TTFT | Avg Decode | Avg Total | Avg tok/s |
|----------|--------------|----------|------------|-----------|-----------|
| 400 | 533 | 0.02s | 54.8s | 54.8s | 11.8 |
| 1000 | 636 | 0.02s | 53.9s | 53.9s | 11.7 |
| 2000 | 779 | 0.02s | 54.1s | 54.1s | 11.7 |
| 5000 | 1032 | 0.02s | 51.7s | 51.7s | 11.9 |
| unlimited | 1095 | 0.02s | 52.2s | 52.2s | 12.2 |

**关键发现**：
- **LP 截断长度对单卡推理速度几乎没有影响**
- Input tokens 从 533→1095 翻了一倍，但总耗时基本持平（54.8s vs 52.2s）
- TTFT 固定 0.02s，prefill 开销可忽略不计
- 瓶颈完全在 decode 阶段：output tokens 数量差不多（616~647），所以总时间差不多
- unlimited 甚至略快（12.2 tok/s），可能因为更丰富的上下文让模型更快收敛到 EOS
- **结论**：BF16 单 GPU 下，截断 LP 内容对吞吐几乎无加速效果。截断主要价值在于：(1) 节省显存/KV cache；(2) 满足 context window 限制；(3) 对超长 LP（10K+ chars）的 prefill 防御

---

## Random200 No-CoT + 截断 2000 结果（2026-04-13）

### 配置
- 模型：BF16 Gemma 4 26B-A4B-it
- 模式：No-CoT + `--max_lp_chars 2000`
- 数据：Random200（200 条）
- 8 GPU 数据并行

### 结果

| 指标 | Random200 CoT | Random200 No-CoT+Trunc2000 | 差异 |
|------|--------------|---------------------------|------|
| **Fully Compliant** | 99.0% | **98.5%** | -0.5pp |
| All 5 tags present | 99.0% | 98.5% | -0.5pp |
| Total time | — | 1702s (8.5s/sample effective) | — |

**分析**：
- No-CoT + 截断 2000 后 compliance 仅下降 0.5pp（99.0% → 98.5%），几乎无影响
- 8 GPU 并行有效吞吐 8.5s/sample（vs 单卡 ~52s）
- 3 条不合规样本需检查具体原因（格式问题 or 截断导致信息丢失）

---

## 两步生成（Two-Step）速度 Benchmark（2026-04-13）

### 背景

两步生成方案将 prompt 生成拆为两步：
1. **Step 1（Scene Planning）**：生成 5 个不同视角的场景描述（batch=1，max_new_tokens=256）
2. **Step 2（Batch Expand）**：将 5 个场景批量扩展为完整 prompt（batch=5，max_new_tokens=512）

目标是通过场景预规划提升 5 个 prompt 的多样性。

### 脚本

- 推理脚本：`Gemma4/inference_gemma4_two_step.py`
- 速度 benchmark：`Gemma4/benchmark_speed_two_step.py`
  - Step 1 TTFT 使用 `TextIteratorStreamer` 测量
  - Step 2 prefill 使用单独 `model.forward()` 测量
  - 输出 prefill/decode 分离计时

### 运行命令

```bash
conda activate gemma4

# 推理测试（5 条）
CUDA_VISIBLE_DEVICES=0 python Gemma4/inference_gemma4_two_step.py \
    --model_id /vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma-4-26B-A4B-it/ \
    --input_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
    --num_samples 5 \
    --no_think \
    --temperature 1.0 \
    --output_file /vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/Gemma4_results/gemma4_two_step_nothink_test.jsonl

# 速度 benchmark（4 条 + 2 warmup）
CUDA_VISIBLE_DEVICES=0 python Gemma4/benchmark_speed_two_step.py \
    --model_id /vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma-4-26B-A4B-it/ \
    --input_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
    --num_samples 4 \
    --warmup 2 \
    --no_think \
    --output_file /vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/Gemma4_results/benchmark_two_step.json
```

### 推理质量测试结果（5 条）

- Format compliance: **100%**（5/5）
- 5 个场景覆盖不同视角（close-up / lifestyle / environmental / outcome / mood）
- 扩展后的 prompt 字数均在 108-132 words，符合 80-150 words 要求
- Step 2 batch=5 输出 tokens 对齐（均为 176-179 tokens），batch padding 行为正常

### 速度 Benchmark 结果（单卡 A100-80GB, BF16, no-think）

| 阶段 | Avg Prefill | Avg Decode | Avg Output Tokens | Avg tok/s |
|------|-------------|------------|-------------------|-----------|
| Step 1（场景规划, batch=1） | 0.02s | 10.29s | 116 | 11.3 |
| Step 2（批量扩展, batch=5） | 1.56s | 38.19s* | 1304（5 条合计） | 35.5 |
| **总计** | — | — | 1420 | — |

*Step 2 decode 含一个异常值（Sample 1: 2560 tok / 79.9s，5 个序列全部 hit max_new_tokens=512 上限）

**排除异常值后 Step 2 正常值**：~25s, ~885 tok, ~36 tok/s

#### 时间分解（avg per sample）

```
Step1 prefill:  0.02s  ( 0%)
Step1 decode:   10.29s (21%)
Step2 prefill:  1.56s  ( 3%)
Step2 decode:   38.19s (76%)
─────────────────────────────
Total:          50.1s
```

#### 与一步生成对比

| 指标 | 一步生成（No-CoT） | 两步生成 | 变化 |
|------|-------------------|---------|------|
| Avg time/sample | 52.0s | ~35.5s* | **-32%** |
| Avg output tokens | 632 | ~1000* | +58% |
| Step 1 时间 | — | 10.3s | — |
| Step 2 时间 | — | ~25s* | — |
| Decode tok/s | 12.1 | 35.5（batch=5 aggregate） | **+194%** |

*排除异常值

**分析**：
- Step 2 batch=5 的 aggregate decode tok/s（35.5）约为单序列（11.3）的 **3 倍**，batch 并行有效利用 GPU
- 总 output tokens 更多（~1000 vs 632），但由于 batch 并行，总耗时反而可能更少
- Step 1 prefill 极快（0.02s），Step 2 prefill 1.5s（5 条序列，含完整 LP 内容）
- **76% 时间在 Step 2 decode**，这是主要瓶颈
- 个别 sample 会 hit max_new_tokens 上限导致极慢（2560 tok / 80s），后续可考虑降低 max_new_tokens 或加 stop token

### Step 1 单步使用场景：仅场景规划（~30 词输出）

如果业务只需要简短的场景描述（不需要完整的 80-150 word prompt），可以只运行 Step 1（场景规划），跳过 Step 2（扩展）。

**Step 1 输出格式**：5 个场景描述，每个 8-15 词，总输出约 50-70 英文单词。

**示例输出**（来自推理质量测试）：
```
<Scene1>Close-up of premium wireless headphones on marble surface with warm light</Scene1>
<Scene2>Young professional commuting on train, wearing headphones, peaceful expression</Scene2>
<Scene3>Headphones resting on home office desk beside laptop and coffee</Scene3>
<Scene4>Person relaxing in park, eyes closed, immersed in music</Scene4>
<Scene5>Moody evening scene with headphones silhouetted against golden hour window</Scene5>
```

**速度数据**（来自 benchmark_speed_two_step.py，单卡 A100-80GB, BF16, no-think）：

| 指标 | 值 |
|------|-----|
| Avg output tokens | 116 |
| Avg prefill (TTFT) | 0.02s |
| Avg decode time | 10.29s |
| **Avg total time** | **~10.3s** |
| Decode tok/s | 11.3 |

**对比完整两步生成和一步生成**：

| 方案 | 输出内容 | 总 output tokens | Avg time/sample |
|------|---------|-----------------|-----------------|
| Step 1 only（仅场景规划） | 5 个短场景描述（~30 词/场景） | ~116 | **~10.3s** |
| 一步生成（No-CoT） | 5 个完整 prompt（80-150 词/prompt） | ~632 | ~52.0s |
| 两步生成完整流程 | 5 个完整 prompt + 场景规划 | ~1000-1400 | ~35-50s |

**适用场景**：
- 快速生成创意方向/大纲，由人工或下游系统扩展为完整 prompt
- 低延迟场景（~10s vs ~50s），牺牲 prompt 细节换取速度
- 批量生成大量场景方向后筛选，再对优选场景做 Step 2 扩展

---

## Gemma 4 E4B-it 速度 Benchmark（2026-04-13）

### 背景

测试 Gemma 4 E4B-it（Dense, 4.5B 有效参数 / 8B 总参数）的推理速度，与 26B-A4B-it（MoE, 3.8B 激活 / 25.2B 总参数）对比。

### 模型信息

| 维度 | gemma-4-26B-A4B-it | gemma-4-E4B-it |
|------|-------------------|----------------|
| 架构 | MoE, 激活 3.8B | **Dense, 4.5B 有效** |
| 总参数 | 25.2B | 8B（含 PLE 嵌入表） |
| 上下文 | 256K | 128K |
| 多模态 | Text + Image | Text + Image + **Audio** |
| License | — | Apache 2.0 |

### 配置
- 硬件：单卡 A100-SXM4-80GB
- 精度：BF16
- 模式：No-CoT, no-think
- Attention: SDPA
- 数据：`dpo_combined_eval_cot.jsonl`，10 条 benchmark + 2 warmup

### 运行命令

```bash
conda activate gemma4
bash Gemma4/benchmark_speed_e4b.sh
```

### 结果

| 指标 | E4B-it | 26B-A4B-it (No-CoT) | 差异 |
|------|--------|---------------------|------|
| Avg input tokens | 1257 | ~1095 | — |
| **Avg output tokens** | **559** | **632** | **-12%** |
| **Avg tok/s** | **12.9** | **12.1** | **+7%** |
| Median tok/s | 12.8 | 12.2 | +5% |
| Min tok/s | 12.7 | — | — |
| Max tok/s | 13.0 | — | — |
| **Avg time/sample** | **43.5s** | **52.0s** | **-16%** |
| Median time/sample | 43.4s | — | — |
| TTFT (prefill) | 0.02s | 0.02s | 0 |

### 分析

1. **tok/s 几乎相同**（12.9 vs 12.1，仅 +7%）— E4B 虽然模型更小（4.5B vs 3.8B 激活），但 Dense 架构 vs MoE 架构在 A100 上单条推理速度差异不大
2. **总耗时减少 16%**（43.5s vs 52.0s）— 主要原因是 E4B 输出更短（559 vs 632 tokens），而非 decode 速度显著更快
3. **TTFT 完全一致**（0.02s）— 两个模型的 prefill 开销都可忽略
4. **tok/s 极其稳定**（12.7-13.0 范围）— 与输入长度无关（input 564-2345 tokens），瓶颈完全在 decode
5. **E4B 显存优势显著**：8B 总参数 BF16 ~16GB vs 26B ~52GB，单卡可多副本部署

### 关键结论

- **E4B 的速度优势不如预期**：Dense 4.5B 的 decode 速度与 MoE 3.8B 激活参数几乎持平，MoE 的 expert 路由开销在 A100 上可忽略
- **E4B 的主要优势是显存**：~16GB vs ~52GB，同一张 A100 可跑 3-4 个 E4B 副本（vs 1 个 26B-A4B）
- **输出质量已验证**（见下方 evaluate.py 结果）：format compliance 100%，但 prompt 偏短、forbidden words 偏高

### E4B-it evaluate.py 质量评估（2026-04-13）

对 benchmark 输出（10 条样本，no-think 模式）运行 `evaluate.py`：

```bash
python QwenFinetune/evaluate.py \
    --generated_file .../gemma4_e4b_benchmark.jsonl \
    --gt_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
    --report_file .../gemma4_e4b_eval_report.json
```

| 指标 | E4B-it | 说明 |
|------|--------|------|
| All 5 tags present | **100%** | 格式完全合规 |
| All 5 prompts unique | **100%** | 无重复 |
| Fully compliant | **100%** | 5 tags + unique + ≤150 words 全部通过 |
| Prompts within 150 words | **5.0 / 5** | 全部在 150 词以内 |
| Avg word count per prompt | **82.4** | 偏短（目标 80-150 词，刚踩到下限） |
| `<think>` block present | 0% | 预期：使用 no-think 模式 |
| All 6 CoT fields present | 0% | 预期：使用 no-think 模式 |
| Prompts with quality hints | **1.5 / 5** | 偏低，prompt 中嵌入的质量约束不够 |
| Prompts with forbidden words | **2.1 / 5** | ⚠️ 偏高，平均每条有 2.1 个 prompt 含 forbidden 词 |
| Avg LP keyword coverage | 0% | N/A：benchmark 输出无 lp_fields，无法计算 |

#### 与 26B-A4B-it 对比（需在 26B-A4B benchmark 输出上跑相同评估做对比）

| 维度 | E4B-it | 备注 |
|------|--------|------|
| Format compliance | 100% | 优秀 |
| Avg word count | 82.4 | 偏短，可能缺细节 |
| Quality hints | 1.5/5 | 需要更强的 system prompt 引导 |
| Forbidden words | 2.1/5 | 需要优化，可能是小模型约束跟随能力弱 |

#### 分析

1. **格式能力强**：E4B 100% format compliance，说明 4.5B 参数足够理解 `<Prompt1>...<Prompt5>` XML 格式
2. **Prompt 偏短**（82.4 词 vs 目标 80-150）：刚踩到下限，缺少细节描述
3. **Forbidden words 偏高**（2.1/5）：小模型对 "no watermark/logo/stock photo" 等排除约束的遵循较弱
4. **Quality hints 偏低**（1.5/5）：prompt 中很少主动嵌入 "sharp focus", "correct anatomy" 等质量关键词
5. **结论**：E4B 格式能力合格，但**内容质量明显弱于大模型**（约束遵循和细节丰富度不足），不建议直接替代 26B-A4B-it 用于生产

---

## Gemma 4 E2B-it 速度 Benchmark + 质量评估（2026-04-13）

### 背景

测试 Gemma 4 E2B-it（Dense, 2B 有效参数）的推理速度和输出质量，与 E4B-it 和 26B-A4B-it 对比。

### 模型信息

| 维度 | gemma-4-26B-A4B-it | gemma-4-E4B-it | gemma-4-E2B-it |
|------|-------------------|----------------|----------------|
| 架构 | MoE, 激活 3.8B | Dense, 4.5B 有效 | **Dense, 2B 有效** |
| 总参数 | 25.2B | 8B | ~4B |
| 上下文 | 256K | 128K | 128K |

### 配置
- 硬件：单卡 A100-SXM4-80GB
- 精度：BF16
- 模式：No-CoT, no-think
- Attention: SDPA
- 数据：`dpo_combined_eval_cot.jsonl`，10 条 benchmark + 2 warmup

### 运行命令

```bash
conda activate gemma4
bash Gemma4/benchmark_speed_e2b.sh
```

### Speed Benchmark 结果

| 指标 | E2B-it | E4B-it | 26B-A4B-it | E2B vs E4B | E2B vs 26B |
|------|--------|--------|------------|------------|------------|
| Avg input tokens | 1257 | 1257 | ~1095 | — | — |
| **Avg output tokens** | **522** | **559** | **632** | -7% | -17% |
| **Avg tok/s** | **15.9** | **12.9** | **12.1** | **+23%** | **+31%** |
| Median tok/s | 15.9 | 12.8 | 12.2 | +24% | +30% |
| Min tok/s | 15.7 | 12.7 | — | — | — |
| Max tok/s | 16.1 | 13.0 | — | — | — |
| **Avg time/sample** | **32.7s** | **43.5s** | **52.0s** | **-25%** | **-37%** |
| Median time/sample | 32.9s | 43.4s | — | — | — |
| TTFT (prefill) | 0.02s | 0.02s | 0.02s | 0 | 0 |

### Evaluate.py 质量评估

```bash
python QwenFinetune/evaluate.py \
    --generated_file .../gemma4_e2b_benchmark.jsonl \
    --gt_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
    --report_file .../gemma4_e2b_eval_report.json
```

| 指标 | E2B-it | E4B-it | 说明 |
|------|--------|--------|------|
| All 5 tags present | **20%** | 100% | ⚠️ E2B 仅 2/10 样本输出完整 5 个标签 |
| All 5 prompts unique | **20%** | 100% | 仅完整样本可判断 |
| Fully compliant | **20%** | 100% | E2B 格式合规率极低 |
| Prompts within 150 words | **4.2 / 5** | 5.0 / 5 | E2B 部分 prompt 超长或缺失 |
| Avg word count per prompt | **75.9** | 82.4 | E2B 更短 |
| Prompts with quality hints | **2.4 / 5** | 1.5 / 5 | E2B 反而更多嵌入质量关键词 |
| Prompts with forbidden words | **1.4 / 5** | 2.1 / 5 | E2B 反而更少含 forbidden 词 |
| Avg LP keyword coverage | 0% | 0% | N/A：benchmark 输出无 lp_fields |

### 三模型速度 vs 质量综合对比

| 维度 | E2B-it (2B) | E4B-it (4.5B) | 26B-A4B-it (3.8B act) |
|------|-------------|---------------|----------------------|
| **Decode tok/s** | **15.9** | 12.9 | 12.1 |
| **Avg time/sample** | **32.7s** | 43.5s | 52.0s |
| Format compliance | **20%** | 100% | — |
| Avg word count | 75.9 | 82.4 | ~100+ |
| Quality hints | 2.4/5 | 1.5/5 | — |
| Forbidden words | 1.4/5 | 2.1/5 | — |
| 显存 (BF16 估算) | **~8GB** | ~16GB | ~52GB |

### 分析

1. **E2B 速度最快**：15.9 vs 12.9 tok/s（+23%），总耗时 32.7s vs 43.5s（-25%）
2. **E2B 格式合规率极差**：仅 20%（2/10）样本正确输出全部 5 个 `<PromptN>` 标签，说明 2B 参数不足以可靠遵循复杂 XML 格式指令
3. **E2B 内容质量反而更好**：quality hints 2.4/5 vs E4B 1.5/5，forbidden words 1.4/5 vs E4B 2.1/5 — 但格式不合规导致无法使用
4. **E4B 是小模型中的最优选**：100% format compliance + 合理速度，E2B 格式不可靠不适合生产

### 关键结论

- **E2B 不可用于生产**：格式合规率仅 20%，80% 的输出无法正确解析出 5 个 prompt
- **E4B 是小模型最优选**：100% format compliance，速度尚可（12.9 tok/s），显存友好（~16GB）
- **格式遵循能力是模型选择的硬门槛**：E2B 内容质量指标反而更好，但格式不合规导致完全不可用
- **26B-A4B 仍是质量标杆**：需要在 26B-A4B benchmark 输出上跑 evaluate.py 做直接对比
- **下一步**：跑 26B-A4B evaluate.py 做完整质量对比

---

## Two-Step Benchmark：无 stop_strings、默认 max_new_tokens 基线结果

**日期**：2026-04-13
**配置**：
- 模型：gemma-4-26B-A4B-it BF16
- Step 1：`max_new_tokens=128`，无 `stop_strings`
- Step 2：`max_new_tokens=256`（batch=5），无 `stop_strings`
- 4 benchmark samples + 2 warmup

### 原始输出

```
  [WARMUP] S1: prefill=0.14s decode=11.20s (97tok) | S2: prefill=4.47s decode=12.0s (330tok) | Total: 27.8s
  [WARMUP] S1: prefill=0.02s decode=8.36s (87tok) | S2: prefill=2.58s decode=40.0s (1280tok) | Total: 51.0s
  [Sample 1/4] S1: prefill=0.02s decode=7.84s (84tok) | S2: prefill=2.45s decode=38.3s (1280tok) | Total: 48.6s
  [Sample 2/4] S1: prefill=0.02s decode=8.10s (92tok) | S2: prefill=1.69s decode=9.7s (325tok) | Total: 19.5s
  [Sample 3/4] S1: prefill=0.02s decode=7.62s (89tok) | S2: prefill=0.83s decode=34.4s (1280tok) | Total: 42.9s
  [Sample 4/4] S1: prefill=0.02s decode=7.24s (84tok) | S2: prefill=1.18s decode=38.8s (1280tok) | Total: 47.2s
```

### 汇总统计

| 指标 | 值 |
|------|-----|
| **Step 1 Avg prefill** | 0.019s |
| **Step 1 Avg decode** | 7.70s (87 tokens, 11.3 tok/s) |
| **Step 2 Avg prefill** | 1.539s |
| **Step 2 Avg decode** | 30.3s (1041 tokens, 34.3 tok/s) |
| **Avg total time** | **39.6s/sample** |
| **Median total time** | 45.1s |
| **P95 total time** | 48.4s |

### 时间分解

| 阶段 | 时间 | 占比 |
|------|------|------|
| Step 1 prefill | 0.02s | 0% |
| Step 1 decode | 7.70s | 19% |
| Step 2 prefill | 1.54s | 4% |
| Step 2 decode | 30.30s | **77%** |
| **Total** | **39.6s** | 100% |

### 关键发现

1. **Step 2 decode 是瓶颈**（77% 时间），4 个 sample 中有 3 个 Step 2 打满 1280 tokens（256 × 5 sequences）
2. **大量无用输出**：实际有用 prompt 内容 ~60-80 tokens/prompt（30-50 词），5 个合计 ~300-400 tokens，但模型平均输出 1041 tokens
3. **根因**：模型在输出 `</Prompt>` 标签后继续生成无用内容直到 max_new_tokens 上限
4. **优化方案**：添加 `stop_strings=["</Prompt>"]` + 降低 `max_new_tokens`（Step 1: 128→80, Step 2: 256→128），预计总时间从 39.6s → ~20s

---

## Two-Step Benchmark：添加 stop_strings + 降低 max_new_tokens 优化后结果

**日期**：2026-04-13
**配置**：
- 模型：gemma-4-26B-A4B-it BF16
- Step 1：`max_new_tokens=120`，`stop_strings=["</Scene5>"]`
- Step 2：`max_new_tokens=128`（batch=5），`stop_strings=["</Prompt>"]`
- 4 benchmark samples + 2 warmup

### 原始输出

```
  [WARMUP] S1: prefill=0.29s decode=12.45s (93tok) | S2: prefill=4.16s decode=12.2s (295tok) | Total: 29.1s
  [WARMUP] S1: prefill=0.20s decode=10.30s (92tok) | S2: prefill=2.42s decode=12.0s (340tok) | Total: 24.9s
  [Sample 1/4] S1: prefill=0.18s decode=9.56s (85tok) | S2: prefill=2.25s decode=10.6s (295tok) | Total: 22.5s
  [Sample 2/4] S1: prefill=0.17s decode=10.09s (94tok) | S2: prefill=1.61s decode=10.6s (315tok) | Total: 22.5s
  [Sample 3/4] S1: prefill=0.17s decode=9.15s (85tok) | S2: prefill=0.81s decode=9.8s (325tok) | Total: 20.0s
  [Sample 4/4] S1: prefill=0.16s decode=8.56s (77tok) | S2: prefill=1.10s decode=9.9s (310tok) | Total: 19.7s
```

### 汇总统计

| 指标 | 值 |
|------|-----|
| **Step 1 Avg prefill** | 0.169s |
| **Step 1 Avg decode** | 9.34s (85 tokens, 9.1 tok/s) |
| **Step 2 Avg prefill** | 1.445s |
| **Step 2 Avg decode** | 10.2s (311 tokens, 30.5 tok/s) |
| **Avg total time** | **21.2s/sample** |
| **Median total time** | 21.2s |
| **P95 total time** | 22.5s |

### 时间分解

| 阶段 | 时间 | 占比 |
|------|------|------|
| Step 1 prefill | 0.17s | 1% |
| Step 1 decode | 9.34s | 44% |
| Step 2 prefill | 1.45s | 7% |
| Step 2 decode | 10.23s | 48% |
| **Total** | **21.2s** | 100% |

### 优化前后对比

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| **Avg total time** | 39.6s | **21.2s** | **-46%** |
| Step 2 output tokens | 1041 | 311 | -70% |
| Step 2 decode time | 30.3s | 10.2s | -66% |
| Step 2 decode 占比 | 77% | 48% | 均衡 |
| 错误率 | 3/4 打满 token cap | 0 errors | 全部正常停止 |

### 关键结论

1. **stop_strings 生效**：Step 2 不再打满 max_new_tokens，所有 sample 在 `</Prompt>` 后立即停止
2. **Step 2 decode 时间下降 66%**（30.3s → 10.2s），无用 token 被完全消除
3. **时间分布均衡**：Step 1 decode (44%) 和 Step 2 decode (48%) 各占一半，无明显瓶颈
4. **总时间减半**：39.6s → 21.2s，接近单步生成速度（~19s），两步方案的额外开销可控

---

## UHRS Human Label 对比：Gemma4 Zero-shot vs GPT5（Random 200 LPs）

**日期**：2026-04-13
**数据来源**：
- Gemma4: `Gemma4/data/UHRS_Task_lp_relevance_labeling_0410.tsv`（995 images × 3 judges）
- GPT5: `QwenFinetune/RawData/UHRS_Task_lp_quality_labeling_0324_GPT5.tsv`（1000 images × 3 judges）

**方法**：每张图由 3 位 UHRS judge 独立标注 Good/Fair/Bad，取 max-vote 作为最终标签。排除 Imageloadfail/N/A 后，Gemma4 有效 994 张，GPT5 有效 998 张。

### Image Level Good Rate（max-vote）

| 指标 | Gemma4 Zero-shot | GPT5 | 差异 |
|------|-----------------|------|------|
| **Total images** | 994 | 998 | — |
| **Good** | 749 (75.4%) | 653 (65.4%) | **+10.0pp** |
| **Fair** | 73 (7.3%) | 65 (6.5%) | +0.8pp |
| **Bad** | 172 (17.3%) | 280 (28.1%) | -10.8pp |

### LP Level N/5 Good Distribution

| N/5 Good | Gemma4 (198 LPs) | GPT5 (200 LPs) |
|----------|------------------|-----------------|
| 0/5 | 2 (1.0%) | 1 (0.5%) |
| 1/5 | 5 (2.5%) | 9 (4.5%) |
| 2/5 | 17 (8.6%) | 39 (19.5%) |
| 3/5 | 48 (24.2%) | 65 (32.5%) |
| 4/5 | 69 (34.8%) | 59 (29.5%) |
| 5/5 | 57 (28.8%) | 27 (13.5%) |

> Gemma4 有 198 个 LP（LP_119 仅 2 张图，LP_95 仅 3 张图），GPT5 有 200 个完整 LP。

### LP Level Cumulative（>= N/5 Good）

| 阈值 | Gemma4 | GPT5 | 差异 |
|------|--------|------|------|
| >= 1/5 | 99.0% | 99.5% | -0.5pp |
| >= 2/5 | 96.5% | 95.0% | +1.5pp |
| >= 3/5 | 87.9% | 75.5% | **+12.4pp** |
| >= 4/5 | 63.6% | 43.0% | **+20.6pp** |
| >= 5/5 | 28.8% | 13.5% | **+15.3pp** |

### 关键结论

1. **Image Good Rate: Gemma4 75.4% vs GPT5 65.4%（+10pp）**，Bad Rate 从 28.1% 降至 17.3%
2. **LP Level 质量分布明显右移**：Gemma4 在 4/5 和 5/5 Good 的 LP 占比（63.6%）远超 GPT5（43.0%）
3. **>= 3/5 Good 阈值**：Gemma4 87.9% vs GPT5 75.5%，说明 Gemma4 生成的 prompt 质量更稳定
4. **Gemma4 Zero-shot 无需微调即超越 GPT5 baseline**，验证了 Gemma4 26B-A4B-it 在 image prompt generation 任务上的强大能力

---

## Single-Prompt vLLM 推理实验（2026-04-14）

### 背景

原有方案一次生成 5 个 prompt（`<Prompt1>`~`<Prompt5>`），在 temperature=1.2 时出现：
- 主题偏离（后面的 prompt 与 LP 无关）
- 标签错位（`<Prompt3>` 内容出现在 `<Prompt4>` 标签内）

新方案：每次只生成 1 个 prompt，同一输入调用 5 次，再组合为 `<Prompt1>`~`<Prompt5>` 格式。

### 方案对比

| 维度 | 原方案（5-prompt） | 新方案（single-prompt x5） |
|------|-------------------|--------------------------|
| 每次输出 | 5 个 prompt | 1 个 prompt |
| 多样性来源 | 模型自主变化 | temperature 采样 |
| 标签错位风险 | 高（temperature↑） | 无 |
| 推理后端 | HF Transformers | vLLM offline |

### 实现要点

1. **System Prompt 改为只生成 1 个 prompt**，输出 `<Prompt>...</Prompt>`
2. **User Message 修复**：eval 数据中的 "Generate 5 image" 替换为 "Generate 1 image"，避免与 system prompt 冲突
3. **vLLM 离线模式**：`LLM.generate()` 一次提交 N×K 个 prompt，利用 continuous batching + PagedAttention 实现真并行
4. **Prefix caching**：同一输入的 5 个副本共享 KV cache，prefill 只算一次
5. **TTFT 测量**：vLLM V1 离线模式 `metrics=None`，改用 `max_tokens=1` 探测 prefill latency

### 全量评估结果（190 条，no-CoT，temperature=1.2，TP=2）

**推理性能：**

| 指标 | 值 |
|------|-----|
| Total samples | 190 |
| Total requests | 950 (190 x 5) |
| Total inference time | 122.7s (0.6s/sample) |
| TTFT (prefill) | 0.059s/prompt |
| Decode throughput | 1,299 tok/s |
| Avg input tokens | 1,457 |
| Avg output tokens | 168 |

**质量评估（evaluate.py text-only）：**

| 指标 | 值 |
|------|-----|
| All 5 tags present | 100.0% |
| All 5 prompts unique | 100.0% |
| Fully compliant | 98.9% |
| Avg word count | 126.5 |
| Quality hints | 4.2/5 |
| Forbidden words | 4.0/5 |
| LP keyword coverage | 0.0% |

### 分析

1. **格式合规 100%**：single-prompt 方案彻底消除了标签错位问题
2. **多样性 100%**：5 次独立采样保证了多样性
3. **Forbidden words 4.0/5 偏高**：几乎每个 prompt 都含违禁词，需后续优化 system prompt
4. **LP keyword coverage 0%**：evaluate.py 需要输出 JSONL 中包含 LP 字段才能计算，当前输出格式缺少此字段
5. **推理速度极快**：vLLM offline 模式 190 条仅需 ~2 分钟，decode 吞吐 1,299 tok/s

---

## Two-Step vLLM 推理实验（2026-04-14）

### 背景

此前已有 HF Transformers 版本的 two-step 推理（`inference_gemma4_two_step.py`），分两步生成：
1. Step 1：生成 5 个不同视角的场景概念（5-10 words 短语）
2. Step 2：将每个场景扩展为 30-50 word 的详细 prompt

该方案多样性较好（5 个 scene 强制不同视觉角度），但 Transformers 单卡逐条推理速度较慢（21.2s/sample）。
本次实验将 two-step 方案迁移到 vLLM offline 模式，利用 continuous batching 大幅提升吞吐。

### 脚本

- `Gemma4/inference_gemma4_two_step_vllm.py`
- 基于 `inference_gemma4_single_prompt_vllm.py`（vLLM 模式）和 `inference_gemma4_two_step.py`（two-step 逻辑）

### 核心设计

- **Step 1**：所有 N 条记录的 scene planning prompt 放入一次 `llm.generate()`
  - `SamplingParams(max_tokens=120, stop=["</Scene5>"])`
- **Step 2**：N×5 个 expansion prompt 放入一次 `llm.generate()`
  - `SamplingParams(max_tokens=128, stop=["</Prompt>"])`
- **TTFT 探测**：用 `max_tokens=1` 单独测量 prefill 延迟
- 输出 JSONL 包含 `scenes`, `generated_prompts`, `raw_output`，兼容 evaluate.py

### 运行命令

```bash
# 2 条测试
python Gemma4/inference_gemma4_two_step_vllm.py \
    --model_id /vc_data/.../gemma-4-26B-A4B-it \
    --input_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
    --num_samples 2 \
    --temperature 1.0 \
    --tensor_parallel_size 2 \
    --output_file Gemma4/results/gemma4_two_step_vllm_test.jsonl

# 全量 190 条
python Gemma4/inference_gemma4_two_step_vllm.py \
    --model_id /vc_data/.../gemma-4-26B-A4B-it \
    --input_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
    --temperature 1.0 \
    --tensor_parallel_size 2 \
    --output_file Gemma4/results/gemma4_two_step_vllm_full.jsonl
```

### 配置

- 模型：gemma-4-26B-A4B-it BF16
- 硬件：2×A100-SXM4-80GB（tensor_parallel_size=2）
- 模式：No-CoT, no-think
- 数据：`dpo_combined_eval_cot.jsonl`，190 条

### 全量结果（190 条）

| 指标 | 值 |
|------|-----|
| Records | 190 |
| Full scenes (5/5) | 190/190 (100.0%) |
| Format compliance | 190/190 (100.0%) |
| All prompts parsed | 190/190 (100.0%) |
| **TTFT (prefill)** | **0.052s/prompt** |
| **Step 1 (scenes)** | **14.2s** (0.07s/sample) |
| **Step 2 (expand)** | **54.6s** (0.06s/prompt) |
| **Total inference** | **68.8s** (0.36s/sample) |
| Step 1 input tokens | 284,756 |
| Step 1 output tokens | 16,936 |
| Step 2 input tokens | 1,382,955 |
| Step 2 output tokens | 57,067 |
| Total input tokens | 1,667,711 |
| Total output tokens | 74,003 |
| **Decode throughput** | **1,075 tok/s** |

### 与 Transformers Two-Step 对比

| 指标 | Transformers (1×A100) | vLLM (2×A100) | 提升 |
|------|----------------------|---------------|------|
| **每条记录耗时** | 21.2s | **0.36s** | **~59x** |
| **190 条预计总耗时** | ~4028s (~67min) | **68.8s (~1.1min)** | **~59x** |
| Decode throughput | 30.5 tok/s (Step2 batch=5) | **1,075 tok/s** | **~35x** |
| Step 2 prefill | 1.45s | ~0.05s | **~29x** |
| Format compliance | 100% (4条) | 100% (190条) | — |
| GPU 资源 | 1×A100 | 2×A100 | 2x |

### 与 Single-Prompt vLLM 对比

| 指标 | Single-Prompt vLLM | Two-Step vLLM | 差异 |
|------|---------------------|---------------|------|
| **总耗时** | 122.7s | **68.8s** | **-44%** |
| 每条记录 | 0.6s | 0.36s | -40% |
| Format compliance | 100% | 100% | — |
| Decode throughput | 1,299 tok/s | 1,075 tok/s | -17% |
| 总输出 tokens | 159,474 | 74,003 | -54% |
| TTFT | 0.405s/prompt | 0.052s/prompt | -87% |

### 分析

1. **vLLM continuous batching 带来 ~59x 加速**：Transformers 逐条推理 21.2s/sample → vLLM 批量 0.36s/sample
2. **Two-Step 比 Single-Prompt 快 44%**：主要因为 prompt 长度目标 30-50 words（vs 80-150 words），总输出 token 减少 54%
3. **throughput 略低**（1,075 vs 1,299 tok/s）：短输出序列 decode 效率稍差，但总时间显著缩短
4. **TTFT 显著降低**（0.052s vs 0.405s）：Step 1 scene prompt 比 single-prompt 的 system prompt 更短
5. **100% format compliance**：Scene 解析和 Prompt 扩展均正常，stop tokens 生效
6. **多样性优势**：5 个 scene 强制不同视角（close-up / lifestyle / environmental / outcome / mood），天然保证多样性

---

## Two-Step vLLM 单卡 vs 双卡对比（2026-04-14）

### 背景

测试 Two-Step vLLM 在单卡 A100 上的表现，与双卡（tensor_parallel_size=2）对比，评估是否需要双卡。

### 运行命令

```bash
# 单卡
python Gemma4/inference_gemma4_two_step_vllm.py \
    --model_id /vc_data/.../gemma-4-26B-A4B-it \
    --input_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
    --temperature 1.0 \
    --tensor_parallel_size 1 \
    --output_file Gemma4/results/gemma4_two_step_vllm_1gpu_full.jsonl
```

### 单卡全量结果（190 条）

| 指标 | 值 |
|------|-----|
| Records | 190 |
| Full scenes (5/5) | 189/190 (99.5%) |
| Format compliance | 189/190 (99.5%) |
| All prompts parsed | 189/190 (99.5%) |
| **TTFT (prefill)** | **0.088s/prompt** |
| **Step 1 (scenes)** | **15.8s** (0.08s/sample) |
| **Step 2 (expand)** | **69.2s** (0.07s/prompt) |
| **Total inference** | **84.9s** (0.45s/sample) |
| Step 1 input tokens | 284,756 |
| Step 1 output tokens | 16,974 |
| Step 2 input tokens | 1,381,681 |
| Step 2 output tokens | 56,865 |
| Total output tokens | 73,839 |
| **Decode throughput** | **869.5 tok/s** |

### 单卡 vs 双卡 vs Transformers 全对比

| 指标 | Transformers (1×A100) | vLLM (1×A100) | vLLM (2×A100) |
|------|----------------------|---------------|---------------|
| **总耗时** | ~4028s (~67min) | **84.9s (~1.4min)** | **68.8s (~1.1min)** |
| **每条记录** | 21.2s | **0.45s** | **0.36s** |
| Format compliance | 100% (4条) | 99.5% (190条) | 100% (190条) |
| Decode throughput | 30.5 tok/s | **869.5 tok/s** | **1,075 tok/s** |
| TTFT | 0.02s | 0.088s | 0.052s |
| GPU 资源 | 1×A100 | 1×A100 | 2×A100 |

### 分析

1. **单卡完全可行**：26B BF16 (~52GB) 在 A100 80GB 上加载无问题，84.9s 跑完 190 条
2. **双卡仅快 19%**（84.9s → 68.8s）：TP=2 的加速不显著，瓶颈在 decode 而非 prefill
3. **单卡性价比更高**：省一张 A100，速度仅慢 16s（1.4min vs 1.1min）
4. **单卡 throughput 仍达 869.5 tok/s**：相比 Transformers 的 30.5 tok/s 仍有 ~28x 加速
5. **1 条 scene 解析失败**（99.5% vs 100%）：单卡略有差异，可能是随机采样导致，非系统性问题
6. **建议**：日常实验用单卡即可，大批量或低延迟要求时用双卡

---

## Two-Step vLLM 单卡全量评估（evaluate.py）

**日期**：2026-04-13
**评估文件**：`Gemma4/results/gemma4_two_step_vllm_1gpu_full.jsonl`（190 条）
**评估脚本**：`QwenFinetune/evaluate.py`（text-only，无 LLM Judge）

### 评估结果

| 指标 | 值 | 判定 |
|------|-----|------|
| **All 5 tags present** | 99.5% | 合格 |
| **All 5 prompts unique** | 99.5% | 合格 |
| **Fully compliant** | 99.5% | 合格 |
| **Prompts within 150 words** | 5.0/5 | 合格 |
| **Avg word count** | 42.4 words | 合格（目标 30-50 words） |
| **Quality hints** | 1.8/5 | 正常（zero-shot 无 quality hint 要求） |
| **Forbidden words** | 0.1/5 | 合格（几乎无违规词） |
| **CoT compliance** | 0% | 预期（zero-shot 不使用 CoT） |
| **LP keyword coverage** | 0% | 已知限制（见下） |

### 与 Single-Prompt vLLM 评估对比

| 指标 | Two-Step | Single-Prompt | 说明 |
|------|----------|---------------|------|
| Format compliance | 99.5% | 100% | 均极高 |
| Avg word count | **42.4** | 126.5 | Two-step 目标 30-50，single 目标 80-150 |
| Forbidden words | **0.1/5** | 4.0/5 | Two-step 大幅改善 |
| Quality hints | 1.8/5 | 4.2/5 | 短 prompt 自然包含更少修饰语 |

### 分析

1. **Forbidden words 大幅改善**：从 single-prompt 的 4.0/5 降至 0.1/5，two-step 的短 prompt（30-50 words）天然减少了违禁词出现概率
2. **Word count 完美命中目标区间**：42.4 words 在 30-50 范围内，vs single-prompt 的 126.5 words（目标 80-150）
3. **Word vs Token 区分**：42.4 是 word count，实际输出 ~60 tokens/prompt（1 word ≈ 1.4 tokens，包含 XML 标签和 subword tokenization 开销）
4. **LP keyword coverage 0%**：evaluate.py 从输出 JSONL 的 `lp_fields` 字段提取关键词，但 two-step 输出中未包含该字段，非生成质量问题

---

## Two-Step vLLM AWQ 4-bit 量化推理实验（2026-04-14）

### 背景

测试 AWQ 4-bit 量化模型在 vLLM 下的 two-step 推理速度，对比 BF16 全精度模型。
量化模型显存仅需 ~13GB（vs BF16 ~52GB），单卡 A100 可运行多个副本。

GPTQ 4-bit 模型（`gemma-4-26B-A4B-it-GPTQ-Int4`）因权重格式与 vLLM Gemma4 loader 不兼容（`KeyError: 'layers.0.moe.experts.0.down_proj'`），无法加载。

### 配置

- 模型：`gemma-4-26B-A4B-it-AWQ-4bit`（cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit）
- 硬件：1×A100-SXM4-80GB
- 精度：float16（GPTQ/AWQ 要求 fp16，不支持 bf16）
- 模式：No-CoT, no-think
- 数据：`dpo_combined_eval_cot.jsonl`，190 条

### 运行命令

```bash
python Gemma4/inference_gemma4_two_step_vllm.py \
    --model_id /vc_data/.../gemma-4-26B-A4B-it-AWQ-4bit \
    --input_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
    --temperature 1.0 \
    --tensor_parallel_size 1 \
    --dtype half \
    --output_file Gemma4/results/gemma4_two_step_vllm_awq_full.jsonl
```

### 全量结果（190 条）

| 指标 | 值 |
|------|-----|
| Records | 190 |
| Full scenes (5/5) | 188/190 (98.9%) |
| Format compliance | 188/190 (98.9%) |
| All prompts parsed | 188/190 (98.9%) |
| **TTFT (prefill)** | **0.085s/prompt** |
| **Step 1 (scenes)** | **12.5s** (0.07s/sample) |
| **Step 2 (expand)** | **57.5s** (0.06s/prompt) |
| **Total inference** | **70.0s** (0.37s/sample) |
| Total input tokens | 1,648,479 |
| Total output tokens | 77,359 |
| **Decode throughput** | **1,105.2 tok/s** |

### 质量评估（evaluate.py）

| 指标 | 值 |
|------|-----|
| All 5 tags present | 98.9% |
| All 5 prompts unique | 98.9% |
| Fully compliant | 98.9% |
| Prompts within 150 words | 5.0/5 |
| Avg word count | 40.6 words |
| Quality hints | 1.7/5 |
| Forbidden words | 0.1/5 |
| CoT compliance | 0% (预期) |
| LP keyword coverage | 0% (已知限制) |

### 全配置对比

| 指标 | BF16 1×A100 | AWQ 4-bit 1×A100 | BF16 2×A100 |
|------|-------------|------------------|-------------|
| **总耗时** | 84.9s | **70.0s** | 68.8s |
| 每条记录 | 0.45s | **0.37s** | 0.36s |
| Decode throughput | 869.5 tok/s | **1,105.2 tok/s** | 1,075 tok/s |
| Format compliance | 99.5% | 98.9% | 100% |
| Forbidden words | — | 0.1/5 | — |
| TTFT | 0.088s | 0.085s | 0.052s |
| 模型权重显存 | ~52GB | **~19GB** | ~26GB/卡 |
| 推理时总显存 | ~75GB | **~75GB** | ~38GB/卡 |

### 显存分析

vLLM 默认 `gpu_memory_utilization=0.9`，会预分配 A100 80GB × 90% ≈ 72GB：
- **加载时 ~19GB**：AWQ 4-bit 模型权重（26B × 4bit ≈ 13GB + 框架开销）
- **推理时 ~75GB**：权重 + KV cache pool（vLLM 将剩余显存全部分配给 KV cache）
- **BF16 推理时也是 ~75GB**：权重 ~52GB + KV cache ~23GB

AWQ 的显存优势不是总占用更少，而是**模型权重更小 → KV cache pool 更大 → 能同时 batch 更多请求 → throughput 更高**（1,105 vs 869 tok/s）。

如需在同一张卡上跑多副本，需降低 `--gpu_memory_utilization`（如 2 副本各用 0.45）。

### 分析

1. **AWQ 4-bit 单卡比 BF16 单卡快 18%**（84.9s → 70.0s）：模型权重更小，更多显存留给 KV cache，batch 容量更大
2. **throughput 超过 BF16 双卡**（1,105 vs 1,075 tok/s）：单卡 AWQ 的吞吐已与双卡 BF16 持平
3. **模型权重省 63%**（~19GB vs ~52GB）：但 vLLM 会将剩余显存分配给 KV cache，总显存占用相近
4. **质量几乎无损**：format compliance 98.9%（vs BF16 99.5-100%）、forbidden words 0.1/5（与 BF16 two-step 相同）
5. **GPTQ 不可用**：第三方 GPTQ 权重与 vLLM Gemma4 loader 不兼容，AWQ 是当前唯一可用的 4-bit 量化方案
6. **推荐配置**：AWQ 4-bit + 单卡 A100 是最优方案，速度与双卡 BF16 持平，且有更大 KV cache 容量支持高并发

---

## DLIS 线上部署（2026-04-15）

### 目标

将 Gemma4 T2I Prompt Generation 模型部署到 DLIS 线上服务，使用 OaaS LLM Template。

### 参考

- 线上已有模型 `ImgLPRelevance6`（Qwen3-VL，多模态图文输入，Image-LP Relevance 分类任务）
- 部署流程文档：DLIS Model Deployment Guide Using OaaS Template（ChangXu）
- OaaS LLM Template 源码：`OaaS_LLMTemplate/`

### 关键差异分析：ImgLPRelevance6 vs Gemma4

| 维度 | ImgLPRelevance6 (Qwen3-VL) | Gemma4 T2I Prompt Gen |
|------|---------------------------|----------------------|
| 模型类型 | 多模态（图+文） | 纯文本 |
| 输入 | 图片 + Landing Page 内容 | Landing Page 内容 + URL |
| 输出 | Good/Fair/Bad 分类 + Score | 5 个 image generation prompts |
| 依赖 | `qwen_vl_utils`, `PIL` | 无额外依赖 |
| 推理方式 | 单步（每张图一次推理） | **两步**（先生成 scenes，再展开为 prompts） |

### Two-Step 推理在 DLIS 框架中的实现

#### 问题

DLIS OaaS 框架的标准流程是 `preprocess → vLLM推理 → postprocess` 单轮调用。但 Gemma4 的 two-step 推理（`inference_gemma4_two_step_vllm.py`）需要两轮 vLLM 调用：
1. Step 1: 生成 5 个 diverse scene concepts（短语）
2. Step 2: 将每个 scene 展开为完整 image prompt（batch 5 个推理）

#### 方案评估

| 方案 | Latency | 可实现性 | 说明 |
|------|---------|----------|------|
| 合并为一步（单 prompt） | 最低 | 高 | 一个 system prompt 让模型在 `<think>` 中先规划 scenes 再展开 |
| 客户端两次调用 | 高（2x round-trip） | 中 | 需要调用方配合改逻辑 |
| **修改 model.py 支持两轮推理** | **中** | **高** | 在 model.py 中增加一轮 `oaas_wrapper.run()` 调用 |

#### 最终方案：修改 model.py + dlis_inter.py 实现真正 two-step

通过分析 OaaS Template 源码（`OaaS_LLMTemplate/dlis_model/model/model.py`），发现：
- `PreAndPostProcessor()` 初始化时不传参（processor=None）
- `oaas_wrapper.run(prompts)` 接受 prompt list，内部调用 vLLM engine batch 推理
- 可以在 `Eval()` 中调用两次 `oaas_wrapper.run()`

#### 实现的文件

**`Gemma4Deploy/dlis_inter.py`** — 三个接口：

```
preprocess(data)
  → 输入：原始请求 JSON（landing_page_content, url）
  → 输出：Step1 prompt（生成 5 个 scene concepts）+ metadata

build_step2_prompts(step1_output, metadata)
  → 输入：Step1 生成的 scenes 文本
  → 输出：5 个 Step2 prompts（逐个展开 scene）+ 更新后的 metadata

postprocess(step2_outputs, metadata)
  → 输入：5 个 Step2 生成文本
  → 输出：最终结构化结果（generated_prompts, scenes, format_compliant, Status）
```

**`Gemma4Deploy/model.py`** — 修改版 OaaS model.py：

```python
# Eval() 流程
preprocess(data) → oaas_wrapper.run(step1) → build_step2_prompts() → oaas_wrapper.run(step2) → postprocess()
```

`EvalBatch()` 同样支持 batch：所有请求的 Step1 一起 batch，所有 Step2 一起 batch，最大化 vLLM throughput。

**`Gemma4Deploy/Modelfile`** — Gemma chat template 配置。

#### System Prompts

直接复用 `inference_gemma4_two_step_vllm.py` 的原始 system prompts：
- Step 1（`SYSTEM_PROMPT_STEP1`）：5 种不同视角的 scene concepts（close-up / lifestyle / environmental / outcome / mood）
- Step 2（`SYSTEM_PROMPT_STEP2`）：将 scene 展开为 30-50 words 的 detailed prompt

### 下一步

- [ ] 上传模型权重和配置文件到 Gen1
- [ ] Gen1 → Gen2 数据迁移
- [ ] Docker 镜像构建（如需自定义 template）
- [x] 本地 Mock 测试 (test_local.py) — 通过
- [ ] 本地 Docker 测试 (A6000)
- [ ] 创建 Polaris Job
- [ ] 构建 DLIS Service

---

## 本地验证（2026-04-15）

### Mock 测试结果

`Gemma4Deploy/test_local.py` — 不依赖 GPU/vLLM，mock 引擎输出验证逻辑：

| 测试 | 内容 | 结果 |
|------|------|------|
| 1 | 完整流程 preprocess → build_step2 → postprocess | PASS |
| 2 | Step 1 输出截断恢复 (缺少 `</Scene5>` 闭合) | PASS |
| 3 | Step 2 部分输出无 `<Prompt>` 标签 fallback | PASS |
| 4 | preprocess 接受 JSON string 输入 | PASS |
| 5 | LP content 超长截断 | PASS |

修复: `dlis_inter.py` 第 84 行 `print(os.listdir("/Model"))` 加 `os.path.exists` 保护，避免非容器环境崩溃。

### A6000 Docker 端到端验证步骤

#### 前置条件
- SSH 到 A6000 机器: `ssh jinjinchen@BR1T45-S1-17`
- 确认 Docker + NVIDIA Container Toolkit 已安装: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

**A6000 机器路径:**
- 模型权重: `/home/jinjinchen/data/gemma-4-26B-A4B-it-AWQ-4bit/`
- 代码仓库: `/home/jinjinchen/ms-image-quality-filters-aether-module-main/`

#### Step 1: 准备 Docker 镜像

```bash
# 方案 A: 使用已有 OaaS vLLM 镜像 (推荐)
docker pull dlisproddockerrepo.azurecr.io/dlis/llm_framework_vllm:latest

# 方案 B: 如需本地构建
cd /path/to/OaaS_LLMTemplate
SOURCE_BRANCH=main ./pipeline/build_vllm_image.sh
```

#### Step 2: 启动容器（交互式 bash）

```bash
# 端口 8888 可能被占用，改用 8886
docker run -it --rm --gpus all \
  -p 8886:8886 \
  -v /home/jinjinchen/ms-image-quality-filters-aether-module-main/Gemma4Deploy:/Model \
  -v /home/jinjinchen/data/gemma-4-26B-A4B-it-AWQ-4bit:/Model/model \
  -e _ModelPath_=/dlis_model/run.sh \
  -e _ListeningPort_=8886 \
  -e EnableOaas=true \
  -e AB_MAX_SEQ_LEN=16 \
  -e AB_INSTANCE_GROUP_COUNT=1 \
  dlisproddockerrepo.azurecr.io/dlis/llm_framework_vllm:latest \
  /bin/bash
```

#### Step 3: 容器内配置并启动服务

```bash
# 1. 复制自定义代码覆盖模板
cp /Model/model.py /dlis_model/model/model.py
cp /Model/dlis_inter.py /dlis_model/model/dlis_inter.py

# 2. 设置 PYTHONPATH (llm_opt 在容器 / 目录下)
export PYTHONPATH=/:$PYTHONPATH

# 3. 设置必要环境变量 (不可改)
export _ListeningPort_=8886
export AB_MAX_SEQ_LEN=16
export AB_INSTANCE_GROUP_COUNT=1
export AB_ENTRYPOINT=/v2/models/gemma4/versions/1/infer

# 4. 启动 HTTP 服务
cd /dlis_model && ./run.sh http
```

**说明:**
- `-v .../Gemma4Deploy:/Model` — 挂载 `dlis_inter.py` 和 `model.py` 到容器 `/Model`
- `-v .../gemma-4-26B-A4B-it-AWQ-4bit:/Model/model` — 挂载 AWQ 量化权重到容器 `/Model/model`
- `cp /Model/*.py /dlis_model/model/` — 覆盖模板代码，否则加载的是模板自带的 model.py
- `export PYTHONPATH=/` — `llm_opt` 目录在容器根目录 `/llm_opt/`，需加入 Python path
- `AB_ENTRYPOINT` — 模型推理入口路径（不可改）

#### Step 4: 发送测试请求

```bash
# 在另一个终端
curl -X POST http://localhost:8886/ \
  -H "Content-Type: text/plain" \
  -d '{
    "landing_page_content": "Welcome to TrailMaster Outdoor Gear. Premium hiking boots, ultralight backpacks, and camping essentials for your next adventure. Free shipping on orders over $100.",
    "url": "https://trailmaster.example.com",
    "num_prompts": 5,
    "max_lp_chars": 5000
  }'
```

#### Step 5: 验证返回结果

期望返回 JSON:
```json
{
  "generated_prompts": ["prompt1", "prompt2", "prompt3", "prompt4", "prompt5"],
  "scenes": ["scene1", "scene2", "scene3", "scene4", "scene5"],
  "raw_output": "<Prompt1>...</Prompt1>\n\n<Prompt2>...</Prompt2>...",
  "step1_raw": "<Scene1>...</Scene1>...<Scene5>...</Scene5>",
  "format_compliant": true,
  "Status": "Success"
}
```

**检查清单:**
- [ ] `Status` = `"Success"`
- [ ] `generated_prompts` 包含 5 个非空 prompt
- [ ] `scenes` 包含 5 个 scene concept
- [ ] `format_compliant` = `true`
- [ ] 每个 prompt 30-50 words
- [ ] Latency 检查: 响应头 `UnderlyingModelLatencyInUs` 值合理 (两次 vLLM 调用)
- [ ] 无 CUDA 错误或异常日志

---

### DLIS 容器启动 Debug 记录

#### 问题 1: 端口 8888 被占用
```
ERROR: Port 8888 was already in use
```
**修复**: 改用端口 8886，`docker run` 加 `-p 8886:8886`

#### 问题 2: `ModuleNotFoundError: No module named 'llm_opt'`
容器内 `llm_opt` 在根目录 `/llm_opt/`。
**修复**: `export PYTHONPATH=/:$PYTHONPATH`

#### 问题 3: 缺少 `AB_ENTRYPOINT` 环境变量
**修复**: `export AB_ENTRYPOINT=/v2/models/gemma4/versions/1/infer`

#### 问题 4: 模板 model.py 被加载而非自定义版本
容器 `/dlis_model/model/model.py` 是模板自带的，不是我们的。
**修复**: 
```bash
cp /Model/model.py /dlis_model/model/model.py
cp /Model/dlis_inter.py /dlis_model/model/dlis_inter.py
```

#### 问题 5: `RuntimeError: Cannot find model folder /dlis_model/Model`
OaasWrapper 期望模型在 `/dlis_model/Model/` 目录。
**修复**: `ln -s /Model/model /dlis_model/Model`

#### 问题 6: `RuntimeError: DLIS integration not validated`
OaasWrapper 检查 `dlis_integration_validated` 标记文件。
**修复**: `touch /dlis_model/Model/dlis_integration_validated`

#### 问题 7: `model type 'gemma4' not recognized` + `Could not get any available runner`
默认走 `load_org()` 路径 → 使用 BaseLLM (transformers AutoModelForCausalLM) → 不认识 gemma4 架构。
需要走优化路径 `load_core()` → 使用 vLLM 引擎。

**修复**: 创建优化目录结构
```bash
mkdir -p /dlis_model/Model/gemma4_opt
printf "llm" > /dlis_model/Model/gemma4_opt/opt_type.txt  # 注意: echo 会带换行, 用 printf

cat > /dlis_model/Model/gemma4_opt/best_setting.json << 'EOF'
{
    "llm_type": "vllm",
    "model": "gemma-4-26B-A4B-it-AWQ-4bit",
    "quantization": "awq",
    "max_output_len": 256,
    "temperature": 0.8,
    "top_p": 0.95,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "trust_remote_code": true,
    "dtype": "auto",
    "max_model_len": 8192
}
EOF
```

**关键**: `opt_type.txt` 内容必须是 `llm`（无换行），用 `printf` 而非 `echo`。

#### 问题 8: `EngineArgs.__init__() got an unexpected keyword argument 'num_scheduler_steps'`
容器内 vLLM 版本 0.19.0 不支持 `num_scheduler_steps` 参数，但 OaaS wrapper (`/llm_opt/vllm/vllm_runner.py:399`) 硬编码传入。

**修复**: Patch 容器内 vllm_runner.py 删除该参数
```bash
# 方法 1: sed
sed -i '/num_scheduler_steps/d' /llm_opt/vllm/vllm_runner.py

# 方法 2: python patch (如 sed 不好用)
python3 -c "
import re
with open('/llm_opt/vllm/vllm_runner.py', 'r') as f:
    content = f.read()
content = re.sub(r'.*num_scheduler_steps.*\n', '', content)
with open('/llm_opt/vllm/vllm_runner.py', 'w') as f:
    f.write(content)
print('Patched: removed num_scheduler_steps lines')
"
```

Patch 后重启服务：
```bash
cd /dlis_model && ./run.sh http
```

#### 完整容器内启动流程（汇总）

```bash
# === 在容器内依次执行 ===

# 1. 复制自定义代码
cp /Model/model.py /dlis_model/model/model.py
cp /Model/dlis_inter.py /dlis_model/model/dlis_inter.py

# 2. 创建模型目录 symlink
ln -s /Model/model /dlis_model/Model

# 3. 创建 DLIS 验证标记
touch /dlis_model/Model/dlis_integration_validated

# 4. 创建优化目录
mkdir -p /dlis_model/Model/gemma4_opt
printf "llm" > /dlis_model/Model/gemma4_opt/opt_type.txt
cat > /dlis_model/Model/gemma4_opt/best_setting.json << 'EOF'
{
    "llm_type": "vllm",
    "model": "gemma-4-26B-A4B-it-AWQ-4bit",
    "quantization": "awq",
    "max_output_len": 256,
    "temperature": 0.8,
    "top_p": 0.95,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "trust_remote_code": true,
    "dtype": "auto",
    "max_model_len": 8192
}
EOF

# 5. Patch vllm_runner.py (移除不兼容参数)
sed -i '/num_scheduler_steps/d' /llm_opt/vllm/vllm_runner.py

# 6. 设置环境变量
export PYTHONPATH=/:$PYTHONPATH
export _ListeningPort_=8886
export AB_MAX_SEQ_LEN=16
export AB_INSTANCE_GROUP_COUNT=1
export AB_ENTRYPOINT=/v2/models/gemma4/versions/1/infer

# 7. 启动服务
cd /dlis_model && ./run.sh http
```

#### 测试请求
```bash
curl -X POST http://localhost:8886/ \
  -H "Content-Type: text/plain" \
  -d '{
    "landing_page_content": "Welcome to TrailMaster Outdoor Gear. Premium hiking boots, ultralight backpacks, and camping essentials for your next adventure.",
    "url": "https://trailmaster.example.com",
    "num_prompts": 5
  }'
```

#### 问题 9: `LLM.generate()` 参数签名不兼容
vLLM 0.19.0 的 `LLM.generate()` 不接受 `prompt_ids` 作为位置参数。

**修复**: Patch `/llm_opt/vllm/vllm_runner.py`
```bash
sed -i 's/outputs = self.engine.generate(prompts, self.sampling_params, prompt_ids)/outputs = self.engine.generate(prompts, sampling_params=self.sampling_params)/' /llm_opt/vllm/vllm_runner.py
```

#### 问题 10: `run.sh` 内 `unset CUDA_VISIBLE_DEVICES`
**修复**:
```bash
sed -i 's/^unset CUDA_VISIBLE_DEVICES/#unset CUDA_VISIBLE_DEVICES/' /dlis_model/run.sh
```

#### 问题 11: vLLM runner 返回嵌套 list
`runner.run()` 返回 `[[text], [text], ...]`，`dlis_inter.py` 需要递归展开。

**修复**: 用 `while isinstance(x, list)` 递归展开 step1_output 和 step2_outputs。

#### 里程碑: DLIS 服务首次成功返回结果 ✅

首次 curl 测试返回 `Status: Success`，5 个 scenes + 5 个 prompts。

**已知质量问题**:
- 模型输出包含 `thought` 前缀（Gemma4 thinking 模式），部分 prompt 质量差
- Prompt 1 解析为空，Prompt 3 陷入重复循环
- 需要在 vLLM sampling params 中设置合适的 stop tokens 或增加 `max_output_len`

**完整容器内启动流程（更新版）**

```bash
# === 在容器内依次执行 ===

# 1. 复制自定义代码
cp /Model/model.py /dlis_model/model/model.py
cp /Model/dlis_inter.py /dlis_model/model/dlis_inter.py

# 2. 创建模型目录 symlink
ln -s /Model/model /dlis_model/Model

# 3. 创建 DLIS 验证标记
touch /dlis_model/Model/dlis_integration_validated

# 4. 创建优化目录
mkdir -p /dlis_model/Model/gemma4_opt
printf "llm" > /dlis_model/Model/gemma4_opt/opt_type.txt
cat > /dlis_model/Model/gemma4_opt/best_setting.json << 'EOF'
{
    "llm_type": "vllm",
    "model": "gemma-4-26B-A4B-it-AWQ-4bit",
    "quantization": "awq",
    "max_output_len": 256,
    "temperature": 0.8,
    "top_p": 0.95,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "trust_remote_code": true,
    "dtype": "auto",
    "max_model_len": 8192
}
EOF

# 5. Patch vllm_runner.py
sed -i '/num_scheduler_steps/d' /llm_opt/vllm/vllm_runner.py
python3 -c "
with open('/llm_opt/vllm/vllm_runner.py', 'r') as f:
    lines = f.readlines()
new_lines = []
for i, line in enumerate(lines):
    if line.strip() == '),' and new_lines and new_lines[-1].strip() == '),':
        continue
    new_lines.append(line)
with open('/llm_opt/vllm/vllm_runner.py', 'w') as f:
    f.writelines(new_lines)
"
sed -i 's/outputs = self.engine.generate(prompts, self.sampling_params, prompt_ids)/outputs = self.engine.generate(prompts, sampling_params=self.sampling_params)/' /llm_opt/vllm/vllm_runner.py

# 6. Patch run.sh
sed -i 's/^unset CUDA_VISIBLE_DEVICES/#unset CUDA_VISIBLE_DEVICES/' /dlis_model/run.sh

# 7. 升级 transformers
python3 -m pip install "transformers>=5.5.0"

# 8. 设置环境变量
export PYTHONPATH=/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
export _ListeningPort_=8886
export AB_MAX_SEQ_LEN=16
export AB_INSTANCE_GROUP_COUNT=1
export AB_ENTRYPOINT=/v2/models/gemma4/versions/1/infer

# 9. 启动服务
cd /dlis_model && ./run.sh http
```

### DLIS 容器重建 Debug 记录（2026-04-15 晚）

#### 背景

A6000 机器 (BR1T45-S1-17) 重启后，昨天的容器（`--rm` 参数）被清除。
重新用 `latest` 镜像启动后发现 vLLM 版本变为 0.10.0（昨天是 0.19.0），`latest` tag 可能被别人更新。

#### 环境对比

| 组件 | 昨天的 `latest` 容器 | 今天的 `latest` 容器 | A100 conda 环境 (gemma4-vllm) |
|------|---------------------|---------------------|-------------------------------|
| vLLM | 0.19.0 | 0.10.0 | 0.19.0 |
| transformers | 5.5.x (升级后) | 5.3.0 (默认) | 5.5.3 |
| torch | ? (cu130?) | 2.7.1+cu126 | 2.10.0+cu128 |
| Python | 3.12 | 3.12 | 3.10 |

#### A100 上的安装流程（已验证可用）

```bash
conda create -n gemma4-vllm python=3.10 -y
conda init bash && exec bash
conda activate gemma4-vllm
pip install vllm                    # 自动装 torch + vllm 0.19.0
pip install "transformers>=5.5.0"   # 升级到 5.5.3
```

最终版本：torch 2.10.0+cu128, vllm 0.19.0, transformers 5.5.3

#### 尝试过的修复方案

| # | 方案 | 结果 |
|---|------|------|
| 1 | `latest` 容器 + `pip3 install "transformers>=5.5.0"` | `rope_scaling should have a 'rope_type' key` — vLLM 0.10.0 不支持 gemma4 |
| 2 | `latest` 容器 + `pip3 install "vllm==0.19.0" --no-deps` | `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation` — vLLM 0.19.0 (cu130) 与容器 PyTorch 2.7.1+cu126 不兼容 |
| 3 | `20260228-0834-merge` 镜像 (vLLM 0.13.0) | 别人的容器，非 jinjinchen 创建 |

#### 根本原因

- vLLM PyPI wheel 是 cu130 编译的，与容器内 PyTorch cu126 的 CUDA symbols 不兼容
- vLLM 0.10.0 太旧，不认识 gemma4 的 rope_scaling 配置
- 需要升级 PyTorch 到 cu128/cu130 以匹配 vLLM 0.19.0，或在容器内重建完整环境

#### 解决方案

基于 `latest` 镜像创建持久化容器 `gemma4-dlis`（不带 `--rm`），升级核心包复现 A100 环境：

```bash
# 创建持久化容器
docker run -d --name gemma4-dlis --runtime nvidia --gpus all \
  -p 8886:8886 \
  -v /home/jinjinchen/models/gemma-4-26B-A4B-it-AWQ-4bit:/Model/model \
  dlisproddockerrepo.azurecr.io/dlis/llm_framework_vllm:latest \
  sleep infinity

# 升级核心包
docker exec gemma4-dlis pip3 install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
docker exec gemma4-dlis pip3 install vllm==0.19.0 --no-deps
docker exec gemma4-dlis pip3 install "transformers>=5.5.0"
docker exec gemma4-dlis pip3 install -U huggingface_hub
```

最终版本：torch 2.10.0+cu128, vllm 0.19.0, transformers 5.5.3

#### DLIS 代码修复（容器内 patch）

**问题 1: vLLM runner 返回嵌套 list**

`oaas_wrapper.run()` 返回 `[[text]]`，`dlis_inter.py` 的旧版代码只做了 `step1_output[0]`（取到的还是 list）。

修复：`build_step2_prompts` 和 `postprocess` 都改为 `while isinstance(x, list)` 递归展开。

**问题 2: Gemma4 thinking 模式导致输出退化**

`_format_gemma_chat` fallback 模板没有关闭 thinking 模式，模型先输出大段推理分析再生成结构化内容，导致：
- Step 1: thinking 内容里出现 `</Scene5>` 误触发 stop token
- Step 2: 部分 prompt 出现数字/文字重复退化

修复：在 `_format_gemma_chat` 末尾加上 `<|channel>thought\n<channel|>` 跳过 thinking：

```python
parts.append("<start_of_turn>model\n<|channel>thought\n<channel|>")
```

**问题 3: 缺少 stop tokens**

vllm_runner 的 `SamplingParams` 没有 stop 参数，模型生成 `</Prompt>` 后继续输出直到 max_tokens，导致重复退化。

修复：patch `/llm_opt/vllm/vllm_runner.py`，在 `_create_sampling_params` 中加入：
```python
"stop": ["</Prompt>", "</Scene5>", "<end_of_turn>"],
```

#### A6000 DLIS 性能测试结果

**配置**：
- 模型：gemma-4-26B-A4B-it-AWQ-4bit (AWQ 4-bit)
- GPU：NVIDIA RTX A6000 (48GB)
- best_setting.json：max_output_len=256, temperature=0.8, top_p=0.95
- stop tokens：`</Prompt>`, `</Scene5>`, `<end_of_turn>`
- thinking 模式：已关闭（`<|channel>thought\n<channel|>`）

**单请求耗时**：

| 阶段 | 耗时 |
|------|------|
| preprocess | 0.000s |
| Step 1 推理 | 0.896s |
| build_step2 | 0.001s |
| Step 2 推理 (batch=5) | 0.818s |
| postprocess | 0.000s |
| **总计** | **1.715s** |

**与 A100 对比**：

| 指标 | A100 (80GB) | A6000 (48GB) | 倍率 |
|------|-------------|--------------|------|
| 单请求总耗时 | ~0.35s (warmup 后) | ~1.72s | ~5x |
| 显存带宽 | ~2 TB/s | ~768 GB/s | 2.6x |

A6000 慢于 A100 主要因为显存带宽差距（LLM decode 是 memory-bandwidth bound）。

**输出质量**：5/5 prompt 格式正确，内容多样，无退化，`format_compliant=true`。

#### 完整容器内启动流程（A6000 更新版）

```bash
# === 在容器内依次执行 ===

# 1. 复制自定义代码
cp /Model/model.py /dlis_model/model/model.py
cp /Model/dlis_inter.py /dlis_model/model/dlis_inter.py

# 2. 创建模型目录 symlink
ln -s /Model/model /dlis_model/Model

# 3. 创建 DLIS 验证标记
touch /dlis_model/Model/dlis_integration_validated

# 4. 创建优化目录
mkdir -p /dlis_model/Model/gemma4_opt
printf "llm" > /dlis_model/Model/gemma4_opt/opt_type.txt
cat > /dlis_model/Model/gemma4_opt/best_setting.json << 'EOF'
{
    "llm_type": "vllm",
    "model": "gemma-4-26B-A4B-it-AWQ-4bit",
    "quantization": "awq",
    "max_output_len": 256,
    "temperature": 0.8,
    "top_p": 0.95,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "trust_remote_code": true,
    "dtype": "auto",
    "max_model_len": 8192
}
EOF

# 5. Patch vllm_runner.py
sed -i '/num_scheduler_steps/d' /llm_opt/vllm/vllm_runner.py
python3 -c "
with open('/llm_opt/vllm/vllm_runner.py', 'r') as f:
    lines = f.readlines()
new_lines = []
for i, line in enumerate(lines):
    if line.strip() == '),' and new_lines and new_lines[-1].strip() == '),':
        continue
    new_lines.append(line)
with open('/llm_opt/vllm/vllm_runner.py', 'w') as f:
    f.writelines(new_lines)
"
sed -i 's/outputs = self.engine.generate(prompts, self.sampling_params, prompt_ids)/outputs = self.engine.generate(prompts, sampling_params=self.sampling_params)/' /llm_opt/vllm/vllm_runner.py

# 5b. Patch vllm_runner.py: 添加 stop tokens
python3 -c "
p = '/llm_opt/vllm/vllm_runner.py'
t = open(p).read()
old = '\"max_tokens\": self.config.max_output_len,\n        }\n\n        self.sampling_params = SamplingParams(**params)'
new = '\"max_tokens\": self.config.max_output_len,\n            \"stop\": [\"</Prompt>\", \"</Scene5>\", \"<end_of_turn>\"],\n        }\n\n        self.sampling_params = SamplingParams(**params)'
open(p,'w').write(t.replace(old, new))
print('OK')
"

# 5c. Patch dlis_inter.py: 递归展开嵌套 list + 关闭 thinking
python3 -c "
p = '/dlis_model/model/dlis_inter.py'
t = open(p).read()
# 递归展开 step1_output
old = '''        # Handle both single string and list inputs
        if isinstance(step1_output, list):
            step1_text = step1_output[0] if step1_output else \"\"
        else:
            step1_text = step1_output'''
new = '''        # Handle nested list from vLLM runner: [[text], [text], ...]
        step1_text = step1_output
        while isinstance(step1_text, list):
            step1_text = step1_text[0] if step1_text else \"\"'''
t = t.replace(old, new)
# 关闭 thinking 模式
t = t.replace(
    '<start_of_turn>model\\\n\")',
    '<start_of_turn>model\\\n<|channel>thought\\\n<channel|>\")'
)
open(p,'w').write(t)
print('OK')
"

# 6. Patch run.sh
sed -i 's/^unset CUDA_VISIBLE_DEVICES/#unset CUDA_VISIBLE_DEVICES/' /dlis_model/run.sh

# 7. 升级 transformers (容器内已通过 pip3 升级，此步可跳过)
# python3 -m pip install "transformers>=5.5.0"

# 8. 设置环境变量
export PYTHONPATH=/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export _ListeningPort_=8886
export AB_MAX_SEQ_LEN=16
export AB_INSTANCE_GROUP_COUNT=1
export AB_ENTRYPOINT=/v2/models/gemma4/versions/1/infer

# 9. 启动服务
cd /dlis_model && ./run.sh http
```

#### 容器持久化

为防止容器丢失，建议 commit 为自定义镜像：
```bash
docker commit gemma4-dlis gemma4-dlis:v1
```

### DLIS 部署方案迁移：OaaS_LLMTemplate 分支方式（2026-04-16）

#### 背景

之前手动在容器内 patch 的方式非常脆弱：
- 容器重启就丢失所有修改
- `latest` 镜像版本不可控（被别人更新后 vLLM 版本变了）
- 手动 patch 步骤多，难以复现

#### 新方案

基于 `OaaS_LLMTemplate` 仓库（`C:\Users\jinjinchen\OneDrive - Microsoft\OaaS_LLMTemplate`）创建分支 `jinjinchen/Gemma4-v1`，将所有自定义修改写入代码，利用仓库自带的 CI/CD pipeline 自动构建 Docker 镜像并上传。

**优势**：
- 代码变更可追踪（git 管理）
- PR 或 merge 自动触发 `pipeline/build_vllm_image.sh` 构建镜像
- 非 main 分支镜像 tag 格式：`YYYYMMDD-HHMM-jinjinchen-Gemma4-v1`
- 镜像推送到 `dlisproddockerrepo.azurecr.io/dlis/llm_framework_vllm:<tag>`
- 本地可先手动 `docker build` 测试

#### 修改清单（7 个文件）

| # | 文件 | 修改内容 |
|---|------|----------|
| 1 | `pipeline/Dockerfile_vllm_0.10.0` | `vllm==0.10.0` → `vllm==0.19.0`，添加 `transformers>=5.5.0` |
| 2 | `llm_opt/vllm/vllm_runner.py` | (a) 删除 `num_scheduler_steps` 引擎参数 (b) `_create_sampling_params` 添加 stop tokens 支持 (c) `engine.generate()` 改为关键字参数 `sampling_params=` |
| 3 | `llm_opt/vllm/vllm_util.py` | `VLLMConfig` dataclass 添加 `stop: list[str] \| None = None` 字段 |
| 4 | `dlis_model/model/model.py` | 替换为 two-step 版本：`preprocess → run(step1) → build_step2 → run(step2) → postprocess` |
| 5 | `dlis_model/model/dlis_inter.py` | **新增** — Gemma4 `PreAndPostProcessor`（系统 prompt、场景解析、thinking 关闭、嵌套 list 展开） |
| 6 | `dlis_model/run.sh` | 注释 `unset CUDA_VISIBLE_DEVICES` → `#unset CUDA_VISIBLE_DEVICES` |
| 7 | `requirements-vllm.txt` | 添加 `transformers>=5.5.0`、`huggingface_hub>=0.20.0` |

#### 各修改对应的原始 patch 来源

| 修改 | 对应的容器内 patch |
|------|-------------------|
| Dockerfile vLLM 0.19.0 | `docker exec gemma4-dlis pip3 install vllm==0.19.0` |
| Dockerfile transformers | `docker exec gemma4-dlis pip3 install "transformers>=5.5.0"` |
| 删 num_scheduler_steps | `sed -i '/num_scheduler_steps/d' /llm_opt/vllm/vllm_runner.py` |
| 添加 stop tokens | patch `_create_sampling_params` 加 `"stop": ["</Prompt>", "</Scene5>", "<end_of_turn>"]` |
| engine.generate 参数 | `sed -i 's/...prompt_ids)/...sampling_params=self.sampling_params)/'` |
| 注释 unset CUDA | `sed -i 's/^unset CUDA_VISIBLE_DEVICES/#unset CUDA_VISIBLE_DEVICES/' /dlis_model/run.sh` |
| model.py two-step | 来自 `Gemma4Deploy/model.py` |
| dlis_inter.py | 来自 `Gemma4Deploy/dlis_inter.py`（含 thinking 关闭 + 嵌套 list 修复） |

#### best_setting.json 配置（需在模型部署时放入 `<model_dir>/<model_name>_opt/` 目录）

```json
{
    "llm_type": "vllm",
    "model": "gemma-4-26B-A4B-it-AWQ-4bit",
    "quantization": "awq",
    "max_output_len": 256,
    "temperature": 0.8,
    "top_p": 0.95,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "trust_remote_code": true,
    "dtype": "auto",
    "max_model_len": 8192,
    "stop": ["</Prompt>", "</Scene5>", "<end_of_turn>"]
}
```

#### 下一步

- [ ] 本地 `docker build` 测试镜像是否能正常构建
- [ ] 在 A6000 上用本地构建的镜像启动服务验证
- [ ] 验证通过后 push 分支、提交 PR
- [ ] PR 触发 pipeline 自动构建并上传镜像到 ACR

---

## UHRS 人工标注对比：Gemma4 Two-Step vs One-Step（Random 200 LPs）

**日期**: 2026-04-13  
**数据来源**:
- Two-Step: `gemma4_random200_two_step_vllm_t2i_UHRS_Task_lp_relevance_labeling_0415.tsv`
- One-Step: `Gemma4_1Step_UHRS_Task_lp_relevance_labeling_0410.tsv`

**标注规则**: 每张图 3 个 UHRS judge，取 max-vote 作为最终标签（Good/Fair/Bad）

### 1. Image Level Good Rate

| | Two-Step | One-Step | Diff |
|---|---|---|---|
| Good | 755 (76.0%) | 749 (75.3%) | +0.7pp |
| Fair | 94 (9.5%) | 73 (7.3%) | +2.1pp |
| Bad | 145 (14.6%) | 173 (17.4%) | -2.8pp |
| Total | 994 | 995 | — |

### 2. LP Level N/5 Good Distribution

| N/5 | Two-Step (200 LPs) | One-Step (200 LPs) |
|---|---|---|
| 0/5 | 1 (0.5%) | 2 (1.0%) |
| 1/5 | 4 (2.0%) | 5 (2.5%) |
| 2/5 | 21 (10.5%) | 18 (9.0%) |
| 3/5 | 46 (23.0%) | 49 (24.5%) |
| 4/5 | 69 (34.5%) | 69 (34.5%) |
| 5/5 | 59 (29.5%) | 57 (28.5%) |

### 3. LP Level Cumulative Good Rate

| Threshold | Two-Step | One-Step | Diff |
|---|---|---|---|
| >= 3/5 | 87.0% | 87.5% | -0.5pp |
| >= 4/5 | 64.0% | 63.0% | +1.0pp |
| >= 5/5 | 29.5% | 28.5% | +1.0pp |

### 4. 不完整 LP（图片数 < 5）

- **Two-Step**: LP 165, 166, 195（各 3 张图）
- **One-Step**: LP 119（2 张图）, LP 95（3 张图）

### 5. 结论

Two-Step 与 One-Step 在 UHRS 人工标注质量上几乎一致：
- Image Level Good Rate: 76.0% vs 75.3%（+0.7pp）
- LP Level ≥4/5 Good: 64.0% vs 63.0%（+1.0pp）
- Two-Step Bad Rate 更低: 14.6% vs 17.4%（-2.8pp）

Two-Step 的优势主要体现在**推理速度**和**prompt 长度控制**：
- 速度: 84.9s vs 122.7s（-30.8%）

---

## 2026-04-16 Dockerfile 全量 Build 调试记录

### 目标

从手动 patch 容器迁移到 Dockerfile 全量 build，使 CI/CD pipeline 能自动构建可用的 DLIS 镜像。

分支：`jinjinchen/Gemma4-v1`（OaaS_LLMTemplate repo）

### 遇到的问题及解决方式

#### 问题 1：torch ABI 不匹配 — `vllm/_C.abi3.so: undefined symbol`

**现象**：容器里 `torch==2.7.0+cu126`，但 `vllm==0.19.0` 编译时用的是 `torch==2.10.0+cu128`，ABI 不兼容。

**根因**：`requirements-vllm.txt` 里的 `torchvision==0.22.0` 在 `pip install -r` 时把 torch 从 2.10.0 降级到了 2.7.0（torchvision 0.22.0 依赖 torch==2.7.0）。

**解决**：
- Dockerfile 里在装 vllm 之前，先用 `--index-url cu128` 显式装 `torch==2.10.0 torchvision==0.25.0`
- `requirements-vllm.txt` 里移除 torch、torchvision、transformers（全由 Dockerfile 处理）

#### 问题 2：`torchvision::nms operator does not exist`

**现象**：旧的 `torchvision==0.22.0` 残留在容器里，和新的 `torch==2.10.0` 不兼容。

**解决**：卸载旧 torchvision，装 `torchvision==0.25.0 --index-url cu128`。

#### 问题 3：`Gemma4VideoProcessor requires the Torchvision library`

**现象**：卸载 torchvision 后 vllm 的 Gemma4 多模态处理器找不到 torchvision。

**解决**：装回 `torchvision==0.25.0`（匹配 torch 2.10.0）。

#### 问题 4：`DLIS integration not validated`

**现象**：`RuntimeError: ⚠️ DLIS integration not validated`

**根因**：`dlis_integration_validated` 文件路径错误。代码在 `oaas_wrapper_v2.py` 里检查 `os.path.join(model_folder, "dlis_integration_validated")`，实际路径应该是 `/dlis_model/Model/dlis_integration_validated`。

**解决**：`touch /dlis_model/Model/dlis_integration_validated`

#### 问题 5：`Unknown opt type vllm` / `Unknown opt type llm\n`

**现象**：`opt_type.txt` 内容不对或有换行符。

**解决**：用 `printf "llm"` 而非 `echo "llm"` 写入（避免末尾换行符）。

#### 问题 6：`best_setting.json` 缺少必要字段

**现象**：`Failed to create runner: 'llm_type'`

**根因**：`best_setting.json` 只写了 `tensor_parallel_size` 和 `gpu_memory_utilization`，缺少 `llm_type`、`model` 等必填字段。

**解决**：使用完整的 `best_setting.json`（见下方配置）。

#### 问题 7：`Invalid repository ID or local directory specified: 'Model'`

**现象**：不管 `best_setting.json` 里 model 字段怎么改，vllm 始终收到 `model='Model'`。

**根因**：`QUANTIZATION_MAP`（`vllm_util.py`）里没有 `"awq"` 这个 key（只有 `"int4_awq"`），导致 `best_setting.json` 里的 `"quantization": "awq"` 被映射为 `None`。此时代码走 `else` 分支：`model_path = self.root_model_path = "Model"`，而 `Model/` 目录下没有 `config.json`（挂载路径不对）。

**解决**：
- 不在 `best_setting.json` 里设 `quantization` 字段（vllm 会从模型的 `config.json` 自动检测 AWQ）
- 或者保持原有 `QUANTIZATION_MAP` 不变，让 quantization 为 None 走 else 分支
- 关键是**模型挂载路径必须正确**：挂载到 `/Model/model`，然后软链接 `ln -s /Model/model /dlis_model/Model`，使 `/dlis_model/Model/` 直接包含 `config.json` 等模型文件

#### 问题 8：模型挂载路径错误

**现象**：`/dlis_model/Model/config.json` 不存在。

**根因**：docker run 时挂载到了 `/dlis_model/Model/model`（多了一层），或挂载到 `/dlis_model/Model`（被 Dockerfile 里已有目录覆盖）。宿主机路径也写错了（`/home/jinjinchen/models/` → 实际在 `/home/jinjinchen/data/`）。

**解决**：
- 宿主机路径：`/home/jinjinchen/data/gemma-4-26B-A4B-it-AWQ-4bit`
- 挂载到：`/Model/model`
- 容器内：`ln -s /Model/model /dlis_model/Model`

### 最终成功的完整测试流程

```bash
# 1. Build 镜像（A6000 上）
cd /path/to/OaaS_LLMTemplate && git pull
export SOURCE_BRANCH="test"
sudo bash pipeline/build_vllm_image.sh

# 2. 创建测试容器
IMAGE_TAG="20260416-1226-test"  # 替换为实际 tag
sudo docker run -d --name gemma4-test-v3 --runtime nvidia --gpus all \
  -v /home/jinjinchen/data/gemma-4-26B-A4B-it-AWQ-4bit:/Model/model \
  my-vllm-final:$IMAGE_TAG sleep infinity

# 3. 配置并测试
sudo docker exec gemma4-test-v3 bash -c '
rm -rf /dlis_model/Model
ln -s /Model/model /dlis_model/Model
touch /dlis_model/Model/dlis_integration_validated
mkdir -p /dlis_model/Model/gemma4_opt
printf "llm" > /dlis_model/Model/gemma4_opt/opt_type.txt
cat > /dlis_model/Model/gemma4_opt/best_setting.json << EOF
{
    "llm_type": "vllm",
    "model": "gemma-4-26B-A4B-it-AWQ-4bit",
    "max_output_len": 256,
    "temperature": 0.8,
    "top_p": 0.95,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "trust_remote_code": true,
    "dtype": "auto",
    "max_model_len": 8192,
    "stop": ["</Prompt>", "</Scene5>", "<end_of_turn>"]
}
EOF
echo "{\"landing_page_content\": \"Welcome to TrailMaster.\", \"url\": \"https://example.com\", \"num_prompts\": 3, \"max_lp_chars\": 5000}" > /tmp/input.json
cd /dlis_model && ./run.sh offline /tmp/input.json /tmp/output.json'
```

### 测试结果

```
[TIMING] preprocess=0.000s  step1_infer=2.476s  build_step2=0.001s  step2_infer=0.614s  postprocess=0.000s  total=3.091s
```

两步推理成功，性能正常。

### 对 OaaS_LLMTemplate repo 的修改总结（分支 `jinjinchen/Gemma4-v1`）

| 文件 | 修改内容 | 原因 |
|------|---------|------|
| `pipeline/Dockerfile_vllm_0.10.0` | 在 `pip install vllm==0.19.0` 之前添加 `pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128` | 防止 vllm 从 PyPI 拉默认 torch（可能不带 CUDA 或版本不对） |
| `requirements-vllm.txt` | 只保留 `huggingface_hub==1.6.0`，移除 `torchvision==0.22.0` 和 `transformers==5.5.3` | torchvision 0.22.0 会降级 torch；transformers 已由 Dockerfile 处理 |
| `llm_opt/vllm/vllm_runner.py` | 添加 stop tokens 支持、移除 `num_scheduler_steps` | Gemma4 两步推理需要 stop tokens 控制生成 |
| `llm_opt/vllm/vllm_util.py` | `VLLMConfig` 添加 `stop` 字段 | 配合 vllm_runner.py 的 stop tokens 支持 |
| `llm_opt/__init__.py` | 创建空文件 | 修复 `ModuleNotFoundError: No module named 'llm_opt'` |
| `llm_opt/base/__init__.py` | 创建空文件 | 修复 `No module named 'llm_opt.base'` |
| `llm_opt/trtllm/__init__.py` | 创建空文件 | 修复模块导入错误 |
| `dlis_model/model/model.py` | Two-step inference ModelImp | Gemma4 两步推理流程 |
| `dlis_model/model/dlis_inter.py` | Two-step PreAndPostProcessor | 场景生成 → prompt 扩展 |

### 关键注意事项

1. **模型挂载路径**：必须挂载到 `/Model/model`，然后 `ln -s /Model/model /dlis_model/Model`。不要直接挂载到 `/dlis_model/Model`（会被 Dockerfile 已有目录覆盖）
2. **best_setting.json 不要设 quantization**：`QUANTIZATION_MAP` 没有 `"awq"` 映射，设了会导致模型路径拼接错误。vllm 会从 `config.json` 自动检测 AWQ
3. **opt_type.txt 用 `printf` 不要用 `echo`**：避免末尾换行符导致类型识别失败
4. **torch/torchvision 版本必须从 cu128 源安装**：PyPI 默认源可能给 CPU 版本
5. **`requirements-vllm.txt` 不要加 torch/torchvision/transformers**：这些包需要特殊的安装源或顺序，由 Dockerfile 统一管理
6. **CI pipeline 的 apt-get / PyPI 超时是网络问题**：重跑即可
- Prompt 长度: 42 words vs 127 words（-67%）
- 质量持平的前提下，Two-Step 是更优方案

---

## 改动记录：CI 清华源 + Kusto 日志 + 快速构建分支（2026-04-16）

### 改动一：CI Pipeline 网络超时修复 + num_scheduler_steps 恢复

**问题**：CI pipeline 构建镜像时频繁遇到 PyPI / apt-get 超时（SGLang、vLLM、TensorRT-LLM job 均受影响），导致构建失败。

**方案**：参考同事 siwenzhu 的 PR，在 CI 中加入清华 pip 镜像源。

**改动**：
| 文件 | 修改内容 |
|------|---------|
| `pipeline/azure-pipelines-unified.yml` | vLLM job 添加 `PIP_INDEX_URL` 和 `UV_INDEX_URL` 环境变量，指向 `https://pypi.tuna.tsinghua.edu.cn/simple` |
| `pipeline/build_vllm_image.sh` | `docker build` 添加 `--build-arg PIP_INDEX_URL/UV_INDEX_URL/PIP_EXTRA_INDEX_URL/UV_EXTRA_INDEX_URL` 等参数传入 Dockerfile；新增 `PIP_ARGS` 变量用于 `docker exec pip install` 命令 |
| `pipeline/Dockerfile_vllm_0.10.0` | 添加 `ARG PIP_INDEX_URL UV_INDEX_URL` 和 `ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL` 接收构建参数 |
| `llm_opt/vllm/vllm_runner.py` | 恢复之前误删的 `num_scheduler_steps` 配置解析和 engine kwargs 传入（两处） |

**关键教训**：
- 清华源参数要通过 `--build-arg` 传入 Dockerfile、通过 `PIP_ARGS` 变量传给 `docker exec pip install`，不能用 `docker exec -e` 环境变量方式
- `num_scheduler_steps` 被意外删除会影响 vLLM 调度性能，修改时要注意不要误删配置项

### 改动二：Kusto 日志集成

**目的**：给 Gemma4 DLIS 部署添加 Kusto 日志功能，实现生产环境请求监控。参考同事 siwenzhu 在 `users/siwenzhu/VLLM_MML_localbuild_pypi_kustoc_certificate` 分支的实现。

**方案**：保持 OaasWrapper 两步推理流程不变，只加日志基础设施。

**新建文件**：
| 文件 | 说明 |
|------|------|
| `dlis_model/model/config.py` | Kusto 日志配置，`application_name='Gemma4PromptGen'`，EventHub namespace、证书路径等 |
| `dlis_model/model/eventhub_sink.py` | EventHub 消息发送，使用 `CertificateCredential` 认证 |
| `dlis_model/model/kusto_log.py` | `KustoLogHandler`：info/warn/err/perf 四个 sink，deque 缓冲 + 定时批量上传 |

**修改文件**：
| 文件 | 修改内容 |
|------|---------|
| `dlis_model/model/utils.py` | 添加 `get_tracking_data()` 函数，生成请求追踪字段（requestid、trackingid 等） |
| `dlis_model/model/model.py` | 添加 `KustoLogHandler` + `BackgroundScheduler` 初始化；`Eval()` / `EvalBatch()` 中添加 tracking_data 解析和 timing 日志；`print()` → `logger.info()` |
| `requirements-vllm.txt` | 添加 `APScheduler`、`pydantic-settings`、`azure-eventhub`、`azure-identity` |

**日志架构**：
- `BackgroundScheduler` 每 0.5s 触发 `KustoLogHandler.send()`，从 deque 批量取消息发送到 EventHub
- 四个 EventHub topic：`appsvc_info`、`appsvc_warn`、`appsvc_err`、`appsvc_perf`
- 每条日志携带 requestid/trackingid/sessionid/customerid + duration 等字段

### 改动三：快速构建分支 `jinjinchen/Gemma4-v1-fast-build`

**问题**：当前 `Dockerfile_vllm_0.10.0` 从 `nvidia/cuda:12.8.1-devel-ubuntu22.04` 开始构建，编译 FlashInfer AOT 内核和 DeepGEMM 源码，整个 CI 构建非常慢。

**方案**：新建分支，使用 `vllm/vllm-openai:latest` 作为基础镜像，跳过所有编译步骤。

**改动**：
| 文件 | 修改内容 |
|------|---------|
| `pipeline/Dockerfile_vllm_fast`（新建） | `FROM vllm/vllm-openai:latest`，只安装 `huggingface_hub==1.6.0` |
| `pipeline/build_vllm_image.sh` | BLOCK 2 改用 `Dockerfile_vllm_fast` 替代 `Dockerfile_vllm_0.10.0` |

**注意事项**：
- `vllm/vllm-openai:latest` 自带 vllm/torch/transformers，版本可能与之前 pin 的不同，需 CI 验证兼容性
- FlashInfer AOT 和 DeepGEMM 被跳过，可能影响推理性能但功能可用
- 如版本不匹配可改为 `vllm/vllm-openai:v0.19.0` 等具体 tag

---

## 本地测试记录：fast-build 分支验证（2026-04-17）

### 测试环境
- 机器：`BR1T45-S1-17`
- 模型：`~/data/gemma-4-26B-A4B-it-AWQ-4bit/`
- 分支：`jinjinchen/Gemma4-v1-fast-build`

### 测试流程与命令

#### Step 1: 构建基础镜像

```bash
cd ~/0417_test/OaaS_LLMTemplate
git checkout jinjinchen/Gemma4-v1-fast-build
IMAGE_TAG="fast-build-test"

sudo docker build \
    -t my-vllm-base:$IMAGE_TAG \
    --file pipeline/Dockerfile_vllm_fast \
    pipeline/
```

**结果**：构建仅耗时 **0.8 秒**（对比原 Dockerfile_vllm_0.10.0 需要几十分钟），因为基础镜像 `vllm/vllm-openai:latest` 已包含所有编译好的包。

#### Step 2: 创建最终镜像（模拟 CI BLOCK 3 流程）

```bash
sudo docker rm -f temp-oaas-container 2>/dev/null || true
sudo docker run -d --name temp-oaas-container my-vllm-base:$IMAGE_TAG sleep infinity
sudo docker cp . temp-oaas-container:/
sudo docker exec temp-oaas-container chmod +x /dlis_model/run.sh
sudo docker exec temp-oaas-container chmod +x /dlis_model/async_run.sh
sudo docker exec temp-oaas-container chmod +x /LLMModelOptimization.sh
sudo docker exec temp-oaas-container python3 -m pip install -r /requirements-common.txt
sudo docker exec temp-oaas-container python3 -m pip install -r /requirements-vllm.txt
sudo docker exec temp-oaas-container python3 -m pip install -e /
sudo docker commit temp-oaas-container gemma4-fast:$IMAGE_TAG
sudo docker rm -f temp-oaas-container
```

#### Step 3: 验证包版本

```bash
sudo docker run --rm --entrypoint python3 gemma4-fast:$IMAGE_TAG -c "
import vllm; print(f'vllm: {vllm.__version__}')
import torch; print(f'torch: {torch.__version__}')
import transformers; print(f'transformers: {transformers.__version__}')
import huggingface_hub; print(f'huggingface_hub: {huggingface_hub.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

**输出**：
```
vllm: 0.19.0
torch: 2.10.0+cu129
transformers: 4.57.6
huggingface_hub: 0.36.2
CUDA available: False  # 因为验证命令未挂载 GPU
```

#### Step 4: 挂载模型测试推理

```bash
sudo docker run --rm -it --runtime nvidia --gpus all \
    --entrypoint bash \
    -v ~/data/gemma-4-26B-A4B-it-AWQ-4bit:/vllm-workspace/Model \
    gemma4-fast:$IMAGE_TAG \
    -c "
        printf 'vllm' > /dlis_model/opt_type.txt
        python3 -c \"
import sys; sys.path.insert(0, '/dlis_model/model')
from llm_opt.oaas_wrapper_v2 import OaasWrapper
w = OaasWrapper('Model', is_llm_model=True)
out = w.run(['Hello, who are you?'])
print('Output:', out)
\"
    "
```

**注意**：`vllm/vllm-openai` 镜像的默认 entrypoint 是 `vllm serve`，必须用 `--entrypoint bash` 覆盖，否则 bash 命令会被当作 vllm 参数解析报错。

### 遇到的问题与修复

#### 问题 1: `huggingface_hub==1.6.0` 版本冲突

**报错**：
```
ImportError: huggingface-hub>=0.34.0,<1.0 is required for a normal functioning
of this module, but found huggingface-hub==1.6.0.
```

**原因**：`Dockerfile_vllm_fast` 和 `requirements-vllm.txt` 中 pin 了 `huggingface_hub==1.6.0`（版本 >= 1.0），但基础镜像自带的 `transformers` 要求 `<1.0`。

**修复**：移除 `huggingface_hub==1.6.0` 的 pin，使用基础镜像自带的 0.36.2 版本。

#### 问题 2: `num_scheduler_steps` 不再被 EngineArgs 接受

**报错**：
```
Failed to load sync VLLM engine: EngineArgs.__init__() got an unexpected
keyword argument 'num_scheduler_steps'
```

**原因**：`vllm/vllm-openai:latest` 的 vllm 0.19.0 将 `num_scheduler_steps` 从 `EngineArgs` 中移除（改为其他配置方式）。

**修复**：在 fast-build 分支上从 `vllm_runner.py`（配置解析 + engine kwargs）和 `vllm_util.py`（VLLMConfig + DefaultSettings）中移除 `num_scheduler_steps` 相关代码。

**注意**：此修改仅在 `jinjinchen/Gemma4-v1-fast-build` 分支，`jinjinchen/Gemma4-v1` 分支（使用 Dockerfile_vllm_0.10.0 自行安装 vllm）仍保留 `num_scheduler_steps`。

#### 问题 3: Docker 权限和 Entrypoint

- 所有 `docker` 命令需要加 `sudo`
- `vllm/vllm-openai` 的 entrypoint 是 `vllm serve`，运行 bash 命令必须 `--entrypoint bash`
- 模型需要挂载到 `/vllm-workspace/Model`（OaasWrapper 使用相对路径 `"Model"`，而镜像工作目录是 `/vllm-workspace`）

### 当前状态

- 基础镜像构建：✅ 通过（0.8s）
- 包版本兼容性：✅ vllm 0.19.0 + torch 2.10.0+cu129 + transformers 4.57.6
- 推理测试：⏳ 待 GPU 环境验证

---

## PR CI 构建验证 + 本地测试修复总结（2026-04-17）

### PR CI 构建结果

提交 `bc9abfc`（`jinjinchen/Gemma4-v1-fast-build` 分支）触发 CI 自动构建 vLLM 镜像，使用 `Dockerfile_vllm_fast`。

**构建日志关键信息**：
- 基础镜像：`vllm/vllm-openai:latest@sha256:d9a5c1c1614c...`
- `Dockerfile_vllm_fast` 加载正常（592B）
- 构建速度极快（跳过了 FlashInfer AOT 和 DeepGEMM 源码编译）

**结论**：CI pipeline 使用 `Dockerfile_vllm_fast` 构建成功 ✅

### 本地测试遇到的额外问题与修复

#### 问题 4: `transformers 4.57.6` 不识别 Gemma4 架构

**报错**：模型加载时无法识别 `gemma4` 架构类型。

**原因**：`vllm/vllm-openai:latest` 基础镜像自带 `transformers==4.57.6`，该版本尚未支持 Gemma4。

**修复**：在 `Dockerfile_vllm_fast` 中添加 `RUN python3 -m pip install transformers==5.5.3`（本地测试验证 5.5.3 可用）。

#### 问题 5: Pydantic v2 要求 BaseSettings 字段必须有类型注解

**报错**：
```
PydanticUserError: A non-annotated attribute was detected: `eventhub_namespace = '...'`.
All model fields require a type annotation.
```

**原因**：基础镜像的 pydantic v2 严格要求所有 `BaseSettings` 字段必须有类型注解，而 `config.py` 中 4 个字段缺少 `str` 注解。

**修复**：为 `eventhub_namespace`、`client_id`、`tenant_id`、`corp_tenant_id` 添加 `: str` 类型注解。

#### 问题 6: 模型路径需要双重 symlink

**现象**：`OaasWrapper("Model")` 查找模型时，路径解析为 `{cwd}/Model`。

**修复**：
- 模型挂载到 `/vllm-workspace/Model`（容器工作目录）
- `ln -s /vllm-workspace/Model /Model`（供 config.py 证书路径使用）
- `ln -s /vllm-workspace/Model /dlis_model/Model`（供 http_server.py 从 `/dlis_model` 目录启动时使用）

#### 问题 7: GPU OOM

**现象**：GPU 0 显存不足。

**修复**：改用 GPU 1 启动：`--gpus '"device=1"'`

### 提交记录

| 提交 | 仓库 | 内容 |
|------|------|------|
| `bc9abfc` | OaaS_LLMTemplate | config.py 类型注解 + Dockerfile transformers==5.5.3 |
| `59152de` | ms-image-quality-filters | debug log 本地测试记录 |

### 当前状态

- CI 构建：✅ 通过
- 本地镜像构建：✅ 通过（0.8s）
- 包兼容性修复：✅ transformers==5.5.3, pydantic 类型注解
- http_server.py 启动：✅ 成功
- 端到端推理：✅ 通过

---

## 端到端推理验证结果（2026-04-17）

### 测试环境

- **机器**：BR1T45-S1-17，GPU 1
- **镜像**：`gemma4-fast:fast-build-test`（基于 `vllm/vllm-openai:latest`）
- **模型**：`gemma-4-26B-A4B-it-AWQ-4bit`，挂载到 `/vllm-workspace/Model`
- **关键版本**：vllm 0.19.0 / torch 2.10.0+cu129 / transformers 5.5.3
- **容器启动手动修复**：symlink（`/Model`, `/dlis_model/Model`）、sed 修复 config.py 类型注解

### 启动命令

```bash
sudo docker run -it --gpus '"device=1"' \
  -v /home/jinjinchen/data/gemma-4-26B-A4B-it-AWQ-4bit:/vllm-workspace/Model \
  -p 8889:8888 --entrypoint bash --name gemma4-fast-test \
  gemma4-fast:fast-build-test
```

容器内修复：
```bash
ln -s /vllm-workspace/Model /Model
ln -s /vllm-workspace/Model /dlis_model/Model
pip install transformers==5.5.3
sed -i 's/eventhub_namespace="/eventhub_namespace: str = "/' /dlis_model/model/config.py
sed -i 's/client_id="/client_id: str = "/' /dlis_model/model/config.py
sed -i 's/tenant_id="/tenant_id: str = "/' /dlis_model/model/config.py
sed -i 's/corp_tenant_id="/corp_tenant_id: str = "/' /dlis_model/model/config.py
cd /dlis_model && python3 model/http_server.py
```

### 测试请求

```bash
curl -X POST http://localhost:8889/ \
  -H "Content-Type: application/json" \
  -d '{"landing_page_content": "Welcome to TrailMaster Outdoor Gear. Premium hiking boots, ultralight backpacks, and camping essentials for your next adventure. Free shipping on orders over $99.", "url": "https://trailmaster.example.com", "num_prompts": 5}'
```

**注意**：路由是根路径 `/`，不是 `/score`（tornado 注册的是 `r"/"`）。

### 测试结果

- **Status**: Success，`format_compliant: true`
- **两步推理正常**：scene generation → prompt expansion
- **生成 5 个 prompt**，内容质量合理（hiking boots、backpack、camping gear 相关场景）

### 性能数据

| 阶段 | 耗时 |
|------|------|
| preprocess | 0.000s |
| step1_infer（scene generation，1 条） | 2.705s |
| build_step2 | 0.001s |
| step2_infer（prompt expansion，5 条） | 0.765s |
| postprocess | 0.000s |
| **总计** | **3.471s** |

- Step 1 吞吐：input 109.02 toks/s，output 32.11 toks/s
- Step 2 吞吐：input 1636.76 toks/s，output 293.87 toks/s（5 条并行，batch 效率高）

### 结论

Fast-build 分支（`jinjinchen/Gemma4-v1-fast-build`）使用 `vllm/vllm-openai:latest` 基础镜像，端到端推理功能完全正常。相比原 `Dockerfile_vllm_0.10.0` 构建时间从数十分钟降至秒级，推理功能无损。

**待办**：
- 将手动修复（symlink、config.py 注解）固化到 Dockerfile 和代码中（已提交 `bc9abfc`）
- ~~触发 CI 重新构建镜像，验证完整自动化流程~~ ✅ 已完成
- 部署到 DLIS 进行线上验证

---

## 2026-04-17 Docker 镜像构建优化总结

### 背景

DLIS 部署 Gemma4 模型需要构建包含 vLLM 推理框架的 Docker 镜像。原方案基于 `nvidia/cuda:12.8.1-devel-ubuntu22.04` 基础镜像，从源码编译 FlashInfer AOT 内核和 DeepGEMM，构建过程极其缓慢。

### 优化方案

| | 旧方案 | 新方案 |
|---|---|---|
| **基础镜像** | `nvidia/cuda:12.8.1-devel-ubuntu22.04` | `vllm/vllm-openai:latest` |
| **Dockerfile** | `Dockerfile_vllm_0.10.0` | `Dockerfile_vllm_fast` |
| **构建内容** | 从源码编译 torch、vllm、FlashInfer AOT、DeepGEMM | 仅安装 `transformers==5.5.3`，创建 python symlink |
| **构建时间** | ~2 小时 | ~8 分钟 |
| **分支** | `jinjinchen/Gemma4-v1` | `jinjinchen/Gemma4-v1-fast-build` |

### 关键改动

1. **新建 `pipeline/Dockerfile_vllm_fast`**：基于 `vllm/vllm-openai:latest`，跳过所有源码编译，只做最小必要配置
2. **修改 `pipeline/build_vllm_image.sh`**：BLOCK 2 切换到 `Dockerfile_vllm_fast`
3. **`ENTRYPOINT []` 重置**（commit `b838ca2`）：`vllm/vllm-openai:latest` 自带 ENTRYPOINT 会拦截 DLIS 的 CMD，必须显式重置
4. **安全修复**（commit `0145bca`）：对 CSV 日志中的用户可控 tracking 字段添加清洗，防止日志注入

### 踩坑记录

- **ENTRYPOINT 覆盖问题**：`vllm/vllm-openai:latest` 的 ENTRYPOINT 会将 DLIS 传入的 CMD（`./dlis_model/run.sh http`）作为 vllm CLI 参数执行，导致容器启动后跑的是 vllm serve 而非 Tornado 服务器。`docker commit` 会保留基础镜像的 ENTRYPOINT，所以必须在 Dockerfile 中显式设置 `ENTRYPOINT []`
- **transformers 版本不兼容**：`vllm/vllm-openai:latest` 自带的 transformers 版本不识别 gemma4 架构，需要升级到 `transformers==5.5.3`
- **python symlink 缺失**：基础镜像只有 `python3`，DLIS 框架调用 `python`，需要创建 symlink

### 结论

通过切换到预构建的 `vllm/vllm-openai:latest` 基础镜像，**CI 构建时间从 2 小时降低到 8 分钟**，提升约 15 倍。推理功能经本地容器验证无损。

---

## 2026-04-18 DLIS 部署调试 — OaasWrapper 模型加载路径问题

### 背景

镜像构建成功后部署到 DLIS，容器持续 crash 重启。日志显示模型加载走了 BaseLLM（transformers）fallback 路径，16GB AWQ 模型被 `AutoModelForCausalLM.from_pretrained().to("cuda")` 加载时触发 OOM Killed。

### 容器启动完整调用链

```
run.sh http
  → python3 model/main.py http
    → ModelImp.__init__()
      → PreAndPostProcessor()                    # 从 /Model 导入 dlis_inter
      → OaasWrapper("Model/gemma-4-26B-A4B-it-AWQ-4bit", is_llm_model=True)
        → _validate_and_setup_model_folder()      # 拼接 cwd 得到完整路径
        → is_dlis_validated(model_folder)          # 检查 model_folder/dlis_integration_validated
        → get_optimized_model(model_folder)        # 核心分支点 ↓
          → get_optimized_model_dir(model_folder)
            → os.listdir(model_folder)             # 扫描模型目录下的所有条目
            → 检查每个条目名是否以 "_opt" 结尾
            → 找到 → self.optimized_model = 该子目录路径
            → 没找到 → self.optimized_model = None
        → _create_runner(model_folder)
          ├─ [optimized_model 存在] → _create_optimized_runner()
          │    → create_runner(optimized_model_folder)
          │      → 读取 opt_type.txt → "llm"
          │      → LLMRunner.load_core()
          │        → 读取 best_setting.json → llm_type="vllm"
          │        → VLLM().load_engine()           # ✅ vLLM 引擎，原生支持 AWQ
          │
          └─ [optimized_model 不存在 & is_llm_model=True] → create_pytorch_runner_for_LLM()
               → LLMRunner.load_org()
                 → BaseLLM().load_engine()
                   → AutoModelForCausalLM.from_pretrained().to("cuda")  # ❌ OOM Killed
```

### 关键日志证据

容器 model 日志（16:28-16:55 时段）反复输出：

```
['gemma-4-26B-A4B-it-AWQ-4bit', 'Gemma4Deploy', 'complete.txt', '__placeholder__', 'gemma4-26B-AWQ-v1.tar']
⚠️ DLIS integration not validated - continuing without validation file
Could not find optimized model in Model/gemma-4-26B-A4B-it-AWQ-4bit
Compressing model: 100%
Loading weights: 100%
./dlis_model/run.sh: line 64: 9 Killed python3 $DIR/model/main.py http
```

每 3-4 分钟循环一次（容器 OOM → 重启 → 再 OOM）。

### 问题分析

1. **`_opt` 目录未找到**：`get_optimized_model_dir()` 用 `os.listdir()` 扫描 `Model/gemma-4-26B-A4B-it-AWQ-4bit/` 内部，寻找以 `_opt` 结尾的子目录。日志显示没找到，说明 `_opt` 目录要么没上传到正确位置，要么 Cosmos 数据未同步。

2. **`dlis_integration_validated` 也未找到**：同样是在 `Model/gemma-4-26B-A4B-it-AWQ-4bit/` 下查找该文件，也不存在。

3. **路径层级问题**：怀疑上传时把 `gemma-4-26B-A4B-it-AWQ-4bit_opt/` 放到了 `/Model/` 下面（与模型目录平级），而不是放在 `/Model/gemma-4-26B-A4B-it-AWQ-4bit/` 里面。

### 正确的 Cosmos 目录结构

```
/Model/gemma-4-26B-A4B-it-AWQ-4bit/
    ├── config.json
    ├── model-00001-of-00003.safetensors
    ├── ...（其他模型文件）
    ├── dlis_integration_validated              ← 空文件，放在模型目录内部
    └── gemma-4-26B-A4B-it-AWQ-4bit_opt/        ← _opt 目录，必须在模型目录内部
        ├── opt_type.txt                         ← 内容: llm（无换行）
        └── best_setting.json                    ← vLLM 配置
```

**错误结构**（`_opt` 放在外层 `/Model/` 下）：
```
/Model/
    ├── gemma-4-26B-A4B-it-AWQ-4bit/            ← 模型目录
    └── gemma-4-26B-A4B-it-AWQ-4bit_opt/        ← ❌ 错误位置，代码扫描不到
```

### `best_setting.json` 参考配置（基于本地调试验证）

```json
{
    "llm_type": "vllm",
    "model": "gemma-4-26B-A4B-it-AWQ-4bit",
    "quantization": "awq",
    "dtype": "float16",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.95,
    "tensor_parallel_size": 1,
    "trust_remote_code": true,
    "max_num_seqs": 32,
    "enforce_eager": true
}
```

### 待确认项更新（2026-04-18）

- [x] 确认 Cosmos 上 `_opt` 目录是否在 `gemma-4-26B-A4B-it-AWQ-4bit/` 内部 → **已确认路径正确**
- [ ] 确认 `dlis_integration_validated` 文件是否在 `gemma-4-26B-A4B-it-AWQ-4bit/` 内部
- [x] 如果路径正确，检查 Cosmos 数据是否需要重新部署/同步才能生效 → **已确认：不会自动同步，见下方分析**

---

## DLIS 部署调试续：Polaris 配置分析与根因确认（2026-04-18）

### Job 信息

| 字段 | 值 |
|------|-----|
| JobId | `b09f29f2-92b3-4c9d-9ed5-470ff0b188f7` |
| 镜像 | `dlisproddockerrepo.azurecr.io/dlis/llm_framework_vllm:20260418-1333-merge` |
| ModelDataPath | `abfs://dlisstore@dlisstoregen2.dfs.core.windows.net/dlismodelrepository-c09/local/users/jinjinchen/gemma4-26B-AWQ-v1.rar` |
| 机器类型 | `DLIS-Linux-A100-BN_23Cores` (A100 GPU) |
| 环境变量 | `DLIS_MODEL_DATA_TARGET_PATH=/Model; GPU_MEMORY_UTILIZATION=0.7` |
| 资源配置 | MaxMemory=20GB, MaxCpuCores=16, NumGpuDevices=1 |

### Polaris 运行时间线（PST / UTC）

| 时间 (PST) | 时间 (UTC) | 事件 |
|------------|-----------|------|
| 09:25 | 16:25 | Job 提交 |
| 09:27:28 | 16:27:28 | Polaris 开始执行 |
| 09:27:41 | 16:27:41 | Talk2InferenceServiceProxy 初始化完成 |
| 09:27:47 | 16:27:47 | 开始等待模型就绪，WaitingModelReadyInMin=30 |
| 09:27:53 | 16:27:53 | 容器第一次启动，打印 `/Model` 列表 |
| 09:28:03 | 16:28:03 | `Could not find optimized model` → BaseLLM fallback |
| 09:28:05 | 16:28:05 | Compressing + Loading weights → OOM Killed |
| 09:29~09:55 | 16:29~16:55 | **Crash loop: 每 3-4 分钟重复一次**（共约 8 次） |
| 09:57:47 | 16:57:47 | WaitingModelReady 超时，最终 heartbeat 仍显示 `CurrentStatus=Starting` |

### Heartbeat XML 关键数据

容器最后一次 heartbeat 上报的资源使用：
```
CurrentCpuUsagePercent: 0.17%
CurrentMemoryUsageInBytes: 9,255,145,472 (~9 GB CPU RAM)
MaxMemoryUsageInBytes: 9,255,145,472 (~9 GB)
CurrentGpuMemoryUsageInBytes: 9,437,184 (~9 MB)
MaxGpuMemoryUsageInBytes: 85,899,345,920 (~80 GB)
CurrentStatus: Starting
CopyingStatus: ImportedContainer
```

**关键发现：GPU 显存只用了 9MB，CPU 内存用了 9GB。**
- 说明 BaseLLM 的 `AutoModelForCausalLM.from_pretrained()` 在 CPU 上加载权重（"Compressing model" + "Loading weights"）
- 加载完成后调用 `.to("cuda")` 搬到 GPU 时，CPU 内存峰值超过 20GB 限制，被 Linux OOM Killer 杀掉
- A100 GPU 有 80GB 显存完全够用，但 CPU 内存 20GB 限制导致加载过程中 OOM

### 环境变量 `GPU_MEMORY_UTILIZATION=0.7` 无效

`GPU_MEMORY_UTILIZATION` 只在 vLLM 引擎路径（`LLMRunner.load_core()`）中使用。当前走的是 BaseLLM（transformers）fallback 路径，这个环境变量完全没有被读取。

### `/Model` 目录列表（每次启动完全相同）

```
['gemma-4-26B-A4B-it-AWQ-4bit', 'Gemma4Deploy', 'complete.txt', '__placeholder__', 'gemma4-26B-AWQ-v1.tar']
```

**没有 `gemma-4-26B-A4B-it-AWQ-4bit_opt` 目录。**

### 根因分析

**DLIS 的 ModelDataPath 机制：**
- `ModelDataPath` 指向 `gemma4-26B-AWQ-v1.rar`，但 rar 本身并不使用，**仅用于定位 Cosmos 目录**
- DLIS 实际是将 rar 所在的**整个父目录**挂载到 `/Model`
- 容器内 `/Model` 列表 = Cosmos 目录 `local/users/jinjinchen/` 下的内容
- `_opt` 目录已上传到 `gemma-4-26B-A4B-it-AWQ-4bit/gemma-4-26B-A4B-it-AWQ-4bit_opt/`（模型目录内部）
- 但 `os.listdir("Model/gemma-4-26B-A4B-it-AWQ-4bit")` 始终没有列出 `_opt` 子目录

**可能原因（待排查）：**
1. **Cosmos 缓存/快照**：_opt 文件在 09:21 上传，Job 在 09:25 提交，DLIS 可能使用了缓存的目录快照而非实时拉取
2. **DLIS 数据下载时机**：DLIS 可能在机器分配阶段就已经下载/缓存了数据，容器启动时不会重新拉取
3. **`_opt` 目录层级错误**：需要确认 `_opt` 是在 `gemma-4-26B-A4B-it-AWQ-4bit/` **内部**还是与其**同级**

### 解决方案

| 方案 | 描述 | 优缺点 |
|------|------|--------|
| **A. 改代码绕过 `_opt` 检测** | 修改 `model.py` 或 `oaas_wrapper_v2.py`，直接构造 vLLM runner，不依赖 `_opt` 目录 | ✅ 最快，无需重新上传模型数据；⚠️ 需要重新构建镜像 |
| **B. 等待 Cosmos 同步后重试** | 等待足够时间让 Cosmos 数据同步，然后重新提交 job | ✅ 不改代码不改数据；⚠️ 不确定需要等多久 |
| **C. 混合方案** | 在代码中检查环境变量 `GPU_MEMORY_UTILIZATION` 存在时直接走 vLLM | ✅ 灵活；⚠️ 需要重新构建镜像 |

---

## 第二次部署尝试（2025-06-05）— 确认 Cosmos 同步不是问题

### Job 信息

| 字段 | 值 |
|------|-----|
| Job ID | `f35200e8-00d8-4d2b-ba25-50c264aee71c` |
| 提交时间 | 距第一次部署约 7+ 小时后 |
| 结果 | **失败 — 同样的 OOM 崩溃循环** |

### Heartbeat XML 关键数据

```
CurrentMemoryUsageInBytes: 16,885,006,336  (~16 GB)  ← 比第一次的 9GB 更高
CurrentGpuMemoryUsageInBytes: 9,437,184     (~9 MB)   ← GPU 依然未使用
RestartCount: 2
```

### 失败模式

与第一次完全一致：
1. 容器启动 → BaseLLM 加载（CPU）→ 内存飙到 16GB → OOM 被杀
2. 自动重启 2 次，每次都 OOM
3. 30 分钟超时 → `SetupEnvironmentException: Model failed to initialize in time`
4. Polaris 最终报错：`TryPing` 持续失败，Job 标记为 Failed

### 关键结论

1. **Cosmos 同步不是问题** — 第二次部署距离 `_opt` 上传已过 7+ 小时，`_opt` 仍然不在容器内
2. **DLIS 数据挂载机制不会拾取手动添加的子目录** — 这是 DLIS 平台行为，非 Cosmos 延迟问题
3. **CPU 内存用量升到 16GB** — 更接近 20GB 限制，说明加载过程中峰值必然超过限制
4. **GPU 仍然只有 9MB** — 确认走的是 BaseLLM（transformers）路径，vLLM 从未被激活

### 最终判断

**方案 B（等待 Cosmos 同步）已被排除。** 必须走方案 A 或 C，通过改代码绕过 `_opt` 检测，直接使用 vLLM 引擎。

---

## 第三次部署尝试（2026-04-18）— ModelDataPath 改为 best_setting.json

### 变更点

用户修改了 Polaris job config：
- **ModelDataPath** 从 `gemma4-26B-AWQ-v1.rar` 改为 `best_setting.json`（希望刷新 Cosmos 同步）
- **ModelPath** 更新为 `llm_framework_vllm:20260418-1333-merge`（新构建的镜像）
- 其他配置不变

### Job 信息

| 字段 | 值 |
|------|-----|
| Job ID | `591a0a93-670b-42db-a48e-0a538e80d35a` |
| 机器 | `BN2BEAP0000495E` |
| 时间范围 | 20:22:02 ~ 20:52:19 UTC (2026-04-18) |
| 结果 | **失败 — 同样的 OOM + 30 分钟超时** |

### Heartbeat XML 关键数据

```
CurrentMemoryUsageInBytes: 18,408,464,384  (~18.4 GB)  ← 历次最高，非常接近 20GB 限制
CurrentGpuMemoryUsageInBytes: 9,437,184     (~9 MB)     ← GPU 依然未使用
RestartCount: 1
```

### 分析

1. **ModelDataPath 指向单个文件 `best_setting.json`**：DLIS 挂载其父目录 `local/users/jinjinchen/` 到 `/Model`
2. 容器内 `/Model` 目录应包含 `gemma-4-26B-A4B-it-AWQ-4bit/` 等内容（与之前相同的父目录）
3. `best_setting.json` 被单独放在 `local/users/jinjinchen/` 下，而非 `gemma-4-26B-A4B-it-AWQ-4bit/gemma-4-26B-A4B-it-AWQ-4bit_opt/` 内
4. OaasWrapper 依然找不到 `_opt` 子目录 → BaseLLM fallback → CPU 加载 → OOM
5. CPU 内存达到 18.4GB，是三次部署中最高，更加接近 20GB 限制

### 关键结论

- **改 ModelDataPath 为单个文件不能解决 `_opt` 检测问题** — OaasWrapper 的 `get_optimized_model_dir()` 需要在模型目录内找到 `*_opt` 子目录
- 三次部署全部失败，模式完全一致：BaseLLM CPU 加载 → OOM
- **必须改代码**，没有其他绕过方式
---

## 代码修复：model.py 内创建 _opt 目录（Plan A 实施）

### 日期：2026-04-18

### 问题根因

DLIS 容器通过 Cosmos 挂载数据到 `/Model`，但手动上传到模型目录内的 `_opt` 子目录始终不可见。三次部署全部因此失败：
- OaasWrapper `get_optimized_model_dir()` 扫描不到 `*_opt` 子目录
- Fallback 到 `create_pytorch_runner_for_LLM()` → `LLMRunner.load_org()` → `BaseLLM` (transformers)
- BaseLLM 用 `AutoModelForCausalLM.from_pretrained().to("cuda")` 先在 CPU 加载完整模型 → 超过 20GB 内存限制 → OOM Kill

### 修复方案

在 `model.py` 的 `ModelImp.__init__()` 中，**在调用 `OaasWrapper()` 之前**，用代码创建 `_opt` 子目录并写入配置文件：

```python
model_dir_name = "gemma-4-26B-A4B-it-AWQ-4bit"
opt_dir = os.path.join("/Model", model_dir_name, f"{model_dir_name}_opt")
if not os.path.exists(opt_dir):
    os.makedirs(opt_dir, exist_ok=True)
    with open(os.path.join(opt_dir, "opt_type.txt"), "w") as f:
        f.write("llm")
    best_setting = {
        "llm_type": "vllm",
        "model": model_dir_name,
        "max_output_len": 256,
        "temperature": 0.8,
        "top_p": 0.95,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9")),
        "trust_remote_code": True,
        "dtype": "auto",
        "max_model_len": 8192,
        "stop": ["</Prompt>", "</Scene5>", "<end_of_turn>"]
    }
    with open(os.path.join(opt_dir, "best_setting.json"), "w") as f:
        json.dump(best_setting, f, indent=2)
```

### 与本地成功测试的对比验证

本地 AWQ 模型测试（debug log 第 2921-2958 行）成功使用了以下配置：

| 配置项 | 本地成功测试 | 代码修复 | 匹配？ |
|--------|-------------|---------|--------|
| opt_type.txt | `llm`（printf "llm"） | `llm` | ✅ |
| llm_type | `vllm` | `vllm` | ✅ |
| quantization 字段 | **未设置** | **未设置** | ✅ |
| model | `gemma-4-26B-A4B-it-AWQ-4bit` | `gemma-4-26B-A4B-it-AWQ-4bit` | ✅ |
| max_model_len | 8192 | 8192 | ✅ |
| gpu_memory_utilization | 0.9 | 0.9（通过环境变量可配） | ✅ |
| trust_remote_code | True | True | ✅ |
| dtype | `auto` | `auto` | ✅ |

### 关键发现：quantization 字段

- `QUANTIZATION_MAP`（`vllm_util.py`）只有 `int4_awq`, `int_w8a8`, `fp_w8a8`, `bitsandbytes`, `gptq` 这些 key
- **没有 `"awq"` key** — 如果设了 `"quantization": "awq"`，`QUANTIZATION_MAP.get("awq")` 返回 `None`
- quantization=None 时，vllm_runner.py 的路径逻辑：`model_path = root_model_path`（_opt 的父目录），这恰好是正确的
- vLLM 引擎会自动从模型的 `config.json` 中检测到 AWQ 量化，无需手动指定
- **结论：不设 quantization 字段是正确做法，与本地成功测试一致**

### opt_type.txt 值澄清

- `opt_type.txt` 应为 `llm`（不是 `vllm`）
- `runner_factory.py` 中：`opt_type == "llm"` → `LLMRunner().load_core()` → vLLM 引擎 ✅
- `opt_type == "vllm"` 会触发 "Unknown opt type" 异常 ❌
- 本地快速构建测试（第 3129 行）使用了 `printf 'vllm'`，这实际上是错误的（但当时测试目的不同）
- 本地正式测试（第 2941 行）正确使用了 `printf "llm"`

### 预期调用链

```
ModelImp.__init__()
  → 代码创建 /Model/gemma-4-26B-A4B-it-AWQ-4bit/gemma-4-26B-A4B-it-AWQ-4bit_opt/
  → 写入 opt_type.txt ("llm") + best_setting.json (无 quantization 字段)
  → OaasWrapper("Model/gemma-4-26B-A4B-it-AWQ-4bit")
    → get_optimized_model_dir() 找到 _opt 子目录 ✅
    → create_runner(_opt) → LLMRunner.load_core() → 读取 best_setting.json → VLLM().load_engine() ✅
    → vLLM 直接在 GPU 加载 AWQ 模型，不经过 CPU → 不会 OOM ✅
```

### 下一步

1. 重新构建 Docker 镜像（包含修改后的 model.py）
2. 提交 Polaris job
3. 验证容器日志出现 "Creating sync LLMRunner for optimized model" 而非 BaseLLM fallback
4. 验证 GPU 显存使用 >> 9MB，CPU 内存 << 20GB

---

## 路径修复：_opt 创建路径与 OaasWrapper 对齐

### 日期：2026-04-19

### 发现的问题

对比本地测试成功时的完整命令历史，发现一个关键细节：

本地测试第 20 行执行了 `ln -s /vllm-workspace/Model /dlis_model/Model`，说明 OaasWrapper 使用**相对路径** `"Model/gemma-4-26B-A4B-it-AWQ-4bit"` 解析模型目录，工作目录为 `/dlis_model`，实际路径为 `/dlis_model/Model/gemma-4-26B-A4B-it-AWQ-4bit`。

而之前的代码修复中 `_opt` 目录创建在**绝对路径** `/Model/...`，与 OaasWrapper 解析出的路径不一致 → OaasWrapper 仍然找不到 `_opt`。

### 修复内容

将 `_opt` 创建逻辑改为与 `OaasWrapper._validate_and_setup_model_folder()` 相同的路径解析：

```python
# 之前（硬编码绝对路径）：
opt_dir = os.path.join("/Model", model_dir_name, f"{model_dir_name}_opt")

# 之后（与 OaasWrapper 一致的解析逻辑）：
model_folder = os.path.join("Model", model_dir_name)
if not os.path.isdir(model_folder):
    model_folder = os.path.join(os.path.realpath(os.getcwd()), "Model", model_dir_name)
opt_dir = os.path.join(model_folder, f"{model_dir_name}_opt")
```

### 其他风险排查结果

| 检查项 | 结果 |
|--------|------|
| opt_type.txt 尾部换行 | Python `f.write("llm")` 不加换行，`get_opt_type()` 无 `.strip()` → 无问题 ✅ |
| best_setting.json model 字段 | quantization=None 时不使用 model 字段构建路径 → 无影响 ✅ |
| vLLM stop tokens | 与本地测试一致 ✅ |
| certificate_path 硬编码 | EventHub 初始化有 try/catch 容错，不阻塞启动 ✅ |
| config.py 类型注解 | 已在之前的 commit 中修复（加了 `: str`） ✅ |
| transformers 依赖 | 走 vLLM 路径不需要，低风险 ✅ |

### 提交记录

- Commit: `b154584`
- Branch: `jinjinchen/Gemma4-v1-fast-build`
- Repo: `OaaS_LLMTemplate`
- Message: `fix: create _opt directory programmatically to force vLLM engine path`

---

## 第四次部署失败：Read-only filesystem + Unable to find exposed port 8888

### 日期：2026-04-19

### 错误现象

部署后容器日志出现两个错误：

**错误 1：Read-only file system（Errno 30）**
```
OSError: [Errno 30] Read-only file system: '/Model/gemma-4-26B-A4B-it-AWQ-4bit/gemma-4-26B-A4B-it-AWQ-4bit_opt'
```
DLIS 容器将 `/Model` 目录以只读方式挂载（从 Cosmos 数据源），因此 `os.makedirs()` 在 `/Model` 下创建 `_opt` 子目录直接失败。

**错误 2：Unable to find exposed port 8888**
```
Unable to find exposed port 8888 for container bb039b6a952e...
```
DLIS Common Log 显示容器状态为 `running`，但持续（每 3 秒）报 "Unable to find exposed port 8888"，尝试不同宿主端口（55561-55589）映射均失败，持续约 30 分钟。

### 根因分析：两个错误是同一条因果链

```
/Model 只读 → _opt 创建失败 (Errno 30)
  → OaasWrapper 找不到 _opt 目录
  → fallback 到 BaseLLM（transformers CPU 加载 26B 模型）
  → CPU 内存爆炸 → OOM Kill（进程被杀）
  → HTTP server (Tornado on port 8888) 从未启动
  → DLIS 探测不到 port 8888 → "Unable to find exposed port 8888"
```

Port 8888 的配置本身没有问题：
- `Dockerfile_vllm_fast` 已有 `EXPOSE 8888`
- `http_server.py` 监听 `utils.get_listening_port(8888)`

问题在于模型加载阶段就 OOM 被杀了，HTTP server 的代码根本没有执行到。

### 为什么 /Model 必须是只读的（为什么需要 writable mirror 方案）

DLIS（Deep Learning Inference Service）的容器架构设计中，`/Model` 目录是从 Cosmos 存储以**只读方式挂载**的，这是有意为之的设计：

1. **数据完整性保证**：模型权重文件（26B 参数的 AWQ 量化模型，约 14GB）从 Cosmos 下载后以只读挂载，确保推理过程中不会意外修改模型文件，保证每次推理使用的权重完全一致。

2. **多实例共享**：同一个 Cosmos 路径可以被多个容器实例共享挂载。只读挂载避免了多实例间的写冲突。

3. **安全隔离**：只读挂载是容器安全最佳实践，防止容器内恶意或错误代码篡改模型数据。

4. **存储层限制**：Cosmos 作为分布式存储系统，其 FUSE 挂载驱动本身可能不支持写操作，即使容器尝试写入也会被底层拒绝。

### 修复方案：Writable Mirror

既然无法写入 `/Model`，在 `/tmp`（容器可写的临时目录）下创建一个"镜像目录"：

```python
# 在 /tmp 下创建可写的镜像目录
writable_model = os.path.join("/tmp", model_dir_name)
os.makedirs(writable_model, exist_ok=True)

# 用 symlink 指向原始只读模型文件（不复制，零额外存储开销）
for item in os.listdir(src_model):
    os.symlink(os.path.join(os.path.realpath(src_model), item),
               os.path.join(writable_model, item))

# 在可写镜像中创建 _opt 目录 + vLLM 配置
opt_dir = os.path.join(writable_model, f"{model_dir_name}_opt")
os.makedirs(opt_dir, exist_ok=True)
# 写入 opt_type.txt 和 best_setting.json ...

# 传可写路径给 OaasWrapper
self.oaas_wrapper = OaasWrapper(writable_model, is_llm_model=True)
```

优点：
- 模型权重文件通过 symlink 引用，**零额外磁盘占用**
- `_opt` 配置文件总共只有几百字节
- OaasWrapper 正常检测到 `_opt` → 走 vLLM 引擎路径 → GPU 加载 → 不 OOM
- HTTP server 正常启动 → port 8888 可用 → DLIS 健康检查通过

### 修复状态

model.py 已修改（writable mirror 方案），commit `06f39f1` 并 push 到 `jinjinchen/Gemma4-v1-fast-build` 分支。

---

## DLIS 部署 #5：Runner 创建静默失败（2026-04-19）

### 进展

Writable mirror 方案生效，容器启动阶段有显著进展：
- ✅ HTTP server 成功启动（port 8888 正常工作）
- ✅ DLIS 健康检查通过
- ✅ Eval 请求能到达 model.py（日志显示 `Eval request received`）
- ❌ 每个 Eval 请求立即失败，响应时间 ~0.6ms

### 错误

```
RuntimeError: Could not get any available runner
  at oaas_wrapper_v2.py:122
```

每 2 秒重复一次（14:38 - 14:50 UTC），所有请求返回 500。

### 根因分析

`oaas_wrapper_v2.py` 的 `_create_runner()` 方法（第 87-103 行）有一个 try/except 捕获所有异常：

```python
def _create_runner(self, model_folder):
    try:
        if not is_test_original_model() and self.model_is_optimized():
            return self._create_optimized_runner()
        if not self.async_mode and self.is_llm_model:
            return create_pytorch_runner_for_LLM(model_folder)
        return None
    except Exception as e:
        print(f"Failed to create runner: {e}")  # 只输出到 stdout，不进 DLIS 日志
        return None
```

问题链路：
1. `__init__` 阶段调用 `_create_runner()` → 检测到 `_opt` 目录 → 走 `create_runner()` 路径
2. `create_runner()` → `LLMRunner.load_core()` → 尝试初始化 vLLM 引擎
3. vLLM 引擎初始化失败（**实际错误被 catch 吞掉**，只 print 到 stdout）
4. `_create_runner()` 返回 `None` → `self.used_runner = None`
5. 之后每个 `run()` 调用都直接触发 `RuntimeError("Could not get any available runner")`

### 可能的 vLLM 初始化失败原因

1. **GPU 不可见 / CUDA 未就绪**：容器启动时 GPU 驱动可能未就绪
2. **max_model_len 超出 GPU 显存**：best_setting.json 设置 `max_model_len: 8192` 可能需要更多显存
3. **模型路径解析问题**：symlink 路径传给 vLLM 后，vLLM 内部 `from_pretrained` 可能无法正确解析
4. **AWQ 量化支持**：vLLM 版本可能不支持 Gemma4 的 AWQ 量化格式
5. **依赖缺失**：容器镜像中缺少 vLLM 需要的某些依赖包

### 关键问题

实际的 vLLM 错误被 `_create_runner()` 的 `except Exception as e` 吞掉，只 `print()` 到 stdout。DLIS 容器日志默认可能不捕获 stdout，导致真正的错误信息丢失。

### 下一步

1. **方案 A**：在 `model.py` 中 wrap `OaasWrapper()` 调用，捕获并 `logger.error()` 记录初始化失败详情
2. **方案 B**：在 `_create_runner()` 中将 `print` 改为 `logger.error`（需要修改 oaas_wrapper_v2.py）
3. **方案 C**：在 `model.py` 的 `__init__` 中加入 `assert self.oaas_wrapper.used_runner is not None` 并打印详细错误
4. **方案 D**：直接绕过 OaasWrapper，参考 siwenzhu 分支用 `vllm.LLM` 直接加载（最激进但最可控）

### 根因确认

经过代码追踪，找到了确切的 bug：

**`best_setting.json` 缺少 `"quantization": "int4_awq"` 字段。**

`vllm_runner.py:62-66` 中，模型路径选择逻辑依赖 `quantization` 字段：

```python
self.root_model_path = os.path.dirname(optimized_model_folder)  # = /tmp
if self.config.quantization:  # None → False
    model_path = os.path.join(self.root_model_path, model_name)  # 正确：/tmp/gemma-4-26B-A4B-it-AWQ-4bit
else:
    model_path = self.root_model_path  # 错误：/tmp
```

没有 `quantization` 字段 → `model_path = /tmp` → vLLM 在 `/tmp` 找不到模型文件 → 初始化失败 → 被 `_create_runner()` 的 catch-all 吞掉。

同时，`QUANTIZATION_MAP` 中 `"int4_awq"` 映射到 vLLM 的 `"AWQ"` 量化格式，AWQ 4-bit 模型必须指定此参数。

### 修复

1. 在 `model.py` 的 `best_setting` dict 中添加 `"quantization": "int4_awq"`
2. 在 `OaasWrapper()` 初始化后添加 runner None 检查，fail fast 并写入 logger（而非被 `_create_runner` 静默吞掉）
