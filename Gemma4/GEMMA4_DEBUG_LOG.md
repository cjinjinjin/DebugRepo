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

### 待实验

- [ ] TP=2 + `--max-model-len 4096` 重新 benchmark
- [ ] No-CoT 模式 benchmark
- [ ] 截断输入（`--max_lp_chars 2000`）benchmark
- [ ] FP8 量化模型（`RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`）vLLM 部署
- [ ] 多 GPTQ 副本部署（单卡 13GB，可放 4-6 副本/机器）

### Benchmark 脚本

`Gemma4/benchmark_vllm.py` — 基于 asyncio + aiohttp 的并发 benchmark 工具：
- 支持 `--concurrency N` 设置并发数
- 支持 `--no_cot` 切换 system prompt
- 自动发现 vLLM 模型名称
- 输出：吞吐量（req/s）、延迟分布（avg/median/p95）、每条 tok/s
- 可导出详细 JSON 结果

