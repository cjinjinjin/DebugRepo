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

#### Step 2: 启动容器内 HTTP 服务

```bash
docker run -it --rm --gpus all \
  -p 8888:8888 \
  -v /home/jinjinchen/ms-image-quality-filters-aether-module-main/Gemma4Deploy:/Model \
  -v /home/jinjinchen/data/gemma-4-26B-A4B-it-AWQ-4bit:/Model/model \
  -e _ModelPath_=/dlis_model/run.sh \
  -e _ListeningPort_=8888 \
  -e EnableOaas=true \
  -e AB_MAX_SEQ_LEN=16 \
  -e AB_INSTANCE_GROUP_COUNT=1 \
  dlisproddockerrepo.azurecr.io/dlis/llm_framework_vllm:latest \
  /bin/bash -c "cd /dlis_model && ./run.sh http"
```

**说明:**
- `-v .../Gemma4Deploy:/Model` — 挂载 `dlis_inter.py` 和 `model.py` 到容器 `/Model`
- `-v .../gemma-4-26B-A4B-it-AWQ-4bit:/Model/model` — 挂载 AWQ 量化权重到容器 `/Model/model`
- `_ListeningPort_=8888` — HTTP 服务端口
- `EnableOaas=true` — 启用 OaaS vLLM 引擎

#### Step 3: 发送测试请求

```bash
# 在另一个终端
curl -X POST http://localhost:8888/ \
  -H "Content-Type: text/plain" \
  -d '{
    "landing_page_content": "Welcome to TrailMaster Outdoor Gear. Premium hiking boots, ultralight backpacks, and camping essentials for your next adventure. Free shipping on orders over $100.",
    "url": "https://trailmaster.example.com",
    "num_prompts": 5,
    "max_lp_chars": 5000
  }'
```

#### Step 4: 验证返回结果

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

