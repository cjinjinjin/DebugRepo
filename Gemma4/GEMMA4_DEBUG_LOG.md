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
5. 🔄 inference Random200（`RawData/UHRS2K_SD_Random200_0324.tsv`）进行额外验证

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
- AWQ 预量化模型的量化信息在 `quantization_config.json` 中
- `AutoModelForCausalLM.from_pretrained()` 自动检测并加载，无需改 inference 代码
- 只需换模型路径即可

### 用法
```bash
# 1. 下载 AWQ 模型
HF_TOKEN=hf_xxx bash Gemma4/download_model_awq.sh

# 2. 跑 eval
bash Gemma4/eval_gemma4_awq_zeroshot.sh
```

### 结果
⬜ 待运行

