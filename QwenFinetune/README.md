# Qwen LoRA Finetune for Image Prompt Generation

将 GPT5 两步 pipeline（LPUnderstanding → ImagePromptCreator）蒸馏进 Qwen3.5-35B-A3B，实现从 LP 字段直接生成高质量 image prompt。

---

## 目录结构

```
QwenFinetune/
├── prepare_data.py      # 原始标注数据 → SFT/DPO 训练集
├── train_lora_sft.py    # SFT + LoRA 训练（主要方案）
├── train_lora_dpo.py    # DPO + LoRA 训练（进阶偏好对齐）
├── inference.py         # 推理 / 模型部署
├── evaluate.py          # 离线评估（格式、关键词覆盖、VLM）
└── requirements.txt
```

---

## 依赖安装

```bash
pip install torch transformers peft accelerate bitsandbytes datasets trl
```

---

## 原始数据格式

准备一份 JSONL 或 JSON 数组文件，每行一条记录：

```json
{
  "id": "sample_001",
  "lp_fields": {
    "FinalDestinationURLUrl": "https://...",
    "DocumentTitle": "...",
    "VisualTitle": "...",
    "Heading": "...",
    "Title_CB": "...",
    "VisualTitle_CB": "...",
    "Heading_CB": "...",
    "BestSnippet_CB": "...",
    "MetaDescription_CB": "...",
    "PrimaryContentNoTitleNoHeading": "..."
  },
  "versions": [
    {
      "version_id": "v1",
      "prompts": ["prompt1", "prompt2", "prompt3", "prompt4", "prompt5"],
      "labels":  ["good", "bad", "good", "bad", "good"]
    },
    {
      "version_id": "v2",
      "prompts": ["prompt1", "prompt2", "prompt3", "prompt4", "prompt5"],
      "labels":  ["bad", "good", "good", "bad", "bad"]
    }
  ]
}
```

---

## 推荐训练流程

### Step 1 — 数据预处理

```bash
python prepare_data.py \
    --input  raw_data/labeled_1000.jsonl \
    --output_dir data/ \
    --eval_ratio 0.1 \
    --mode both
```

输出：`data/sft_train.jsonl`, `data/sft_eval.jsonl`, `data/dpo_train.jsonl`, `data/dpo_eval.jsonl`

---

### Step 2 — SFT LoRA 训练（监督式微调，必做）

```bash
# 单卡 A100 80G / H100
python train_lora_sft.py \
    --model_name_or_path Qwen/Qwen2.5-35B-Instruct \
    --train_file data/sft_train.jsonl \
    --eval_file  data/sft_eval.jsonl  \
    --output_dir checkpoints/qwen35_sft_lora \
    --load_in_4bit \
    --lora_r 64 \
    --lora_alpha 128 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4

# 多卡（4xA100）
accelerate launch --num_processes 4 train_lora_sft.py [同上参数]
```

---

### Step 3 — DPO LoRA 训练（偏好对齐，可选但推荐）

在 SFT checkpoint 基础上继续 DPO，进一步拉开 good/bad 差距：

```bash
python train_lora_dpo.py \
    --model_name_or_path   checkpoints/qwen35_sft_lora/lora_adapter \
    --base_model_for_adapter Qwen/Qwen2.5-35B-Instruct \
    --load_from_sft_adapter \
    --train_file data/dpo_train.jsonl \
    --eval_file  data/dpo_eval.jsonl  \
    --output_dir checkpoints/qwen35_dpo_lora \
    --load_in_4bit \
    --beta 0.1 \
    --num_train_epochs 2 \
    --learning_rate 5e-5
```

---

### Step 4 — 推理

```bash
# 单条查询
python inference.py \
    --adapter_path checkpoints/qwen35_dpo_lora/lora_adapter \
    --url "https://example.com/product" \
    --title "Best Running Shoes 2025"

# 批量推理
python inference.py \
    --adapter_path checkpoints/qwen35_dpo_lora/lora_adapter \
    --input_file   data/test_lp_fields.jsonl \
    --output_file  results/generated_prompts.jsonl \
    --batch_size 4

# 合并适配器（生产部署）
python inference.py \
    --adapter_path checkpoints/qwen35_dpo_lora/lora_adapter \
    --base_model   Qwen/Qwen2.5-35B-Instruct \
    --merge_and_save production_model/
```

---

### Step 5 — 评估

```bash
python evaluate.py \
    --generated_file results/generated_prompts.jsonl \
    --report_file    results/eval_report.json
```

---

## 关键超参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `lora_r` | 64 | LoRA秩，越大拟合能力越强，显存占用越高 |
| `lora_alpha` | 128 | 通常设为 2×r |
| `load_in_4bit` | True | QLoRA量化，A100 80G可跑35B |
| `gradient_accumulation_steps` | 16 | 等效 batch=16 |
| `learning_rate` SFT | 2e-4 | LoRA标准学习率 |
| `learning_rate` DPO | 5e-5 | DPO需要更小学习率 |
| `beta` DPO | 0.1 | KL惩罚强度，0.1是典型值 |
| `max_length` | 2048 | LP字段+5条prompt的总token数 |

---

## 架构说明

```
原 pipeline (GPT5依赖):
  LP字段 → [LPUnderstanding GPT5] → 结构化理解 → [ImagePromptCreator GPT5] → 5个prompts

Finetune后 (Qwen，无外部依赖):
  LP字段 → [Qwen+LoRA] → 5个good image prompts
```

**训练数据构建逻辑：**
- **SFT 正样本**: 每个版本中 label=good 的 prompt，重复填充到5条，作为 assistant 回复
- **DPO 偏好对**: good 回复 (chosen) vs bad 回复 (rejected)
  - 优先使用跨版本对齐（v1_good vs v2_bad），天然消除LP信息偏差
  - 退而使用同版本内对比

---

## VRAM 需求估算

| 配置 | 显存需求 | 说明 |
|------|----------|------|
| QLoRA 4bit + r=64 | ~40GB | 单A100 80G可用 |
| QLoRA 4bit + r=32 | ~30GB | 单A100 40G可用 |
| 多卡 (4xA100 40G) | ~160GB | 推荐生产训练配置 |
