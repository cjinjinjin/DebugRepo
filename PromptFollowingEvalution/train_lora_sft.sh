#!/bin/bash
# =============================================================
# Qwen3-VL-8B LoRA SFT 训练脚本 (ms-swift)
#
# 使用前确保已安装：
#   pip install ms-swift transformers accelerate deepspeed
#   pip install flash-attn --no-build-isolation
#
# 运行方式：
#   bash train_lora_sft.sh
# =============================================================

# ============================================================
# 路径配置
# ============================================================
MODEL_PATH="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-VL-8B-Instruct"
TRAIN_DATA="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/sft_data/train.json"
VAL_DATA="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/sft_data/val.json"
OUTPUT_DIR="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/Qwen3-VL-8B-PromptFollowing-LoRA"

# ============================================================
# GPU 配置（按实际情况修改）
# ============================================================
NUM_GPUS=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ============================================================
# 训练
# ============================================================
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
swift sft \
    --model             $MODEL_PATH \
    --dataset           $TRAIN_DATA \
    --val_dataset       $VAL_DATA \
    --train_type        lora \
    --lora_rank         8 \
    --lora_alpha        16 \
    --lora_target_modules "all-linear" \
    --freeze_vit        true \
    --num_train_epochs  5 \
    --per_device_train_batch_size  2 \
    --per_device_eval_batch_size   2 \
    --gradient_accumulation_steps  8 \
    --learning_rate     5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio      0.1 \
    --weight_decay      0.05 \
    --max_length        2048 \
    --dataloader_num_workers 4 \
    --bf16              true \
    --gradient_checkpointing true \
    --save_steps        50 \
    --eval_steps        50 \
    --save_total_limit  5 \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --logging_steps     10 \
    --output_dir        $OUTPUT_DIR \
    --report_to         tensorboard \
    --lazy_tokenize     true

# ============================================================
# 说明
# ============================================================
# --freeze_vit true     : 冻结视觉编码器，只训练 LLM 部分的 LoRA
#                         数据量 <5k 时推荐；若数据量充足可改为 false
#
# --lora_rank 16        : LoRA rank，数据量小时 16 足够
#                         数据量大（>10k）可调到 32~64
#
# --gradient_accumulation_steps 8
#                         等效 batch_size = 2 * 8 * num_gpus
#                         8 GPUs 下约等效 128 per-step
#
# --max_length 2048     : Qwen3-VL 图文拼接后长度，通常够用
#                         如 OOM 可降到 1024
#
# merge LoRA 权重（训练完成后）：
#   swift export \
#       --model_type qwen3-vl-8b-instruct \
#       --model_id_or_path $MODEL_PATH \
#       --ckpt_dir $OUTPUT_DIR/checkpoint-best \
#       --merge_lora true \
#       --output_dir ${OUTPUT_DIR}-merged
