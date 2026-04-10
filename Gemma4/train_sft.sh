#!/bin/bash
# Gemma 4 26B-A4B-it SFT training with ms-swift
# Use if zero-shot results are unsatisfactory
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Model — download to local first or use HF cache
MODEL_PATH="google/gemma-4-26B-A4B-it"

# Data — reuse Qwen3 SFT data (same task, same format)
DATA_DIR="${ROOT_DIR}/QwenFinetune/data"
TRAIN_DATA="${DATA_DIR}/sft_train_cot.jsonl"
EVAL_DATA="${DATA_DIR}/sft_eval_cot.jsonl"

# Output
OUTPUT_DIR="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma4_sft_lora_cot"

# Training hyperparams
# Gemma 4 26B MoE is smaller in active params (~4B), so:
# - LoRA rank 16 is sufficient (vs Qwen3's rank 64)
# - Lower LR appropriate for MoE (most params frozen)
# - ZeRO-2 should work fine (no FusedMoE TP issues expected)
LORA_RANK=16
LORA_ALPHA=32
NUM_EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LR=5e-5
MAX_LENGTH=4096
SAVE_STEPS=10
EVAL_STEPS=10
LOGGING_STEPS=5

echo "============================================"
echo "Gemma 4 26B-A4B-it SFT Training"
echo "============================================"
echo "Model:     $MODEL_PATH"
echo "Train:     $TRAIN_DATA"
echo "Eval:      $EVAL_DATA"
echo "LoRA:      rank=$LORA_RANK, alpha=$LORA_ALPHA"
echo "LR:        $LR"
echo "Epochs:    $NUM_EPOCHS"
echo "Output:    $OUTPUT_DIR"
echo ""

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
NCCL_TIMEOUT=7200 \
NCCL_DEBUG=WARN \
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
DS_SKIP_CUDA_CHECK=1 \
swift sft \
    --model                       ${MODEL_PATH} \
    --dataset                     ${TRAIN_DATA} \
    --val_dataset                 ${EVAL_DATA} \
    --tuner_type                  lora \
    --lora_rank                   ${LORA_RANK} \
    --lora_alpha                  ${LORA_ALPHA} \
    --num_train_epochs            ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate               ${LR} \
    --lr_scheduler_type           cosine \
    --warmup_ratio                0.05 \
    --max_length                  ${MAX_LENGTH} \
    --output_dir                  ${OUTPUT_DIR} \
    --bf16                        true \
    --gradient_checkpointing      true \
    --deepspeed                   zero3 \
    --save_steps                  ${SAVE_STEPS} \
    --eval_steps                  ${EVAL_STEPS} \
    --logging_steps               ${LOGGING_STEPS} \
    --ddp_timeout                 7200
