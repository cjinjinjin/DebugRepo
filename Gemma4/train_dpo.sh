#!/bin/bash
# Gemma 4 DPO preference training
# Run after SFT if format compliance needs improvement
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Model — SFT merged checkpoint (update path after SFT completes)
MODEL_PATH="${1:-PLEASE_SET_SFT_MERGED_MODEL_PATH}"

# Data
DATA_DIR="${ROOT_DIR}/QwenFinetune/data"
TRAIN_DATA="${DATA_DIR}/dpo_refine_train_cot.jsonl"
EVAL_DATA="${DATA_DIR}/dpo_refine_eval_cot.jsonl"

# Output
OUTPUT_DIR="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma4_dpo_lora_cot"

# DPO hyperparams
# Key lesson from Qwen3: ckpt-1 was best, likelihood displacement kicks in fast
# So: save every step, use lower beta, expect 1-3 steps max useful training
LORA_RANK=16
LORA_ALPHA=32
NUM_EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=4
LR=5e-6
BETA=0.05
MAX_LENGTH=8192
SAVE_STEPS=1
EVAL_STEPS=5
LOGGING_STEPS=1

echo "============================================"
echo "Gemma 4 DPO Training"
echo "============================================"
echo "Model:     $MODEL_PATH"
echo "Beta:      $BETA (lower than Qwen3's 0.1 to reduce likelihood displacement)"
echo "Save:      every step (ckpt-1 was best for Qwen3)"
echo "Output:    $OUTPUT_DIR"
echo ""

if [ "$MODEL_PATH" = "PLEASE_SET_SFT_MERGED_MODEL_PATH" ]; then
    echo "ERROR: Please provide SFT merged model path as first argument"
    echo "Usage: bash train_dpo.sh /path/to/sft_merged_model"
    exit 1
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
NCCL_TIMEOUT=7200 \
NCCL_DEBUG=WARN \
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
DS_SKIP_CUDA_CHECK=1 \
swift rlhf \
    --rlhf_type                   dpo \
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
    --ddp_timeout                 7200 \
    --beta                        ${BETA}
