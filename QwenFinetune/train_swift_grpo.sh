#!/bin/bash
# GRPO training on top of the SFT checkpoint.
#
# Purpose: regularize output format (correct <think> fields, exactly 5 <PromptN> tags,
#          no repetition) using reward-based RL — no ground-truth labels needed.
#
# Usage:
#   bash train_swift_grpo.sh [sft_adapter_path]
#
# Example:
#   bash train_swift_grpo.sh /vc_data/.../qwen3_sft_lora_cot_8192_v2/v0-20260319-083851/checkpoint-50

MODEL_PATH="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-30B-A3B"
SFT_ADAPTER="${1:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_sft_lora_cot_8192_v2/v0-20260319-083851/checkpoint-50}"
DATA_DIR="./data"
OUTPUT_DIR="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_grpo_lora_cot_v1/lora_64"

# ---------------------------------------------------------------------------
# Reward function (inline Python passed via --reward_funcs / external script)
# ---------------------------------------------------------------------------
# See reward_grpo.py for the full implementation.
# swift grpo expects: reward_funcs=external, external_plugins=./reward_grpo.py
# ---------------------------------------------------------------------------

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
NCCL_TIMEOUT=7200 \
NCCL_DEBUG=WARN \
TORCH_NCCL_BLOCKING_WAIT=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
swift rlhf \
    --rlhf_type                    grpo \
    --model                        "${MODEL_PATH}" \
    ${SFT_ADAPTER:+--adapters "${SFT_ADAPTER}"} \
    --dataset                      ${DATA_DIR}/grpo_train.jsonl \
    --train_type                   lora \
    --lora_rank                    64 \
    --lora_alpha                   128 \
    --num_train_epochs             3 \
    --per_device_train_batch_size  1 \
    --gradient_accumulation_steps  8 \
    --learning_rate                5e-6 \
    --lr_scheduler_type            cosine \
    --warmup_ratio                 0.05 \
    --max_length                   4096 \
    --max_completion_length        2048 \
    --output_dir                   ${OUTPUT_DIR} \
    --bf16                         true \
    --gradient_checkpointing       true \
    --deepspeed                    ./ds_zero3.json \
    --save_steps                   10 \
    --logging_steps                5 \
    --num_generations              4 \
    --use_vllm                     true \
    --vllm_mode                    colocate \
    --vllm_tensor_parallel_size    8 \
    --vllm_gpu_memory_utilization  0.6 \
    --reward_funcs                 format_quality \
    --external_plugins             ./reward_grpo.py
