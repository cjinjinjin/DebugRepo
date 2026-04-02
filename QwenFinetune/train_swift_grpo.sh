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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-30B-A3B}"
SFT_ADAPTER="${1:-${SFT_ADAPTER:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_sft_lora_cot_8192_v2/v0-20260319-083851/checkpoint-50}}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_grpo_lora_cot_v1/lora_64}"
REWARD_PLUGIN="${REWARD_PLUGIN:-${SCRIPT_DIR}/reward_grpo.py}"

GRPO_PRESET="${GRPO_PRESET:-stable_grpo_zero2_qlora}"

case "${GRPO_PRESET}" in
    stable_grpo_zero2_qlora)
        DEEPSPEED_CONFIG_DEFAULT="${SCRIPT_DIR}/ds_zero2.json"
        USE_VLLM_DEFAULT="false"
        VLLM_MODE_DEFAULT="colocate"
        LOAD_IN_4BIT_DEFAULT="true"
        DS3_GATHER_FOR_GENERATION_DEFAULT="false"
        ;;
    experimental_grpo_zero3_vllm)
        DEEPSPEED_CONFIG_DEFAULT="${SCRIPT_DIR}/ds_zero3.json"
        USE_VLLM_DEFAULT="true"
        VLLM_MODE_DEFAULT="colocate"
        LOAD_IN_4BIT_DEFAULT="false"
        DS3_GATHER_FOR_GENERATION_DEFAULT="true"
        ;;
    *)
        echo "Unknown GRPO_PRESET: ${GRPO_PRESET}" >&2
        echo "Expected one of: stable_grpo_zero2_qlora, experimental_grpo_zero3_vllm" >&2
        exit 1
        ;;
esac

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${DEEPSPEED_CONFIG_DEFAULT}}"
USE_VLLM="${USE_VLLM:-${USE_VLLM_DEFAULT}}"
VLLM_MODE="${VLLM_MODE:-${VLLM_MODE_DEFAULT}}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-${LOAD_IN_4BIT_DEFAULT}}"
DS3_GATHER_FOR_GENERATION="${DS3_GATHER_FOR_GENERATION:-${DS3_GATHER_FOR_GENERATION_DEFAULT}}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
SAVE_STEPS="${SAVE_STEPS:-10}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"

# ---------------------------------------------------------------------------
# Reward function plugin
# ---------------------------------------------------------------------------
# stable_grpo_zero2_qlora:
#   - new recommended path to try first
#   - keeps GRPO, but avoids ZeRO-3 rollout and vLLM coupling
#   - turns on QLoRA to make 30B-A3B fit more comfortably
# experimental_grpo_zero3_vllm:
#   - groups the previous ZeRO-3 and vLLM style experiments together
#   - useful only after upgrading/fixing the server environment
# ---------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

echo "GRPO_PRESET=${GRPO_PRESET}"
echo "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"
echo "USE_VLLM=${USE_VLLM}"
echo "LOAD_IN_4BIT=${LOAD_IN_4BIT}"
echo "DS3_GATHER_FOR_GENERATION=${DS3_GATHER_FOR_GENERATION}"

cmd=(
    swift rlhf
    --rlhf_type grpo
    --model "${MODEL_PATH}"
    --dataset "${DATA_DIR}/grpo_train.jsonl"
    --train_type lora
    --lora_rank 64
    --lora_alpha 128
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --lr_scheduler_type cosine
    --warmup_ratio 0.05
    --max_length "${MAX_LENGTH}"
    --max_completion_length "${MAX_COMPLETION_LENGTH}"
    --output_dir "${OUTPUT_DIR}"
    --bf16 true
    --gradient_checkpointing true
    --deepspeed "${DEEPSPEED_CONFIG}"
    --save_steps "${SAVE_STEPS}"
    --logging_steps "${LOGGING_STEPS}"
    --num_generations "${NUM_GENERATIONS}"
    --reward_funcs format_quality
    --external_plugins "${REWARD_PLUGIN}"
)

if [[ -n "${SFT_ADAPTER}" ]]; then
    cmd+=(--adapters "${SFT_ADAPTER}")
fi

if [[ "${LOAD_IN_4BIT}" == "true" ]]; then
    cmd+=(--load_in_4bit true)
fi

if [[ "$(basename "${DEEPSPEED_CONFIG}")" == "ds_zero3.json" ]]; then
    cmd+=(--ds3_gather_for_generation "${DS3_GATHER_FOR_GENERATION}")
fi

if [[ "${USE_VLLM}" == "true" ]]; then
    cmd+=(--use_vllm true --vllm_mode "${VLLM_MODE}")

    if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]]; then
        cmd+=(--vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}")
    fi
    if [[ -n "${VLLM_TENSOR_PARALLEL_SIZE:-}" ]]; then
        cmd+=(--vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}")
    fi
    if [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
        cmd+=(--vllm_max_model_len "${VLLM_MAX_MODEL_LEN}")
    fi
    if [[ "${VLLM_MODE}" == "server" ]]; then
        if [[ -n "${VLLM_SERVER_HOST:-}" ]]; then
            cmd+=(--vllm_server_host "${VLLM_SERVER_HOST}")
        fi
        if [[ -n "${VLLM_SERVER_PORT:-}" ]]; then
            cmd+=(--vllm_server_port "${VLLM_SERVER_PORT}")
        fi
    fi
fi

"${cmd[@]}"
