#!/bin/sh
if [ -z "${BASH_VERSION:-}" ]; then
	exec bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_NAME="grpo_stable_zero2_qlora_len4096_comp1024_gen2"
export SFT_ADAPTER=""

export GRPO_PRESET="stable_grpo_zero2_qlora"
export MODEL_PATH="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_sft_lora_cot_8192_v2/v0-20260319-083851/checkpoint-50/merged_model"
export DATA_DIR="${SCRIPT_DIR}/data"
export OUTPUT_DIR="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_grpo_experiments/${EXPERIMENT_NAME}"
export DEEPSPEED_CONFIG="${SCRIPT_DIR}/ds_zero2.json"
export REWARD_PLUGIN="${SCRIPT_DIR}/reward_grpo.py"

export MAX_LENGTH="4096"
export MAX_COMPLETION_LENGTH="1024"
export NUM_GENERATIONS="2"
export PER_DEVICE_TRAIN_BATCH_SIZE="1"
export GRADIENT_ACCUMULATION_STEPS="8"
export NUM_TRAIN_EPOCHS="1"
export LEARNING_RATE="5e-6"
export SAVE_STEPS="10"
export LOGGING_STEPS="5"
export LORA_RANK="64"
export LORA_ALPHA="128"

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NPROC_PER_NODE="8"

# Ensure no stale vllm env vars leak into swift's arg parser
unset VLLM_MODE
unset VLLM_SERVER_HOST
unset VLLM_SERVER_PORT
unset VLLM_SERVER_BASE_URL

echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

bash "${SCRIPT_DIR}/train_swift_grpo.sh"