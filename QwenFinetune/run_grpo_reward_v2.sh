#!/bin/sh
if [ -z "${BASH_VERSION:-}" ]; then
	exec bash "$0" "$@"
fi
set -euo pipefail

# GRPO training with reward v2 (format + content quality).
#
# Based on comp2048 config (ZeRO-3 + BF16 + no vLLM, 8×A100-80GB).
# Key change: reward_grpo.py is now v2 which penalizes empty-shell
# outputs and rewards content quality (min length, CoT fields, descriptiveness).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_NAME="grpo_reward_v2_comp2048"

export SFT_ADAPTER=""
export GRPO_PRESET="stable_grpo_zero2_qlora"
export MODEL_PATH="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_sft_lora_cot_8192_v2/v0-20260319-083851/checkpoint-50/merged_model"
export DATA_DIR="${SCRIPT_DIR}/data"
export OUTPUT_DIR="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_grpo_experiments/${EXPERIMENT_NAME}"
export DEEPSPEED_CONFIG="zero3"
export USE_VLLM="false"
unset VLLM_MODE
unset VLLM_SERVER_HOST
unset VLLM_SERVER_PORT
export LOAD_IN_4BIT="false"
export REWARD_PLUGIN="${SCRIPT_DIR}/reward_grpo.py"

export LORA_RANK="16"
export LORA_ALPHA="32"

export MAX_LENGTH="4096"
export MAX_COMPLETION_LENGTH="2048"
export NUM_GENERATIONS="2"
export PER_DEVICE_TRAIN_BATCH_SIZE="1"
export GRADIENT_ACCUMULATION_STEPS="8"
export NUM_TRAIN_EPOCHS="1"
export LEARNING_RATE="5e-6"
export SAVE_STEPS="1"
export LOGGING_STEPS="1"

echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "REWARD_PLUGIN=${REWARD_PLUGIN} (v2: format + content quality)"

bash "${SCRIPT_DIR}/train_swift_grpo.sh"
