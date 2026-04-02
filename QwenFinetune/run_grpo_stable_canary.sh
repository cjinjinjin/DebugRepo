#!/bin/sh
if [ -z "${BASH_VERSION:-}" ]; then
	exec bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_NAME="grpo_vllm_server_qlora_canary_len2048_comp512_gen2"
SFT_ADAPTER="${1:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_sft_lora_cot_8192_v2/v0-20260319-083851/checkpoint-50}"

export GRPO_PRESET="stable_grpo_zero2_qlora"
export MODEL_PATH="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-30B-A3B"
export DATA_DIR="${SCRIPT_DIR}/data"
export OUTPUT_DIR="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_grpo_experiments/${EXPERIMENT_NAME}"
export DEEPSPEED_CONFIG="zero3"
export USE_VLLM="true"
export VLLM_MODE="server"
export VLLM_SERVER_HOST="127.0.0.1"
export VLLM_SERVER_PORT="8000"
export LOAD_IN_4BIT="true"
export REWARD_PLUGIN="${SCRIPT_DIR}/reward_grpo.py"

export MAX_LENGTH="2048"
export MAX_COMPLETION_LENGTH="512"
export NUM_GENERATIONS="2"
export PER_DEVICE_TRAIN_BATCH_SIZE="1"
export GRADIENT_ACCUMULATION_STEPS="8"
export NUM_TRAIN_EPOCHS="1"
export LEARNING_RATE="5e-6"
export SAVE_STEPS="10"
export LOGGING_STEPS="1"

echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"

bash "${SCRIPT_DIR}/train_swift_grpo.sh" "${SFT_ADAPTER}"