#!/bin/sh
if [ -z "${BASH_VERSION:-}" ]; then
	exec bash "$0" "$@"
fi
set -euo pipefail

echo "[DEPRECATED] This experiment is kept only for reference." >&2
echo "[DEPRECATED] Current server findings indicate ZeRO-3 + vLLM colocate is not a viable default path." >&2
echo "[DEPRECATED] Prefer run_grpo_stable_canary.sh first." >&2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_NAME="grpo_experimental_zero3_vllm_len4096_comp2048_gen2"
SFT_ADAPTER="${1:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_sft_lora_cot_8192_v2/v0-20260319-083851/checkpoint-50}"

export GRPO_PRESET="experimental_grpo_zero3_vllm"
export MODEL_PATH="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-30B-A3B"
export DATA_DIR="${SCRIPT_DIR}/data"
export OUTPUT_DIR="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_grpo_experiments/${EXPERIMENT_NAME}"
export DEEPSPEED_CONFIG="${SCRIPT_DIR}/ds_zero3.json"
export REWARD_PLUGIN="${SCRIPT_DIR}/reward_grpo.py"

export USE_VLLM="true"
export VLLM_MODE="colocate"
export VLLM_GPU_MEMORY_UTILIZATION="0.2"
export VLLM_TENSOR_PARALLEL_SIZE="1"
export LOAD_IN_4BIT="false"
export DS3_GATHER_FOR_GENERATION="true"

export MAX_LENGTH="4096"
export MAX_COMPLETION_LENGTH="2048"
export NUM_GENERATIONS="2"
export PER_DEVICE_TRAIN_BATCH_SIZE="1"
export GRADIENT_ACCUMULATION_STEPS="8"
export NUM_TRAIN_EPOCHS="1"
export LEARNING_RATE="5e-6"
export SAVE_STEPS="10"
export LOGGING_STEPS="5"

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NPROC_PER_NODE="8"

echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

bash "${SCRIPT_DIR}/train_swift_grpo.sh" "${SFT_ADAPTER}"