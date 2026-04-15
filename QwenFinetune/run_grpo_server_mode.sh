#!/bin/sh
if [ -z "${BASH_VERSION:-}" ]; then
	exec bash "$0" "$@"
fi
set -euo pipefail

# ============================================================================
# GRPO Training with External vLLM Server (server mode)
# ============================================================================
#
# Architecture:
#   - GPUs 0,1:   vLLM rollout server (TP=2, BF16, ~30 GB/GPU)
#   - GPUs 2..7:  ZeRO-2 + QLoRA training (4-bit model + LoRA + optimizer)
#
# Requirements:
#   - ms-swift 4.1.0.dev0, vllm >= 0.8.5, torch 2.6.0+cu124
#   - Qwen3-30B-A3B merged SFT checkpoint
#
# Two-step launch:
#   Step 1 (terminal A): start vLLM rollout server on GPUs 0,1
#     CUDA_VISIBLE_DEVICES=0,1 bash start_rollout_server.sh
#
#   Step 2 (terminal B): once "Uvicorn running" appears, launch training
#     bash run_grpo_server_mode.sh
#
# Weight sync:
#   ms-swift syncs merged (base + LoRA) weights to the vLLM server after
#   each rollout via NCCL-backed tensor broadcast. For MoE models, LoRA-only
#   adapter sync (--vllm_enable_lora) is NOT supported — full weight sync
#   with merge/unmerge is used instead. This is slower but correct.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_NAME="grpo_server_zero2_qlora_len2048_comp512_gen2"

# ---------------------------------------------------------------------------
# Model & data
# ---------------------------------------------------------------------------
export MODEL_PATH="${MODEL_PATH:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_sft_lora_cot_8192_v2/v0-20260319-083851/checkpoint-50/merged_model}"
export DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_grpo_experiments/${EXPERIMENT_NAME}}"
export REWARD_PLUGIN="${REWARD_PLUGIN:-${SCRIPT_DIR}/reward_grpo.py}"
export SFT_ADAPTER=""

# ---------------------------------------------------------------------------
# GRPO preset: server mode with ZeRO-2 + QLoRA on training GPUs only
# ---------------------------------------------------------------------------
export GRPO_PRESET="stable_grpo_zero2_qlora"
export DEEPSPEED_CONFIG="${SCRIPT_DIR}/ds_zero2.json"

# Training uses GPUs 2-7 (6 GPUs); GPUs 0-1 are reserved for vLLM server
export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"
export NPROC_PER_NODE="6"

# ---------------------------------------------------------------------------
# vLLM server connection
# ---------------------------------------------------------------------------
export USE_VLLM="true"
export VLLM_MODE="server"
export VLLM_SERVER_HOST="${VLLM_SERVER_HOST:-localhost}"
export VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"
# Timeout for vLLM server responses (seconds) — generation on 30B MoE can be
# slow, especially for the initial full weight sync via NCCL.
export VLLM_SERVER_TIMEOUT="${VLLM_SERVER_TIMEOUT:-600}"

# These are ignored for external vLLM (server mode) — the server controls
# its own TP/memory. Set them anyway so train_swift_grpo.sh doesn't error.
export VLLM_TENSOR_PARALLEL_SIZE=""
export VLLM_GPU_MEMORY_UTILIZATION=""
export VLLM_MAX_MODEL_LEN=""

# QLoRA on the training side
export LOAD_IN_4BIT="true"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
export LORA_RANK="${LORA_RANK:-64}"
export LORA_ALPHA="${LORA_ALPHA:-128}"

export MAX_LENGTH="2048"
export MAX_COMPLETION_LENGTH="512"
export NUM_GENERATIONS="2"
export PER_DEVICE_TRAIN_BATCH_SIZE="1"
export GRADIENT_ACCUMULATION_STEPS="8"
export NUM_TRAIN_EPOCHS="1"
export LEARNING_RATE="5e-6"
export SAVE_STEPS="10"
export LOGGING_STEPS="1"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "============================================================"
echo "EXPERIMENT: ${EXPERIMENT_NAME}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "TRAINING GPUS: ${CUDA_VISIBLE_DEVICES}  (${NPROC_PER_NODE} processes)"
echo "VLLM SERVER:   ${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}"
echo "============================================================"
echo ""
echo "Make sure the vLLM rollout server is already running:"
echo "  CUDA_VISIBLE_DEVICES=0,1 bash ${SCRIPT_DIR}/start_rollout_server.sh"
echo ""

bash "${SCRIPT_DIR}/train_swift_grpo.sh"
