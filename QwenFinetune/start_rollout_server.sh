#!/bin/sh
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

# ============================================================================
# Start the vLLM rollout server for GRPO server-mode training.
#
# This uses `swift rollout` which:
#   - Loads the model with load_format=dummy (no disk read — weights are
#     synced from the trainer process via NCCL at the start of training)
#   - Exposes HTTP endpoints for inference (/infer/) and weight sync
#     (/update_named_param/, /update_flattened_params/, etc.)
#   - Supports NCCL communicator init between trainer and server
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 bash start_rollout_server.sh
#
# For server-mode GRPO (2 GPUs for vLLM, 6 for training):
#   Terminal A:  CUDA_VISIBLE_DEVICES=0,1 bash start_rollout_server.sh
#   Terminal B:  bash run_grpo_server_mode.sh
# ============================================================================

MODEL_PATH="${MODEL_PATH:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_sft_lora_cot_8192_v2/v0-20260319-083851/checkpoint-50/merged_model}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

echo "============================================================"
echo "vLLM Rollout Server Configuration"
echo "============================================================"
echo "MODEL_PATH=${MODEL_PATH}"
echo "VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE}"
echo "VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}"
echo "PORT=${VLLM_SERVER_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "============================================================"
echo ""
echo "Waiting for 'Uvicorn running' before starting training..."
echo ""

swift rollout \
    --model "${MODEL_PATH}" \
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --port "${VLLM_SERVER_PORT}"
