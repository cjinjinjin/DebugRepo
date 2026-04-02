#!/bin/sh
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-30B-A3B}"
LORA_RANK="${LORA_RANK:-16}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"

echo "MODEL_PATH=${MODEL_PATH}"
echo "LORA_RANK=${LORA_RANK}"
echo "VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE}"
echo "VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}"
echo "PORT=${VLLM_SERVER_PORT}"
echo ""
echo "Waiting for 'Uvicorn running' before starting training..."

swift rollout \
    --model "${MODEL_PATH}" \
    --vllm_enable_lora true \
    --vllm_max_lora_rank "${LORA_RANK}" \
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --port "${VLLM_SERVER_PORT}"
