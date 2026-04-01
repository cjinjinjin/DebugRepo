#!/bin/bash
# Run single-GPU inference latency benchmark for the quantized model.
#
# Usage:
#   bash run_benchmark.sh [quantized_model_path]
#
# Example:
#   bash run_benchmark.sh /vc_data/.../merged_model_gptq_int4

QUANTIZED_MODEL_PATH="${1:-/vc_data//shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_dpo_lora_cot_refine/v3-20260320-155846/checkpoint-50/merged_model_gptq_int4}"
PYTHON_INFER="/home/aiscuser/.conda/envs/vllm_infer/bin/python3.10"
DATA_DIR="./data"
EVAL_DATA="${DATA_DIR}/sft_eval_cot.jsonl"
RESULTS_DIR="./results"
OUTPUT_JSON="${RESULTS_DIR}/latency_benchmark.json"

echo "============================================"
echo "Model      : ${QUANTIZED_MODEL_PATH}"
echo "GPU        : CUDA_VISIBLE_DEVICES=0 (single GPU)"
echo "Python env : vllm_infer"
echo "Prompt src : ${EVAL_DATA}"
echo "Output     : ${OUTPUT_JSON}"
echo "============================================"

if [ ! -f "${EVAL_DATA}" ]; then
    echo "[ERROR] Eval data not found: ${EVAL_DATA}"
    exit 1
fi
PROMPT_ARGS="--prompt_file ${EVAL_DATA} --sample_idx 0"

mkdir -p "${RESULTS_DIR}"

CUDA_VISIBLE_DEVICES=0 \
${PYTHON_INFER} benchmark_single_request.py \
    --model           "${QUANTIZED_MODEL_PATH}" \
    --n_runs          5 \
    --warmup          1 \
    --max_tokens      1024 \
    --max_model_len   8192 \
    --quantization    gptq \
    --enable_reasoning \
    --output_json     "${OUTPUT_JSON}" \
    ${PROMPT_ARGS}

if [ $? -ne 0 ]; then
    echo "[ERROR] Benchmark failed."
    exit 1
fi

echo ""
echo "============================================"
echo "Done. Results saved to: ${OUTPUT_JSON}"
echo "============================================"
