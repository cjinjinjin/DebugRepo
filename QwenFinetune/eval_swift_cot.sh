#!/bin/bash
# Evaluate the fine-tuned CoT model using swift infer (batch inference)
# then run evaluate.py on the results.
#
# Usage:
#   bash eval_swift_cot.sh [checkpoint_dir]
#
# Example:
#   bash eval_swift_cot.sh /vc_data/.../checkpoint-6
#
# If no checkpoint_dir is given, ADAPTER_PATH below is used.

MODEL_PATH="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-30B-A3B"
ADAPTER_PATH="${1:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_sft_lora_cot/v8-20260306-060130/checkpoint-6}"
DATA_DIR="./data"
RESULTS_DIR="./results"
RESULT_FILE="${RESULTS_DIR}/eval_swift_output.jsonl"
REPORT_FILE="${RESULTS_DIR}/eval_report.json"

mkdir -p "${RESULTS_DIR}"

echo "============================================"
echo "Model    : ${MODEL_PATH}"
echo "Adapter  : ${ADAPTER_PATH}"
echo "Val data : ${DATA_DIR}/sft_eval_cot.jsonl"
echo "Output   : ${RESULT_FILE}"
echo "============================================"

# ── Step 1: batch inference ──────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift infer \
    --model        "${MODEL_PATH}" \
    --adapters     "${ADAPTER_PATH}" \
    --dataset      "${DATA_DIR}/sft_eval_cot.jsonl" \
    --max_length   4096 \
    --bf16         true \
    --result_path  "${RESULT_FILE}"

echo ""
echo "Inference done. Running evaluate.py ..."

# ── Locate the actual output file ────────────────────────────────────────────
# swift infer may ignore --result_path and write to its own output dir.
# Search for the most recently modified .jsonl under common locations.
if [ ! -f "${RESULT_FILE}" ]; then
    echo "[WARN] ${RESULT_FILE} not found. Searching for swift output ..."
    FOUND=$(find . /tmp ~/ms-image-quality-filters-aether-module-main \
        -name "*.jsonl" -newer "${DATA_DIR}/sft_eval_cot.jsonl" \
        -not -path "*/data/*" \
        2>/dev/null | head -5)
    echo "Candidate files:"
    echo "${FOUND}"
    # Pick the most recently modified one
    LATEST=$(find . /tmp ~/ms-image-quality-filters-aether-module-main \
        -name "*.jsonl" -newer "${DATA_DIR}/sft_eval_cot.jsonl" \
        -not -path "*/data/*" \
        2>/dev/null | xargs ls -t 2>/dev/null | head -1)
    if [ -n "${LATEST}" ]; then
        echo "[INFO] Using: ${LATEST}"
        RESULT_FILE="${LATEST}"
    else
        echo "[ERROR] No swift output file found. Check swift infer logs above."
        exit 1
    fi
fi

# ── Step 2: evaluation ───────────────────────────────────────────────────────
EVAL_ARGS="--generated_file ${RESULT_FILE} --report_file ${REPORT_FILE} --gt_file ${DATA_DIR}/sft_eval_cot.jsonl"

# Uncomment to enable LLM-as-Judge (requires OPENAI_API_KEY):
# export OPENAI_API_KEY="your-key-here"
# export OPENAI_API_BASE="https://api.openai.com/v1"   # or Azure endpoint
# EVAL_ARGS="${EVAL_ARGS} --llm_judge --llm_model gpt-4o"

python evaluate.py ${EVAL_ARGS}

echo ""
echo "Report saved to ${REPORT_FILE}"
