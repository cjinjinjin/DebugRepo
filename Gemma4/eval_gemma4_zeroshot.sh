#!/bin/bash
# Gemma 4 zero-shot evaluation: 8-GPU data-parallel inference (no-think mode)
#
# Usage:
#   bash Gemma4/eval_gemma4_zeroshot.sh
#   bash Gemma4/eval_gemma4_zeroshot.sh [model_path]

MODEL_ID="${1:-./gemma-4-26B-A4B-it}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_DATA="${ROOT_DIR}/QwenFinetune/data/dpo_combined_eval_cot.jsonl"
OUTPUT_FILE="${SCRIPT_DIR}/results/gemma4_zeroshot_eval.jsonl"
REPORT_FILE="${SCRIPT_DIR}/results/gemma4_zeroshot_report.json"

echo "============================================"
echo "Model    : ${MODEL_ID}"
echo "Eval data: ${EVAL_DATA}"
echo "Output   : ${OUTPUT_FILE}"
echo "Report   : ${REPORT_FILE}"
echo "============================================"

# ── Step 0: generate combined eval data if missing ──────────────────────────
if [ ! -f "${EVAL_DATA}" ]; then
    echo "[INFO] ${EVAL_DATA} not found, generating ..."
    cd "${ROOT_DIR}/QwenFinetune" && python combine_dpo_data.py && cd "${ROOT_DIR}"
fi

# ── Step 1: 8-GPU parallel inference (no-think mode) ────────────────────────
python "${SCRIPT_DIR}/inference_gemma4_multi_gpu.py" \
    --model_id "${MODEL_ID}" \
    --input_file "${EVAL_DATA}" \
    --output_file "${OUTPUT_FILE}" \
    --num_gpus 8 \
    --max_new_tokens 2048 \
    --no_think

if [ $? -ne 0 ]; then
    echo "[ERROR] Inference failed. Exiting."
    exit 1
fi

# ── Step 2: evaluation ──────────────────────────────────────────────────────
echo ""
echo "Inference done. Running evaluate.py ..."

python "${ROOT_DIR}/QwenFinetune/evaluate.py" \
    --generated_file "${OUTPUT_FILE}" \
    --report_file "${REPORT_FILE}"

echo ""
echo "Report saved to ${REPORT_FILE}"
