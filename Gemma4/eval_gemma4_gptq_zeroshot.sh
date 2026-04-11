#!/bin/bash
# Gemma 4 GPTQ 4-bit zero-shot evaluation on DPO 196 samples
# Compares GPTQ quantized model against BF16 results (95.9% compliant, avg 89.9 words)
#
# Usage:
#   bash Gemma4/eval_gemma4_gptq_zeroshot.sh
#   bash Gemma4/eval_gemma4_gptq_zeroshot.sh [model_path]

CKPT_ROOT="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT"
MODEL_ID="${1:-./gemma-4-26B-A4B-it-GPTQ-Int4}"
PROCESSOR_ID="${CKPT_ROOT}/gemma-4-26B-A4B-it"
# Remote paths (uncomment when running on vc_data):
# MODEL_ID="${1:-${CKPT_ROOT}/gemma-4-26B-A4B-it-GPTQ-Int4}"

# If local path doesn't exist, try vc_data
if [ ! -d "${MODEL_ID}" ]; then
    MODEL_ID="${CKPT_ROOT}/gemma-4-26B-A4B-it-GPTQ-Int4"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_DATA_COMBINED="${ROOT_DIR}/QwenFinetune/data/dpo_combined_eval_cot.jsonl"

# Final results on vc_data mount
VC_RESULTS="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Gemma4_results"
OUTPUT_FILE="${VC_RESULTS}/gemma4_gptq_zeroshot_eval.jsonl"
REPORT_FILE="${VC_RESULTS}/gemma4_gptq_zeroshot_report.json"

mkdir -p "${VC_RESULTS}"

# ── Step 0: generate combined eval data if missing ─────────────────────────
EVAL_DATA="${EVAL_DATA_COMBINED}"
if [ ! -f "${EVAL_DATA}" ]; then
    echo "[INFO] Combined eval not found, generating ..."
    cd "${ROOT_DIR}/QwenFinetune"
    python prepare_dpo_format.py && python combine_dpo_data.py
    cd "${ROOT_DIR}"
    if [ ! -f "${EVAL_DATA}" ]; then
        echo "[ERROR] Failed to generate ${EVAL_DATA}"
        exit 1
    fi
fi

echo "============================================"
echo "Model    : ${MODEL_ID} (GPTQ 4-bit)"
echo "Eval data: ${EVAL_DATA}"
echo "Output   : ${OUTPUT_FILE}"
echo "Report   : ${REPORT_FILE}"
echo "============================================"

# Verify model exists
if [ ! -d "${MODEL_ID}" ]; then
    echo "[ERROR] Model not found: ${MODEL_ID}"
    echo "  Download first: HF_TOKEN=hf_xxx bash Gemma4/download_model_gptq.sh"
    exit 1
fi

# ── Step 1: 8-GPU parallel inference (no-think mode) ───────────────────────
# GPTQ 4-bit ~13GB per replica, fits easily on A100-80GB
# Use --processor_id to load processor from original BF16 model if needed
python "${SCRIPT_DIR}/inference_gemma4_multi_gpu.py" \
    --model_id "${MODEL_ID}" \
    --processor_id "${PROCESSOR_ID}" \
    --input_file "${EVAL_DATA}" \
    --output_file "${OUTPUT_FILE}" \
    --num_gpus 8 \
    --max_new_tokens 2048 \
    --no_think

if [ $? -ne 0 ]; then
    echo "[ERROR] Inference failed. Exiting."
    exit 1
fi

# ── Step 2: evaluation ─────────────────────────────────────────────────────
echo ""
echo "Inference done. Running evaluate.py ..."

python "${ROOT_DIR}/QwenFinetune/evaluate.py" \
    --generated_file "${OUTPUT_FILE}" \
    --report_file "${REPORT_FILE}"

echo ""
echo "Report saved to ${REPORT_FILE}"
