#!/bin/bash
# Gemma 4 Random200 inference: BF16 No-CoT + LP truncation
# TSV → JSONL → 8-GPU parallel inference → evaluate
#
# Usage:
#   bash Gemma4/eval_gemma4_random200_nocot_truncate.sh
#   bash Gemma4/eval_gemma4_random200_nocot_truncate.sh [model_path] [max_lp_chars]

set -e

MODEL_ID="${1:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma-4-26B-A4B-it}"
MAX_LP_CHARS="${2:-2000}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Input TSV
INPUT_TSV="${ROOT_DIR}/QwenFinetune/RawData/UHRS2K_SD_Random200_0324.tsv"

# Intermediate JSONL (local)
INFER_INPUT="${SCRIPT_DIR}/data/random200_infer_input.jsonl"

# Final results on vc_data mount
VC_RESULTS="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Gemma4_results"
OUTPUT_FILE="${VC_RESULTS}/gemma4_random200_nocot_truncate${MAX_LP_CHARS}_eval.jsonl"
REPORT_FILE="${VC_RESULTS}/gemma4_random200_nocot_truncate${MAX_LP_CHARS}_report.json"

mkdir -p "${VC_RESULTS}"
mkdir -p "$(dirname "${INFER_INPUT}")"

echo "============================================"
echo "Gemma 4 Random200 — BF16 No-CoT + Truncate"
echo "============================================"
echo "Model        : ${MODEL_ID}"
echo "Input TSV    : ${INPUT_TSV}"
echo "Max LP chars : ${MAX_LP_CHARS}"
echo "Output       : ${OUTPUT_FILE}"
echo "Report       : ${REPORT_FILE}"
echo "============================================"

# ── Step 0: check input TSV exists ─────────────────────────────────────────
if [ ! -f "${INPUT_TSV}" ]; then
    echo "[ERROR] Input TSV not found: ${INPUT_TSV}"
    exit 1
fi

# ── Step 1: preprocess TSV → JSONL ─────────────────────────────────────────
if [ -f "${INFER_INPUT}" ]; then
    echo "[INFO] JSONL already exists: ${INFER_INPUT}, skipping preprocessing."
else
    echo "[INFO] Preprocessing ${INPUT_TSV} -> ${INFER_INPUT} ..."
    python "${ROOT_DIR}/QwenFinetune/prepare_infer_input.py" \
        --input_tsv "${INPUT_TSV}" \
        --output_jsonl "${INFER_INPUT}"

    if [ $? -ne 0 ]; then
        echo "[ERROR] Preprocessing failed. Exiting."
        exit 1
    fi
fi

TOTAL=$(wc -l < "${INFER_INPUT}")
echo "[INFO] Input records: ${TOTAL}"

# ── Step 2: 8-GPU parallel inference (No-CoT + truncation) ────────────────
echo ""
echo "[INFO] Starting 8-GPU inference ..."
python "${SCRIPT_DIR}/inference_gemma4_multi_gpu.py" \
    --model_id "${MODEL_ID}" \
    --input_file "${INFER_INPUT}" \
    --output_file "${OUTPUT_FILE}" \
    --num_gpus 8 \
    --max_new_tokens 2048 \
    --no_think \
    --no_cot \
    --max_lp_chars "${MAX_LP_CHARS}"

if [ $? -ne 0 ]; then
    echo "[ERROR] Inference failed. Exiting."
    exit 1
fi

# ── Step 3: evaluation ─────────────────────────────────────────────────────
echo ""
echo "[INFO] Inference done. Running evaluate.py ..."

python "${ROOT_DIR}/QwenFinetune/evaluate.py" \
    --generated_file "${OUTPUT_FILE}" \
    --report_file "${REPORT_FILE}"

echo ""
echo "[OK] Report saved to ${REPORT_FILE}"
