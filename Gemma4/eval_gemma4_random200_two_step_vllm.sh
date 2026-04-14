#!/bin/bash
# Gemma 4 Random200 inference: Two-Step + vLLM (AWQ 4-bit)
# TSV → JSONL → Two-Step vLLM inference → evaluate
#
# Usage:
#   bash Gemma4/eval_gemma4_random200_two_step_vllm.sh
#   bash Gemma4/eval_gemma4_random200_two_step_vllm.sh [model_path] [tensor_parallel_size]

set -e

CKPT_ROOT="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT"
MODEL_ID="${1:-${CKPT_ROOT}/gemma-4-26B-A4B-it-AWQ-4bit}"
TP_SIZE="${2:-1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Input TSV (try without _GPT5 first, then with)
INPUT_TSV="${ROOT_DIR}/QwenFinetune/RawData/UHRS2K_SD_Random200_0324.tsv"
if [ ! -f "${INPUT_TSV}" ]; then
    INPUT_TSV="${ROOT_DIR}/QwenFinetune/RawData/UHRS2K_SD_Random200_0324_GPT5.tsv"
fi

# Intermediate JSONL (local)
INFER_INPUT="${SCRIPT_DIR}/data/random200_infer_input.jsonl"

# Final results on vc_data mount
VC_RESULTS="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Gemma4_results"
OUTPUT_FILE="${VC_RESULTS}/gemma4_random200_two_step_vllm_eval.jsonl"
REPORT_FILE="${VC_RESULTS}/gemma4_random200_two_step_vllm_report.json"

mkdir -p "${VC_RESULTS}"
mkdir -p "$(dirname "${INFER_INPUT}")"

echo "============================================"
echo "Gemma 4 Random200 — Two-Step + vLLM (AWQ 4-bit)"
echo "============================================"
echo "Model        : ${MODEL_ID}"
echo "Input TSV    : ${INPUT_TSV}"
echo "TP size      : ${TP_SIZE}"
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

# ── Step 2: Two-Step vLLM inference ────────────────────────────────────────
echo ""
echo "[INFO] Starting Two-Step vLLM inference (TP=${TP_SIZE}) ..."
python "${SCRIPT_DIR}/inference_gemma4_two_step_vllm.py" \
    --model_id "${MODEL_ID}" \
    --input_file "${INFER_INPUT}" \
    --temperature 1.0 \
    --tensor_parallel_size "${TP_SIZE}" \
    --dtype half \
    --output_file "${OUTPUT_FILE}"

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

# ── Step 4: extract prompts for T2I ──────────────────────────────────────
T2I_FILE="${VC_RESULTS}/gemma4_random200_two_step_vllm_t2i.txt"

echo ""
echo "[INFO] Extracting prompts for T2I ..."
python "${SCRIPT_DIR}/extract_prompts_for_t2i.py" \
    --infer_file "${OUTPUT_FILE}" \
    --input_file "${INFER_INPUT}" \
    --output_file "${T2I_FILE}"

echo "[OK] T2I file saved to ${T2I_FILE}"
